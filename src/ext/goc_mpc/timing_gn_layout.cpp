#include "timing_gn_layout.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include <unsupported/Eigen/AutoDiff>

namespace gn_timing {

namespace {
using ADS = Eigen::AutoDiffScalar<Eigen::VectorXd>;
using ADVec = Eigen::Matrix<ADS, Eigen::Dynamic, 1>;
}  // namespace

Eigen::VectorXd CumsumWithZero(const Eigen::VectorXd& x, int n) {
	Eigen::VectorXd y(n + 1);
	double s = 0.0;
	for (int i = 0; i < n + 1; ++i) {
		y(i) = s;
		s += x(i);
	}
	return y;
}

void AssembleObjective(
		const ProblemLayout& layout,
		double acceleration_cost,
		double time_cost,
		double time_cost2,
		const Eigen::VectorXd& x,
		double* f_out,
		Eigen::VectorXd* grad_out,
		std::vector<Eigen::Triplet<double>>* hess_out) {

	double f = 0.0;
	Eigen::VectorXd grad = Eigen::VectorXd::Zero(layout.n);
	std::vector<Eigen::Triplet<double>> triplets;

	// time_cost / time_cost2: exact, no approximation.
	for (const auto& seg : layout.segments) {
		const double tau = x(seg.tau_idx);
		f += time_cost * tau + time_cost2 * tau * tau;
		grad(seg.tau_idx) += time_cost + 2.0 * time_cost2 * tau;
		if (time_cost2 > 0.0) {
			triplets.emplace_back(seg.tau_idx, seg.tau_idx, 2.0 * time_cost2);
		}
	}

	if (acceleration_cost > 0.0) {
		const double sqrt_accel = std::sqrt(acceleration_cost);

		for (const auto& seg : layout.segments) {
			const double tau_val = x(seg.tau_idx);

			for (const auto& off : seg.spline->block_offsets_) {
				if (off.type == CubicConfigurationSpline::Block::Type::SO3Mat) {
					throw std::runtime_error(
						"gn_timing::AssembleObjective: SO3Mat blocks are not "
						"supported (matches BlockPositionDelta's own "
						"unsupported-SO3Mat stub)");
				}

				const int tN = off.tangent_size;
				const bool v0_free = seg.v0_idx >= 0;
				const bool v1_free = seg.v1_idx >= 0;
				const int L = 1 + (v0_free ? tN : 0) + (v1_free ? tN : 0);

				std::vector<int> global_idx;
				global_idx.reserve(L);
				global_idx.push_back(seg.tau_idx);
				if (v0_free) {
					for (int k = 0; k < tN; ++k) {
						global_idx.push_back(seg.v0_idx + off.tangent_offset + k);
					}
				}
				if (v1_free) {
					for (int k = 0; k < tN; ++k) {
						global_idx.push_back(seg.v1_idx + off.tangent_offset + k);
					}
				}

				// Seed local AutoDiff vars: [tau, v0_blk?, v1_blk?].
				const ADS tau(tau_val, L, 0);
				ADVec v0(tN), v1(tN);
				int next = 1;
				for (int k = 0; k < tN; ++k) {
					v0(k) = v0_free
						? ADS(x(seg.v0_idx + off.tangent_offset + k), L, next++)
						: ADS(seg.v0_const(off.tangent_offset + k));
				}
				for (int k = 0; k < tN; ++k) {
					v1(k) = v1_free
						? ADS(x(seg.v1_idx + off.tangent_offset + k), L, next++)
						: ADS(seg.v1_const(off.tangent_offset + k));
				}

				const ADS tau_inv = 1.0 / tau;
				const ADS tau_inv_sqrt = sqrt(tau_inv);

				for (int k = 0; k < tN; ++k) {
					const ADS D = seg.disp(off.tangent_offset + k)
						- 0.5 * tau * (v0(k) + v1(k));
					const ADS V = v1(k) - v0(k);
					const ADS r_D = (std::sqrt(12.0) * sqrt_accel) * tau_inv * tau_inv_sqrt * D;
					const ADS r_V = sqrt_accel * tau_inv_sqrt * V;

					const ADS residuals[2] = {r_D, r_V};
					for (const ADS& r : residuals) {
						f += r.value() * r.value();
						const Eigen::VectorXd& dr = r.derivatives();
						for (int a = 0; a < L; ++a) {
							grad(global_idx[a]) += 2.0 * r.value() * dr(a);
						}
						for (int a = 0; a < L; ++a) {
							for (int b = 0; b <= a; ++b) {
								const int ga = global_idx[a], gb = global_idx[b];
								triplets.emplace_back(
									std::max(ga, gb), std::min(ga, gb),
									2.0 * dr(a) * dr(b));
							}
						}
					}
				}
			}
		}
	}

	if (f_out) *f_out = f;
	if (grad_out) *grad_out = std::move(grad);
	if (hess_out) *hess_out = std::move(triplets);
}

ProblemLayout BuildProblemLayout(
		const GraphOfConstraints& graph,
		const std::vector<CubicConfigurationSpline>& splines,
		const std::vector<int>& remaining_vertices,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& assignments,
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const Eigen::VectorXd& t_by_node,
		std::vector<Eigen::MatrixXd>* wps_list_out,
		std::vector<std::vector<int>>* agent_nodes_list_out,
		const std::vector<Eigen::MatrixXd>& prev_vs_list,
		const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
		const std::vector<std::vector<int>>& prev_agent_nodes_list,
		const std::map<int, int>& prev_agent_spline_length_map) {

	ProblemLayout layout;
	const int num_agents = graph.num_agents;
	const int ambient_dim = splines.at(0).ambient_dim();
	const int tangent_dim = splines.at(0).tangent_dim();

	auto [parents, agent_nodes, agent_interactions] =
		graph.get_agent_paths(remaining_vertices, assignments, t_by_node);

	layout.agents.resize(num_agents);
	wps_list_out->resize(num_agents);
	agent_nodes_list_out->resize(num_agents);

	// First pass: settle every agent's tau/v ranges (an interaction row can
	// reference any agent regardless of loop order, so every range must be
	// known before the interaction-row pass below).
	std::vector<Eigen::MatrixXd> wps_list(num_agents);
	int n = 0;
	for (int i = 0; i < num_agents; ++i) {
		const std::vector<int>& agent_i_nodes = agent_nodes[i];
		(*agent_nodes_list_out)[i] = agent_i_nodes;
		const int K = static_cast<int>(agent_i_nodes.size());
		layout.agents[i].K = K;
		if (K == 0) continue;

		Eigen::MatrixXd wps_i(K, ambient_dim);
		for (int j = 0; j < K; ++j) {
			const int node = agent_i_nodes[j];
			for (int k = 0; k < ambient_dim; ++k) {
				wps_i(j, k) = waypoints(node, i * ambient_dim + k);
			}
		}
		wps_list[i] = wps_i;
		(*wps_list_out)[i] = wps_i;

		layout.agents[i].tau_offset = n;
		n += K;
		if (K > 1) {
			layout.agents[i].v_offset = n;
			n += (K - 1) * tangent_dim;
		}
	}
	layout.n = n;

	// Default initial guess: tau=10 (interior to [kTauMin, kTauMax]), v=1 --
	// same generic guess add_agent_timing_segments used. Overwritten below
	// wherever a warm-start match is found.
	layout.x_init = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < num_agents; ++i) {
		const AgentLayout& al = layout.agents[i];
		for (int j = 0; j < al.K; ++j) layout.x_init(al.tau_offset + j) = 10.0;
		if (al.K > 1) {
			for (int k = al.v_offset; k < al.v_offset + (al.K - 1) * tangent_dim; ++k) {
				layout.x_init(k) = 1.0;
			}
		}
	}

	// Second pass: per-segment layout.
	for (int i = 0; i < num_agents; ++i) {
		const AgentLayout& al = layout.agents[i];
		if (al.K == 0) continue;
		const Eigen::MatrixXd& wps_i = wps_list[i];
		const Eigen::VectorXd x0_i = x0.segment(i * ambient_dim, ambient_dim);
		const Eigen::VectorXd v0_i = v0.segment(i * tangent_dim, tangent_dim);
		const CubicConfigurationSpline& spline = splines.at(i);

		for (int j = 0; j < al.K; ++j) {
			SegmentLayout seg;
			seg.tau_idx = al.tau_offset + j;
			seg.spline = &spline;

			Eigen::VectorXd xJm1(ambient_dim);
			if (j == 0) {
				xJm1 = x0_i;
				seg.v0_idx = -1;
				seg.v0_const = v0_i;
			} else {
				xJm1 = wps_i.row(j - 1).transpose();
				seg.v0_idx = al.v_offset + (j - 1) * tangent_dim;
			}

			const Eigen::VectorXd xJ = wps_i.row(j).transpose();
			if (j == al.K - 1) {
				seg.v1_idx = -1;
				seg.v1_const = Eigen::VectorXd::Zero(tangent_dim);
			} else {
				seg.v1_idx = al.v_offset + j * tangent_dim;
			}

			seg.disp = spline.PositionDelta<double>(xJ, xJm1);
			layout.segments.push_back(std::move(seg));
		}
	}

	// Interaction rows -- exactly linear, contiguous per-agent tau ranges.
	for (const AgentInteraction& p : agent_interactions) {
		InteractionRow ir;
		ir.tau_offset_i = layout.agents[p.agent_i].tau_offset;
		ir.count_i = p.agent_i_depth + 1;
		ir.tau_offset_j = layout.agents[p.agent_j].tau_offset;
		ir.count_j = p.agent_j_depth + 1;
		ir.type = p.type;
		const auto key = std::make_pair(p.node_u, p.node_v);
		ir.min_tau = graph.edge_to_min_tau_map.contains(key)
			? graph.edge_to_min_tau_map.at(key) : 0.0;
		layout.interactions.push_back(ir);
	}

	// Warm start: node-id-matched against the previous cycle's converged
	// solution -- same purpose as GraphTimingMPC::solve()'s own warm-start
	// block (see that function's comment: an NLP solver re-solving a
	// barely-changed problem from a generic guess every cycle can land in a
	// very different local optimum cycle to cycle).
	for (int i = 0; i < num_agents; ++i) {
		if (!prev_agent_spline_length_map.contains(i)) continue;
		const AgentLayout& al = layout.agents[i];
		const std::vector<int>& new_nodes = (*agent_nodes_list_out)[i];
		const std::vector<int>& old_nodes = prev_agent_nodes_list[i];

		for (int j_new = 0; j_new < static_cast<int>(new_nodes.size()); ++j_new) {
			const auto it = std::find(old_nodes.begin(), old_nodes.end(), new_nodes[j_new]);
			if (it == old_nodes.end()) continue;
			const int j_old = static_cast<int>(std::distance(old_nodes.begin(), it));

			if (j_old < prev_time_deltas_list[i].size()) {
				layout.x_init(al.tau_offset + j_new) = prev_time_deltas_list[i](j_old);
			}
			if (j_old < prev_vs_list[i].rows() && j_new < al.K - 1) {
				for (int c = 0; c < tangent_dim; ++c) {
					layout.x_init(al.v_offset + j_new * tangent_dim + c) =
						prev_vs_list[i](j_old, c);
				}
			}
		}
	}

	return layout;
}

}  // namespace gn_timing
