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
						"GraphTimingMPC: SO3Mat blocks are not supported "
						"(matches BlockPositionDelta's own unsupported-"
						"SO3Mat stub)");
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

namespace {

// Where each agent's own dims[i]-wide block starts within a concatenated x0
// (ambient) or v0 (tangent) vector -- agents need not share one width (e.g.
// a Torus-only mobile base vs. an SO3Quat end effector), so plain
// `i * splines.at(0).ambient_dim()`-style indexing silently misaligns every
// agent after the first whenever widths differ. That misalignment doesn't
// throw: it just reads/writes the WRONG slice of a big enough buffer, or
// walks past the end of one -- surfacing later, often as an unrelated
// `malloc(): unsorted double linked list corrupted` abort inside some other
// Eigen allocation entirely (confirmed via gdb), not at the actual
// out-of-bounds write.
std::vector<int> CumulativeOffsets(const std::vector<int>& dims) {
	std::vector<int> offsets(dims.size());
	int off = 0;
	for (size_t i = 0; i < dims.size(); ++i) {
		offsets[i] = off;
		off += dims[i];
	}
	return offsets;
}

// Shared core: builds segments/interactions/x_init given each agent's
// already-resolved (node_ids, wps) pair (real node ids, or -1 for a
// synthetic/traced interior point in the dense case) and already-resolved
// interactions -- both BuildProblemLayout (sparse) and
// BuildDenseProblemLayout share this; they differ only in HOW agent_wps/
// agent_node_ids/interactions get resolved before calling in.
ProblemLayout BuildProblemLayoutCore(
		const std::vector<CubicConfigurationSpline>& splines,
		const std::vector<std::vector<int>>& agent_node_ids,
		const std::vector<Eigen::MatrixXd>& agent_wps,
		const std::vector<AgentInteraction>& agent_interactions,
		const std::map<std::pair<int, int>, double>& edge_to_min_tau_map,
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<Eigen::MatrixXd>& prev_vs_list,
		const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
		const std::vector<std::vector<int>>& prev_agent_nodes_list,
		const std::map<int, int>& prev_agent_spline_length_map) {

	ProblemLayout layout;
	const int num_agents = static_cast<int>(splines.size());
	std::vector<int> ambient_dims(num_agents), tangent_dims(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		ambient_dims[i] = splines[i].ambient_dim();
		tangent_dims[i] = splines[i].tangent_dim();
	}
	const std::vector<int> ambient_offsets = CumulativeOffsets(ambient_dims);
	const std::vector<int> tangent_offsets = CumulativeOffsets(tangent_dims);

	layout.agents.resize(num_agents);

	// First pass: settle every agent's tau/v ranges (an interaction row can
	// reference any agent regardless of loop order, so every range must be
	// known before the interaction-row pass below).
	int n = 0;
	for (int i = 0; i < num_agents; ++i) {
		const int K = static_cast<int>(agent_node_ids[i].size());
		layout.agents[i].K = K;
		if (K == 0) continue;

		layout.agents[i].tau_offset = n;
		n += K;
		if (K > 1) {
			layout.agents[i].v_offset = n;
			n += (K - 1) * tangent_dims[i];
		}
	}
	layout.n = n;

	// Default initial guess: tau=10 (interior to [kTauMin, kTauMax]), v=1 --
	// same generic guess add_agent_timing_segments used. Overwritten below,
	// per row, by whichever of the two later passes actually has something
	// better to say about that row: an exact warm-start match against the
	// previous cycle's own converged solution where one exists, else a
	// distance/speed-based estimate (see that pass's own comment) -- this
	// flat value only ever survives to `RunTrustRegionSqp` for a tau row
	// neither pass could inform (in practice: no previous cycle to warm
	// start from at all, i.e. this episode's very first solve).
	layout.x_init = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < num_agents; ++i) {
		const AgentLayout& al = layout.agents[i];
		for (int j = 0; j < al.K; ++j) layout.x_init(al.tau_offset + j) = 10.0;
		if (al.K > 1) {
			for (int k = al.v_offset; k < al.v_offset + (al.K - 1) * tangent_dims[i]; ++k) {
				layout.x_init(k) = 1.0;
			}
		}
	}

	// Second pass: per-segment layout.
	for (int i = 0; i < num_agents; ++i) {
		const AgentLayout& al = layout.agents[i];
		if (al.K == 0) continue;
		const Eigen::MatrixXd& wps_i = agent_wps[i];
		const Eigen::VectorXd x0_i = x0.segment(ambient_offsets[i], ambient_dims[i]);
		const Eigen::VectorXd v0_i = v0.segment(tangent_offsets[i], tangent_dims[i]);
		const CubicConfigurationSpline& spline = splines.at(i);

		for (int j = 0; j < al.K; ++j) {
			SegmentLayout seg;
			seg.tau_idx = al.tau_offset + j;
			seg.spline = &spline;

			Eigen::VectorXd xJm1(ambient_dims[i]);
			if (j == 0) {
				xJm1 = x0_i;
				seg.v0_idx = -1;
				seg.v0_const = v0_i;
			} else {
				xJm1 = wps_i.row(j - 1).transpose();
				seg.v0_idx = al.v_offset + (j - 1) * tangent_dims[i];
			}

			const Eigen::VectorXd xJ = wps_i.row(j).transpose();
			if (j == al.K - 1) {
				seg.v1_idx = -1;
				seg.v1_const = Eigen::VectorXd::Zero(tangent_dims[i]);
			} else {
				seg.v1_idx = al.v_offset + j * tangent_dims[i];
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
		ir.min_tau = edge_to_min_tau_map.contains(key) ? edge_to_min_tau_map.at(key) : 0.0;
		layout.interactions.push_back(ir);
	}

	// Warm start: node-id-matched against the previous cycle's converged
	// solution -- same purpose as GraphTimingMPC::solve()'s own warm-start
	// block used to have (see git history): an NLP/QP solver re-solving a
	// barely-changed problem from a generic guess every cycle can land in a
	// very different local optimum cycle to cycle. Skips synthetic (-1)
	// rows in the dense case -- unlike a real graph node, a synthetic
	// traced interior point can legitimately change point-for-point between
	// cycles (the traced polyline itself shifts as x0 moves), so matching
	// one arbitrary -1 to another would seed the solver from an unrelated
	// point instead of just leaving it at the generic guess above.
	//
	// tau_matched tracks which tau rows this loop actually overwrote, so
	// the distance-based fallback below (for every row it COULDN'T match --
	// e.g. the ever-present x0 -> first-remaining-node leg, always -1
	// since x0 itself is never a graph node, or any genuine RDP-surviving
	// interior point on an obstacle-routed edge) knows which ones to leave
	// alone.
	std::vector<bool> tau_matched(n, false);
	for (int i = 0; i < num_agents; ++i) {
		if (!prev_agent_spline_length_map.contains(i)) continue;
		const AgentLayout& al = layout.agents[i];
		const std::vector<int>& new_nodes = agent_node_ids[i];
		const std::vector<int>& old_nodes = prev_agent_nodes_list[i];

		for (int j_new = 0; j_new < static_cast<int>(new_nodes.size()); ++j_new) {
			if (new_nodes[j_new] < 0) continue;
			const auto it = std::find(old_nodes.begin(), old_nodes.end(), new_nodes[j_new]);
			if (it == old_nodes.end()) continue;
			const int j_old = static_cast<int>(std::distance(old_nodes.begin(), it));

			if (j_old < prev_time_deltas_list[i].size()) {
				layout.x_init(al.tau_offset + j_new) = prev_time_deltas_list[i](j_old);
				tau_matched[al.tau_offset + j_new] = true;
			}
			if (j_old < prev_vs_list[i].rows() && j_new < al.K - 1) {
				for (int c = 0; c < tangent_dims[i]; ++c) {
					layout.x_init(al.v_offset + j_new * tangent_dims[i] + c) =
						prev_vs_list[i](j_old, c);
				}
			}
		}
	}

	// Distance/speed-based fallback for every tau row warm-start matching
	// above left untouched, in place of the flat, cycle-independent
	// tau=10.0 guess the first pass set every row to. That flat guess is
	// wrong by up to 3 orders of magnitude for a short leg (e.g. this
	// suite's ~0.3m UR5e hops, or a single control cycle's worth of
	// residual distance to a not-yet-reached node) -- confirmed
	// empirically (`ur5e_multi_waypoint_experiment.py`, see PR/commit
	// description) to let the trust-region loop wander into a materially
	// different schedule cycle to cycle purely because ITS OWN starting
	// guess for one segment (almost always the x0 -> first-remaining-node
	// leg, which can never be node-id-matched -- x0 is never a graph node)
	// was off by 3 orders of magnitude, not because the underlying problem
	// itself changed.
	//
	// Rather than hardcoding any particular speed (this solver is generic
	// across robot types/units -- meters for a mobile base, radians for a
	// joint-space arm, etc., so there's no one "reasonable m/s" constant
	// that's valid everywhere), this estimates a speed FROM the very rows
	// warm-start matching above just confirmed are trustworthy this same
	// cycle: avg_speed = mean(segment displacement / matched tau) over
	// every successfully-matched row, then applies that same speed to each
	// unmatched row's own (always-known) displacement. Falls back to the
	// original flat default only when there's nothing to learn a speed
	// from at all (e.g. the very first solve of an episode, before any
	// row has ever been matched).
	{
		double speed_sum = 0.0;
		int speed_count = 0;
		for (const auto& seg : layout.segments) {
			if (!tau_matched[seg.tau_idx]) continue;
			const double tau = layout.x_init(seg.tau_idx);
			if (tau > 1e-6) {
				speed_sum += seg.disp.norm() / tau;
				++speed_count;
			}
		}
		if (speed_count > 0) {
			const double avg_speed = speed_sum / speed_count;
			if (avg_speed > 1e-6) {
				for (const auto& seg : layout.segments) {
					if (tau_matched[seg.tau_idx]) continue;
					const double est_tau = seg.disp.norm() / avg_speed;
					layout.x_init(seg.tau_idx) = std::clamp(est_tau, kTauMin, kTauMax);
				}
			}
		}
	}

	return layout;
}

}  // namespace

ProblemLayout BuildProblemLayout(
		const GraphOfConstraints& graph,
		const std::vector<CubicConfigurationSpline>& splines,
		const std::vector<int>& remaining_vertices,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& var_assignments,
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const Eigen::VectorXd& t_by_node,
		std::vector<Eigen::MatrixXd>* wps_list_out,
		std::vector<std::vector<int>>* agent_nodes_list_out,
		const std::vector<Eigen::MatrixXd>& prev_vs_list,
		const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
		const std::vector<std::vector<int>>& prev_agent_nodes_list,
		const std::map<int, int>& prev_agent_spline_length_map) {

	const int num_agents = graph.num_agents;

	auto [parents, agent_nodes, agent_interactions] =
		graph.get_agent_paths(remaining_vertices, var_assignments, t_by_node);

	wps_list_out->resize(num_agents);
	agent_nodes_list_out->resize(num_agents);
	std::vector<Eigen::MatrixXd> agent_wps(num_agents);

	for (int i = 0; i < num_agents; ++i) {
		const std::vector<int>& agent_i_nodes = agent_nodes[i];
		(*agent_nodes_list_out)[i] = agent_i_nodes;
		const int K = static_cast<int>(agent_i_nodes.size());
		if (K == 0) continue;

		// Per-agent width/column-offset, not a shared splines.at(0) width --
		// agent_col_offset is `waypoints`' own authoritative column layout
		// (agent i's block need not start at i * ambient_dim once widths
		// differ -- see CumulativeOffsets' own comment above).
		const int ambient_dim_i = splines.at(i).ambient_dim();
		const int col_offset_i = graph.agent_col_offset(i);
		Eigen::MatrixXd wps_i(K, ambient_dim_i);
		for (int j = 0; j < K; ++j) {
			const int node = agent_i_nodes[j];
			for (int k = 0; k < ambient_dim_i; ++k) {
				wps_i(j, k) = waypoints(node, col_offset_i + k);
			}
		}
		agent_wps[i] = wps_i;
		(*wps_list_out)[i] = wps_i;
	}

	return BuildProblemLayoutCore(
		splines, *agent_nodes_list_out, agent_wps, agent_interactions,
		graph.edge_to_min_tau_map, x0, v0,
		prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
		prev_agent_spline_length_map);
}

ProblemLayout BuildDenseProblemLayout(
		const std::vector<CubicConfigurationSpline>& splines,
		const std::vector<Eigen::MatrixXd>& agent_dense_wps,
		const std::vector<std::vector<int>>& agent_dense_node_ids,
		const std::vector<AgentInteraction>& agent_interactions,
		const std::map<std::pair<int, int>, double>& edge_to_min_tau_map,
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		std::vector<Eigen::MatrixXd>* wps_list_out,
		std::vector<std::vector<int>>* agent_nodes_list_out,
		const std::vector<Eigen::MatrixXd>& prev_vs_list,
		const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
		const std::vector<std::vector<int>>& prev_agent_nodes_list,
		const std::map<int, int>& prev_agent_spline_length_map) {

	*wps_list_out = agent_dense_wps;
	*agent_nodes_list_out = agent_dense_node_ids;

	return BuildProblemLayoutCore(
		splines, agent_dense_node_ids, agent_dense_wps, agent_interactions,
		edge_to_min_tau_map, x0, v0,
		prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
		prev_agent_spline_length_map);
}

}  // namespace gn_timing
