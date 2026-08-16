#include "admm_short_path_mpc.hpp"

#include <algorithm>
#include <utility>

#include "obstacle_projection.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

namespace {

// A single quadratic-cost term's coefficient pattern: `weight * (sum_i
// coeff_i * s[idx_i] - target)^2`, where `s` is the per-axis state vector
// [p_0, v_0, p_1, v_1, ..., p_{H-1}, v_{H-1}]. Every term used by this
// solver (tracking, velocity-tracking, acceleration, obstacle proximal
// pulls) has this exact shape -- see build_hessian/build_rhs's own
// comments for the actual per-term coefficients and their derivation.
using Entry = std::pair<int, double>;

void add_term_h(Eigen::MatrixXd* H, const std::vector<Entry>& a, double weight) {
	for (const auto& [i, ci] : a) {
		for (const auto& [j, cj] : a) {
			(*H)(i, j) += weight * ci * cj;
		}
	}
}

void add_term_g(Eigen::VectorXd* g, const std::vector<Entry>& a, double target, double weight) {
	for (const auto& [i, ci] : a) {
		(*g)(i) += weight * target * ci;
	}
}

int idx_p(int i) { return 2 * i; }
int idx_v(int i) { return 2 * i + 1; }

// Builds the (2*H x 2*H) Hessian for ONE spatial axis of ONE agent's smooth
// cost (tracking + velocity-tracking + acceleration-smoothing, identical to
// GraphShortPathMPC's items 1/1b/2 -- see graph_short_path_mpc.cpp) plus a
// UNIFORM diagonal boost of `rho * num_candidates` at every position index,
// representing every registered obstacle's ADMM proximal pull applying at
// every horizon step (see this file's own top comment for why that's the
// right simplification here). This matrix is IDENTICAL across every axis
// of a given agent (only the right-hand side, built by build_rhs below,
// differs per axis) and across every ADMM outer iteration within one
// solve() call (candidates/rho are fixed for the whole call) -- callers
// factorize it exactly once and reuse that factorization throughout.
//
// Acceleration term derivation: GraphShortPathMPC's acc_norm term is
// ||a6_tau + b2||^2 where (per axis, tau = time_per_step):
//   a6_tau = (6/tau^2)(-2(xK-xKm1) + tau(vK+vKm1))
//   b2     = (2/tau^2)( 3(xK-xKm1) - tau(vK+2 vKm1))
// which simplifies (see this file's own derivation notes, verified against
// GraphShortPathMPC's no-obstacle output -- see the Python cross-check) to:
//   a6_tau + b2 = -(6/tau^2)(xK - xKm1) + (2/tau)(2 vK + vKm1)
// a LINEAR combination of (xK, xKm1, vK, vKm1) with no cross terms across
// axes -- exactly the (a . s - target)^2 shape this solver accumulates
// term by term.
Eigen::MatrixXd build_hessian(int num_steps, double tau, int num_candidates, double rho) {
	const int n = 2 * num_steps;
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	const double tau2 = tau * tau;

	for (int i = 0; i < num_steps; ++i) {
		add_term_h(&H, {{idx_p(i), 1.0}}, 1.0);  // tracking
		add_term_h(&H, {{idx_v(i), 1.0}}, 1.0);  // velocity tracking
		if (i == 0) {
			add_term_h(&H, {{idx_p(0), -6.0 / tau2}, {idx_v(0), 4.0 / tau}}, 1.0);
		} else {
			add_term_h(&H,
				   {{idx_p(i), -6.0 / tau2}, {idx_p(i - 1), 6.0 / tau2},
				    {idx_v(i), 4.0 / tau}, {idx_v(i - 1), 2.0 / tau}},
				   1.0);
		}
		if (num_candidates > 0) {
			add_term_h(&H, {{idx_p(i), 1.0}}, rho * num_candidates);
		}
	}
	return H;
}

// Right-hand side for ONE spatial axis `k` of one agent -- same term
// structure as build_hessian, but each term's TARGET is axis-specific
// (reference position/velocity, x0/v0, and the current ADMM z-u targets
// for whichever candidates are close enough to matter). `z`/`u` are
// indexed [candidate][step], each a workspace_dim-vector -- only
// component `k` of each is used here, but they were computed (and will be
// updated) as full vectors by the z-update, since projection genuinely
// couples all workspace axes together (unlike everything else in this
// per-axis solve).
Eigen::VectorXd build_rhs(int k, int num_steps, double tau, const Eigen::VectorXd& ref_p_axis,
			   const Eigen::VectorXd& ref_v_axis, double x0_k, double v0_k,
			   int workspace_dim, double rho,
			   const std::vector<std::vector<Eigen::VectorXd>>& z,
			   const std::vector<std::vector<Eigen::VectorXd>>& u) {
	const int n = 2 * num_steps;
	Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
	const double tau2 = tau * tau;
	const int num_candidates = static_cast<int>(z.size());

	for (int i = 0; i < num_steps; ++i) {
		add_term_g(&g, {{idx_p(i), 1.0}}, ref_p_axis(i), 1.0);
		add_term_g(&g, {{idx_v(i), 1.0}}, ref_v_axis(i), 1.0);
		if (i == 0) {
			// See build_hessian's doc comment for the simplified
			// acceleration expression; x0/v0 are CONSTANTS (not decision
			// variables) here, so they fold into this term's target
			// rather than its coefficient pattern -- target = -(the
			// constant offset), see this file's derivation notes.
			const double target0 = -(6.0 / tau2) * x0_k - (2.0 / tau) * v0_k;
			add_term_g(&g, {{idx_p(0), -6.0 / tau2}, {idx_v(0), 4.0 / tau}}, target0, 1.0);
		} else {
			add_term_g(&g,
				   {{idx_p(i), -6.0 / tau2}, {idx_p(i - 1), 6.0 / tau2},
				    {idx_v(i), 4.0 / tau}, {idx_v(i - 1), 2.0 / tau}},
				   0.0, 1.0);
		}
		if (k < workspace_dim) {
			for (int m = 0; m < num_candidates; ++m) {
				const double target = z[m][i](k) - u[m][i](k);
				add_term_g(&g, {{idx_p(i), 1.0}}, target, rho);
			}
		}
	}
	return g;
}

// Candidate/project_out/gather_candidates (obstacle-proximity geometry) now
// live in obstacle_projection.hpp, shared with SqpShortPathMPC's own
// safety-projection pass -- see that header for the sphere/box math.

}  // namespace

AdmmShortPathMPC::AdmmShortPathMPC(const GraphOfConstraints& graph, unsigned int num_steps,
				    unsigned int num_agents, unsigned int dim, double time_per_step,
				    const ObstacleSet& obstacles, double rho,
				    unsigned int num_iterations, double point_cloud_query_margin)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step),
	  _obstacles(&obstacles),
	  _rho(rho),
	  _num_iterations(num_iterations),
	  _point_cloud_query_margin(point_cloud_query_margin) {
	_times = Eigen::VectorXd(_num_steps);
	for (unsigned int i = 0; i < _num_steps; ++i) {
		_times(i) = (i + 1) * _time_per_step;
	}
	_points = Eigen::MatrixXd::Zero(_num_steps, _num_agents * _dim);
	_vels = Eigen::MatrixXd::Zero(_num_steps, _num_agents * _dim);
}

bool AdmmShortPathMPC::solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& v0,
			      const Eigen::VectorXi& /*var_assignments*/,
			      const std::vector<int>& /*remaining_vertices*/,
			      const std::vector<CubicConfigurationSpline>& references) {
	_timer.Start();
	try {
		const int H = static_cast<int>(_num_steps);
		const int a_dim = references.at(0).ambient_dim();
		const int t_dim = references.at(0).tangent_dim();
		const int workspace_dim = _graph->workspace_dim;

		Eigen::MatrixXd ref_points(H, _num_agents * a_dim);
		Eigen::MatrixXd ref_velocities(H, _num_agents * t_dim);
		for (unsigned int ag = 0; ag < _num_agents; ++ag) {
			const auto& [q_ag, qdot_ag] = references[ag].eval_multiple(_times);
			ref_points.block(0, ag * a_dim, H, a_dim) = q_ag;
			ref_velocities.block(0, ag * t_dim, H, t_dim) = qdot_ag;
		}

		Eigen::MatrixXd points(H, _num_agents * _dim);
		Eigen::MatrixXd vels(H, _num_agents * _dim);

		for (unsigned int ag = 0; ag < _num_agents; ++ag) {
			const Eigen::MatrixXd ref_p_agent = ref_points.block(0, ag * _dim, H, _dim);
			const Eigen::MatrixXd ref_v_agent = ref_velocities.block(0, ag * _dim, H, _dim);
			const Eigen::VectorXd x0_agent = x0.segment(ag * _dim, _dim);
			const Eigen::VectorXd v0_agent = v0.segment(ag * _dim, _dim);

			const std::vector<Candidate> candidates = gather_candidates(
				*_obstacles, ref_p_agent.leftCols(workspace_dim), workspace_dim,
				_point_cloud_query_margin);
			const int num_candidates = static_cast<int>(candidates.size());

			// Factorize ONCE per agent -- reused across every axis AND
			// every outer ADMM iteration below (see build_hessian's own
			// comment for why this stays valid the whole time).
			const Eigen::MatrixXd H_mat =
				build_hessian(H, _time_per_step, num_candidates, _rho);
			const Eigen::LDLT<Eigen::MatrixXd> factorization(H_mat);

			// z/u state, one workspace_dim-vector per (candidate, step) --
			// see build_rhs's own comment. Initialized fresh each solve()
			// call (not warm-started across MPC cycles yet -- candidates
			// can differ cycle to cycle as the point cloud updates; a
			// future refinement could warm-start where a candidate
			// persists).
			std::vector<std::vector<Eigen::VectorXd>> z(
				num_candidates, std::vector<Eigen::VectorXd>(H));
			std::vector<std::vector<Eigen::VectorXd>> u(
				num_candidates, std::vector<Eigen::VectorXd>(H));
			for (int m = 0; m < num_candidates; ++m) {
				for (int i = 0; i < H; ++i) {
					z[m][i] = ref_p_agent.row(i).head(workspace_dim).transpose();
					u[m][i] = Eigen::VectorXd::Zero(workspace_dim);
				}
			}

			Eigen::MatrixXd agent_points(H, _dim);
			Eigen::MatrixXd agent_vels(H, _dim);

			for (unsigned int outer = 0; outer < _num_iterations; ++outer) {
				// x-update: one linear solve per axis, same cached
				// factorization every time.
				for (int k = 0; k < static_cast<int>(_dim); ++k) {
					const Eigen::VectorXd g =
						build_rhs(k, H, _time_per_step, ref_p_agent.col(k),
							  ref_v_agent.col(k), x0_agent(k), v0_agent(k),
							  workspace_dim, _rho, z, u);
					const Eigen::VectorXd s = factorization.solve(g);
					for (int i = 0; i < H; ++i) {
						agent_points(i, k) = s(idx_p(i));
						agent_vels(i, k) = s(idx_v(i));
					}
				}
				// z/u-update: closed-form projection per (step,
				// candidate) -- always well-defined.
				if (num_candidates > 0) {
					for (int i = 0; i < H; ++i) {
						const Eigen::VectorXd p_i =
							agent_points.row(i).head(workspace_dim).transpose();
						for (int m = 0; m < num_candidates; ++m) {
							z[m][i] = project_out(p_i + u[m][i], candidates[m]);
							u[m][i] += p_i - z[m][i];
						}
					}
				}
			}

			// Final safety pass: Dykstra-style ALTERNATING projection of
			// the last x-update's position against every candidate,
			// ITERATED -- guarantees the RETURNED trajectory clears every
			// registered obstacle regardless of how well the ADMM loop
			// above converged in `_num_iterations` (fixed, not
			// residual-based -- see this class's own doc comment). This is
			// the actual feasibility guarantee; the ADMM loop's job above
			// is only to make this pass's corrections small.
			//
			// A SINGLE sequential sweep through the candidates is NOT
			// enough once obstacles are close enough to interact:
			// projecting onto candidate 2 can push a point back out of
			// candidate 1's feasible region even though it satisfied
			// candidate 1 a moment ago (confirmed empirically -- a 500-
			// trial randomized stress test with 1-3 nearby obstacles per
			// trial found violations up to 0.37 units with a single
			// sweep). Repeating the sweep several times is the standard
			// fix (alternating/Dykstra-style projection onto an
			// intersection of sets) -- cheap here regardless, since each
			// full sweep is a handful of O(workspace_dim) closed-form
			// projections.
			constexpr int kSafetyPassRounds = 10;
			for (int round = 0; round < kSafetyPassRounds; ++round) {
				for (int i = 0; i < H; ++i) {
					Eigen::VectorXd p = agent_points.row(i).head(workspace_dim).transpose();
					for (const Candidate& c : candidates) {
						p = project_out(p, c);
					}
					agent_points.row(i).head(workspace_dim) = p.transpose();
				}
			}

			points.block(0, ag * _dim, H, _dim) = agent_points;
			vels.block(0, ag * _dim, H, _dim) = agent_vels;
		}

		_points = points;
		_vels = vels;
		_has_solved = true;
		_last_solve_time = _timer.Tick();
		return true;
	} catch (const std::exception& e) {
		std::cout << "Caught exception in ADMM short path solver: " << e.what() << std::endl;
		return false;
	}
}
