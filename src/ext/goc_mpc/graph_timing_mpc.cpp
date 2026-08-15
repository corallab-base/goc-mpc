#include "graph_timing_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <qpOASES.hpp>

#include "timing_gn_layout.hpp"

using gn_timing::AgentLayout;
using gn_timing::AssembleObjective;
using gn_timing::BuildDenseProblemLayout;
using gn_timing::BuildProblemLayout;
using gn_timing::CumsumWithZero;
using gn_timing::kInf;
using gn_timing::kTauMax;
using gn_timing::kTauMin;
using gn_timing::ProblemLayout;

namespace {
using RealMat = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RealVec = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, 1>;
}  // namespace

struct GraphTimingMPC::QpState {
	qpOASES::SQProblem qp;
	int n = 0, m_eff = 0;  // m_eff: max(real interaction-row count, 1) -- see solve()'s own comment
	bool initialized = false;

	QpState(int n_, int m_eff_) : qp(n_, m_eff_), n(n_), m_eff(m_eff_) {
		qp.setPrintLevel(qpOASES::PL_NONE);
	}
};

GraphTimingMPC::GraphTimingMPC(const GraphOfConstraints& graph,
				std::vector<CubicConfigurationSpline> splines,
				double time_cost,
				double time_cost2,
				double acceleration_cost,
				double energy_cost,
				double arclength_cost,
				std::vector<double> max_vel,
				std::vector<double> max_acc,
				std::vector<double> max_jerk,
				int max_iterations,
				double initial_trust_radius,
				double max_trust_radius,
				double min_trust_radius,
				double grad_tol)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _time_cost(time_cost),
	  _time_cost2(time_cost2),
	  _acceleration_cost(acceleration_cost),
	  _max_iterations(max_iterations),
	  _initial_trust_radius(initial_trust_radius),
	  _max_trust_radius(max_trust_radius),
	  _min_trust_radius(min_trust_radius),
	  _grad_tol(grad_tol) {

	// See this class's own doc comment for why these are rejected (loudly)
	// rather than silently dropped. max_jerk isn't checked at all -- it was
	// already an inert no-op in the implementation this class replaces.
	if (energy_cost != 0.0) {
		throw std::runtime_error(
			"GraphTimingMPC: energy_cost is not supported by this "
			"(trust-region SQP) implementation -- see this class's own doc "
			"comment. Pass 0.0 (the default) or omit it entirely.");
	}
	if (arclength_cost != 0.0) {
		throw std::runtime_error(
			"GraphTimingMPC: arclength_cost is not supported by this "
			"(trust-region SQP) implementation -- see this class's own doc "
			"comment. Pass 0.0 (the default) or omit it entirely.");
	}
	auto check_unbounded = [](const std::vector<double>& v, const char* name) {
		for (double val : v) {
			if (val > 0.0) {
				throw std::runtime_error(
					std::string("GraphTimingMPC: ") + name + " hard bounds are "
					"not supported by this (trust-region SQP) implementation "
					"-- see this class's own doc comment. Pass <= 0 "
					"(unbounded) for every block, or omit " + name + " entirely.");
			}
		}
	};
	check_unbounded(max_vel, "max_vel");
	check_unbounded(max_acc, "max_acc");

	const int num_agents = _graph->num_agents;
	const int num_nodes = _graph->structure.num_nodes();
	// Assuming all the same -- same assumption the implementation this
	// replaces made.
	const int ambient_dim = _splines->at(0).ambient_dim();
	const int tangent_dim = _splines->at(0).tangent_dim();

	_wps_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _wps_list[i] = Eigen::MatrixXd::Zero(num_nodes, ambient_dim);
	_vs_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _vs_list[i] = Eigen::MatrixXd::Zero(num_nodes, tangent_dim);
	_time_deltas_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) _time_deltas_list[i] = Eigen::VectorXd::Zero(num_nodes);
}

// Defined here (not defaulted in the header) because QpState is only a
// complete type in this translation unit -- unique_ptr<QpState>'s deleter
// (and, for the move constructor/assignment below, its move operations)
// need that at the point of instantiation.
GraphTimingMPC::~GraphTimingMPC() = default;
GraphTimingMPC::GraphTimingMPC(GraphTimingMPC&&) noexcept = default;
GraphTimingMPC& GraphTimingMPC::operator=(GraphTimingMPC&&) noexcept = default;

namespace {

// Shared trust-region Gauss-Newton SQP loop, given an already-built
// ProblemLayout -- solve()/solve_dense() differ only in how they build
// `layout` (see BuildProblemLayout vs. BuildDenseProblemLayout) and what
// they do with the caller-visible remaining_vertices/agent_interactions
// inputs; everything from here on (the QP loop itself, and reading the
// solution back out) is identical. Returns the final iterate `x` and
// diagnostics; storing it into the class's own per-agent buffers is the
// caller's job (see solve()/solve_dense()), since only they know whether
// those buffers need resizing (solve_dense()'s dense sequences aren't
// bounded by the graph's own node count the way solve()'s are).
struct SqpResult {
	Eigen::VectorXd x;
	int iterations = 0;
	double trust_radius = 0.0;
};

SqpResult RunTrustRegionSqp(
		const ProblemLayout& layout,
		double acceleration_cost,
		double time_cost,
		double time_cost2,
		int max_iterations,
		double initial_trust_radius,
		double max_trust_radius,
		double min_trust_radius,
		double grad_tol,
		std::unique_ptr<GraphTimingMPC::QpState>* qp_state) {

	const int n = layout.n;
	const int m = static_cast<int>(layout.interactions.size());
	// qpOASES::SQProblem always gets at least one (possibly trivially-
	// inactive, all-zero-row) constraint row rather than special-casing nC=0.
	const int m_eff = std::max(m, 1);

	// (Re)build the persistent QpState on the first call or whenever (n, m)
	// changed since last time (a node completing, or a dense re-trace
	// changing point count, resizes the problem) -- qpOASES::SQProblem is
	// fixed-size once constructed. Every other cycle at a stable size
	// reuses the same instance, which is what lets hotstart() actually
	// warm-start cycle-to-cycle, not just iteration-to-iteration within one
	// call.
	if (!*qp_state || (*qp_state)->n != n || (*qp_state)->m_eff != m_eff) {
		*qp_state = std::make_unique<GraphTimingMPC::QpState>(n, m_eff);
	}
	qpOASES::SQProblem& qp = (*qp_state)->qp;

	Eigen::VectorXd x = layout.x_init;
	double f_current;
	Eigen::VectorXd grad;
	std::vector<Eigen::Triplet<double>> hess_triplets;
	AssembleObjective(layout, acceleration_cost, time_cost, time_cost2,
			   x, &f_current, &grad, &hess_triplets);

	if (n == 0) {
		return SqpResult{x, 0, initial_trust_radius};
	}

	// Constraint matrix A -- fixed for the whole call (every interaction
	// row is exactly linear, same +/-1 pattern every outer iteration; only
	// its RHS bounds shift with the current iterate x).
	RealMat A = RealMat::Zero(m_eff, n);
	for (int r = 0; r < m; ++r) {
		const auto& ir = layout.interactions[r];
		for (int k = 0; k < ir.count_i; ++k) A(r, ir.tau_offset_i + k) += 1.0;
		for (int k = 0; k < ir.count_j; ++k) A(r, ir.tau_offset_j + k) -= 1.0;
	}

	// Per-variable box bounds (tau in [kTauMin, kTauMax]; v unbounded).
	Eigen::VectorXd var_lb = Eigen::VectorXd::Constant(n, -kInf);
	Eigen::VectorXd var_ub = Eigen::VectorXd::Constant(n, kInf);
	for (const auto& seg : layout.segments) {
		var_lb(seg.tau_idx) = kTauMin;
		var_ub(seg.tau_idx) = kTauMax;
	}

	double trust_radius = initial_trust_radius;
	int iter = 0;

	for (; iter < max_iterations; ++iter) {
		if (grad.norm() < grad_tol) break;
		// <=, not <: the clamp below (std::max(..., min_trust_radius)) can
		// only ever land EXACTLY at the floor, never strictly below it, so
		// a strict "<" here would never fire -- silently defeating this
		// stall check and leaving the loop to burn every remaining
		// iteration re-solving noise-dominated QPs at a radius so small
		// that actual/predicted reduction are both dominated by floating-
		// point cancellation (both near the true optimum's f value),
		// making rho itself meaningless at that scale.
		if (iter > 0 && trust_radius <= min_trust_radius) break;

		// Dense symmetric Gauss-Newton Hessian, mirrored from the
		// lower-triangle-only triplets AssembleObjective returns.
		RealMat H = RealMat::Zero(n, n);
		for (const auto& t : hess_triplets) {
			H(t.row(), t.col()) += static_cast<qpOASES::real_t>(t.value());
			if (t.row() != t.col()) {
				H(t.col(), t.row()) += static_cast<qpOASES::real_t>(t.value());
			}
		}
		// Tiny Tikhonov floor: a numerical-conditioning safety net for
		// qpOASES' internal Cholesky factorization on a PSD-but-not-
		// necessarily-PD Hessian -- doesn't change which point is a
		// stationary point of the true objective (H_GN is PSD by
		// construction regardless -- see this file's own header comment),
		// only the QP subproblem's conditioning.
		for (int i = 0; i < n; ++i) H(i, i) += qpOASES::real_t(1e-10);

		const RealVec dx_lb = (var_lb - x).cwiseMax(-trust_radius).cast<qpOASES::real_t>();
		const RealVec dx_ub = (var_ub - x).cwiseMin(trust_radius).cast<qpOASES::real_t>();

		RealVec lbA = RealVec::Constant(m_eff, qpOASES::real_t(-kInf));
		RealVec ubA = RealVec::Constant(m_eff, qpOASES::real_t(kInf));
		for (int r = 0; r < m; ++r) {
			const auto& ir = layout.interactions[r];
			double gi = 0.0, gj = 0.0;
			for (int k = 0; k < ir.count_i; ++k) gi += x(ir.tau_offset_i + k);
			for (int k = 0; k < ir.count_j; ++k) gj += x(ir.tau_offset_j + k);
			const double g_val = gi - gj;
			if (ir.type == AgentInteraction::Type::LESS_THAN) {
				lbA(r) = qpOASES::real_t(-kInf);
				ubA(r) = static_cast<qpOASES::real_t>(-ir.min_tau - g_val);
			} else {
				lbA(r) = ubA(r) = static_cast<qpOASES::real_t>(-g_val);
			}
		}

		const RealVec grad_q = grad.cast<qpOASES::real_t>();

		qpOASES::int_t nWSR = 200;
		qpOASES::returnValue status;
		if (!(*qp_state)->initialized) {
			status = qp.init(H.data(), grad_q.data(), A.data(),
					  dx_lb.data(), dx_ub.data(), lbA.data(), ubA.data(), nWSR);
			(*qp_state)->initialized = (status == qpOASES::SUCCESSFUL_RETURN);
		} else {
			status = qp.hotstart(H.data(), grad_q.data(), A.data(),
					      dx_lb.data(), dx_ub.data(), lbA.data(), ubA.data(), nWSR);
		}

		if (status != qpOASES::SUCCESSFUL_RETURN) {
			// The QP subproblem itself is always solvable by construction
			// (PSD Hessian, exactly-linear feasible region -- see this
			// file's own header comment), so a non-success status here
			// means the active-set homotopy struggled numerically, not
			// that no solution exists. Retry with a full re-init (discard
			// the possibly-confused warm start) at a smaller trust region
			// rather than trusting a possibly-garbage primal solution.
			trust_radius *= 0.25;
			(*qp_state)->initialized = false;
			continue;
		}

		RealVec dx_q(n);
		qp.getPrimalSolution(dx_q.data());
		const Eigen::VectorXd dx = dx_q.cast<double>();

		// The QP found essentially no improving direction -- dx (approx.)
		// solving the trust-region model to optimality at dx=0 IS the
		// first-order stationarity condition, and is a better-scaled
		// convergence signal than a raw gradient norm for this problem
		// (tau/v have an O(1-10) physical scale, so a tiny STEP is
		// meaningful regardless of the objective's absolute curvature).
		if (dx.norm() < 1e-9) {
			break;
		}

		double f_new;
		AssembleObjective(layout, acceleration_cost, time_cost, time_cost2,
				   x + dx, &f_new, nullptr, nullptr);

		const Eigen::MatrixXd H_d = H.cast<double>();
		const double predicted_reduction = -(grad.dot(dx) + 0.5 * dx.dot(H_d * dx));
		const double actual_reduction = f_current - f_new;
		// predicted_reduction ~ 0 means the quadratic model already thinks
		// x is stationary within this trust region -- treat as converged
		// (rho=1) rather than dividing by ~0.
		const double rho = (predicted_reduction > 1e-14)
			? actual_reduction / predicted_reduction : 1.0;

		const double step_inf_norm = dx.lpNorm<Eigen::Infinity>();
		if (rho < 0.25) {
			trust_radius = std::max(0.25 * trust_radius, min_trust_radius);
		} else if (rho > 0.75 && step_inf_norm > 0.9 * trust_radius) {
			trust_radius = std::min(2.0 * trust_radius, max_trust_radius);
		}

		if (rho > 1e-8) {
			// Accept: advance x and refresh (f, grad, Hessian) at the new
			// iterate for the next loop pass.
			x += dx;
			f_current = f_new;
			AssembleObjective(layout, acceleration_cost, time_cost, time_cost2,
					   x, &f_current, &grad, &hess_triplets);
		}
		// else: reject -- x/f_current/grad/hess_triplets all stay at the
		// previous (still fully valid) iterate; only trust_radius moved.
	}

	return SqpResult{x, iter, trust_radius};
}

// Writes `layout`/`result` into the class's own per-agent buffers, given
// `_agent_nodes_list`/`_wps_list` are already set by the caller. Always
// resizes _time_deltas_list[i]/_vs_list[i] to exactly this cycle's K/K-1
// rows rather than writing into whatever was there before -- solve_dense()
// routinely needs MORE rows than solve()'s own num_nodes-bounded buffers
// ever did, and with Eigen's own bounds-assertions compiled out in this
// build's RelWithDebInfo config, writing into a stale, too-small buffer
// would silently corrupt memory instead of throwing.
void StoreResult(
		const GraphOfConstraints& graph,
		const ProblemLayout& layout,
		const SqpResult& result,
		int tangent_dim,
		std::vector<Eigen::MatrixXd>* vs_list,
		std::vector<Eigen::VectorXd>* time_deltas_list,
		std::map<int, int>* agent_spline_length_map,
		const std::vector<std::vector<int>>& agent_nodes_list) {

	for (int i = 0; i < graph.num_agents; ++i) {
		const int K = static_cast<int>(agent_nodes_list[i].size());
		if (K == 0) continue;
		(*agent_spline_length_map)[i] = K + 1;

		const AgentLayout& al = layout.agents[i];
		(*time_deltas_list)[i] = Eigen::VectorXd::Zero(K);
		for (int j = 0; j < K; ++j) {
			(*time_deltas_list)[i](j) = result.x(al.tau_offset + j);
		}
		(*vs_list)[i] = Eigen::MatrixXd::Zero(std::max(K - 1, 0), tangent_dim);
		if (K > 1) {
			for (int row = 0; row < K - 1; ++row) {
				for (int c = 0; c < tangent_dim; ++c) {
					(*vs_list)[i](row, c) = result.x(al.v_offset + row * tangent_dim + c);
				}
			}
		}
	}
}

}  // namespace

bool GraphTimingMPC::solve(
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<int>& remaining_vertices,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& assignments,
		const Eigen::VectorXd& t_by_node) {

	_timer.Start();

	// Snapshotted before this cycle's own values overwrite them below --
	// lets BuildProblemLayout's warm start seed this cycle's initial guess
	// from the last cycle's converged solution.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	ProblemLayout layout;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	try {
		layout = BuildProblemLayout(
			*_graph, *_splines, remaining_vertices, waypoints, assignments, x0, v0,
			t_by_node, &wps_list, &agent_nodes_list,
			prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
			prev_agent_spline_length_map);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in GraphTimingMPC problem construction: "
			  << e.what() << std::endl;
		return false;
	}

	_wps_list = wps_list;
	_agent_nodes_list = agent_nodes_list;

	const SqpResult result = RunTrustRegionSqp(
		layout, _acceleration_cost, _time_cost, _time_cost2,
		_max_iterations, _initial_trust_radius, _max_trust_radius,
		_min_trust_radius, _grad_tol, &_qp_state);

	_last_iterations = result.iterations;
	_last_trust_radius = result.trust_radius;
	_last_solve_time = _timer.Tick();

	StoreResult(*_graph, layout, result, _splines->at(0).tangent_dim(),
		    &_vs_list, &_time_deltas_list, &_agent_spline_length_map, _agent_nodes_list);
	return true;
}

bool GraphTimingMPC::solve_dense(
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<Eigen::MatrixXd>& agent_dense_wps,
		const std::vector<std::vector<int>>& agent_dense_node_ids,
		const std::vector<AgentInteraction>& agent_interactions) {

	_timer.Start();

	// Same warm-start purpose as solve()'s. Unlike solve()'s node-id
	// matching (skips synthetic -1 rows -- see BuildProblemLayoutCore's own
	// comment), this is handled uniformly inside BuildDenseProblemLayout
	// via the same shared core.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	ProblemLayout layout;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	try {
		layout = BuildDenseProblemLayout(
			*_splines, agent_dense_wps, agent_dense_node_ids, agent_interactions,
			_graph->edge_to_min_tau_map, x0, v0, &wps_list, &agent_nodes_list,
			prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
			prev_agent_spline_length_map);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in GraphTimingMPC dense problem construction: "
			  << e.what() << std::endl;
		return false;
	}

	_wps_list = wps_list;
	_agent_nodes_list = agent_nodes_list;

	const SqpResult result = RunTrustRegionSqp(
		layout, _acceleration_cost, _time_cost, _time_cost2,
		_max_iterations, _initial_trust_radius, _max_trust_radius,
		_min_trust_radius, _grad_tol, &_qp_state);

	_last_iterations = result.iterations;
	_last_trust_radius = result.trust_radius;
	_last_solve_time = _timer.Tick();

	StoreResult(*_graph, layout, result, _splines->at(0).tangent_dim(),
		    &_vs_list, &_time_deltas_list, &_agent_spline_length_map, _agent_nodes_list);
	return true;
}

int GraphTimingMPC::get_agent_spline_length(int agent) const {
	if (!_agent_spline_length_map.contains(agent)) {
		return 0;
	} else {
		return _agent_spline_length_map.at(agent);
	}
}

std::vector<int> GraphTimingMPC::get_agent_spline_nodes(int agent) const {
	if (agent < 0 || agent > _agent_nodes_list.size()) {
		return std::vector<int>();
	} else {
		return _agent_nodes_list.at(agent);
	}
}

std::set<int> GraphTimingMPC::set_progressed_time(double delta, double tau_cutoff) {
	// This function, instead of resolving for all the vertices and taus, as
	// above, just updates the first tau of each remaining active spline.
	std::set<int> passed_nodes;

	for (int i = 0; i < _graph->num_agents; ++i) {
		if (_agent_spline_length_map[i] > 0) {
			// Walking forward, accumulating tau, handles both a sparse
			// (real-node-only, from solve()) and dense (from solve_dense(),
			// several synthetic -1 rows ahead of each real one) node list
			// identically -- degenerates to a single-row check when every
			// row is real.
			double cumulative_tau = 0.0;
			for (int j = 0; j < static_cast<int>(_agent_nodes_list[i].size()); ++j) {
				cumulative_tau += _time_deltas_list[i](j);
				if (delta < cumulative_tau - tau_cutoff) {
					break;  // haven't reached even this row yet -- neither has any later one
				}
				const int node_id = _agent_nodes_list[i][j];
				if (node_id >= 0) {
					passed_nodes.insert(node_id);
				}
				// else: a synthetic traced interior point was passed --
				// nothing further to do for it.
			}
		}
	}

	return passed_nodes;
}

void GraphTimingMPC::fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
					 const Eigen::VectorXd& x0,
					 const Eigen::VectorXd& v0) const {
	const int a_d = splines[0]->ambient_dim();
	const int t_d = splines[0]->tangent_dim();

	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(spline_length_i, a_d);
			wps_i.row(0) = x0_i;
			wps_i.bottomRows(spline_length_i - 1) = _wps_list[i];

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(spline_length_i, t_d);
			vs_i.row(0) = v0_i;
			vs_i.block(1, 0, spline_length_i - 2, t_d) = _vs_list[i];
			vs_i.row(spline_length_i - 1).setZero();

			Eigen::VectorXd times_i = CumsumWithZero(_time_deltas_list[i], spline_length_i - 1);

			splines[i]->set(wps_i, vs_i, times_i);
		} else {
			// Dummy spline that stays at x0, and comes to a stop after 1 second.
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(2, a_d);
			wps_i.row(0) = x0_i;
			wps_i.row(1) = x0_i;

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(2, t_d);
			vs_i.row(0) = v0_i;
			vs_i.row(1).setZero();

			Eigen::VectorXd times_i(2);
			times_i << 0.0, 1.0;

			splines[i]->set(wps_i, vs_i, times_i);
		}
	}
}

const std::vector<double> GraphTimingMPC::get_next_taus() const {
	std::vector<double> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1) {
			result.push_back(_time_deltas_list[i](0));
		}
	}
	return result;
}

const std::vector<int> GraphTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1 && !_agent_nodes_list[i].empty()) {
			const int node = _agent_nodes_list[i].at(0);
			if (node >= 0) {
				result.push_back(node);
			}
			// else: first row is a synthetic traced interior point (only
			// possible after solve_dense()) -- no real "next node" to
			// report for this agent yet.
		}
	}
	return result;
}
