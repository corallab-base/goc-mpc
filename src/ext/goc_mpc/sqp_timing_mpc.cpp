#include "sqp_timing_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <qpOASES.hpp>

#include "timing_gn_layout.hpp"

using gn_timing::AgentLayout;
using gn_timing::AssembleObjective;
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

struct SqpTimingMPC::QpState {
	qpOASES::SQProblem qp;
	int n = 0, m_eff = 0;  // m_eff: max(real interaction-row count, 1) -- see solve()'s own comment
	bool initialized = false;

	QpState(int n_, int m_eff_) : qp(n_, m_eff_), n(n_), m_eff(m_eff_) {
		qp.setPrintLevel(qpOASES::PL_NONE);
	}
};

SqpTimingMPC::SqpTimingMPC(const GraphOfConstraints& graph,
			    std::vector<CubicConfigurationSpline> splines,
			    double time_cost,
			    double time_cost2,
			    double acceleration_cost,
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

	const int num_agents = _graph->num_agents;
	const int num_nodes = _graph->structure.num_nodes();
	// Assuming all the same -- same assumption GnTimingMPC/GraphTimingMPC's
	// own constructors make.
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
// needs that at the point of instantiation.
SqpTimingMPC::~SqpTimingMPC() = default;

bool SqpTimingMPC::solve(
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<int>& remaining_vertices,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& assignments,
		const Eigen::VectorXd& t_by_node) {

	_timer.Start();

	// Snapshotted before this cycle's own values overwrite them below --
	// same warm-start purpose as GnTimingMPC::solve()'s own snapshot.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	ProblemLayout layout;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	Eigen::VectorXd x;
	double f_current;
	Eigen::VectorXd grad;
	std::vector<Eigen::Triplet<double>> hess_triplets;
	try {
		layout = BuildProblemLayout(
			*_graph, *_splines, remaining_vertices, waypoints, assignments, x0, v0,
			t_by_node, &wps_list, &agent_nodes_list,
			prev_vs_list, prev_time_deltas_list, prev_agent_nodes_list,
			prev_agent_spline_length_map);
		x = layout.x_init;
		AssembleObjective(layout, _acceleration_cost, _time_cost, _time_cost2,
				   x, &f_current, &grad, &hess_triplets);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in SqpTimingMPC problem construction: "
			  << e.what() << std::endl;
		return false;
	}

	_wps_list = wps_list;
	_agent_nodes_list = agent_nodes_list;

	if (layout.n == 0) {
		_last_solve_time = _timer.Tick();
		_last_iterations = 0;
		_last_trust_radius = _initial_trust_radius;
		return true;
	}

	const int n = layout.n;
	const int m = static_cast<int>(layout.interactions.size());
	// See QpState's own comment -- qpOASES::SQProblem always gets at least
	// one (possibly trivially-inactive, all-zero-row) constraint row rather
	// than special-casing nC=0.
	const int m_eff = std::max(m, 1);

	// (Re)build the persistent QpState on the first solve() or whenever
	// (n, m) changed since last time (a node completing resizes the
	// problem) -- qpOASES::SQProblem is fixed-size once constructed. Every
	// other cycle at a stable size reuses the same instance, which is what
	// lets hotstart() actually warm-start cycle-to-cycle, not just
	// iteration-to-iteration within one solve() call.
	if (!_qp_state || _qp_state->n != n || _qp_state->m_eff != m_eff) {
		_qp_state = std::make_unique<QpState>(n, m_eff);
	}
	qpOASES::SQProblem& qp = _qp_state->qp;

	// Constraint matrix A -- fixed for the whole solve() call (every
	// interaction row is exactly linear, same +/-1 pattern every outer
	// iteration; only its RHS bounds shift with the current iterate x).
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

	double trust_radius = _initial_trust_radius;
	int iter = 0;

	for (; iter < _max_iterations; ++iter) {
		if (grad.norm() < _grad_tol) break;
		// <=, not <: the clamp below (std::max(..., _min_trust_radius))
		// can only ever land EXACTLY at the floor, never strictly below
		// it, so a strict "<" here would never fire -- silently defeating
		// this stall check and leaving the loop to burn every remaining
		// iteration re-solving noise-dominated QPs at a radius so small
		// that actual/predicted reduction are both dominated by floating-
		// point cancellation (both near the true optimum's f value),
		// making rho itself meaningless at that scale.
		if (iter > 0 && trust_radius <= _min_trust_radius) break;

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
		// construction regardless -- see this file's own header
		// comment), only the QP subproblem's conditioning.
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
		if (!_qp_state->initialized) {
			status = qp.init(H.data(), grad_q.data(), A.data(),
					  dx_lb.data(), dx_ub.data(), lbA.data(), ubA.data(), nWSR);
			_qp_state->initialized = (status == qpOASES::SUCCESSFUL_RETURN);
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
			_qp_state->initialized = false;
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
		// Without this, a converged run keeps re-solving the QP and
		// shrinking trust_radius down to its floor every single call
		// instead of recognizing convergence -- same final answer either
		// way (this is a diagnostic/efficiency fix, not a correctness one:
		// _last_iterations/_last_trust_radius were the only things that
		// were misleading), but it burns iterations for no reason.
		if (dx.norm() < 1e-9) {
			break;
		}

		double f_new;
		AssembleObjective(layout, _acceleration_cost, _time_cost, _time_cost2,
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
			trust_radius = std::max(0.25 * trust_radius, _min_trust_radius);
		} else if (rho > 0.75 && step_inf_norm > 0.9 * trust_radius) {
			trust_radius = std::min(2.0 * trust_radius, _max_trust_radius);
		}

		if (rho > 1e-8) {
			// Accept: advance x and refresh (f, grad, Hessian) at the new
			// iterate for the next loop pass.
			x += dx;
			f_current = f_new;
			AssembleObjective(layout, _acceleration_cost, _time_cost, _time_cost2,
					   x, &f_current, &grad, &hess_triplets);
		}
		// else: reject -- x/f_current/grad/hess_triplets all stay at the
		// previous (still fully valid) iterate; only trust_radius moved.
	}

	_last_iterations = iter;
	_last_trust_radius = trust_radius;
	_last_solve_time = _timer.Tick();

	const int tangent_dim = _splines->at(0).tangent_dim();
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int K = static_cast<int>(_agent_nodes_list[i].size());
		if (K == 0) continue;
		_agent_spline_length_map[i] = K + 1;

		const AgentLayout& al = layout.agents[i];
		for (int j = 0; j < K; ++j) {
			_time_deltas_list[i](j) = x(al.tau_offset + j);
		}
		if (K > 1) {
			for (int row = 0; row < K - 1; ++row) {
				for (int c = 0; c < tangent_dim; ++c) {
					_vs_list[i](row, c) = x(al.v_offset + row * tangent_dim + c);
				}
			}
		}
	}
	return true;
}

int SqpTimingMPC::get_agent_spline_length(int agent) const {
	if (!_agent_spline_length_map.contains(agent)) {
		return 0;
	} else {
		return _agent_spline_length_map.at(agent);
	}
}

std::vector<int> SqpTimingMPC::get_agent_spline_nodes(int agent) const {
	if (agent < 0 || agent > _agent_nodes_list.size()) {
		return std::vector<int>();
	} else {
		return _agent_nodes_list.at(agent);
	}
}

std::set<int> SqpTimingMPC::set_progressed_time(double delta, double tau_cutoff) {
	// Verbatim copy of GnTimingMPC::set_progressed_time's logic (itself a
	// copy of GraphTimingMPC's) -- see that function's own comment for the
	// "walk forward accumulating tau" rationale.
	std::set<int> passed_nodes;

	for (int i = 0; i < _graph->num_agents; ++i) {
		if (_agent_spline_length_map[i] > 0) {
			double cumulative_tau = 0.0;
			for (int j = 0; j < static_cast<int>(_agent_nodes_list[i].size()); ++j) {
				cumulative_tau += _time_deltas_list[i](j);
				if (delta < cumulative_tau - tau_cutoff) {
					break;
				}
				const int node_id = _agent_nodes_list[i][j];
				if (node_id >= 0) {
					passed_nodes.insert(node_id);
				}
			}
		}
	}

	return passed_nodes;
}

void SqpTimingMPC::fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				       const Eigen::VectorXd& x0,
				       const Eigen::VectorXd& v0) const {
	// Verbatim copy of GnTimingMPC::fill_cubic_splines's logic.
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

const std::vector<double> SqpTimingMPC::get_next_taus() const {
	std::vector<double> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1) {
			result.push_back(_time_deltas_list[i](0));
		}
	}
	return result;
}

const std::vector<int> SqpTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);
		if (spline_length_i > 1 && !_agent_nodes_list[i].empty()) {
			const int node = _agent_nodes_list[i].at(0);
			if (node >= 0) {
				result.push_back(node);
			}
		}
	}
	return result;
}
