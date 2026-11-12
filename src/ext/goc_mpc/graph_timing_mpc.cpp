#include "graph_timing_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <optional>
#include <stdexcept>

#include <proxsuite/proxqp/dense/dense.hpp>
#include <proxsuite/proxqp/sparse/sparse.hpp>

#include "timing_gn_layout.hpp"

using gn_timing::AgentLayout;
using gn_timing::AssembleObjective;
using gn_timing::BuildDenseProblemLayout;
using gn_timing::BuildProblemLayout;
using gn_timing::CumsumWithZero;
using gn_timing::InteractionRow;
using gn_timing::kInf;
using gn_timing::kTauMax;
using gn_timing::kTauMin;
using gn_timing::ProblemLayout;
using gn_timing::VelocityConstraintRow;

namespace {
namespace psp = proxsuite::proxqp;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;
using Trip = Eigen::Triplet<double, long long>;
// proxqp treats any bound whose magnitude reaches helpers::infinite_bound
// (sqrt(DBL_MAX)) as +/-infinity.
const double kProxInf = proxsuite::helpers::infinite_bound<double>::value();
double ProxBound(double v) { return std::clamp(v, -kProxInf, kProxInf); }

// RunReducedNewton's per-tau relative trust region (|dtau_j| <= radius *
// tau_j) and convergence test (every |dtau_j| / tau_j below
// kRelativeStepTol), and the interaction violation up to which a start
// counts as feasible for it.
constexpr double kInitialRelativeRadius = 0.5;
constexpr double kMaxRelativeRadius = 10.0;
constexpr double kMinRelativeRadius = 1e-8;
constexpr double kRelativeStepTol = 1e-7;
constexpr double kFeasibilityTol = 1e-6;

// Compressed sparsity-pattern signature (outer + inner index arrays) for the
// update()-vs-init() decision in RunTrustRegionSqp.
std::vector<long long> PatternKey(const SpMat& m) {
	std::vector<long long> key(m.outerIndexPtr(), m.outerIndexPtr() + m.outerSize() + 1);
	key.insert(key.end(), m.innerIndexPtr(), m.innerIndexPtr() + m.nonZeros());
	return key;
}
}  // namespace

// Sparse ProxQP subproblem solver, hot-started across outer iterations and
// across solve() calls: update() (values only, keeps the symbolic
// factorization) + WARM_START_WITH_PREVIOUS_RESULT while the H/C sparsity
// pattern is unchanged, a full init() otherwise. The variable box is passed
// as identity inequality rows after the interaction/velocity rows (the
// sparse backend has no separate box argument).
struct GraphTimingMPC::QpState {
	psp::sparse::QP<double, long long> qp;
	int n = 0, n_in = 0;
	bool initialized = false;
	std::vector<long long> h_pattern;
	std::vector<long long> c_pattern;

	QpState(int n_, int n_in_) : qp(n_, 0, n_in_), n(n_), n_in(n_in_) {
		qp.settings.verbose = false;
		qp.settings.compute_timings = false;
		qp.settings.eps_abs = 1.0e-7;
		qp.settings.eps_rel = 0.0;
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
				double grad_tol,
				double interaction_penalty_weight)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _time_cost(time_cost),
	  _time_cost2(time_cost2),
	  _acceleration_cost(acceleration_cost),
	  _interaction_penalty_weight(interaction_penalty_weight),
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

	// Per-agent, NOT a shared splines->at(0) width -- agents need not share
	// one ambient/tangent width (see timing_gn_layout.cpp's CumulativeOffsets
	// comment for why that assumption used to silently corrupt memory).
	_wps_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i)
		_wps_list[i] = Eigen::MatrixXd::Zero(num_nodes, _splines->at(i).ambient_dim());
	_vs_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i)
		_vs_list[i] = Eigen::MatrixXd::Zero(num_nodes, _splines->at(i).tangent_dim());
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

// Sums agent i's (resp. j's) cumulative tau over `ir`'s own depth+1 range
// at `x` -- shared by the per-iteration lbA/ubA RHS below (built fresh
// every outer iteration) and TotalInteractionViolation's merit-function
// evaluator, so the two can never silently disagree about what "this row's
// current value" means.
std::pair<double, double> InteractionGiGj(const Eigen::VectorXd& x, const InteractionRow& ir) {
	double gi = 0.0, gj = 0.0;
	for (int k = 0; k < ir.count_i; ++k) gi += x(ir.tau_offset_i + k);
	for (int k = 0; k < ir.count_j; ++k) gj += x(ir.tau_offset_j + k);
	return {gi, gj};
}

// Sum of every interaction row's ACTUAL (not QP-model) constraint
// violation at `x`: LESS_THAN contributes how far gi exceeds gj - min_tau
// (0 if already satisfied); EQUAL contributes the absolute cross-agent
// timing mismatch. Used by the merit function below (actual, not
// predicted, violation at a candidate point) -- mirrors
// graph_short_path_mpc.cpp's EvaluateObstacleViolation/
// EvaluateAgentPairViolation's own role there.
// A velocity constraint row's value at `x` (sum of coeffs * x[cols]).
double VelocityRowValue(const Eigen::VectorXd& x, const VelocityConstraintRow& vr) {
	double value = 0.0;
	for (std::size_t c = 0; c < vr.cols.size(); ++c) value += vr.coeffs[c] * x(vr.cols[c]);
	return value;
}

double TotalInteractionViolation(const Eigen::VectorXd& x, const ProblemLayout& layout) {
	double violation = 0.0;
	for (const auto& ir : layout.interactions) {
		const auto [gi, gj] = InteractionGiGj(x, ir);
		const double g_val = gi - gj;
		if (ir.type == AgentInteraction::Type::LESS_THAN) {
			violation += std::max(0.0, g_val + ir.min_tau);
		} else {
			violation += std::abs(g_val);
		}
	}
	// Velocity constraint rows (layout.velocity_rows): how far a . x is
	// outside [lb, ub]
	for (const auto& vr : layout.velocity_rows) {
		const double value = VelocityRowValue(x, vr);
		violation += std::max(0.0, vr.lb - value) + std::max(0.0, value - vr.ub);
	}
	return violation;
}

// The interaction rows over the taus alone, in local tau indices: EQUAL rows
// A tau = 0, LESS_THAN rows C tau <= u.
struct TauConstraints {
	std::vector<int> cols;   // local index -> global tau column
	std::vector<int> local;  // global column -> local index (-1: not a tau)
	Eigen::MatrixXd A, C;
	Eigen::VectorXd u;
};

TauConstraints BuildTauConstraints(const ProblemLayout& layout) {
	TauConstraints tc;
	tc.local.assign(layout.n, -1);
	for (const auto& al : layout.agents)
		for (int j = 0; j < al.K; ++j) {
			tc.local[al.tau_offset + j] = static_cast<int>(tc.cols.size());
			tc.cols.push_back(al.tau_offset + j);
		}
	const int nt = static_cast<int>(tc.cols.size());
	int n_eq = 0, n_lt = 0;
	for (const auto& ir : layout.interactions)
		(ir.type == AgentInteraction::Type::EQUAL ? n_eq : n_lt)++;
	tc.A = Eigen::MatrixXd::Zero(n_eq, nt);
	tc.C = Eigen::MatrixXd::Zero(n_lt, nt);
	tc.u = Eigen::VectorXd::Zero(n_lt);
	int r_eq = 0, r_lt = 0;
	for (const auto& ir : layout.interactions) {
		const bool eq = ir.type == AgentInteraction::Type::EQUAL;
		auto row = eq ? tc.A.row(r_eq) : tc.C.row(r_lt);
		for (int k = 0; k < ir.count_i; ++k) row(tc.local[ir.tau_offset_i + k]) += 1.0;
		for (int k = 0; k < ir.count_j; ++k) row(tc.local[ir.tau_offset_j + k]) -= 1.0;
		if (eq) {
			++r_eq;
		} else {
			tc.u(r_lt++) = -ir.min_tau;
		}
	}
	return tc;
}

// min 1/2 z'Hz + g'z  s.t.  A z = b_eq, C z <= u_lt, lo <= z <= hi, over the
// taus (dense ProxQP); nullopt if it doesn't solve.
std::optional<Eigen::VectorXd> SolveTauQp(const Eigen::MatrixXd& H, const Eigen::VectorXd& g,
					  const TauConstraints& tc, const Eigen::VectorXd& b_eq,
					  const Eigen::VectorXd& u_lt, const Eigen::VectorXd& lo,
					  const Eigen::VectorXd& hi) {
	const int nt = static_cast<int>(H.rows()), n_eq = static_cast<int>(tc.A.rows()),
		  n_lt = static_cast<int>(tc.C.rows());
	Eigen::MatrixXd C(n_lt + nt, nt);
	C << tc.C, Eigen::MatrixXd::Identity(nt, nt);
	Eigen::VectorXd l(n_lt + nt), u(n_lt + nt);
	l << Eigen::VectorXd::Constant(n_lt, -kProxInf), lo;
	u << u_lt, hi;
	psp::dense::QP<double> qp(nt, n_eq, n_lt + nt);
	qp.settings.eps_abs = 1e-10;
	if (n_eq > 0) {
		qp.init(H, g, tc.A, b_eq, C, l, u);
	} else {
		qp.init(H, g, proxsuite::nullopt, proxsuite::nullopt, C, l, u);
	}
	qp.solve();
	if (qp.results.info.status != psp::QPSolverOutput::PROXQP_SOLVED) return std::nullopt;
	return qp.results.x;
}

// Moves the taus of `x` to the nearest point (in relative change,
// sum ((tau - tau0) / tau0)^2) satisfying every interaction row exactly and
// the tau box, so the solve starts feasible. Leaves `x` unchanged if the rows
// can't all hold.
void ProjectTausOntoInteractions(const ProblemLayout& layout, Eigen::VectorXd* x) {
	if (layout.interactions.empty()) return;
	const TauConstraints tc = BuildTauConstraints(layout);
	const int nt = static_cast<int>(tc.cols.size());
	const Eigen::VectorXd tau0 = (*x)(tc.cols);
	const Eigen::VectorXd w = tau0.cwiseInverse();
	const std::optional<Eigen::VectorXd> tau = SolveTauQp(
		w.cwiseProduct(w).asDiagonal(), -w, tc, Eigen::VectorXd::Zero(tc.A.rows()), tc.u,
		Eigen::VectorXd::Constant(nt, kTauMin), Eigen::VectorXd::Constant(nt, kTauMax));
	if (!tau) return;
	Eigen::VectorXd projected = *x;
	projected(tc.cols) = tau->cwiseMax(kTauMin).cwiseMin(kTauMax);
	if (TotalInteractionViolation(projected, layout) < TotalInteractionViolation(*x, layout)) *x = projected;
}

// The objective (time_cost + time_cost2 + acceleration_cost * psi, as
// AssembleObjective) with its exact gradient and dense Hessian. Per merged
// block segment, with T the sum of its taus, s = v0 + v1, d = v1 - v0 and
// D = disp - T s / 2, psi's term is 12 / T^3 |D|^2 + 1 / T |d|^2.
void EvaluateExact(const ProblemLayout& layout, double acceleration_cost, double time_cost,
		   double time_cost2, const Eigen::VectorXd& x, double* f, Eigen::VectorXd* grad,
		   Eigen::MatrixXd* hess) {
	*f = 0.0;
	grad->setZero(layout.n);
	hess->setZero(layout.n, layout.n);
	for (const auto& al : layout.agents)
		for (int j = 0; j < al.K; ++j) {
			const int idx = al.tau_offset + j;
			const double tau = x(idx);
			*f += time_cost * tau + time_cost2 * tau * tau;
			(*grad)(idx) += time_cost + 2.0 * time_cost2 * tau;
			(*hess)(idx, idx) += 2.0 * time_cost2;
		}
	if (acceleration_cost <= 0.0) return;
	const double w = acceleration_cost;
	for (const auto& bs : layout.block_segments) {
		const auto& off = bs.spline->block_offsets_[bs.block_idx];
		if (off.type == CubicConfigurationSpline::Block::Type::SO3Mat)
			throw std::runtime_error("GraphTimingMPC: SO3Mat blocks are not supported");
		const int tN = off.tangent_size;
		double T = 0.0;
		for (int t : bs.tau_indices) T += x(t);
		const Eigen::VectorXd v0 = bs.v0_idx >= 0 ? Eigen::VectorXd(x.segment(bs.v0_idx, tN)) : bs.v0_const;
		const Eigen::VectorXd v1 = bs.v1_idx >= 0 ? Eigen::VectorXd(x.segment(bs.v1_idx, tN)) : bs.v1_const;
		const Eigen::VectorXd s = v0 + v1, d = v1 - v0, D = bs.disp - 0.5 * T * s;
		const double T2 = T * T, T3 = T2 * T, T4 = T3 * T, T5 = T4 * T;
		const double DD = D.squaredNorm(), Ds = D.dot(s), ss = s.squaredNorm(), dd = d.squaredNorm();

		*f += w * (12.0 / T3 * DD + dd / T);
		const double c_T = w * (-36.0 / T4 * DD - 12.0 / T3 * Ds - dd / T2);
		const double c_TT = w * (144.0 / T5 * DD + 72.0 / T4 * Ds + 6.0 / T3 * ss + 2.0 / T3 * dd);
		const Eigen::VectorXd c_v0 = w * (-12.0 / T2 * D - 2.0 / T * d);
		const Eigen::VectorXd c_v1 = w * (-12.0 / T2 * D + 2.0 / T * d);
		const Eigen::VectorXd c_Tv0 = w * (24.0 / T3 * D + 6.0 / T2 * s + 2.0 / T2 * d);
		const Eigen::VectorXd c_Tv1 = w * (24.0 / T3 * D + 6.0 / T2 * s - 2.0 / T2 * d);

		for (int a : bs.tau_indices) {
			(*grad)(a) += c_T;
			for (int b : bs.tau_indices) (*hess)(a, b) += c_TT;
		}
		for (int k = 0; k < tN; ++k) {
			const int i0 = bs.v0_idx >= 0 ? bs.v0_idx + k : -1;
			const int i1 = bs.v1_idx >= 0 ? bs.v1_idx + k : -1;
			if (i0 >= 0) {
				(*grad)(i0) += c_v0(k);
				(*hess)(i0, i0) += w * 8.0 / T;
				for (int a : bs.tau_indices) {
					(*hess)(a, i0) += c_Tv0(k);
					(*hess)(i0, a) += c_Tv0(k);
				}
			}
			if (i1 >= 0) {
				(*grad)(i1) += c_v1(k);
				(*hess)(i1, i1) += w * 8.0 / T;
				for (int a : bs.tau_indices) {
					(*hess)(a, i1) += c_Tv1(k);
					(*hess)(i1, a) += c_Tv1(k);
				}
			}
			if (i0 >= 0 && i1 >= 0) {
				(*hess)(i0, i1) += w * 4.0 / T;
				(*hess)(i1, i0) += w * 4.0 / T;
			}
		}
	}
}

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

// Reduced (velocity-eliminated) solve, for a layout with no velocity
// constraints whose start `x` satisfies every interaction row. For fixed taus
// the objective is a strictly convex quadratic in the knot velocities, so
// v*(tau) is one linear solve, and the solve minimizes
// F(tau) = f(tau, v*(tau)) over the taus alone, subject to the interaction
// rows (held exactly) and the tau box: trust-region Newton on F's exact
// Hessian (the Schur complement f_tt - f_tv f_vv^-1 f_vt, shifted to be
// positive definite where it isn't), each step bounded per tau relative to
// that tau, |dtau_j| <= radius * tau_j. F's minimizers are exactly f's.
SqpResult RunReducedNewton(const ProblemLayout& layout, double acceleration_cost, double time_cost,
			   double time_cost2, int max_iterations, Eigen::VectorXd x) {
	const TauConstraints tc = BuildTauConstraints(layout);
	const int nt = static_cast<int>(tc.cols.size());
	std::vector<int> v_cols;
	for (int c = 0; c < layout.n; ++c)
		if (tc.local[c] < 0) v_cols.push_back(c);

	struct Reduced {
		double F;
		Eigen::VectorXd g;
		Eigen::MatrixXd H;
	};
	// Sets x's velocities to v*(tau) and returns F, its gradient and Hessian.
	auto reduce = [&](Eigen::VectorXd& xr) -> Reduced {
		double f;
		Eigen::VectorXd grad;
		Eigen::MatrixXd hess;
		EvaluateExact(layout, acceleration_cost, time_cost, time_cost2, xr, &f, &grad, &hess);
		if (v_cols.empty()) return {f, grad(tc.cols), hess(tc.cols, tc.cols)};
		const Eigen::LLT<Eigen::MatrixXd> llt(hess(v_cols, v_cols));
		xr(v_cols) -= llt.solve(Eigen::VectorXd(grad(v_cols)));
		EvaluateExact(layout, acceleration_cost, time_cost, time_cost2, xr, &f, &grad, &hess);
		const Eigen::MatrixXd H_vt = hess(v_cols, tc.cols);
		return {f, grad(tc.cols), hess(tc.cols, tc.cols) - H_vt.transpose() * llt.solve(H_vt)};
	};

	Reduced cur = reduce(x);
	double radius = kInitialRelativeRadius;
	int iter = 0;
	for (; iter < max_iterations && nt > 0; ++iter) {
		const Eigen::VectorXd tau = x(tc.cols);
		Eigen::MatrixXd H = cur.H;
		const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(H, Eigen::EigenvaluesOnly);
		const double floor = 1e-8 * std::max(1.0, eig.eigenvalues().cwiseAbs().maxCoeff());
		if (eig.eigenvalues().minCoeff() < floor)
			H.diagonal().array() += floor - eig.eigenvalues().minCoeff();

		const Eigen::VectorXd lo = (Eigen::VectorXd::Constant(nt, kTauMin) - tau).cwiseMax(-radius * tau);
		const Eigen::VectorXd hi = (Eigen::VectorXd::Constant(nt, kTauMax) - tau).cwiseMin(radius * tau);
		const std::optional<Eigen::VectorXd> step =
			SolveTauQp(H, cur.g, tc, -tc.A * tau, tc.u - tc.C * tau, lo, hi);
		if (!step) {
			radius *= 0.25;
			if (radius < kMinRelativeRadius) break;
			continue;
		}
		const Eigen::VectorXd& d = *step;
		const double predicted = -(cur.g.dot(d) + 0.5 * d.dot(H * d));
		if (d.cwiseQuotient(tau).cwiseAbs().maxCoeff() < kRelativeStepTol ||
		    predicted <= 1e-12 * std::max(1.0, std::abs(cur.F))) {
			++iter;
			break;
		}

		Eigen::VectorXd trial = x;
		trial(tc.cols) += d;
		Reduced next = reduce(trial);
		const double rho = (cur.F - next.F) / predicted;
		const bool binding = (d.array().abs() > 0.9 * radius * tau.array()).any();
		if (rho < 0.25) {
			radius *= 0.25;
		} else if (rho > 0.75 && binding) {
			radius = std::min(2.0 * radius, kMaxRelativeRadius);
		}
		if (rho > 1e-8) {
			x = std::move(trial);
			cur = std::move(next);
		}
		if (radius < kMinRelativeRadius) {
			++iter;
			break;
		}
	}
	return SqpResult{x, iter, radius};
}

SqpResult RunTrustRegionSqp(
		const ProblemLayout& layout,
		double acceleration_cost,
		double time_cost,
		double time_cost2,
		double interaction_penalty_weight,
		int max_iterations,
		double initial_trust_radius,
		double max_trust_radius,
		double min_trust_radius,
		double grad_tol,
		std::unique_ptr<GraphTimingMPC::QpState>* qp_state) {

	// n_smooth: the tau/v decision-vector width alone -- what
	// AssembleObjective's f/grad/Hessian, and `x` itself, are sized to
	// throughout this function (matches GraphShortPathMPC's own n_smooth/n
	// split). Interaction-row slack columns (see below) are appended after
	// these, purely a QP-assembly artifact the smooth cost model never
	// sees.
	// Interaction rows first, then the velocity constraint rows (both
	// slack-relaxed -- see below).
	const int m_interactions = static_cast<int>(layout.interactions.size());
	const int m = m_interactions + static_cast<int>(layout.velocity_rows.size());

	// Slack-column layout for the interaction rows' Sl1QP relaxation (see
	// this file's own header comment's CORRECTION paragraph): a LESS_THAN
	// row gets one nonnegative slack (`(Adx)_r - s_r <= ubA(r)`), an EQUAL
	// row (two-sided) gets a nonnegative plus/minus PAIR
	// (`(Adx)_r + s_plus - s_minus = target`, the elastic-variable
	// reformulation of an equality). Computed once here since it depends
	// only on `layout.interactions`' TYPES, not the current iterate --
	// same "row count/shape fixed for the whole call" discipline
	// graph_short_path_mpc.cpp's pruning uses.
	std::vector<int> slack_offset(m);
	int n_slack = 0;
	for (int r = 0; r < m; ++r) {
		slack_offset[r] = n_slack;
		// A velocity row is two-sided in general (lb <= a . x <= ub): a pair
		n_slack += (r < m_interactions && layout.interactions[r].type == AgentInteraction::Type::LESS_THAN) ? 1 : 2;
	}
	const int n_smooth = layout.n;
	const int n = n_smooth + n_slack;
	const int n_in = m + n;  // constraint rows, then one identity row per variable (the box)

	// (Re)build the persistent QpState on the first call or whenever (n,
	// n_in) changed since last time (a node completing, or a dense re-trace
	// changing point count, resizes the problem); the QP is fixed-size once
	// constructed. Every other cycle at a stable size reuses the same
	// instance, so update() warm-starts cycle-to-cycle, not just
	// iteration-to-iteration within one call.
	if (!*qp_state || (*qp_state)->n != n || (*qp_state)->n_in != n_in) {
		*qp_state = std::make_unique<GraphTimingMPC::QpState>(n, n_in);
	}
	GraphTimingMPC::QpState& st = **qp_state;

	// Per-variable box bounds (tau in [kTauMin, kTauMax]; v unbounded but
	// for the single-variable velocity constraints, layout.velocity_bounds)
	// -- smooth (tau/v) columns only; slack columns' own (>= 0) bound is
	// applied directly to dx_lb/dx_ub below every iteration instead (no
	// `var_lb`/`var_ub` entry needed since slack has no accumulated `x` of
	// its own to clamp a step against -- see this file's header comment).
	Eigen::VectorXd var_lb = Eigen::VectorXd::Constant(n_smooth, -kInf);
	Eigen::VectorXd var_ub = Eigen::VectorXd::Constant(n_smooth, kInf);
	for (const auto& al : layout.agents) {
		for (int j = 0; j < al.K; ++j) {
			var_lb(al.tau_offset + j) = kTauMin;
			var_ub(al.tau_offset + j) = kTauMax;
		}
	}
	for (const auto& vb : layout.velocity_bounds) {
		var_lb(vb.col) = std::max(var_lb(vb.col), vb.lb);
		var_ub(vb.col) = std::max(var_lb(vb.col), std::min(var_ub(vb.col), vb.ub));  // conflicting: lb wins
	}
	// Fixed columns (an equality velocity constraint): their gradient
	// doesn't count towards convergence below
	std::vector<int> fixed_cols;
	for (int c = 0; c < n_smooth; ++c)
		if (var_lb(c) == var_ub(c)) fixed_cols.push_back(c);

	// The start inside the bounds (x_init already is, unless two velocity
	// bounds on one column conflict)
	Eigen::VectorXd x = layout.x_init.cwiseMax(var_lb).cwiseMin(var_ub);
	ProjectTausOntoInteractions(layout, &x);
	if (layout.velocity_bounds.empty() && layout.velocity_rows.empty() &&
	    TotalInteractionViolation(x, layout) <= kFeasibilityTol)
		return RunReducedNewton(layout, acceleration_cost, time_cost, time_cost2, max_iterations, x);
	double f_current;
	Eigen::VectorXd grad;
	std::vector<Eigen::Triplet<double>> hess_triplets;
	AssembleObjective(layout, acceleration_cost, time_cost, time_cost2,
			   x, &f_current, &grad, &hess_triplets);
	double violation_current = TotalInteractionViolation(x, layout);
	double phi_current = f_current + interaction_penalty_weight * violation_current;

	if (n_smooth == 0) {
		return SqpResult{x, 0, initial_trust_radius};
	}

	// Constraint matrix C -- fixed for the whole call (every interaction
	// row is exactly linear, same +/-1 pattern every outer iteration, incl.
	// the slack columns' own fixed +/-1 coefficients; only its RHS bounds
	// shift with the current iterate x). Rows m..m+n-1 are the box.
	std::vector<Trip> c_trips;
	for (int r = 0; r < m; ++r) {
		const int scol = n_smooth + slack_offset[r];
		if (r >= m_interactions) {  // a velocity constraint row, elastic both ways
			const auto& vr = layout.velocity_rows[r - m_interactions];
			for (std::size_t c = 0; c < vr.cols.size(); ++c) c_trips.emplace_back(r, vr.cols[c], vr.coeffs[c]);
			c_trips.emplace_back(r, scol, 1.0);       // s_plus
			c_trips.emplace_back(r, scol + 1, -1.0);  // s_minus
			continue;
		}
		const auto& ir = layout.interactions[r];
		for (int k = 0; k < ir.count_i; ++k) c_trips.emplace_back(r, ir.tau_offset_i + k, 1.0);
		for (int k = 0; k < ir.count_j; ++k) c_trips.emplace_back(r, ir.tau_offset_j + k, -1.0);
		if (ir.type == AgentInteraction::Type::LESS_THAN) {
			c_trips.emplace_back(r, scol, -1.0);
		} else {
			c_trips.emplace_back(r, scol, 1.0);       // s_plus
			c_trips.emplace_back(r, scol + 1, -1.0);  // s_minus
		}
	}
	for (int c = 0; c < n; ++c) c_trips.emplace_back(m + c, c, 1.0);
	SpMat C_sp(n_in, n);
	C_sp.setFromTriplets(c_trips.begin(), c_trips.end());  // sums duplicate (r, col) entries
	C_sp.makeCompressed();
	std::vector<long long> c_key = PatternKey(C_sp);


	double trust_radius = initial_trust_radius;
	int iter = 0;

	for (; iter < max_iterations; ++iter) {
		Eigen::VectorXd free_grad = grad;
		for (int c : fixed_cols) free_grad(c) = 0.0;
		if (free_grad.norm() < grad_tol) break;
		// <=, not <: the clamp below (std::max(..., min_trust_radius)) can
		// only ever land EXACTLY at the floor, never strictly below it, so
		// a strict "<" here would never fire -- silently defeating this
		// stall check and leaving the loop to burn every remaining
		// iteration re-solving noise-dominated QPs at a radius so small
		// that actual/predicted reduction are both dominated by floating-
		// point cancellation (both near the true optimum's f value),
		// making rho itself meaningless at that scale.
		if (iter > 0 && trust_radius <= min_trust_radius) break;

		// Gauss-Newton Hessian, upper triangle (what proxqp reads) from the
		// lower-triangle triplets AssembleObjective returns, in the smooth
		// (tau/v) block; the slack block gets just the Tikhonov floor (slack
		// appears LINEARLY in the true cost, `interaction_penalty_weight *
		// slack`). The floor is a conditioning safety net on a PSD-but-not-
		// necessarily-PD Hessian; it doesn't move the true objective's
		// stationary points.
		std::vector<Trip> h_trips;
		h_trips.reserve(hess_triplets.size() + n);
		for (const auto& t : hess_triplets)
			h_trips.emplace_back(std::min(t.row(), t.col()), std::max(t.row(), t.col()), t.value());
		for (int i = 0; i < n; ++i) h_trips.emplace_back(i, i, 1e-10);
		SpMat H_sp(n, n);
		H_sp.setFromTriplets(h_trips.begin(), h_trips.end());
		H_sp.makeCompressed();

		Eigen::VectorXd cl = Eigen::VectorXd::Constant(n_in, -kProxInf);
		Eigen::VectorXd cu = Eigen::VectorXd::Constant(n_in, kProxInf);
		for (int r = 0; r < m; ++r) {
			if (r >= m_interactions) {
				const auto& vr = layout.velocity_rows[r - m_interactions];
				const double value = VelocityRowValue(x, vr);
				cl(r) = vr.lb <= -kInf ? -kProxInf : ProxBound(vr.lb - value);
				cu(r) = vr.ub >= kInf ? kProxInf : ProxBound(vr.ub - value);
				continue;
			}
			const auto& ir = layout.interactions[r];
			const auto [gi, gj] = InteractionGiGj(x, ir);
			const double g_val = gi - gj;
			if (ir.type == AgentInteraction::Type::LESS_THAN) {
				cu(r) = ProxBound(-ir.min_tau - g_val);
			} else {
				cl(r) = cu(r) = ProxBound(-g_val);
			}
		}
		for (int c = 0; c < n_smooth; ++c) {
			cl(m + c) = ProxBound(std::max(var_lb(c) - x(c), -trust_radius));
			cu(m + c) = ProxBound(std::min(var_ub(c) - x(c), trust_radius));
		}
		for (int c = 0; c < n_slack; ++c) cl(m + n_smooth + c) = 0.0;

		Eigen::VectorXd g_qp(n);
		g_qp.head(n_smooth) = grad;
		for (int c = 0; c < n_slack; ++c) g_qp(n_smooth + c) = interaction_penalty_weight;

		std::vector<long long> h_key = PatternKey(H_sp);
		if (st.initialized && h_key == st.h_pattern && c_key == st.c_pattern) {
			st.qp.settings.initial_guess = psp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
			st.qp.update(H_sp, g_qp, proxsuite::nullopt, proxsuite::nullopt, C_sp, cl, cu,
				     /*update_preconditioner=*/false);
		} else {
			st.qp.settings.initial_guess = psp::InitialGuessStatus::NO_INITIAL_GUESS;
			st.qp.init(H_sp, g_qp, proxsuite::nullopt, proxsuite::nullopt, C_sp, cl, cu);
			st.h_pattern = std::move(h_key);
			st.c_pattern = c_key;
		}
		st.initialized = true;
		st.qp.solve();

		if (st.qp.results.info.status != psp::QPSolverOutput::PROXQP_SOLVED) {
			// Every QP subproblem is feasible by construction (dx=0,
			// slack=|current violation| satisfies every row, independent of
			// trust_radius), so a non-solved status is a numerical hiccup.
			// Retry from a cold init at a smaller trust region.
			trust_radius *= 0.25;
			st.initialized = false;
			continue;
		}

		const Eigen::VectorXd& z = st.qp.results.x;
		const Eigen::VectorXd dx = z.head(n_smooth);
		const double predicted_slack_sum = n_slack > 0 ? z.tail(n_slack).sum() : 0.0;

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
		const double violation_new = TotalInteractionViolation(x + dx, layout);
		const double phi_new = f_new + interaction_penalty_weight * violation_new;

		double dx_H_dx = 0.0;
		for (const auto& t : hess_triplets)
			dx_H_dx += (t.row() == t.col() ? 1.0 : 2.0) * t.value() * dx(t.row()) * dx(t.col());
		const double predicted_smooth_reduction = -(grad.dot(dx) + 0.5 * dx_H_dx);
		// The QP's own slack values ARE exactly the post-step violation at
		// the QP optimum (interaction_penalty_weight > 0 drives every
		// slack to its tight lower bound given the row's RHS), so this is
		// the model's own prediction of the post-step violation -- same
		// "predicted reduction" role graph_short_path_mpc.cpp's
		// predicted_violation_reduction plays.
		const double predicted_violation_reduction = violation_current - predicted_slack_sum;
		const double predicted_reduction =
			predicted_smooth_reduction + interaction_penalty_weight * predicted_violation_reduction;
		const double actual_reduction = phi_current - phi_new;
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
			// Accept: advance x and refresh (f, grad, Hessian, violation)
			// at the new iterate for the next loop pass.
			x += dx;
			f_current = f_new;
			violation_current = violation_new;
			phi_current = phi_new;
			AssembleObjective(layout, acceleration_cost, time_cost, time_cost2,
					   x, &f_current, &grad, &hess_triplets);
		}
		// else: reject -- x/f_current/grad/hess_triplets/violation_current/
		// phi_current all stay at the previous (still fully valid) iterate;
		// only trust_radius moved.
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
		const std::vector<CubicConfigurationSpline>& splines,
		std::vector<Eigen::MatrixXd>* vs_list,
		std::vector<Eigen::VectorXd>* time_deltas_list,
		std::map<int, int>* agent_spline_length_map,
		const std::vector<std::vector<int>>& agent_nodes_list) {

	for (int i = 0; i < graph.num_agents; ++i) {
		// Per-agent, not a single shared width -- see this file's
		// constructor and timing_gn_layout.cpp's CumulativeOffsets comment
		// for why a shared width silently corrupts memory once agents'
		// tangent widths actually differ.
		const int tangent_dim = splines.at(i).tangent_dim();
		const int K = static_cast<int>(agent_nodes_list[i].size());
		if (K == 0) {
			// No active spline for this agent this cycle -- erase (not
			// leave stale) so fill_cubic_splines/get_next_taus/
			// get_next_nodes see "absent" instead of a leftover nonzero
			// spline_length_i from whatever earlier K>0 cycle last
			// touched this agent, which no longer matches _wps_list[i]
			// (already freshly resized to 0 rows this cycle by
			// BuildDenseProblemLayout/BuildProblemLayout) -- otherwise
			// fill_cubic_splines reads the stale spline_length_i against
			// the fresh, too-small _wps_list[i]/_vs_list[i], and with
			// Eigen's bounds-assertions compiled out in this build, that
			// shape-mismatched block assignment silently corrupts memory
			// instead of throwing (confirmed via gdb: SIGSEGV inside
			// fill_cubic_splines' Eigen::Block assignment).
			agent_spline_length_map->erase(i);
			(*vs_list)[i] = Eigen::MatrixXd(0, tangent_dim);
			(*time_deltas_list)[i] = Eigen::VectorXd(0);
			continue;
		}
		(*agent_spline_length_map)[i] = K + 1;

		const AgentLayout& al = layout.agents[i];
		(*time_deltas_list)[i] = Eigen::VectorXd::Zero(K);
		for (int j = 0; j < K; ++j) {
			(*time_deltas_list)[i](j) = result.x(al.tau_offset + j);
		}
		// Interior-knot velocities are packed per block and only for knots
		// the block treats as real (a bridged block has no variable at a
		// skipped knot). Write each BlockSegment's k_hi endpoint velocity
		// (every interior active knot is exactly one block-segment's k_hi);
		// knots a block bridged stay 0 -- fill_cubic_splines' spline set()
		// ignores them for that block anyway.
		(*vs_list)[i] = Eigen::MatrixXd::Zero(std::max(K - 1, 0), tangent_dim);
		if (K > 1) {
			for (const auto& bs : layout.block_segments) {
				if (bs.agent != i || bs.v1_idx < 0) continue;
				const auto& off = bs.spline->block_offsets_[bs.block_idx];
				for (int c = 0; c < off.tangent_size; ++c) {
					(*vs_list)[i](bs.k_hi - 1, off.tangent_offset + c) =
						result.x(bs.v1_idx + c);
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
		const Eigen::VectorXi& var_assignments,
		const Eigen::VectorXd& t_by_node) {

	_timer.Start();

	// Kept for fill_cubic_splines, which runs right after this and asks the
	// graph (constrained_columns) which config blocks each node pins --
	// var_agent_q(...) references there resolve through this.
	_last_var_assignments = var_assignments;

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
			*_graph, *_splines, remaining_vertices, waypoints, var_assignments, x0, v0,
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
	// Single source of truth for the per-block bridged knots -- fill_cubic_splines
	// reads this instead of recomputing the mask from constrained_columns.
	_agent_block_active_list = layout.agent_block_active;

	const SqpResult result = RunTrustRegionSqp(
		layout, _acceleration_cost, _time_cost, _time_cost2, _interaction_penalty_weight,
		_max_iterations, _initial_trust_radius, _max_trust_radius,
		_min_trust_radius, _grad_tol, &_qp_state);

	_last_iterations = result.iterations;
	_last_trust_radius = result.trust_radius;
	_last_solve_time = _timer.Tick();

	StoreResult(*_graph, layout, result, *_splines,
		    &_vs_list, &_time_deltas_list, &_agent_spline_length_map, _agent_nodes_list);
	return true;
}

bool GraphTimingMPC::solve_dense(
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const std::vector<Eigen::MatrixXd>& agent_dense_wps,
		const std::vector<std::vector<int>>& agent_dense_node_ids,
		const std::vector<AgentInteraction>& agent_interactions,
		const Eigen::VectorXi& var_assignments) {

	_timer.Start();

	// Kept for fill_cubic_splines' per-node knot mask, which asks the graph
	// (constrained_columns) which config blocks each real node pins --
	// var_agent_q(...) references resolve through this. The caller
	// (TracedTimingMPC) already handed the same vector to get_agent_paths
	// when resolving the dense sequence. Plain copy, not left stale from an
	// earlier solve().
	_last_var_assignments = var_assignments;

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
			*_graph, *_splines, agent_dense_wps, agent_dense_node_ids,
			agent_interactions, _last_var_assignments,
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
	// Single source of truth for the per-block bridged knots -- fill_cubic_splines
	// reads this instead of recomputing the mask from constrained_columns.
	_agent_block_active_list = layout.agent_block_active;

	const SqpResult result = RunTrustRegionSqp(
		layout, _acceleration_cost, _time_cost, _time_cost2, _interaction_penalty_weight,
		_max_iterations, _initial_trust_radius, _max_trust_radius,
		_min_trust_radius, _grad_tol, &_qp_state);

	_last_iterations = result.iterations;
	_last_trust_radius = result.trust_radius;
	_last_solve_time = _timer.Tick();

	StoreResult(*_graph, layout, result, *_splines,
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
	// Per-agent width AND cumulative offset -- not a shared splines[0] width
	// times a flat agent index -- once agents' ambient/tangent widths
	// differ, agent i's own block in x0/v0 no longer starts at i * a_d/t_d
	// (see timing_gn_layout.cpp's CumulativeOffsets comment).
	std::vector<int> a_offset(_graph->num_agents), t_offset(_graph->num_agents);
	{
		int a_off = 0, t_off = 0;
		for (int i = 0; i < _graph->num_agents; ++i) {
			a_offset[i] = a_off;
			t_offset[i] = t_off;
			a_off += splines[i]->ambient_dim();
			t_off += splines[i]->tangent_dim();
		}
	}

	for (int i = 0; i < _graph->num_agents; ++i) {
		const int a_d = splines[i]->ambient_dim();
		const int t_d = splines[i]->tangent_dim();
		const int spline_length_i = get_agent_spline_length(i);

		if (spline_length_i > 1) {
			Eigen::VectorXd x0_i = x0.segment(a_offset[i], a_d);
			Eigen::MatrixXd wps_i(spline_length_i, a_d);
			wps_i.row(0) = x0_i;
			wps_i.bottomRows(spline_length_i - 1) = _wps_list[i];

			Eigen::VectorXd v0_i = v0.segment(t_offset[i], t_d);
			Eigen::MatrixXd vs_i(spline_length_i, t_d);
			vs_i.row(0) = v0_i;
			vs_i.block(1, 0, spline_length_i - 2, t_d) = _vs_list[i];
			vs_i.row(spline_length_i - 1).setZero();

			Eigen::VectorXd times_i = CumsumWithZero(_time_deltas_list[i], spline_length_i - 1);

			// Per-(knot, block) active mask (1 == real knot, 0 == bridge the
			// block past it) -- computed once during the last solve()/
			// solve_dense() layout build (gn_timing::AgentBlockActiveMask,
			// stored on ProblemLayout::agent_block_active), so the output
			// spline bridges exactly the knots the timing solve merged its
			// BlockSegments across. Row 0 is x0, rows 1.. this agent's
			// resolved node sequence; shape is (spline_length_i x NB).
			// CubicConfigurationSpline::set() force-activates rows 0 and N-1
			// regardless. An empty/stale-shaped entry (agent had no active
			// spline that cycle, or a size skew) falls back to all-knot.
			const int NB = static_cast<int>(splines[i]->block_offsets_.size());
			static const Eigen::MatrixXi kNoMask;
			const Eigen::MatrixXi& mask =
				(i < static_cast<int>(_agent_block_active_list.size()))
					? _agent_block_active_list[i] : kNoMask;
			if (mask.rows() == spline_length_i && mask.cols() == NB)
				splines[i]->set_block_active_mask(mask);
			else
				splines[i]->set_block_active_mask(Eigen::MatrixXi());

			splines[i]->set(wps_i, vs_i, times_i);
		} else {
			// Dummy spline that stays at x0, and comes to a stop after 1 second.
			Eigen::VectorXd x0_i = x0.segment(a_offset[i], a_d);
			Eigen::MatrixXd wps_i(2, a_d);
			wps_i.row(0) = x0_i;
			wps_i.row(1) = x0_i;

			Eigen::VectorXd v0_i = v0.segment(t_offset[i], t_d);
			Eigen::MatrixXd vs_i(2, t_d);
			vs_i.row(0) = v0_i;
			vs_i.row(1).setZero();

			Eigen::VectorXd times_i(2);
			times_i << 0.0, 1.0;

			// Drop any mask left over from an earlier, longer cycle -- its
			// row count no longer matches and set() would throw.
			splines[i]->set_block_active_mask(Eigen::MatrixXi());
			splines[i]->set(wps_i, vs_i, times_i);
		}
	}
}

const std::vector<double> GraphTimingMPC::get_next_taus() const {
	std::vector<double> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = get_agent_spline_length(i);
		if (spline_length_i > 1) {
			result.push_back(_time_deltas_list[i](0));
		}
	}
	return result;
}

const std::vector<int> GraphTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = get_agent_spline_length(i);
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
