#include "local_nlp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/SparseCholesky>
#include <lpi/lpi.h>
#include <proxsuite/proxqp/sparse/sparse.hpp>

namespace {

using SparseRows = LocalNLP::SparseRows;
using SpMat = Eigen::SparseMatrix<double>;
using Clock = std::chrono::steady_clock;
constexpr double kInf = std::numeric_limits<double>::infinity();

#define LPI_CHECK(call)                                                                 \
	do {                                                                            \
		if ((call) != SCIP_OKAY) throw std::runtime_error("Sl1LP: LP call failed: " #call); \
	} while (0)

double Since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

// The pattern of M'M plus the diagonal, zero-valued.
SpMat GramPattern(const SparseRows& M) {
	SparseRows ones = M;
	ones.coeffs().setOnes();
	SpMat eye(M.cols(), M.cols());
	eye.setIdentity();
	SpMat P = SpMat(ones.transpose() * ones) + eye;
	P.coeffs().setZero();
	P.makeCompressed();
	return P;
}

// An augmented Lagrangian (Rockafellar's form for l <= h <= u) on the rows h
// a derived class penalizes, each inner problem by Levenberg-Marquardt: the
// derived class's step from y for a damping mu, accepted when it lowers the
// merit c.y + 1/2 |res|^2, res = sqrt(rho) (s - clamp(s, l, u)),
// s = h + lambda/rho.
class AugmentedLagrangianLM : public LocalSolver {
public:
	void Solve(const LocalNLP& nlp, Eigen::VectorXd* y, LocalSolveRecord* record) final {
		_nlp = &nlp;
		Setup(y, record);
		const int m = static_cast<int>(_l.size());
		Eigen::VectorXd h, lambda = Eigen::VectorXd::Zero(m);
		double rho = 10.0;
		auto merit = [&](const Eigen::VectorXd& yy, Eigen::VectorXd* res) {
			if (!yy.allFinite()) return kInf;
			const auto t0 = Clock::now();
			Rows(yy, &h);
			record->eval_time += Since(t0);
			++record->trials;
			if (!h.allFinite()) return kInf;
			const Eigen::VectorXd shifted = h + lambda / rho;
			*res = std::sqrt(rho) * (shifted - shifted.cwiseMax(_l).cwiseMin(_u));
			return nlp.c.dot(yy) + 0.5 * res->squaredNorm();
		};

		double prev_viol = kInf;
		// The violations of the consecutive outers at the capped rho with
		// the linear rows met, for the stall exit.
		std::vector<double> capped_viol;
		int outer = 0;
		for (; outer < kMaxOuter; ++outer) {
			Eigen::VectorXd res;
			double phi = merit(*y, &res);
			if (!std::isfinite(phi)) throw std::runtime_error("the local solve's iterate can't be evaluated");
			double mu = 1e-3;
			for (int it = 0; it < kMaxInner; ++it) {
				// Gauss-Newton on the active penalty rows (res is 0 on the rest).
				const Eigen::VectorXd grad = nlp.c + std::sqrt(rho) * (_J.transpose() * res);
				++record->inner;
				const Eigen::VectorXd active = (res.array() != 0.0).cast<double>();
				if (!Prepare(*y, grad, active.asDiagonal() * _J, rho)) break;
				bool accepted = false;
				while (mu < 1e12) {
					const auto t0 = Clock::now();
					Eigen::VectorXd y_new, res_new;
					const bool stepped = Trial(*y, mu, &y_new);
					record->factor_time += Since(t0);
					const double phi_new = stepped ? merit(y_new, &res_new) : kInf;
					if (phi_new < phi) {
						accepted = phi - phi_new > 1e-14 * std::max(1.0, std::abs(phi));
						*y = y_new;
						res = res_new;
						phi = phi_new;
						mu = std::max(mu / 3.0, 1e-12);
						break;
					}
					mu *= 4.0;
				}
				if (!accepted) break;
			}
			// Outer: multipliers, and a stiffer penalty when progress stalls.
			merit(*y, &res);
			const double viol = (_l - h).cwiseMax(h - _u).cwiseMax(0.0).lpNorm<Eigen::Infinity>();
			const Eigen::VectorXd shifted = h + lambda / rho;
			lambda = rho * (shifted - shifted.cwiseMax(_l).cwiseMin(_u));
			if (viol < kFeasTol) break;
			// Stalled: at the capped penalty, with the linear rows met, the
			// violation hasn't halved over the last kStallWindow such outers.
			const Eigen::VectorXd ay = nlp.A * *y + nlp.a0;
			const double lin_viol =
				(nlp.l_lin - ay).cwiseMax(ay - nlp.u_lin).cwiseMax(0.0).lpNorm<Eigen::Infinity>();
			if (rho >= kMaxRho && lin_viol <= kStallLinearTol) {
				capped_viol.push_back(viol);
				const size_t k = capped_viol.size();
				if (k > kStallWindow && viol > 0.5 * capped_viol[k - 1 - kStallWindow]) {
					record->stalled = true;
					break;
				}
			} else {
				capped_viol.clear();
			}
			if (viol > 0.25 * prev_viol) rho = std::min(rho * 10.0, kMaxRho);
			prev_viol = viol;
		}
		record->outer = outer + (outer < kMaxOuter ? 1 : 0);
		record->rho = rho;
	}

protected:
	static constexpr int kMaxOuter = 300, kMaxInner = 1000;
	static constexpr double kFeasTol = 1e-9, kMaxRho = 1e10;
	static constexpr size_t kStallWindow = 5;
	// Under SCIP's feasibility tolerance, so a stalled iterate's linear
	// rows pass its check.
	static constexpr double kStallLinearTol = 1e-7;

	// Once per solve: the penalized rows' bounds _l, _u and Jacobian pattern
	// _J; may move the start *y.
	virtual void Setup(Eigen::VectorXd* y, LocalSolveRecord* record) = 0;
	// The penalized rows h at y, their Jacobian's values into _J.
	virtual void Rows(const Eigen::VectorXd& y, Eigen::VectorXd* h) = 0;
	// Once per inner iteration, the Gauss-Newton model at y: the merit's
	// gradient and the active rows' Jacobian. False when y is stationary.
	virtual bool Prepare(const Eigen::VectorXd& y, const Eigen::VectorXd& grad, const SparseRows& Ja,
			     double rho) = 0;
	// The step's endpoint for damping mu; false when there's none.
	virtual bool Trial(const Eigen::VectorXd& y, double mu, Eigen::VectorXd* y_new) = 0;

	const LocalNLP* _nlp = nullptr;
	Eigen::VectorXd _l, _u;
	SparseRows _J;
};

// Every row penalized, the linear ones (first) included. Steps are clipped
// to the bounds; variables a bound holds leave the step.
class PenalizedLM final : public AugmentedLagrangianLM {
	void Setup(Eigen::VectorXd* y, LocalSolveRecord*) override {
		const LocalNLP& nlp = *_nlp;
		const int n = nlp.n(), m_lin = static_cast<int>(nlp.A.rows()), m_g = static_cast<int>(nlp.l.size());
		_l.resize(m_lin + m_g);
		_u.resize(m_lin + m_g);
		_l << nlp.l_lin, nlp.l;
		_u << nlp.u_lin, nlp.u;
		std::vector<Eigen::Triplet<double>> t;
		for (int r = 0; r < m_lin; ++r)
			for (SparseRows::InnerIterator it(nlp.A, r); it; ++it) t.emplace_back(r, it.col(), it.value());
		for (int r = 0; r < m_g; ++r)
			for (SparseRows::InnerIterator it(nlp.J, r); it; ++it) t.emplace_back(m_lin + r, it.col(), 0.0);
		_J.resize(m_lin + m_g, n);
		_J.setFromTriplets(t.begin(), t.end());
		_J.makeCompressed();
		_gJ = nlp.J;
		_gJ.makeCompressed();
		// g's Jacobian values are copied in behind A's.
		_A_nnz = static_cast<int>(nlp.A.nonZeros());
		if (_J.nonZeros() != nlp.A.nonZeros() + _gJ.nonZeros())
			throw std::runtime_error("PenalizedLM: A or J has duplicate entries");
		_H0 = GramPattern(_J);
		_ldlt.analyzePattern(_H0);
		*y = y->cwiseMax(nlp.lb).cwiseMin(nlp.ub);
	}

	void Rows(const Eigen::VectorXd& y, Eigen::VectorXd* h) override {
		const LocalNLP& nlp = *_nlp;
		nlp.evaluate(y, &_g, &_gJ);
		h->resize(_l.size());
		*h << nlp.A * y + nlp.a0, _g;
		std::copy(_gJ.valuePtr(), _gJ.valuePtr() + _gJ.nonZeros(), _J.valuePtr() + _A_nnz);
	}

	bool Prepare(const Eigen::VectorXd& y, const Eigen::VectorXd& grad, const SparseRows& Ja,
		     double rho) override {
		const LocalNLP& nlp = *_nlp;
		const int n = nlp.n();
		std::vector<char> held(n, 0);
		double pg = 0.0;
		for (int k = 0; k < n; ++k) {
			held[k] = (y(k) <= nlp.lb(k) && grad(k) > 0) || (y(k) >= nlp.ub(k) && grad(k) < 0);
			if (!held[k]) pg = std::max(pg, std::abs(grad(k)));
		}
		if (pg < kInnerTol) return false;
		_H = rho * SpMat(Ja.transpose() * Ja) + _H0;
		for (int k = 0; k < _H.outerSize(); ++k)
			for (SpMat::InnerIterator it(_H, k); it; ++it)
				if (held[it.row()] || held[it.col()]) it.valueRef() = it.row() == it.col() ? 1.0 : 0.0;
		_grad = grad;
		// Marquardt's scaling plus Levenberg's identity: bounded steps on
		// variables the penalty doesn't curve.
		_diag = _H.diagonal().array() + 1.0;
		for (int k = 0; k < n; ++k)
			if (held[k]) _grad(k) = _diag(k) = 0.0;
		return true;
	}

	bool Trial(const Eigen::VectorXd& y, double mu, Eigen::VectorXd* y_new) override {
		SpMat M = _H;
		for (int k = 0; k < M.cols(); ++k) M.coeffRef(k, k) += mu * _diag(k);
		_ldlt.factorize(M);
		if (_ldlt.info() != Eigen::Success) return false;
		*y_new = (y + _ldlt.solve(-_grad)).cwiseMax(_nlp->lb).cwiseMin(_nlp->ub);
		return true;
	}

	static constexpr double kInnerTol = 1e-9;
	SparseRows _gJ;
	Eigen::VectorXd _g, _grad, _diag;
	int _A_nnz = 0;
	SpMat _H0, _H;
	Eigen::SimplicialLDLT<SpMat> _ldlt;
};

// g penalized; the linear rows and bounds held exactly by solving each step
//   min_d  grad.d + 1/2 d'(rho Ja'Ja + mu D) d
//   s.t.   l_lin <= A (y + d) + a0 <= u_lin,  lb <= y + d <= ub
// as a sparse QP with ProxQP (D: Marquardt's scaling plus Levenberg's
// identity), warm-started from the previous step. The start is first moved
// onto the linear rows (the nearest point, by the same QP with H = I).
class ProxQPLM final : public AugmentedLagrangianLM {
	using QP = proxsuite::proxqp::sparse::QP<double, int>;
	using QPMat = Eigen::SparseMatrix<double, Eigen::ColMajor, int>;

	void Setup(Eigen::VectorXd* y, LocalSolveRecord*) override {
		const LocalNLP& nlp = *_nlp;
		const int n = nlp.n(), m_lin = static_cast<int>(nlp.A.rows());
		_l = nlp.l;
		_u = nlp.u;
		_J = nlp.J;
		_J.makeCompressed();
		// The constraint rows: A, then the identity for the bounds.
		std::vector<Eigen::Triplet<double, int>> t;
		for (int r = 0; r < m_lin; ++r)
			for (SparseRows::InnerIterator it(nlp.A, r); it; ++it) t.emplace_back(r, it.col(), it.value());
		for (int k = 0; k < n; ++k) t.emplace_back(m_lin + k, k, 1.0);
		_C.resize(m_lin + n, n);
		_C.setFromTriplets(t.begin(), t.end());
		_C.makeCompressed();
		_cl.resize(m_lin + n);
		_cu.resize(m_lin + n);
		_H0 = GramPattern(_J);
		_qp = std::make_unique<QP>(n, 0, m_lin + n);
		_qp->settings.verbose = false;
		_qp->settings.compute_timings = false;
		_qp->settings.eps_abs = kQPTol;
		_qp->settings.eps_rel = 0.0;
		// Every step QP is feasible (d = 0 is, once the start is on the
		// linear rows): ProxQP's infeasibility test only misfires on them.
		_qp->settings.eps_primal_inf = 1e-14;
		_qp->settings.eps_dual_inf = 1e-14;
		_qp->settings.max_iter = kQPMaxIter;
		_initialized = false;

		*y = y->cwiseMax(nlp.lb).cwiseMin(nlp.ub);
		QPMat H = _H0;
		for (int k = 0; k < n; ++k) H.coeffRef(k, k) = 1.0;
		Bounds(*y);
		Eigen::VectorXd d;
		if (SolveQP(H, Eigen::VectorXd::Zero(n), &d)) *y = (*y + d).cwiseMax(nlp.lb).cwiseMin(nlp.ub);
	}

	void Rows(const Eigen::VectorXd& y, Eigen::VectorXd* h) override { _nlp->evaluate(y, h, &_J); }

	bool Prepare(const Eigen::VectorXd& y, const Eigen::VectorXd& grad, const SparseRows& Ja,
		     double rho) override {
		_Hs = rho * SpMat(Ja.transpose() * Ja) + _H0;
		_diag = _Hs.diagonal().array() + 1.0;
		_grad = grad;
		Bounds(y);
		return true;
	}

	bool Trial(const Eigen::VectorXd& y, double mu, Eigen::VectorXd* y_new) override {
		QPMat H = _Hs;
		for (int k = 0; k < H.cols(); ++k) H.coeffRef(k, k) += mu * _diag(k);
		Eigen::VectorXd d;
		if (!SolveQP(H, _grad, &d)) return false;
		*y_new = (y + d).cwiseMax(_nlp->lb).cwiseMin(_nlp->ub);
		return true;
	}

	// The constraint rows' bounds on the step d from y.
	void Bounds(const Eigen::VectorXd& y) {
		const LocalNLP& nlp = *_nlp;
		const int m_lin = static_cast<int>(nlp.A.rows());
		const Eigen::VectorXd ay = nlp.A * y + nlp.a0;
		auto finite = [](double v) { return std::clamp(v, -kProxInf, kProxInf); };
		for (int r = 0; r < m_lin; ++r) {
			_cl(r) = finite(nlp.l_lin(r) - ay(r));
			_cu(r) = finite(nlp.u_lin(r) - ay(r));
		}
		for (int k = 0; k < nlp.n(); ++k) {
			_cl(m_lin + k) = finite(nlp.lb(k) - y(k));
			_cu(m_lin + k) = finite(nlp.ub(k) - y(k));
		}
	}

	// H keeps _H0's pattern, so after the first solve only values change.
	bool SolveQP(const QPMat& H, const Eigen::VectorXd& g, Eigen::VectorXd* d) {
		namespace psp = proxsuite::proxqp;
		if (_initialized) {
			_qp->settings.initial_guess = psp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
			_qp->update(H, g, proxsuite::nullopt, proxsuite::nullopt, _C, _cl, _cu, false);
		} else {
			_qp->settings.initial_guess = psp::InitialGuessStatus::NO_INITIAL_GUESS;
			_qp->init(H, g, proxsuite::nullopt, proxsuite::nullopt, _C, _cl, _cu);
			_initialized = true;
		}
		_qp->solve();
		if (_qp->results.info.status != psp::QPSolverOutput::PROXQP_SOLVED) return false;
		*d = _qp->results.x;
		return d->allFinite();
	}

	static constexpr double kQPTol = 1e-7;
	static constexpr int kQPMaxIter = 100;
	static inline const double kProxInf = proxsuite::helpers::infinite_bound<double>::value();
	QPMat _C;
	Eigen::VectorXd _cl, _cu, _grad, _diag;
	SpMat _H0, _Hs;
	std::unique_ptr<QP> _qp;
	bool _initialized = false;
};

// Sequential l1 linear programming with an infinity-norm trust region on the
// variables g reads. At y, the step d solves the LP
//   min  c.d + nu sum(p + q)
//   s.t. l_lin <= A (y + d) + a0 <= u_lin,  lb <= y + d <= ub,
//        l <= g(y) + J d + p - q <= u,  p, q >= 0,  |d_k| <= Delta (k read by g),
// accepted when the l1 merit c.y + nu V(y) (V: g's summed bound violation)
// drops by at least eta of the LP's predicted drop; a rejected step gets one
// second-order correction (the same LP with g re-evaluated at y + d). The
// linear rows and bounds hold at every accepted iterate, to the LP's
// tolerance, once the first step has moved the start onto them. nu is raised
// while the step's linearized violation stays well above the least the trust
// region allows (Byrd, Nocedal and Waltz's steering rule), so an unreachable g
// ends at a local minimizer of V.
class Sl1LP final : public LocalSolver {
public:
	~Sl1LP() override {
		if (_lpi) SCIPlpiFree(&_lpi);
	}

	void Solve(const LocalNLP& nlp, Eigen::VectorXd* y, LocalSolveRecord* record) override {
		const int n = nlp.n(), m_lin = static_cast<int>(nlp.A.rows()), m = static_cast<int>(nlp.l.size());
		*y = y->cwiseMax(nlp.lb).cwiseMin(nlp.ub);
		SparseRows J = nlp.J;
		J.makeCompressed();
		SparseRows J_trial = J;
		std::vector<char> in_g(n, 0);
		for (int r = 0; r < m; ++r)
			for (SparseRows::InnerIterator it(J, r); it; ++it) in_g[it.col()] = 1;

		Eigen::VectorXd g, g_trial;
		auto evaluate = [&](const Eigen::VectorXd& yy, Eigen::VectorXd* gg, SparseRows* JJ) {
			const auto t0 = Clock::now();
			nlp.evaluate(yy, gg, JJ);
			record->eval_time += Since(t0);
			++record->trials;
		};
		auto violation = [&](const Eigen::VectorXd& gg) {
			return (nlp.l - gg).cwiseMax(gg - nlp.u).cwiseMax(0.0);
		};
		auto linear_violation = [&](const Eigen::VectorXd& yy) {
			const Eigen::VectorXd ay = nlp.A * yy + nlp.a0;
			return (nlp.l_lin - ay).cwiseMax(ay - nlp.u_lin).cwiseMax(0.0).lpNorm<Eigen::Infinity>();
		};
		evaluate(*y, &g, &J);
		if (!g.allFinite()) throw std::runtime_error("the local solve's start can't be evaluated");

		Load(nlp, J);
		const double inf = SCIPlpiInfinity(_lpi);
		auto clamp_inf = [&](double v) { return std::clamp(v, -inf, inf); };
		const int cols = n + 2 * m;
		std::vector<int> all_cols(cols), all_rows(m_lin + m);
		for (int k = 0; k < cols; ++k) all_cols[k] = k;
		for (int r = 0; r < m_lin + m; ++r) all_rows[r] = r;
		std::vector<double> obj(cols), col_lb(cols), col_ub(cols), lhs(m_lin + m), rhs(m_lin + m), x(cols);

		auto set_objective = [&](double c_scale, double nu) {
			for (int k = 0; k < n; ++k) obj[k] = c_scale * nlp.c(k);
			for (int k = n; k < cols; ++k) obj[k] = nu;
			LPI_CHECK(SCIPlpiChgObj(_lpi, cols, all_cols.data(), obj.data()));
		};
		auto set_bounds = [&](const Eigen::VectorXd& yy, double delta) {
			for (int k = 0; k < n; ++k) {
				double lo = nlp.lb(k) - yy(k), hi = nlp.ub(k) - yy(k);
				if (in_g[k]) {
					lo = std::max(lo, -delta);
					hi = std::min(hi, delta);
				}
				col_lb[k] = clamp_inf(std::min(lo, 0.0));
				col_ub[k] = clamp_inf(std::max(hi, 0.0));
			}
			for (int k = n; k < cols; ++k) {
				col_lb[k] = 0.0;
				col_ub[k] = inf;
			}
			LPI_CHECK(SCIPlpiChgBounds(_lpi, cols, all_cols.data(), col_lb.data(), col_ub.data()));
		};
		// The rows at yy, g's linearized about a point where it is gg - J d0.
		auto set_sides = [&](const Eigen::VectorXd& yy, const Eigen::VectorXd& g_const) {
			const Eigen::VectorXd ay = nlp.A * yy + nlp.a0;
			for (int r = 0; r < m_lin; ++r) {
				lhs[r] = clamp_inf(nlp.l_lin(r) - ay(r));
				rhs[r] = clamp_inf(nlp.u_lin(r) - ay(r));
			}
			for (int r = 0; r < m; ++r) {
				lhs[m_lin + r] = clamp_inf(nlp.l(r) - g_const(r));
				rhs[m_lin + r] = clamp_inf(nlp.u(r) - g_const(r));
			}
			LPI_CHECK(SCIPlpiChgSides(_lpi, m_lin + m, all_rows.data(), lhs.data(), rhs.data()));
		};
		auto set_jacobian = [&]() {
			for (int r = 0; r < m; ++r)
				for (SparseRows::InnerIterator it(J, r); it; ++it)
					LPI_CHECK(SCIPlpiChgCoef(_lpi, m_lin + r, static_cast<int>(it.col()), it.value()));
		};
		// Solves the LP; d and the summed elastic slack on success.
		auto solve = [&](Eigen::VectorXd* d, double* slack) {
			const auto t0 = Clock::now();
			LPI_CHECK(SCIPlpiSolveDual(_lpi));
			record->factor_time += Since(t0);
			++record->inner;
			if (!SCIPlpiIsOptimal(_lpi)) return false;
			LPI_CHECK(SCIPlpiGetSol(_lpi, nullptr, x.data(), nullptr, nullptr, nullptr));
			*d = Eigen::Map<const Eigen::VectorXd>(x.data(), n);
			*slack = 0.0;
			for (int k = n; k < cols; ++k) *slack += x[k];
			return d->allFinite();
		};
		auto tr_norm = [&](const Eigen::VectorXd& d) {
			double v = 0.0;
			for (int k = 0; k < n; ++k)
				if (in_g[k]) v = std::max(v, std::abs(d(k)));
			return v;
		};

		double nu = kNu0 * std::max(1.0, nlp.c.lpNorm<Eigen::Infinity>());
		const double nu_max = kNuMaxRatio * nu;
		double delta = kDelta0;
		bool on_linear = linear_violation(*y) <= kLinearTol;
		Eigen::VectorXd d, d_soc, y_trial;
		int it = 0;
		for (; it < kMaxIter; ++it) {
			const double V = violation(g).sum();
			set_jacobian();
			set_sides(*y, g);
			set_objective(1.0, nu);
			set_bounds(*y, delta);
			double slack;
			bool solved = solve(&d, &slack);
			// Off the linear rows, widen the trust region until they're reachable.
			for (double wide = 10.0 * delta; !solved && !on_linear && wide < 1e6; wide *= 10.0) {
				set_bounds(*y, wide);
				solved = solve(&d, &slack);
			}
			if (!solved) break;  // infeasible linear rows or an unbounded LP

			// Steering: when the linearized g can't be met in the trust
			// region at this nu, compare with the least slack it allows.
			if (slack > kFeasTol && nu < nu_max) {
				set_objective(0.0, 1.0);
				Eigen::VectorXd d_min;
				double slack_min;
				if (solve(&d_min, &slack_min)) {
					set_objective(1.0, nu);
					while (nu < nu_max && slack > kFeasTol &&
					       V - slack < kSteer * (V - slack_min) - kFeasTol) {
						nu = std::min(nu * 10.0, nu_max);
						set_objective(1.0, nu);
						if (!solve(&d, &slack)) break;
					}
				}
				set_objective(1.0, nu);
			}

			const double merit = nlp.c.dot(*y) + nu * V;
			const double pred = nu * V - nlp.c.dot(d) - nu * slack;
			if (!on_linear) {
				// The first step moves the start onto the linear rows.
				*y = (*y + d).cwiseMax(nlp.lb).cwiseMin(nlp.ub);
				evaluate(*y, &g, &J);
				on_linear = true;
				continue;
			}
			if (pred <= kPredTol * std::max(1.0, std::abs(merit))) break;  // stationary

			auto try_step = [&](const Eigen::VectorXd& step) {
				y_trial = (*y + step).cwiseMax(nlp.lb).cwiseMin(nlp.ub);
				evaluate(y_trial, &g_trial, &J_trial);
				if (!g_trial.allFinite()) return -kInf;
				const double ared = merit - (nlp.c.dot(y_trial) + nu * violation(g_trial).sum());
				return ared / pred;
			};
			double ratio = try_step(d);
			const Eigen::VectorXd step = d;
			bool accepted = ratio >= kEta;
			if (!accepted && g_trial.allFinite()) {
				// Second-order correction: g re-linearized as g(y + d) - J d.
				set_sides(*y, g_trial - J * step);
				double slack_soc;
				if (solve(&d_soc, &slack_soc)) {
					ratio = try_step(d_soc);
					accepted = ratio >= kEta;
				}
			}
			const double step_norm = tr_norm(step);
			if (_debug)
				std::fprintf(stderr, "sl1lp it %d V %.3e slack %.3e nu %.1e delta %.2e pred %.3e ratio %.3f |d| %.2e %s\n",
					     it, V, slack, nu, delta, pred, ratio, step_norm, accepted ? "acc" : "rej");
			if (accepted) {
				*y = y_trial;
				evaluate(*y, &g, &J);
				if (ratio > 0.75 && step_norm >= 0.99 * delta) delta = std::min(2.0 * delta, kDeltaMax);
			} else {
				delta = 0.25 * step_norm;
				if (delta < kDeltaMin) break;
			}
		}
		record->outer = it;
		record->rho = nu;
		record->stalled = violation(g).lpNorm<Eigen::Infinity>() >= kFeasTol && it < kMaxIter;
	}

private:
	// The LP over (d, p, q), loaded once per solve with A, J's pattern and
	// the elastic columns; every later change is in place, keeping the basis.
	void Load(const LocalNLP& nlp, const SparseRows& J) {
		const int n = nlp.n(), m_lin = static_cast<int>(nlp.A.rows()), m = static_cast<int>(nlp.l.size());
		if (_lpi) SCIPlpiFree(&_lpi);
		LPI_CHECK(SCIPlpiCreate(&_lpi, nullptr, "sl1lp", SCIP_OBJSEN_MINIMIZE));
		LPI_CHECK(SCIPlpiSetIntpar(_lpi, SCIP_LPPAR_LPINFO, FALSE));
		LPI_CHECK(SCIPlpiSetRealpar(_lpi, SCIP_LPPAR_FEASTOL, kLPTol));
		LPI_CHECK(SCIPlpiSetRealpar(_lpi, SCIP_LPPAR_DUALFEASTOL, kLPTol));
		const int cols = n + 2 * m;
		std::vector<Eigen::Triplet<double>> t;
		for (int r = 0; r < m_lin; ++r)
			for (SparseRows::InnerIterator it(nlp.A, r); it; ++it) t.emplace_back(r, it.col(), it.value());
		for (int r = 0; r < m; ++r) {
			for (SparseRows::InnerIterator it(J, r); it; ++it) t.emplace_back(m_lin + r, it.col(), it.value());
			t.emplace_back(m_lin + r, n + r, 1.0);
			t.emplace_back(m_lin + r, n + m + r, -1.0);
		}
		Eigen::SparseMatrix<double, Eigen::ColMajor, int> M(m_lin + m, cols);
		M.setFromTriplets(t.begin(), t.end());
		M.makeCompressed();
		// Every J pattern entry is in M even at a zero value.
		std::vector<int> beg(M.outerIndexPtr(), M.outerIndexPtr() + cols);
		std::vector<double> zeros(cols, 0.0), lo(m_lin + m, 0.0), hi(m_lin + m, 0.0);
		LPI_CHECK(SCIPlpiLoadColLP(_lpi, SCIP_OBJSEN_MINIMIZE, cols, zeros.data(), zeros.data(), zeros.data(),
					   nullptr, m_lin + m, lo.data(), hi.data(), nullptr,
					   static_cast<int>(M.nonZeros()), beg.data(), M.innerIndexPtr(), M.valuePtr()));
	}

	static constexpr int kMaxIter = 200;
	static constexpr double kFeasTol = 1e-9, kLinearTol = 1e-9, kLPTol = 1e-9, kPredTol = 1e-12;
	static constexpr double kEta = 0.1, kSteer = 0.1;
	static constexpr double kNu0 = 10.0, kNuMaxRatio = 1e8;
	static constexpr double kDelta0 = 0.5, kDeltaMin = 1e-10, kDeltaMax = 10.0;
	SCIP_LPI* _lpi = nullptr;
	const bool _debug = std::getenv("GOC_SL1LP_DEBUG") != nullptr;
};

}  // namespace

std::unique_ptr<LocalSolver> MakeLocalSolver(LocalNLPSolver kind) {
	switch (kind) {
	case LocalNLPSolver::kPenalizedLM: return std::make_unique<PenalizedLM>();
	case LocalNLPSolver::kProxQPLM: return std::make_unique<ProxQPLM>();
	case LocalNLPSolver::kSl1LP: return std::make_unique<Sl1LP>();
	}
	throw std::runtime_error("MakeLocalSolver: unknown LocalNLPSolver");
}
