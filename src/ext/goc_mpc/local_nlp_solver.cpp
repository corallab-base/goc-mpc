#include "local_nlp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/SparseCholesky>
#include <proxsuite/proxqp/sparse/sparse.hpp>

namespace {

using SparseRows = LocalNLP::SparseRows;
using SpMat = Eigen::SparseMatrix<double>;
using Clock = std::chrono::steady_clock;
constexpr double kInf = std::numeric_limits<double>::infinity();

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
			if (viol > 0.25 * prev_viol) rho = std::min(rho * 10.0, 1e10);
			prev_viol = viol;
		}
		record->outer = outer + (outer < kMaxOuter ? 1 : 0);
		record->rho = rho;
	}

protected:
	static constexpr int kMaxOuter = 30, kMaxInner = 100;
	static constexpr double kFeasTol = 1e-9;

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

}  // namespace

std::unique_ptr<LocalSolver> MakeLocalSolver(LocalNLPSolver kind) {
	switch (kind) {
	case LocalNLPSolver::kPenalizedLM: return std::make_unique<PenalizedLM>();
	case LocalNLPSolver::kProxQPLM: return std::make_unique<ProxQPLM>();
	}
	throw std::runtime_error("MakeLocalSolver: unknown LocalNLPSolver");
}
