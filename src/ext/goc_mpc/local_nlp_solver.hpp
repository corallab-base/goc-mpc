#pragma once

#include <functional>
#include <memory>

#include <Eigen/Core>
#include <Eigen/SparseCore>

// One local NLP solve: whether it reached the caller's feasibility
// tolerance, its problem size, its iteration counts (outer augmented-
// Lagrangian updates, inner Levenberg-Marquardt steps, trial points
// evaluated), the final penalty weight, and its wall time split into
// building the problem, evaluating rows and Jacobians, and the step
// systems (factorizing/solving the LM systems, or the step QPs); total
// includes the rest. `stalled`: it stopped because, with the linear rows
// met, the violation stopped shrinking at the capped penalty weight, not at
// an iteration limit.
struct LocalSolveRecord {
	bool success = false, stalled = false;
	int vars = 0, rows = 0;
	int outer = 0, inner = 0, trials = 0;
	double rho = 0.0;
	double build_time = 0.0, eval_time = 0.0, factor_time = 0.0, total_time = 0.0;
};

// min c.y  s.t.  lb <= y <= ub,  l_lin <= A y + a0 <= u_lin,  l <= g(y) <= u,
// with g (nonlinear rows, cones) given by `evaluate`.
struct LocalNLP {
	using SparseRows = Eigen::SparseMatrix<double, Eigen::RowMajor>;
	Eigen::VectorXd c, lb, ub;
	SparseRows A;
	Eigen::VectorXd a0, l_lin, u_lin;
	Eigen::VectorXd l, u;
	// g's Jacobian pattern, compressed (its values unused).
	SparseRows J;
	// g at y, and its Jacobian's values into a matrix with J's pattern.
	std::function<void(const Eigen::VectorXd& y, Eigen::VectorXd* g, SparseRows* J)> evaluate;

	int n() const { return static_cast<int>(c.size()); }
};

enum class LocalNLPSolver {
	// Augmented Lagrangian on every row, linear ones included; bounded
	// Levenberg-Marquardt inner steps (clipped to the bounds).
	kPenalizedLM,
	// Augmented Lagrangian on g only; each Levenberg-Marquardt step a sparse
	// QP (ProxQP) holding the linear rows and bounds exactly.
	kProxQPLM,
	// Sequential l1 LP (SCIP's LP interface): each step an LP holding the
	// linear rows and bounds exactly, g linearized with elastic slacks in a
	// trust region; an unreachable g ends at a local minimizer of its
	// violation.
	kSl1LP,
};

// Solves a LocalNLP locally from *y and leaves its last iterate in *y,
// feasible or not: the caller checks.
class LocalSolver {
public:
	virtual ~LocalSolver() = default;
	virtual void Solve(const LocalNLP& nlp, Eigen::VectorXd* y, LocalSolveRecord* record) = 0;
};

std::unique_ptr<LocalSolver> MakeLocalSolver(LocalNLPSolver kind);
