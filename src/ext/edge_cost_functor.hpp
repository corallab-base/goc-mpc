#pragma once

#include <drake/common/autodiff.h>
#include <drake/common/eigen_types.h>

// Abstract interface for an arbitrary waypoint routing arc cost:
// cost(agent, w_a, w_b). Not called "metric" — an arrival-time-style cost
// (e.g. Fast Marching Method output) need not be symmetric or satisfy a
// triangle inequality, so "metric" would overclaim.
//
// Virtual dispatch and templates don't mix, so this mirrors Drake's own
// EvaluatorBase/Cost pattern (drake/solvers/evaluator_base.h): one overload
// per scalar type actually needed (double for plain numeric evaluation,
// AutoDiffXd for a future Drake Cost/AddCost consumer), no generic template
// method. No symbolic::Expression overload — same scope as Drake's
// FunctionEvaluator, which doesn't support symbolic evaluation either.
//
// Concrete implementations must be written in C++ (not overridden from
// Python): a Python-side override could only ever produce a double, and a
// naively-stubbed AutoDiffXd override would silently hand gradient-based
// solvers zero/wrong derivatives instead of failing loudly.
class EdgeCostFunctor {
public:
	virtual ~EdgeCostFunctor() = default;

	virtual double Evaluate(int agent,
				const Eigen::VectorXd& w_a,
				const Eigen::VectorXd& w_b) const = 0;

	virtual drake::AutoDiffXd Evaluate(int agent,
					   const drake::VectorX<drake::AutoDiffXd>& w_a,
					   const drake::VectorX<drake::AutoDiffXd>& w_b) const = 0;
};
