#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "obstacle_set.hpp"
#include "sqp_short_path_layout.hpp"
#include "../configuration_spline.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

// Riemannian trust-region SQP short-horizon obstacle-avoidance MPC -- see
// the project plan ("Riemannian trust-region SQP short-path solver") for
// the full design rationale; summarized here.
//
// Same template as GraphTimingMPC (trust-region Gauss-Newton SQP over
// qpOASES::SQProblem, hot-started both across outer iterations and across
// MPC cycles), generalized in the ways that template's own doc comment
// says it would need for genuinely nonconvex, re-linearized-every-iteration
// constraints (unlike timing's exactly-linear interaction rows):
//
//   1. Every obstacle-avoidance inequality is a SLACK-RELAXED exact-penalty
//      row (Sℓ1QP, Nocedal & Wright Sec. 18.5): `c(x) + a^T dx >= -s`,
//      `s >= 0`, `penalty_weight * s` added to the QP objective, instead of
//      a hard hard AddConstraint. This makes every QP subproblem feasible
//      BY CONSTRUCTION (dx=0, s=max(0,-c(x)) is always feasible) -- the
//      direct, principled fix for GraphShortPathMPC's SLSQP path failing
//      35-53% of the time with several simultaneously-active nonconvex
//      constraints (see that class's own doc comment), not a tuning fix.
//      Accept/reject uses the merit function phi(x) = f(x) + penalty_weight
//      * sum(max(0,-c_i(x))) instead of GraphTimingMPC's plain objective
//      ratio -- see sqp_short_path_mpc.cpp's RunTrustRegionSqp.
//   2. State lives on the Riemannian product manifold R^k x Torus^k
//      (Block::R / Block::Torus -- SO3Quat/SO3Mat are a later stage, see
//      sqp_short_path_layout.hpp's own doc comment), applied via
//      RETRACTION (CubicConfigurationSpline::Retract) every accepted step,
//      not raw ambient addition -- correct Torus wraparound without
//      PymanoptShortPathMPC's jax/pymanopt per-cycle retrace cost. Unlike
//      GraphTimingMPC's flat-Euclidean `x += dx` accumulation across many
//      iterations, this solver re-linearizes fresh from the just-retracted
//      point every outer iteration -- standard Riemannian trust-region
//      practice, and required here (accumulating a raw step across several
//      iterations without retracting isn't meaningful once a Torus
//      component is involved).
//   3. fk fast path only (v1): `fk(q) = q[:workspace_dim]`, so an obstacle
//      constraint's Jacobian w.r.t. the step is a constant 0/1 selection --
//      no chain rule, no C++ autodiff (none exists anywhere in this repo
//      today). General (articulated-robot) fk is explicitly out of scope,
//      not attempted here -- see the project plan.
//
// Interface mirrors GraphShortPathMPC/AdmmShortPathMPC exactly (solve(x0,
// v0, var_assignments, remaining_vertices, references) -> bool plus
// view_points/view_vels/view_times/view_obstacles/get_last_solve_time), so
// it's a drop-in `short_path_mpc=` override on GraphOfConstraintsMPC (same
// hook Admm/PymanoptShortPathMPC already use) -- NOT the default solver
// yet, see the project plan's verification section for why.
struct SqpShortPathMPC {
	const GraphOfConstraints* _graph;
	unsigned int _num_steps, _num_agents, _dim;
	// AMBIENT per-agent stride, derived (not caller-supplied) from
	// `_agent_shapes` in the constructor -- equals `_dim` (== tangent_dim)
	// for R/Torus-only agents (ambient_dim==tangent_dim always holds for
	// those), differs once an agent has an SO3Quat block (ambient 4 vs
	// tangent 3 per block). `_dim` stays the TANGENT/decision-space stride
	// throughout this file (what IdxP/IdxV, vels, v0, agent_radii use);
	// `_ambient_dim` is only for x0/points/ref_points (AMBIENT
	// configuration) slicing. Uniform across agents (same simplifying
	// assumption GraphShortPathMPC's own a_dim/t_dim split already makes),
	// validated in the constructor.
	unsigned int _ambient_dim;
	double _time_per_step;
	Eigen::VectorXd _times;

	const ObstacleSet* _obstacles;
	// One entry per agent -- a pair's combined inter-agent avoidance radius
	// is _agent_radii(a)+_agent_radii(b). Owned (not by-pointer like
	// `_obstacles`): unlike ObstacleSet, there's no live-mutation use case
	// for this yet, so a plain owned copy is simpler and avoids one more
	// caller-must-keep-alive lifetime contract.
	Eigen::VectorXd _agent_radii;

	sqp_short_path::SmoothCostWeights _smooth_cost_weights;
	double _penalty_weight;
	int _max_iterations;
	double _initial_trust_radius, _max_trust_radius, _min_trust_radius, _grad_tol;

	// Fixed for this instance's whole lifetime (depend only on `graph`/
	// `dim`/`time_per_step`, never on a particular solve() call's
	// references/obstacles) -- built once in the constructor.
	std::vector<CubicConfigurationSpline> _agent_shapes;
	std::vector<sqp_short_path::AxisLayout> _axes;
	std::vector<int> _agent_axis_offsets;
	// The smooth (tracking + velocity-tracking + acceleration-smoothing)
	// cost's normal-equations Hessian FOR EVERY R/TORUS AXIS -- block-
	// diagonal across those axes, PROVABLY constant across every outer SQP
	// iteration AND every solve() call for this instance's lifetime (an
	// honest quadratic form in the step, independent of the current
	// iterate -- see sqp_short_path_layout.hpp's own doc comment), so it's
	// built exactly once here rather than per-call like AdmmShortPathMPC's
	// analogous (but per-call-only) build_hessian. Any SO3Quat block's own
	// rows/cols are left ZERO here (see AssembleSmoothHessian's own
	// comment) -- their real, COUPLED, iteration-DEPENDENT contribution is
	// computed fresh every outer iteration instead
	// (sqp_short_path_mpc.cpp's RunTrustRegionSqp, via
	// AccumulateSO3QuatBlock), on top of a per-iteration COPY of this
	// matrix, not this cached member itself.
	Eigen::MatrixXd _smooth_hessian_normal;

	Eigen::MatrixXd _points;
	Eigen::MatrixXd _vels;
	bool _has_solved = false;

	// Pimpl (like GraphTimingMPC::QpState) -- qpOASES::SQProblem is
	// fixed-size once constructed and not cleanly move/resize-able, so a
	// forward-declared incomplete type here, completed in the .cpp, keeps
	// qpOASES.hpp out of every translation unit that includes this header.
	struct QpState;
	std::unique_ptr<QpState> _qp_state;

	drake::SteadyTimer _timer;
	double _last_solve_time = 0.0;
	int _last_iterations = 0;
	double _last_trust_radius = 0.0;

	// No default value for `obstacles` -- same lifetime discipline as
	// GraphShortPathMPC/AdmmShortPathMPC (storing a pointer to a
	// default-constructed temporary would dangle immediately).
	// dim: every agent's TANGENT width (== ambient width for R/Torus-only
	// agents, the only case before Stage 4 -- so this is a fully
	// backward-compatible reinterpretation, not a behavior change for
	// existing callers). x0/points/ref_points are AMBIENT-width instead
	// (`_ambient_dim`, derived from `graph`/`agent_shapes` internally, see
	// that member's own comment) -- for an agent with an SO3Quat block,
	// `dim` must be its tangent_dim (3 per quat block), NOT its ambient_dim
	// (4 per quat block); x0 must be sized with the agent's AMBIENT width
	// accordingly (a quaternion needs 4 numbers to specify, even though its
	// own step lives in a 3-dim tangent space).
	// agent_radii: one entry per agent, default EMPTY meaning "every agent
	// has radius 0" (point agents that still must not occupy the same
	// position at the same step -- see LinearizeAgentPairConstraints's own
	// comment for why this is position-, not velocity-, based). Passing a
	// non-empty vector requires exactly `num_agents` entries.
	// tracking_weight/velocity_tracking_weight/acceleration_weight: relative
	// weight on each smooth-cost term (see SmoothCostWeights's own doc
	// comment in sqp_short_path_layout.hpp) -- defaults (all 1.0) reproduce
	// this solver's original behavior byte-for-byte. The acceleration term
	// already has a large BUILT-IN stiffness relative to tracking (its own
	// coefficients scale as ~1/time_per_step^2), so raising tracking_weight/
	// velocity_tracking_weight (or lowering acceleration_weight) is how to
	// make reference-tracking more competitive with smoothness -- e.g. to
	// stop a short-horizon obstacle detour from drifting off the reference
	// for the rest of the horizon instead of returning to it.
	SqpShortPathMPC(const GraphOfConstraints& graph,
			 unsigned int num_steps,
			 unsigned int num_agents,
			 unsigned int dim,
			 double time_per_step,
			 const ObstacleSet& obstacles,
			 Eigen::VectorXd agent_radii = Eigen::VectorXd(),
			 double tracking_weight = 1.0,
			 double velocity_tracking_weight = 1.0,
			 double acceleration_weight = 1.0,
			 double penalty_weight = 1.0e3,
			 int max_iterations = 30,
			 double initial_trust_radius = 0.5,
			 double max_trust_radius = 5.0,
			 double min_trust_radius = 1.0e-6,
			 double grad_tol = 1.0e-6);

	// Explicit (not defaulted inline) for the same reason as
	// GraphTimingMPC's: QpState is only complete in the .cpp, and pybind's
	// py::init(lambda) factory pattern (goc_mpc.cpp) needs this class
	// movable.
	~SqpShortPathMPC();
	SqpShortPathMPC(SqpShortPathMPC&&) noexcept;
	SqpShortPathMPC& operator=(SqpShortPathMPC&&) noexcept;
	SqpShortPathMPC(const SqpShortPathMPC&) = delete;
	SqpShortPathMPC& operator=(const SqpShortPathMPC&) = delete;

	// `var_assignments`/`remaining_vertices` accepted but UNUSED -- matches
	// GraphShortPathMPC/AdmmShortPathMPC's own signature exactly (see their
	// own comments), letting GraphOfConstraintsMPC's Python call site treat
	// any of the three interchangeably.
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const Eigen::VectorXi& var_assignments,
		   const std::vector<int>& remaining_vertices,
		   const std::vector<CubicConfigurationSpline>& references);

	const Eigen::MatrixXd& view_points() { return _points; }
	const Eigen::MatrixXd& view_vels() { return _vels; }
	const Eigen::VectorXd& view_times() { return _times; }
	const ObstacleSet& view_obstacles() { return *_obstacles; }
	double get_last_solve_time() { return _last_solve_time; }
	int get_last_iterations() { return _last_iterations; }
	double get_last_trust_radius() { return _last_trust_radius; }
};
