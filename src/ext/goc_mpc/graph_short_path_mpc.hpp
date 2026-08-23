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
//      direct, principled fix for a plain-SLSQP formulation (every obstacle
//      constraint a hard AddConstraint) failing 35-53% of the time with
//      several simultaneously-active nonconvex constraints, not a tuning
//      fix. Accept/reject uses the merit function phi(x) = f(x) + penalty_weight
//      * sum(max(0,-c_i(x))) instead of GraphTimingMPC's plain objective
//      ratio -- see graph_short_path_mpc.cpp's RunTrustRegionSqp.
//   2. State lives on the Riemannian product manifold R^k x Torus^k
//      (Block::R / Block::Torus -- SO3Quat/SO3Mat are a later stage, see
//      sqp_short_path_layout.hpp's own doc comment), applied via
//      RETRACTION (CubicConfigurationSpline::Retract) every accepted step,
//      not raw ambient addition -- correct Torus wraparound without the
//      per-cycle retrace cost a jax/pymanopt-based approach would incur.
//      Unlike GraphTimingMPC's flat-Euclidean `x += dx` accumulation across many
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
// Public interface: solve(x0, v0, var_assignments, remaining_vertices,
// references) -> bool plus view_points/view_vels/view_times/view_obstacles/
// get_last_solve_time -- var_assignments/remaining_vertices are accepted but
// UNUSED, kept only so a caller can swap in a different duck-typed solver via
// GraphOfConstraintsMPC's `short_path_mpc=` override (see that class's own
// doc comment) without changing its call site.
struct GraphShortPathMPC {
	const GraphOfConstraints* _graph;
	unsigned int _num_steps, _num_agents;
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
	// Distance-based QP-row pruning margin (sqp_short_path_layout.hpp's
	// PruneObstaclesByDistance/PruneAgentPairsByDistance) -- an obstacle/
	// agent-pair whose reference-trajectory bounding spheres can't come
	// within this extra margin of touching gets no QP row at all this
	// solve() call. Purely a speed lever: ApplySafetyProjection/
	// ApplyAgentPairSafetyProjection (the final hard-feasibility passes)
	// check every registered obstacle/pair regardless, so this never
	// affects correctness, only how much the SQP loop itself gets to
	// smoothly plan around vs. leaving to that closed-form fallback.
	double _constraint_prune_margin;

	// Fixed for this instance's whole lifetime (depend only on `graph`/
	// `time_per_step`, never on a particular solve() call's
	// references/obstacles) -- built once in the constructor. Agents are
	// NOT required to share a tangent_dim/ambient_dim with each other
	// (Stage 5) -- each agent's own entry in `_agent_shapes` is whatever
	// `graph._robot_specs.at(ag)` says, and `_agent_axis_offsets`/
	// `_agent_ambient_offsets` are the CUMULATIVE per-agent offsets every
	// per-agent matrix (points/vels/ref_points/ref_velocities/x0/v0) is
	// sliced with -- NOT a uniform `ag * dim` stride.
	std::vector<CubicConfigurationSpline> _agent_shapes;
	std::vector<sqp_short_path::AxisLayout> _axes;
	std::vector<int> _agent_axis_offsets;
	std::vector<int> _agent_ambient_offsets;
	// The smooth (tracking + velocity-tracking + acceleration-smoothing)
	// cost's normal-equations Hessian FOR EVERY R/TORUS AXIS -- block-
	// diagonal across those axes, PROVABLY constant across every outer SQP
	// iteration AND every solve() call for this instance's lifetime (an
	// honest quadratic form in the step, independent of the current
	// iterate -- see sqp_short_path_layout.hpp's own doc comment), so it's
	// built exactly once here rather than per outer iteration or per
	// solve() call. Any SO3Quat block's own
	// rows/cols are left ZERO here (see AssembleSmoothHessian's own
	// comment) -- their real, COUPLED, iteration-DEPENDENT contribution is
	// computed fresh every outer iteration instead
	// (graph_short_path_mpc.cpp's RunTrustRegionSqp, via
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

	// No default value for `obstacles` -- storing a pointer to a
	// default-constructed temporary would dangle immediately.
	// No `dim` parameter (removed, Stage 5): every agent's tangent_dim/
	// ambient_dim is derived entirely from `graph._robot_specs.at(ag)`
	// (BuildAgentShapes) and agents are no longer required to agree with
	// each other -- one agent can be a Block::R(2) point mass next to
	// another that's Block::Torus(1)+Block::SO3Quat. x0/v0 (and every
	// per-agent matrix this solver builds) are laid out agent-major,
	// AMBIENT-width for x0/points/ref_points and TANGENT-width for v0/vels/
	// agent_radii, using each agent's own width in turn (see
	// `_agent_ambient_offsets`/`_agent_axis_offsets`) -- NOT `num_agents *`
	// a shared per-agent width. For an agent with an SO3Quat block its
	// ambient slice (4 numbers per quat block) is wider than its tangent
	// slice (3 numbers per quat block); this was already true per-agent as
	// of Stage 4, just no longer required to be the SAME width across
	// agents.
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
	// constraint_prune_margin: see `_constraint_prune_margin`'s own comment
	// above. Default (1.0) matches ApplySafetyProjection's own
	// long-standing hardcoded query margin for the same "how far past the
	// reference trajectory's own bounding sphere could this plausibly
	// matter" question -- same conservative choice, now made tunable
	// because unlike that closed-form pass (fixed cost regardless), this
	// margin trades QP row count (and hence solve time) directly against
	// how far from the reference an obstacle/agent-pair the SQP loop is
	// still willing to plan around.
	GraphShortPathMPC(const GraphOfConstraints& graph,
			 unsigned int num_steps,
			 unsigned int num_agents,
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
			 double grad_tol = 1.0e-6,
			 double constraint_prune_margin = 1.0);

	// Explicit (not defaulted inline) for the same reason as
	// GraphTimingMPC's: QpState is only complete in the .cpp, and pybind's
	// py::init(lambda) factory pattern (goc_mpc.cpp) needs this class
	// movable.
	~GraphShortPathMPC();
	GraphShortPathMPC(GraphShortPathMPC&&) noexcept;
	GraphShortPathMPC& operator=(GraphShortPathMPC&&) noexcept;
	GraphShortPathMPC(const GraphShortPathMPC&) = delete;
	GraphShortPathMPC& operator=(const GraphShortPathMPC&) = delete;

	// `var_assignments`/`remaining_vertices` accepted but UNUSED -- kept only
	// so a caller-supplied `short_path_mpc=` override on GraphOfConstraintsMPC
	// (see that class's own doc comment) can be duck-typed to this same
	// signature without changing the Python call site.
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
