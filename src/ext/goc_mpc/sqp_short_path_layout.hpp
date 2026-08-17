#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "graph_of_constraints.hpp"
#include "obstacle_projection.hpp"
#include "obstacle_set.hpp"
#include "../configuration_spline.hpp"

// Problem-layout/assembly math for SqpShortPathMPC (sqp_short_path_mpc.hpp),
// mirroring timing_gn_layout.{hpp,cpp}'s split from graph_timing_mpc.cpp:
// this file has no qpOASES/trust-region-loop code, only the Hessian/RHS/
// constraint-row math the solver assembles into a QP every outer iteration.
//
// Scope: Block::R, Block::Torus, and (Stage 4) Block::SO3Quat
// (BuildAgentShapes still throws on SO3Mat -- out of scope permanently,
// matches CubicConfigurationSpline's own SO3Mat throw everywhere else) and
// fk(q) = q[:workspace_dim] (a constant 0/1 selection Jacobian -- see
// LinearizeObstacleConstraints). R/Torus tangent columns are independent
// scalars with an ITERATION-CONSTANT Hessian (BuildAxisHessianBlock/
// BuildAxisRhs, built once per solve() call, see AssembleSmoothHessian's
// own comment); an SO3Quat block's 3 tangent columns are COUPLED (SO(3)
// has real curvature) and its Hessian genuinely depends on the current
// iterate, so it gets separate, per-outer-iteration treatment
// (AccumulateSO3QuatBlock) instead -- see the project plan's Stage 4
// section for the full rationale.
namespace sqp_short_path {

// One scalar decision axis: a single tangent-space component of one
// agent's configuration, repeated identically at every horizon step.
// `tangent_col` is this axis's column within that agent's OWN (H x
// agent_tangent_dim) step matrix -- exactly the column
// BlockPositionDelta/BlockRetract's per-block outputs are sliced from.
// Used for R/Torus axes only -- an SO3Quat block's 3 tangent columns are
// identified via CubicConfigurationSpline::BlockOffset instead (see
// AccumulateSO3QuatBlock), since they can't be treated independently.
struct AxisLayout {
	int agent;
	int tangent_col;
};

// Per-agent shape (block layout + ambient/tangent dims), one entry per
// agent, built once from graph._robot_specs and reused for every
// BlockPositionDelta/BlockRetract call this solver makes. Throws if any
// agent has a Block::SO3Mat component (SO3Quat is supported, Stage 4).
std::vector<CubicConfigurationSpline> BuildAgentShapes(const GraphOfConstraints& graph, int num_agents);

// Flat axis list across every agent, agent-major / tangent_col-minor (so
// `agent_axis_offset[ag] + k` is axis 0-indexed within the returned
// vector) -- fixed for the lifetime of a solver instance.
std::vector<AxisLayout> BuildAxisList(const std::vector<CubicConfigurationSpline>& agent_shapes);
std::vector<int> BuildAgentAxisOffsets(const std::vector<CubicConfigurationSpline>& agent_shapes);

// Decision-vector indices for axis `axis`, step `step`, within the
// flattened (num_axes * 2 * num_steps) QP position/velocity-step block --
// slack variables (one per linearized inequality row) are appended after
// this block by the solver, not tracked here.
inline int IdxP(int axis, int step, int num_steps) { return axis * 2 * num_steps + 2 * step; }
inline int IdxV(int axis, int step, int num_steps) { return axis * 2 * num_steps + 2 * step + 1; }

// Relative weight on each smooth-cost term -- multiplies that term's
// squared residual (same "just a scalar multiplier on the whole term"
// semantics as GraphTimingMPC's own `acceleration_cost`, not a change to
// any term's internal shape). Defaults (all 1.0) reproduce this solver's
// original, un-tunable behavior byte-for-byte. `acceleration` in
// particular already has a huge built-in stiffness relative to tracking
// (the coast-corrected residual's own coefficients scale as ~1/tau^2, e.g.
// 600 at tau=0.1s vs tracking's un-weighted 1) -- raising `tracking`/
// `velocity_tracking` (or lowering `acceleration`) is how to make
// reference-tracking more competitive with smoothness, e.g. to keep a
// short-horizon obstacle detour from drifting off the reference for the
// rest of the horizon instead of returning to it.
struct SmoothCostWeights {
	double tracking = 1.0;
	double velocity_tracking = 1.0;
	double acceleration = 1.0;
};

// The smooth (tracking + velocity-tracking + acceleration-smoothing)
// cost's per-axis (2*num_steps x 2*num_steps) Hessian block. IDENTICAL for
// every axis -- the coefficient pattern never depends on which axis/agent
// it's for, only `tau`/`weights` (see this class's project-plan design
// decision 3) -- and constant across outer SQP iterations within one
// solve() call, since it's an honest quadratic form in the STEP variables,
// independent of the current iterate.
Eigen::MatrixXd BuildAxisHessianBlock(int num_steps, double tau, const SmoothCostWeights& weights);

// Assembles the full (n x n) smooth-cost Hessian, `n = axes.size() * 2 *
// num_steps` (axes.size() summed over every agent's FULL tangent_dim,
// R/Torus and SO3Quat columns alike -- see BuildAxisList), by placing
// BuildAxisHessianBlock's block once per R/Torus axis. SO3Quat axes are
// deliberately left ZERO here: their real (coupled, iterate-dependent)
// contribution is added separately, every outer iteration, by
// AccumulateSO3QuatBlock -- this function only ever needs calling ONCE per
// solve() call regardless (its own R/Torus content is genuinely constant,
// same as before Stage 4; it just no longer claims to be the WHOLE smooth
// Hessian when SO3Quat blocks are present).
Eigen::MatrixXd AssembleSmoothHessian(const std::vector<CubicConfigurationSpline>& agent_shapes,
				       const std::vector<int>& agent_axis_offsets,
				       int num_steps, double tau, const SmoothCostWeights& weights);

// Per-axis RHS/target vector (length 2*num_steps) for the smooth cost,
// evaluated AT THE CURRENT ITERATE (`points_agent`/`vels_agent`, H x
// agent_dim, already sliced to this axis's agent) against `x0_agent`/
// `v0_agent` (anchors) and `ref_points_agent`/`ref_velocities_agent`.
// Manifold-aware via `agent_shape.PositionDelta<double>` for every
// position-flavored term (tracking's ref-vs-current delta, and the
// acceleration term's inter-step displacement) -- see the .cpp for the
// full derivation (linearizing each already-quadratic-in-absolute-
// coordinates term around the current iterate). Depends on the current
// iterate, unlike the Hessian -- rebuild every outer iteration. `weights`
// MUST match whatever was passed to BuildAxisHessianBlock for this same
// solve -- the two build the two halves (H, g) of the SAME normal
// equations and will silently disagree if they diverge.
Eigen::VectorXd BuildAxisRhs(const AxisLayout& axis,
			      const CubicConfigurationSpline& agent_shape,
			      int num_steps, double tau, const SmoothCostWeights& weights,
			      const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			      const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			      const Eigen::MatrixXd& ref_points_agent,
			      const Eigen::MatrixXd& ref_velocities_agent);

// Identifies one SO3Quat block within one agent for AccumulateSO3QuatBlock
// below: `offset` is the block's own BlockOffset (giving its
// ambient_offset/tangent_offset WITHIN the agent's own ambient/tangent
// vectors -- e.g. points_agent.row(i).segment(offset.ambient_offset, 4) is
// this block's own quaternion at step i), `axis_offset` is where its 3
// tangent columns start in the FLAT (agent-major, tangent-col-minor) axis
// indexing IdxP/IdxV use (== agent_axis_offsets[agent] +
// offset.tangent_offset) -- its 3 columns occupy 3 CONSECUTIVE flat axis
// indices, same layout 3 independent R/Torus axes would get; this struct
// just marks them for AccumulateSO3QuatBlock's separate, coupled,
// per-outer-iteration treatment instead of BuildAxisHessianBlock/
// BuildAxisRhs's per-axis-scalar, iteration-constant one.
struct SO3QuatBlock {
	int agent;
	CubicConfigurationSpline::BlockOffset offset;
	int axis_offset;
};

// Accumulates ONE SO3Quat block's tracking + velocity-tracking +
// acceleration smooth-cost contribution into the FULL (n_smooth x
// n_smooth) Hessian `H` and (n_smooth) RHS `g` -- ADDED (+=) at this
// block's own scattered (IdxP/IdxV-indexed) rows/cols, not overwritten, so
// callers place the constant R/Torus contribution first
// (AssembleSmoothHessian/BuildAxisRhs) then call this once per SO3Quat
// block on top; every OTHER row/col (R/Torus axes, other SO3Quat blocks)
// is left untouched. UNLIKE AssembleSmoothHessian, this must be called
// EVERY outer iteration, not built once and reused: the coupled 3x3
// Jacobian (CubicConfigurationSpline::ComputeSO3QuatResidual) genuinely
// depends on the CURRENT iterate's residual value, not just its target --
// see the project plan's Stage 4 design decision for why R/Torus don't
// have this problem and SO3Quat does.
//
// Same normal-equations convention (`H_n s = g_n`) as
// BuildAxisHessianBlock/BuildAxisRhs -- the caller applies the identical
// qpOASES 2x/-2x conversion uniformly across the whole assembled matrix,
// not per block. Derivation (tracking and acceleration each linearized via
// ComputeSO3QuatResidual's Jacobians, velocity-tracking flat/diagonal same
// as R/Torus): see the .cpp and project_sqp_short_path_mpc memory.
void AccumulateSO3QuatBlock(const SO3QuatBlock& block, const CubicConfigurationSpline& agent_shape,
			     int num_steps, double tau, const SmoothCostWeights& weights,
			     const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			     const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			     const Eigen::MatrixXd& ref_points_agent,
			     const Eigen::MatrixXd& ref_velocities_agent,
			     Eigen::MatrixXd* H, Eigen::VectorXd* g);

// One linearized inequality constraint row `c(x) + a^T (dx) >= 0` (before
// slack relaxation, which the solver -- not this file -- owns): `coeffs`
// gives `a`'s nonzero entries (decision-vector index, coefficient) and
// `value` gives `c(x)` at the current iterate. The solver forms the
// slack-relaxed QP row `a^T dx + s >= -value`, `s >= 0` from these two
// fields directly.
struct ConstraintRow {
	std::vector<std::pair<int, double>> coeffs;
	double value;
};

// Actual (not quadratic-model) smooth-cost VALUE for one agent, at ANY
// given absolute `points_agent`/`vels_agent` -- i.e. the same tracking +
// velocity-tracking + acceleration residuals BuildAxisRhs's normal
// equations are derived from, evaluated directly rather than through the
// per-outer-iteration linear model. Used by the solver's merit function
// (actual, not predicted, cost at a candidate step) -- NOT part of the QP
// assembly itself. `weights` MUST match BuildAxisHessianBlock/BuildAxisRhs's
// (see those functions' own comments) -- this is the merit function's
// "ground truth" f(x); if it scores a different objective than the one the
// QP is actually stepping toward, the trust-region ratio test (predicted
// vs. actual reduction) becomes meaningless.
double EvaluateSmoothCost(const CubicConfigurationSpline& agent_shape, int num_steps, double tau,
			   const SmoothCostWeights& weights,
			   const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			   const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			   const Eigen::MatrixXd& ref_points_agent, const Eigen::MatrixXd& ref_velocities_agent);

// Per-agent obstacle pointers "close enough to plausibly matter" over this
// solve() call's horizon: agent ag's own REFERENCE-trajectory bounding
// sphere (obstacle_projection.hpp's TrajectoryBoundingSphere) vs each
// registered obstacle's own extent (sphere: radius+margin; box: half-
// extents' norm+margin, a conservative circumscribing-sphere proxy) --
// included if the two spheres could plausibly come within `prune_margin`
// of touching. Computed ONCE per solve() call (SqpShortPathMPC::solve(),
// alongside `ref_points`), reused for the whole call -- same "row COUNT
// fixed for the whole solve() call" discipline design decision 6 already
// established for the unpruned case (row COEFFICIENTS still get
// relinearized every outer iteration; which rows EXIST does not, and
// pruning is exactly one more thing computed once and held fixed).
//
// PURELY a speed optimization on the SQP's own optimization path --
// pointers into `obstacles`, valid only for the caller's own solve() call
// (ObstacleSet isn't mutated mid-call). SqpShortPathMPC's own
// ApplySafetyProjection final hard-feasibility pass (sqp_short_path_mpc.cpp)
// checks EVERY registered obstacle for every agent regardless of this
// pruning, so an obstacle wrongly excluded here degrades solution
// SMOOTHNESS around it (handled by that pass's less-smooth closed-form
// fallback instead of the SQP loop optimizing around it), never
// correctness/feasibility -- see that function's own comment for why this
// asymmetry is safe to lean on.
std::vector<std::vector<const Obstacle*>> PruneObstaclesByDistance(
	const Eigen::MatrixXd& ref_points, int num_agents, int ambient_dim, int workspace_dim,
	const ObstacleSet& obstacles, double prune_margin);

// Same idea for inter-agent pairs: (ag_a, ag_b) survives if their own
// reference-trajectory bounding spheres could plausibly bring them within
// `agent_radii(ag_a) + agent_radii(ag_b) + prune_margin` of each other.
// ApplyAgentPairSafetyProjection's final pass checks EVERY pair
// regardless -- same non-issue-for-correctness property as above. This is
// the more consequential of the two prunings in practice: unpruned pair
// count grows as `num_agents*(num_agents-1)/2`, quadratic in agent count,
// while unpruned obstacle rows only grow linearly in obstacle count.
std::vector<std::pair<int, int>> PruneAgentPairsByDistance(
	const Eigen::MatrixXd& ref_points, int num_agents, int ambient_dim, int workspace_dim,
	const Eigen::VectorXd& agent_radii, double prune_margin);

// Total obstacle-constraint violation (Sum of max(0, -c(q)) over every
// (step, agent, obstacle) SURVIVING PruneObstaclesByDistance) at the given
// absolute `points` -- the merit function's penalty term, evaluated with
// the TRUE (non-linearized) sphere/box signed distance, not
// LinearizeObstacleConstraints' local model. Deliberately scoped to the
// SAME pruned set LinearizeObstacleConstraints' rows come from (not the
// full unpruned obstacle list) -- the merit function must score exactly
// what the QP can act on, or the trust-region ratio test (predicted vs.
// actual reduction) becomes meaningless for whatever it silently omitted.
// `ambient_dim`: per-agent AMBIENT stride used to slice `points`
// (points.row(i).segment(ag*ambient_dim, workspace_dim) is agent ag's
// world position at step i) -- NOT the tangent/decision-space stride
// `agent_axis_offsets` is built from; the two differ once an agent has an
// SO3Quat block (ambient 4 vs tangent 3 per block, see the project plan's
// Stage 4). Uniform across agents (same assumption GraphShortPathMPC's own
// a_dim/t_dim split already makes).
double EvaluateObstacleViolation(int num_steps, int num_agents, int ambient_dim, int workspace_dim,
				  const Eigen::MatrixXd& points,
				  const std::vector<std::vector<const Obstacle*>>& per_agent_obstacles);

// Every (step, agent, obstacle) row for `per_agent_obstacles[ag]` (see
// PruneObstaclesByDistance's own comment), linearized at the current
// iterate `points`. Fast path only: assumes `fk(q) = q[:workspace_dim]`
// (points.row(i).segment(ag*ambient_dim, workspace_dim) IS the agent's
// world position, so the constraint's Jacobian w.r.t. the step is a
// constant 0/1 selection onto that agent's leading `workspace_dim` axes at
// that step -- no chain rule beyond the slice). `agent_axis_offsets` is
// still TANGENT-space (see AccumulateSO3QuatBlock's own comment) -- it's
// what the resulting row's Jacobian columns are indexed into, since the
// decision variables (and hence the QP) live in tangent space regardless
// of `ambient_dim`.
std::vector<ConstraintRow> LinearizeObstacleConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents, int ambient_dim,
	int workspace_dim, const Eigen::MatrixXd& points,
	const std::vector<std::vector<const Obstacle*>>& per_agent_obstacles);

// Inter-agent avoidance: every (step, agent-pair) SURVIVING
// PruneAgentPairsByDistance is treated as a sphere constraint on the
// PAIR's separation (same sdf-style shape as LinearizeObstacleConstraints's
// sphere case), except now BOTH endpoints are decision variables (agent
// b's position is agent a's "obstacle center" and vice versa), so the
// linearized row has nonzero coefficients on both agents' position steps.
// Deliberately position-based, not velocity-based (ORCA/velocity-obstacle
// was tried and rejected for this solver -- see the project plan's Stage 2
// notes: this solver already plans full-horizon POSITIONS jointly for
// every agent, which already gives per-step anticipation and rules out the
// reciprocal-double-counting concern velocity-obstacle constructions solve
// for, so there was no offsetting benefit to pay for the indirect,
// weight-dependent, and non-guaranteed velocity-to-position coupling a
// velocity-space constraint would have needed). `agent_radii` is one entry
// per agent (index-aligned with `agent_axis_offsets`); a pair's combined
// avoidance radius is `agent_radii(ag_i) + agent_radii(ag_j)` (0 is a
// valid default -- agents are still treated as points that must not
// occupy the same position at the same step, exactly like every other
// point-agent assumption already made in this fast-path solver).
//
// Fast path only (same assumption as LinearizeObstacleConstraints): agent
// positions are read directly from the leading `workspace_dim` ambient
// columns, no fk chain rule.
double EvaluateAgentPairViolation(int num_steps, int ambient_dim, int workspace_dim,
				   const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
				   const std::vector<std::pair<int, int>>& active_pairs);

std::vector<ConstraintRow> LinearizeAgentPairConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int ambient_dim, int workspace_dim,
	const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
	const std::vector<std::pair<int, int>>& active_pairs);

}  // namespace sqp_short_path
