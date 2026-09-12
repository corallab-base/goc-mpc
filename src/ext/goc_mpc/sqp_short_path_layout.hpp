#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "agent_collision_model.hpp"
#include "graph_of_constraints.hpp"
#include "obstacle_projection.hpp"
#include "obstacle_set.hpp"
#include "../configuration_spline.hpp"

// Problem-layout/assembly math for GraphShortPathMPC (graph_short_path_mpc.hpp),
// mirroring timing_gn_layout.{hpp,cpp}'s split from graph_timing_mpc.cpp:
// this file has no qpOASES/trust-region-loop code, only the Hessian/RHS/
// constraint-row math the solver assembles into a QP every outer iteration.
//
// Scope: Block::R, Block::Torus, and (Stage 4) Block::SO3Quat
// (BuildAgentShapes still throws on SO3Mat -- out of scope permanently,
// matches CubicConfigurationSpline's own SO3Mat throw everywhere else).
// Collision geometry is a per-agent AgentCollisionModel (agent_collision_
// model.hpp): the constraint-row / violation paths loop over each agent's
// workspace spheres and chain d(sdf)/d(centre) through each sphere's
// tangent Jacobian. A trivial model (nothing registered) yields one
// radius-0 sphere at q[:workspace_dim] with a constant [I|0] Jacobian, so
// those paths reduce exactly to the old fk(q) = q[:workspace_dim] fast
// path. R/Torus tangent columns are independent
// scalars with an ITERATION-CONSTANT Hessian (BuildAxisHessianBlock/
// BuildAxisRhs, built once per solve() call, see AssembleSmoothHessian's
// own comment); an SO3Quat block's 3 tangent columns are COUPLED (SO(3)
// has real curvature) and its Hessian genuinely depends on the current
// iterate, so it gets separate, per-outer-iteration treatment
// (AccumulateSO3QuatBlock) instead -- see the project plan's Stage 4
// section for the full rationale.
//
// Stage 5: agents are no longer required to share a common tangent_dim or
// ambient_dim with each other -- each agent's own CubicConfigurationSpline
// (BuildAgentShapes, built from that agent's own graph._robot_specs entry)
// is already fully heterogeneous; the only thing that used to force
// uniformity was GraphShortPathMPC slicing every per-agent matrix with a
// single shared stride (`ag * dim`/`ag * ambient_dim`). Every function here
// now takes `agent_axis_offsets`/`agent_ambient_offsets` (CUMULATIVE
// per-agent offsets, built once by BuildAgentAxisOffsets/
// BuildAgentAmbientOffsets) instead of a uniform stride int, and slices
// agent ag's own columns as `[offsets[ag], offsets[ag] + agent_shapes[ag].
// tangent_dim()/.ambient_dim())` rather than `[ag*dim, (ag+1)*dim)`.
namespace sqp_short_path {

// Per-agent collision models, one entry per agent, index-aligned with
// BuildAgentShapes / BuildAgentAxisOffsets. Every entry is non-null (a
// trivial single-point model stands in when nothing was registered for an
// agent -- see agent_collision_model.hpp / MakeTrivialCollisionModel).
using AgentCollisionModels = std::vector<std::unique_ptr<AgentCollisionModel>>;

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

// AMBIENT-space counterpart to BuildAgentAxisOffsets: `offsets[ag]` is where
// agent `ag`'s own ambient columns start within the flat (agent-major)
// AMBIENT layout x0/points/ref_points use, i.e. `x0.segment(offsets[ag],
// agent_shapes[ag].ambient_dim())` is agent ag's own ambient state. Agents
// are no longer required to share a common ambient (or tangent) width --
// each is whatever `agent_shapes[ag]` itself reports -- so this, like
// BuildAgentAxisOffsets, is a per-agent CUMULATIVE offset, not a uniform
// `ag * ambient_dim` stride.
std::vector<int> BuildAgentAmbientOffsets(const std::vector<CubicConfigurationSpline>& agent_shapes);

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

// A surviving (agent, obstacle) / agent-pair / (agent, grid), with the
// half-open horizon-step range `[step_lo, step_hi)` over which the
// REFERENCE trajectory actually comes within reach -- rows and merit terms
// are emitted only for those steps. `step_lo >= step_hi` means "survived
// the coarse whole-trajectory test but no individual step is close" and
// contributes nothing. The range is contiguous by construction (a
// short-horizon reference approaches a fixed obstacle -- or another agent's
// reference -- at most once); a genuinely bimodal reference would just get
// an over-wide range here, never a too-narrow one, so this can only
// over-include rows, never drop a row that matters.
//
// Computed ONCE per solve() call (GraphShortPathMPC::solve(), alongside
// `ref_points`) and held fixed for the whole call -- design decision 6: row
// COEFFICIENTS relinearize every outer iteration, which rows EXIST does
// not. PURELY a speed lever on the SQP's own optimization path:
// GraphShortPathMPC's ApplySafetyProjection / ApplyAgentPairSafetyProjection
// final hard-feasibility passes check EVERY registered obstacle / pair at
// EVERY step regardless, so a wrongly-excluded (step, obstacle) degrades
// only the SMOOTHNESS of the SQP's avoidance there (handled by the
// closed-form fallback instead), never correctness/feasibility.
// `spheres` / `sphere_pairs` is the PER-STEP per-sphere broadphase (v2 plan
// Stage 3d): entry `s` (for step `step_lo + s`, length `step_hi - step_lo`)
// lists only the agent body spheres (obstacle/grid) -- or ordered sphere
// pairs (a_k, b_k) (inter-agent) -- that come within the effective prune
// margin of the other geometry AT THAT STEP, instead of all K (or K_a*K_b)
// of them. Per-step (not once for the whole `[step_lo, step_hi)` window) so
// two articulated arms grazing past each other don't pay a K_a*K_b row for
// every step of the crossing -- only the few sphere pairs actually close at
// each one. A trivial agent has exactly one radius-0 sphere, so every entry
// is `{0}` / `{{0,0}}` and the row set is unchanged.
struct ActiveObstacle {
	const Obstacle* obstacle;
	int step_lo = 0;
	int step_hi = 0;
	std::vector<std::vector<int>> spheres;  // [step - step_lo]
};
struct ActivePair {
	int ag_a = 0;
	int ag_b = 0;
	int step_lo = 0;
	int step_hi = 0;
	std::vector<std::vector<std::pair<int, int>>> sphere_pairs;  // [step - step_lo]
};

// The workspace spheres agent `ag`'s collision model occupies at every
// REFERENCE-trajectory step -- the geometry the distance pruning
// (Prune*ByDistance) and, in future, the safety projections test against,
// replacing the old "q[:workspace_dim] IS the world position" assumption
// for a non-trivial (articulated / multi-sphere) agent. For a trivial agent
// this is exactly one radius-0 sphere per step at q[:workspace_dim], so the
// pruning reduces to its former behaviour byte-for-byte. Built ONCE per
// solve() (GraphShortPathMPC::solve()) from the reference trajectory, same
// discipline as the pruning it feeds.
struct AgentReferenceSpheres {
	// centers[i]: (num_spheres x workspace_dim), the agent's sphere centres
	// at reference step i. radii: (num_spheres), step-invariant.
	std::vector<Eigen::MatrixXd> centers;
	Eigen::VectorXd radii;
	double max_radius = 0.0;
	// Bounding sphere over every centre at every step (radii NOT folded in
	// -- callers add the relevant radius themselves).
	BoundingSphere bound;
};

std::vector<AgentReferenceSpheres> BuildAgentReferenceSpheres(
	const Eigen::MatrixXd& ref_points, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim,
	const AgentCollisionModels& models);

// Memoizes `AgentCollisionModel::Eval` (a real Drake FK + Jacobian query for
// a non-trivial/articulated agent -- not free) per (agent, horizon step),
// for ONE fixed absolute configuration (some `points` matrix). Every
// consumer below that needs agent `ag`'s spheres at step `i` for that same
// `points` -- Evaluate*Violation AND Linearize*Constraints alike -- goes
// through the same cache instance instead of calling `Eval` itself, so an
// agent whose spheres both an obstacle row and a pair row need at the same
// step pays for the FK query once, not twice.
//
// One cache instance is tied to ONE `points` matrix (the caller is
// responsible for using a fresh instance whenever `points` changes, and may
// keep reusing the same instance across an outer SQP iteration boundary
// when `points` itself didn't change -- see RunTrustRegionSqp's own use:
// the merit function's violation-at-candidate pass and the NEXT iteration's
// linearization-at-that-same-candidate pass are the same configuration, so
// a solve that accepts every step now pays for each (agent, step) FK query
// once per outer iteration instead of twice).
//
// The [ag][i] slot grid itself (num_agents * num_steps `optional`s) is
// allocated once at construction and never resized again: `Reset` (for
// re-tying an existing instance to a NEW `points`) clears every slot back
// to disengaged in place rather than reallocating the grid, so a caller
// that needs one cache per iteration (RunTrustRegionSqp's candidate-point
// cache) can keep a single long-lived instance across the whole solve and
// `Reset` it every iteration instead of paying two nested heap allocations
// (the outer per-agent vector, and one inner per-step vector per agent) on
// every single outer iteration regardless of whether the step is accepted.
class SphereEvalCache {
   public:
	SphereEvalCache(int num_agents, int num_steps) : cache_(num_agents, Row(num_steps)) {}

	// Agent `ag`'s workspace spheres at horizon step `i`, computed via
	// `model.Eval` on first request and cached thereafter. `points`/
	// `ambient_offset`/`model` must describe the SAME configuration this
	// cache instance was constructed for on every call.
	const std::vector<WorkspaceSphere>& Get(int ag, int i, const Eigen::MatrixXd& points, int ambient_offset,
						const AgentCollisionModel& model);

	// Disengages every [ag][i] slot (`optional::reset()`, each) so this
	// instance can be re-tied to a DIFFERENT `points` without reallocating
	// the [ag][i] grid itself (the per-agent vector of per-step slots) --
	// that's the allocation `make_unique<SphereEvalCache>(num_agents,
	// num_steps)` pays every time, which Reset lets a caller amortize
	// across a whole solve() instead of repeating every outer iteration.
	// Each individual slot's `vector<WorkspaceSphere>` is still freshly
	// allocated by `model.Eval()` the next time that (agent, step) is
	// asked for -- Reset does not, and cannot, avoid that part of the
	// cost, since `Eval()` returns its result by value.
	void Reset();

   private:
	using Row = std::vector<std::optional<std::vector<WorkspaceSphere>>>;
	std::vector<Row> cache_;  // [ag][i]
};

// Per-agent obstacle list "close enough to plausibly matter" over this
// solve() call's horizon: agent ag's swept-collision-sphere bounding sphere
// (BuildAgentReferenceSpheres) vs each registered obstacle's extent (sphere:
// radius+margin; box: half-extents' norm+margin, a conservative
// circumscribing-sphere proxy) as a cheap coarse filter, then a per-step
// per-sphere distance check narrowing to `[step_lo, step_hi)` (the
// ActiveObstacle range above). An obstacle that clears neither filter is
// omitted from `per_agent_obstacles[ag]` entirely.
//
// `max_obstacle_pairs_per_step` (0 = unlimited): after the per-step distance
// filter, keep only the this-many CLOSEST (obstacle, sphere) rows for this
// agent at each step, ranked ACROSS every candidate obstacle at that step
// (not per-obstacle) -- an agent's body can sit near several distinct
// obstacle primitives at once (e.g. a wall meshed into several boxes), each
// contributing up to `num_spheres` rows on its own, so unlike
// `PruneAgentPairsByDistance`'s per-pair cap (bounded by the OTHER agent
// count) this is the more consequential prune in a heavily-obstacled scene.
// Mirrors `PruneAgentPairsByDistance`'s own `max_pairs_per_step`: same
// "only a few are ever the binding contact" reasoning, same
// `std::nth_element` top-k selection.
std::vector<std::vector<ActiveObstacle>> PruneObstaclesByDistance(
	int num_steps, int num_agents, int workspace_dim,
	const std::vector<AgentReferenceSpheres>& ref_spheres, const AgentCollisionModels& models,
	const ObstacleSet& obstacles, double prune_margin, int max_obstacle_pairs_per_step = 0);

// Same idea for inter-agent pairs, over each agent's swept collision
// spheres: (ag_a, ag_b) survives the coarse filter if their bounding
// spheres could bring any sphere pair within `prune_margin`, then a
// per-step per-sphere-pair separation check narrows to `[step_lo,
// step_hi)`. A trivial agent contributes its scalar `agent_radii(ag)` as
// its (single) sphere radius here; a non-trivial one contributes its body
// spheres' own radii. This is the more consequential of the prunings:
// unpruned pair count grows as `num_agents*(num_agents-1)/2`. When both
// agents report a broadphase margin hint (a bounded body such as an arm)
// the pruner uses min(prune_margin, hint_a + hint_b) instead of the raw
// `prune_margin`, so an arm pair doesn't inherit the free-particle default.
//
// `max_pairs_per_step` (0 = unlimited): after the per-step distance filter,
// keep only the this-many CLOSEST sphere pairs at each step (by reference-
// trajectory surface separation). Two densely-sphered bodies deep in a
// shared volume can put a hundred-plus pairs within the margin, but only a
// handful are ever the binding contact -- this caps the QP row count (and
// hence the proxqp factorization cost) at `max_pairs_per_step * horizon`
// per agent pair regardless. Like every prune here it is recomputed once
// per solve() from the reference, so pick it with headroom over the
// genuinely-active contact count for the scene.
std::vector<ActivePair> PruneAgentPairsByDistance(
	int num_steps, int num_agents, const std::vector<AgentReferenceSpheres>& ref_spheres,
	const AgentCollisionModels& models, const Eigen::VectorXd& agent_radii, double prune_margin,
	int max_pairs_per_step = 0);

// Total obstacle-constraint violation (Sum of max(0, -c(q)) over every
// (step, agent, obstacle) SURVIVING PruneObstaclesByDistance) at the given
// absolute `points` -- the merit function's penalty term, evaluated with
// the TRUE (non-linearized) sphere/box signed distance, not
// LinearizeObstacleConstraints' local model. Deliberately scoped to the
// SAME pruned set LinearizeObstacleConstraints' rows come from (not the
// full unpruned obstacle list) -- the merit function must score exactly
// what the QP can act on, or the trust-region ratio test (predicted vs.
// actual reduction) becomes meaningless for whatever it silently omitted.
// `agent_ambient_offsets`: per-agent AMBIENT offset used to slice `points`
// (points.row(i).segment(agent_ambient_offsets[ag], workspace_dim) is agent
// ag's world position at step i) -- NOT the tangent/decision-space offsets
// `agent_axis_offsets` holds; the two differ once an agent has an SO3Quat
// block (ambient 4 vs tangent 3 per block, see the project plan's Stage 4),
// and neither is required to be uniform across agents (Stage 5: agents may
// have entirely different specs/tangent widths from one another).
double EvaluateObstacleViolation(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
				  int workspace_dim, const Eigen::MatrixXd& points,
				  const std::vector<std::vector<ActiveObstacle>>& per_agent_obstacles,
				  const AgentCollisionModels& models, SphereEvalCache& sphere_cache);

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
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim, const Eigen::MatrixXd& points,
	const std::vector<std::vector<ActiveObstacle>>& per_agent_obstacles,
	const AgentCollisionModels& models, SphereEvalCache& sphere_cache);

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
double EvaluateAgentPairViolation(int num_steps, const std::vector<int>& agent_ambient_offsets, int workspace_dim,
				   const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
				   const std::vector<ActivePair>& active_pairs, const AgentCollisionModels& models,
				   SphereEvalCache& sphere_cache);

std::vector<ConstraintRow> LinearizeAgentPairConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, const std::vector<int>& agent_ambient_offsets,
	int workspace_dim, const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
	const std::vector<ActivePair>& active_pairs, const AgentCollisionModels& models,
	SphereEvalCache& sphere_cache);

// Value + gradient of one AgentSdfGrid, multilinearly interpolated
// (bilinear for workspace_dim=2, trilinear for workspace_dim=3) at world
// point `p`. `value` already has `grid.margin` subtracted (same convention
// as SphereSdf/BoxSdf's own `sdf.value - obstacle.margin`, see
// sqp_short_path_layout.cpp). `p` outside the grid's own extent is CLAMPED
// to the nearest boundary vertex before interpolating (constant/
// zero-order-hold extrapolation) rather than extrapolated -- a caller-built
// local crop is expected to already cover whatever region the reference
// trajectory (plus prune margin) needs; querying past its edge only
// happens when that margin wasn't quite enough, and clamping is the
// conservative choice (an out-of-crop query never silently reports a large
// spurious clearance, nor extrapolates an unbounded one). Gradient comes
// from `grid.gradient` if the caller supplied one (interpolated the SAME
// multilinear way as `value`), otherwise from differentiating the SAME
// interpolant `value` was computed from (see AgentSdfGrid's own doc
// comment in obstacle_set.hpp for why this is the default). Used both by
// this file's own EvaluateAgentSdfGridViolation/
// LinearizeAgentSdfGridConstraints and by graph_short_path_mpc.cpp's final
// safety-projection pass -- both must query the SAME interpolant, hence
// public/shared rather than a local anonymous-namespace helper.
struct SdfSample {
	double value;
	Eigen::VectorXd grad;  // d(value)/d(p), workspace_dim
};
SdfSample QueryAgentSdfGrid(const AgentSdfGrid& grid, const Eigen::VectorXd& p, int workspace_dim);

// Per-agent registered grid "close enough to plausibly matter" over this
// solve() call's horizon, same distance-pruning idea as
// PruneObstaclesByDistance but singular per agent (a grid is registered
// PER AGENT already, not shared/matched-by-proximity like spheres/boxes --
// see AgentSdfGrid's own comment) and compared against the grid's own AABB
// instead of a sphere/box's radius/half-extents. Returned vector is
// `num_agents` long; entry `ag` has a null `grid` if agent `ag` has no
// registered grid OR its grid's AABB can't come within `prune_margin` of
// the agent's reference-trajectory bounding sphere, and otherwise carries
// the per-step range `[step_lo, step_hi)` the reference is actually within
// range of the grid. Same "purely a speed lever, computed once per solve()
// call, never affects correctness" property as PruneObstaclesByDistance --
// the final safety-projection pass checks every agent's grid regardless.
struct ActiveGrid {
	const AgentSdfGrid* grid = nullptr;
	int step_lo = 0;
	int step_hi = 0;
	std::vector<std::vector<int>> spheres;  // [step - step_lo], per-step broadphase, see ActiveObstacle
};
std::vector<ActiveGrid> PruneAgentSdfGridsByDistance(
	int num_steps, int num_agents, const std::vector<AgentReferenceSpheres>& ref_spheres,
	const AgentCollisionModels& models, const ObstacleSet& obstacles, double prune_margin);

// Total grid-constraint violation (sum of max(0, -value) over every (step,
// agent) with a non-null `active_grids[ag].grid` and step in its range) at
// the given absolute `points` -- the merit function's penalty term, same
// role as EvaluateObstacleViolation but for grid obstacles.
double EvaluateAgentSdfGridViolation(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
				      int workspace_dim, const Eigen::MatrixXd& points,
				      const std::vector<ActiveGrid>& active_grids, const AgentCollisionModels& models,
				      SphereEvalCache& sphere_cache);

// One (step, agent) row per non-null `active_grids[ag].grid`, over its
// `[step_lo, step_hi)` range, linearized at the current iterate `points` --
// same fk fast-path shape as LinearizeObstacleConstraints (constant 0/1
// selection Jacobian), row count `O(steps*agents)` regardless of grid
// resolution (the whole point of a field representation over Stage 3's
// reverted one-row-per-point approach -- see the project plan).
std::vector<ConstraintRow> LinearizeAgentSdfGridConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim, const Eigen::MatrixXd& points,
	const std::vector<ActiveGrid>& active_grids, const AgentCollisionModels& models,
	SphereEvalCache& sphere_cache);

}  // namespace sqp_short_path
