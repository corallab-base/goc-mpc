#pragma once

#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "graph_of_constraints.hpp"
#include "obstacle_set.hpp"
#include "../configuration_spline.hpp"

// Problem-layout/assembly math for SqpShortPathMPC (sqp_short_path_mpc.hpp),
// mirroring timing_gn_layout.{hpp,cpp}'s split from graph_timing_mpc.cpp:
// this file has no qpOASES/trust-region-loop code, only the Hessian/RHS/
// constraint-row math the solver assembles into a QP every outer iteration.
//
// v1 scope: Block::R and Block::Torus only (BuildAgentShapes throws on
// SO3Quat/SO3Mat -- see the project plan's Stage 4 for why quaternion
// blocks need a structurally different, coupled-matrix layout instead of
// this flat per-axis one) and fk(q) = q[:workspace_dim] (a constant 0/1
// selection Jacobian -- see LinearizeObstacleConstraints).
namespace sqp_short_path {

// One scalar decision axis: a single tangent-space component of one
// agent's configuration, repeated identically at every horizon step.
// `tangent_col` is this axis's column within that agent's OWN (H x
// agent_tangent_dim) step matrix -- exactly the column
// BlockPositionDelta/BlockRetract's per-block outputs are sliced from.
struct AxisLayout {
	int agent;
	int tangent_col;
};

// Per-agent shape (block layout + ambient/tangent dims), one entry per
// agent, built once from graph._robot_specs and reused for every
// BlockPositionDelta/BlockRetract call this solver makes. Throws if any
// agent has a Block::SO3Quat/SO3Mat component.
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

// The smooth (tracking + velocity-tracking + acceleration-smoothing)
// cost's per-axis (2*num_steps x 2*num_steps) Hessian block. IDENTICAL for
// every axis -- the coefficient pattern never depends on which axis/agent
// it's for, only `tau` (see this class's project-plan design decision 3) --
// and constant across outer SQP iterations within one solve() call, since
// it's an honest quadratic form in the STEP variables, independent of the
// current iterate.
Eigen::MatrixXd BuildAxisHessianBlock(int num_steps, double tau);

// Assembles the full block-diagonal (n x n) smooth-cost Hessian, `n =
// axes.size() * 2 * num_steps`, by placing BuildAxisHessianBlock's block
// once per axis. Call once per solve() call, reuse across every outer
// iteration.
Eigen::MatrixXd AssembleSmoothHessian(const std::vector<AxisLayout>& axes, int num_steps, double tau);

// Per-axis RHS/target vector (length 2*num_steps) for the smooth cost,
// evaluated AT THE CURRENT ITERATE (`points_agent`/`vels_agent`, H x
// agent_dim, already sliced to this axis's agent) against `x0_agent`/
// `v0_agent` (anchors) and `ref_points_agent`/`ref_velocities_agent`.
// Manifold-aware via `agent_shape.PositionDelta<double>` for every
// position-flavored term (tracking's ref-vs-current delta, and the
// acceleration term's inter-step displacement) -- see the .cpp for the
// full derivation (linearizing each already-quadratic-in-absolute-
// coordinates term around the current iterate). Depends on the current
// iterate, unlike the Hessian -- rebuild every outer iteration.
Eigen::VectorXd BuildAxisRhs(const AxisLayout& axis,
			      const CubicConfigurationSpline& agent_shape,
			      int num_steps, double tau,
			      const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			      const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			      const Eigen::MatrixXd& ref_points_agent,
			      const Eigen::MatrixXd& ref_velocities_agent);

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
// assembly itself.
double EvaluateSmoothCost(const CubicConfigurationSpline& agent_shape, int num_steps, double tau,
			   const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			   const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			   const Eigen::MatrixXd& ref_points_agent, const Eigen::MatrixXd& ref_velocities_agent);

// Total obstacle-constraint violation (Sum of max(0, -c(q)) over every
// (step, agent, obstacle)) at the given absolute `points` -- the merit
// function's penalty term, evaluated with the TRUE (non-linearized)
// sphere/box signed distance, not LinearizeObstacleConstraints' local
// model.
double EvaluateObstacleViolation(int num_steps, int num_agents, int dim, int workspace_dim,
				  const Eigen::MatrixXd& points, const ObstacleSet& obstacles);

// Every (step, agent, obstacle) row in `obstacles.obstacles()` (spheres and
// boxes; point-cloud obstacles are a later stage, not read here),
// linearized at the current iterate `points`. Fast path only: assumes
// `fk(q) = q[:workspace_dim]` (points.row(i).segment(ag*dim, workspace_dim)
// IS the agent's world position, so the constraint's Jacobian w.r.t. the
// step is a constant 0/1 selection onto that agent's leading
// `workspace_dim` axes at that step -- no chain rule beyond the slice).
std::vector<ConstraintRow> LinearizeObstacleConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents, int dim,
	int workspace_dim, const Eigen::MatrixXd& points, const ObstacleSet& obstacles);

}  // namespace sqp_short_path
