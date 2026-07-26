#pragma once

#include <drake/solvers/mathematical_program.h>

#include "graph_of_constraints.hpp"

// Compiles a stored SymbolicNodeConstraint/SymbolicEdgeConstraint record (as
// produced by GraphOfConstraints::add_constraint / add_assignable_constraint /
// add_edge_constraint) into real Drake MathematicalProgram constraints, at
// MILPWaypointMPC build time. This is the same substitution/big-M-gating logic
// that used to live inline in DeferredOp/DeferredEdgeOp builder closures —
// relocated here (and parameterized by the stored record instead of captured
// by a lambda) so the raw Formula stays introspectable for non-MILP consumers
// (e.g. the JAX evolutionary solver's formula_compiler.py).
void CompileSymbolicNodeConstraint(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicNodeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments);

// `previous_X` (this solver's persisted per-node waypoint buffer, indexed by
// GLOBAL node id -- MILPWaypointMPC::_waypoints) is the fallback for a
// relational (non-along_edge) formula's u_node/v_node when that endpoint
// isn't in `subgraph` (it's already passed out of remaining_vertices; the
// edge is still compiled here as long as the OTHER endpoint remains -- see
// SubgraphOfConstraints' constructor). Mirrors the legacy DeferredEdgeOp
// API's previous_X.row(u_node) fallback (Constraint 13 in
// BuildGraphWaypointProblem). MILPWaypointMPC::Solve() keeps a passed node's
// row in `previous_X` live-rebound to the real x0 as of the cycle it passed
// (rather than merely its last-solved value), so this fallback reads the
// real state, not a stale plan.
void CompileSymbolicEdgeConstraint(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const Eigen::MatrixXd& previous_X);

// Compiles an "along the edge" SymbolicEdgeConstraint (rec.along_edge) at a
// single subgraph node -- substituting the plain agent_q/object_q/var_agent_q
// placeholders at sg_node, same as a node constraint would. CompileSymbolic-
// EdgeConstraint calls this at the edge's own u_node/v_node (extra_gate =
// nullptr, unconditional besides any assignable-var assignment gating); the
// MILP build loop (BuildGraphWaypointProblem) additionally calls this
// directly for every OTHER subgraph node the edge might span, passing a
// betweenness-indicator expression as extra_gate -- mirroring how
// DeferredEdgeOp::interior_builder re-applies a per-endpoint invariant at
// interior nodes for the older add_robot_holding_cube_constraint-style API.
void CompileAlongEdgeConstraintAtNode(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	int sg_node,
	const drake::symbolic::Expression* extra_gate);

// Numeric constraint-violation evaluators, used by
// GraphOfConstraints::evaluate_phi/evaluate_edge_phi. Mirrors the eval_fn
// closures the pre-refactor DeferredOp/DeferredEdgeOp used to carry.
double EvaluateSymbolicNodeConstraint(
	const GraphOfConstraints& graph,
	const SymbolicNodeConstraint& rec,
	const Eigen::VectorXd& x,
	int assigned_agent);

// Multi-variable (>1 assignable var referenced) symbolic node constraints have
// no numeric evaluator (matches the pre-refactor eval_fn_multi stub, which
// always returned 0.0) — EvaluateSymbolicNodeConstraint returns 0.0 for those.

double EvaluateSymbolicEdgeConstraint(
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const Eigen::VectorXd& x,
	const Eigen::VectorXi& var_assignments);
