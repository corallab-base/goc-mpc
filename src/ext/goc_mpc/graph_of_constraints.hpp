#pragma once

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <fmt/format.h>

#include <drake/common/symbolic/expression.h>
#include <drake/common/symbolic/expression/environment.h>
#include <drake/common/symbolic/expression/formula.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <drake/math/quaternion.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "../configuration_spline.hpp"
#include "../graphs.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::symbolic::Expression;
using namespace pybind11::literals;
namespace py = pybind11;


struct SubgraphOfConstraints;

enum class RobotKind { kPointMass, kPosYaw, kPosQuat, kPosRotMat, kArticulated };
enum class ConstraintDegree { kLinear, kQuadratic, kGeneral };

enum class DeferredOpKind {
	kLinearEq,
	kLinearIneq,
	kBoundingBox,
	kNonlinearCost,
	kQuadraticCost,
	kNonlinearEq,
	kOther,
	// MultiAgent
	kAgentLinearEq,
	// Symbolic (unified path via drake::symbolic)
	kSymbolic,
};

struct DeferredOp {
	DeferredOpKind kind;
	int id;
	int node;
	std::function<double(const Eigen::VectorXd&,
			     const int)> eval;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const struct SubgraphOfConstraints&,
			   const int,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&)> builder;
};

struct DeferredEdgeOp {
	DeferredOpKind kind;
	int id;
	int u_node;
	int v_node;
	std::set<int> cubes; // edge constraints either do or don't involve a cube (keypoint)
	std::function<double(const Eigen::VectorXd&,
			     const Eigen::VectorXi&)> eval;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const struct SubgraphOfConstraints&,
			   const int,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const Eigen::VectorXd&)> waypoint_builder;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const int,
			   const Eigen::VectorXi&,
			   const drake::solvers::MatrixXDecisionVariable&)> short_path_builder;
	// Optional: for edge constraints that check an independent invariant at
	// each endpoint (rather than a single relation coupling both), applies
	// that same per-node invariant at an interior subgraph node that may end
	// up scheduled between u_node and v_node in the solved route. `gate` is
	// a binary that is 1 whenever the interior node is (weakly) between the
	// two endpoints in time; unset (default) for edge ops that don't have a
	// well-defined per-node interior application (e.g. relational formulas).
	std::function<void(drake::solvers::MathematicalProgram&,
			   const struct SubgraphOfConstraints&,
			   const int,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&,
			   int,
			   const drake::symbolic::Variable&)> interior_builder;
};

struct DeferredVarOp {
	DeferredOpKind kind;
	int id;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const struct SubgraphOfConstraints&,
			   const int,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&)> builder;
};

// A fully-covered manifold block (Torus/SO3Quat/SO3Mat) referenced by an
// Eq-kind formula, detected once at add_constraint/add_edge_constraint time
// (see PopulateBlockResidualGroups) and cached on the owning record below.
// Runtime evaluation (ComputeViolation/ComputeSignedViolation,
// symbolic_constraint_compiler.cpp) uses this instead of checking each
// component independently, which can't express e.g. quaternion geodesic
// distance -- inherently a joint function of all 4 components together --
// any more than raw per-component subtraction can express a Torus angle's
// shortest-arc distance (the same reason add_agent_timing_segments'
// stability_cost had to stop using a raw ambient subtraction -- see
// graph_timing_mpc.hpp's _stability_cost doc comment). `components[j]` is
// the (lhs, rhs) Expression pair pulled directly from whichever Eq leaf
// pins ambient component j of the block; evaluating and feeding them
// through CubicConfigurationSpline::BlockPositionDelta -- the SAME
// per-block residual add_agent_timing_segments' stability_cost and max_acc
// bound already use -- gives the correct wrap-aware/quaternion-log
// residual, whose norm is the group's scalar violation.
struct BlockResidualGroup {
	CubicConfigurationSpline::Block::Type type;
	int agent_id;
	int block_index;   // index into graph._robot_specs[agent_id]
	std::vector<std::pair<drake::symbolic::Expression, drake::symbolic::Expression>> components;
};

// Raw, introspectable records for the unified symbolic constraint API
// (add_constraint / add_assignable_constraint / add_edge_constraint). Unlike
// DeferredOp/DeferredEdgeOp, these store the original drake::symbolic::Formula
// (with placeholders unsubstituted) instead of a pre-built closure, so that
// consumers other than MILPWaypointMPC (e.g. the JAX evolutionary solver) can
// introspect and independently compile the same constraint. Compilation into
// actual Drake constraints happens at build time in symbolic_constraint_compiler.*,
// mirroring what the DeferredOp builder closures used to do inline.
struct SymbolicNodeConstraint {
	int id;
	int node;
	std::optional<int> var_id;         // set iff exactly one assignable var is referenced
	std::vector<int> multi_var_ids;    // set iff >1 assignable vars are referenced (Or-over-combos case)
	drake::symbolic::Formula formula;
	// Cached once at construction (see PopulateBlockResidualGroups), not
	// re-derived from `formula` on every runtime evaluate_phi call (every
	// control cycle). `ungrouped_leaves` is every Eq/Leq/... leaf NOT
	// covered by a complete block group (an R-typed component, a lone
	// incomplete manifold-block reference, or a non-Eq atom) -- evaluated
	// exactly as `formula` was before block grouping existed.
	std::vector<BlockResidualGroup> block_residual_groups;
	std::vector<drake::symbolic::Formula> ungrouped_leaves;
};

struct SymbolicEdgeConstraint {
	int id;
	int u_node;
	int v_node;
	drake::symbolic::Formula formula;
	// true: `formula` is built from the plain agent_q/object_q/var_agent_q
	// placeholders ("along the edge" -- an invariant applied independently
	// at both endpoints, and, in MILP, at any interior node the edge might
	// span). false: `formula` is built from u_agent_q/v (or u_object_q/v)
	// -- a single relation coupling the two endpoints, compiled once.
	bool along_edge;
	std::optional<int> var_id;  // set iff along_edge and exactly one
	                             // var_agent_q placeholder is referenced.
	// Same as SymbolicNodeConstraint's own fields -- see there.
	std::vector<BlockResidualGroup> block_residual_groups;
	std::vector<drake::symbolic::Formula> ungrouped_leaves;
};

// Populates *groups/*ungrouped from `f` -- see BlockResidualGroup's own doc
// comment for what a "group" is and why. Flattens `f`'s top-level
// And-conjunction into its Eq/Leq/.../generic leaf atoms, groups any Eq
// leaf whose lhs or rhs is a single manifold-block-typed Variable
// (Torus/SO3Quat/SO3Mat, resolved against `graph`'s STATIC agent_q/
// agent_q_u/agent_q_v placeholder families only -- var_agent_q's
// dynamically-resolved agent isn't known until evaluation time, so a
// formula referencing it never groups, same limitation _resolve_holds
// (evolutionary_waypoint_solver/spec.py) already has for assignable holds)
// by (side, agent_id, block_index), and keeps only COMPLETE groups (every
// ambient component of the block present as a sibling leaf). Everything
// else (R-typed leaves, incomplete manifold-block references, non-Eq
// atoms) is left in *ungrouped untouched, evaluated exactly as before.
// Defined in graph_of_constraints.cpp; called once per add_constraint/
// add_assignable_constraint/add_edge_constraint call, not on every
// runtime evaluate_phi.
void PopulateBlockResidualGroups(
	const struct GraphOfConstraints& graph, const drake::symbolic::Formula& f,
	std::vector<BlockResidualGroup>* groups,
	std::vector<drake::symbolic::Formula>* ungrouped);

// Canonical, introspectable declaration that `held_point_ids` are rigidly
// held by a robot over edge (u_node -> v_node) -- either a statically
// assigned robot (robot_ag) or one resolved later via an assignable
// variable (var_id); exactly one of the two is set. Populated by
// add_robot_holding_cube_constraint / add_assignable_robot_holding_point_
// constraint (which otherwise only register a proximity check) so that
// MILPWaypointMPC's Constraint 14 (rigid-hold + stationary-object dynamics)
// and any other consumer (e.g. the JAX evolutionary solver) can read a
// single source of truth for "which edges are holds", instead of each
// re-deriving it from DeferredEdgeOp::cubes plus the static/assignable
// edge-phi maps.
struct HoldDeclaration {
	int id;
	int u_node;
	int v_node;
	std::vector<int> held_point_ids;
	std::optional<int> robot_ag;  // set iff statically assigned
	std::optional<int> var_id;    // set iff assignable
};

struct AgentInteraction {
	enum Type { LESS_THAN, EQUAL };

	int agent_i;
	int agent_i_depth;
	int agent_j;
	int agent_j_depth;
	int node_u;
	int node_v;
	Type type;

	AgentInteraction(int i, int i_depth, int j, int j_depth, int u, int v, Type t) :
		agent_i(i),
		agent_i_depth(i_depth),
		agent_j(j),
		agent_j_depth(j_depth),
		node_u(u),
		node_v(v),
		type(t) {}
};

// A family of fixed-width symbolic placeholder Variable vectors, keyed by
// Key, created lazily on first access (Get()/Vars()). Backs every "row
// placeholder" in the unified symbolic constraint API -- agent_q, object_q,
// var_agent_q, agent_link_pos, and their u_/v_ edge counterparts -- which
// were previously ~10 separately-declared GraphOfConstraints members (some
// vector<VectorX<Variable>>, index-keyed and eagerly populated in the
// constructor; some map<Key, VectorX<Variable>>, lazily populated on first
// access) with near-duplicate accessor bodies, near-duplicate constructor
// population loops, and near-duplicate Substitution/Environment-building
// loops scattered across symbolic_constraint_compiler.cpp (one 8-line block
// hand-copied at ~6 call sites -- exactly the kind of duplication that made
// adding agent_link_pos support require touching all 6 by hand).
//
// Laziness is harmless even for the formerly-eager families
// (agent_q/object_q/u_/v_): a formula can only ever reference a placeholder
// Variable that Get()/Vars() already returned (there's no other way to
// spell one), so by the time any formula is compiled/evaluated, every key
// it could possibly reference already exists in `vars_` regardless of
// eager vs. lazy creation -- see ReferencesAny/KeysReferencedBy, which rely
// on exactly this.
template <typename Key>
class PlaceholderVarFamily {
public:
	PlaceholderVarFamily() = default;
	// `width`: fixed size of every key's placeholder vector, the SAME for
	// every key (e.g. `workspace_dim` for agent_link_pos, 1 for param).
	// `namer(key)`: produces each entry's debug name (shown in printed
	// formulas).
	PlaceholderVarFamily(int width, std::function<std::string(const Key&)> namer)
		: width_(width), namer_(std::move(namer)) {}

	// Per-KEY width (agent_q/object_q/var_agent_q and their u_/v_
	// counterparts -- agents/objects/variables need not share a width with
	// each other). `Key` must be `int` (an agent/object/variable id, used
	// directly as the vector index) -- `widths` is a non-owning pointer to
	// a vector the CALLER (GraphOfConstraints, which owns both this family
	// and the vector as sibling members) keeps alive and may mutate/grow in
	// place for this family's whole lifetime; `push_back` never invalidates
	// the vector's own address, only iterators/pointers to individual
	// elements, so this is safe as long as nothing swaps or reassigns the
	// vector wholesale. Mirrors this project's existing store-pointer-
	// caller-keeps-alive convention (e.g. ObstacleSet).
	PlaceholderVarFamily(const std::vector<int>* widths, std::function<std::string(const Key&)> namer)
		: widths_(widths), namer_(std::move(namer)) {}

	// True iff `key` already has a placeholder (Vars()/Get() was called for
	// it before) -- does NOT create one. Used where "was this
	// assignable-var/link already declared" must be checked without the
	// side effect of declaring it (e.g. add_assignable_constraint's
	// DRAKE_DEMAND).
	bool Contains(const Key& key) const { return vars_.contains(key); }

	// Raw placeholder Variable vector for `key`, creating it (named via
	// `namer_`) on first access. Needed (rather than Get()) wherever a
	// caller indexes into a Substitution/Environment, both of which key on
	// Variable, not Expression.
	const drake::VectorX<drake::symbolic::Variable>& Vars(const Key& key) const {
		auto it = vars_.find(key);
		if (it == vars_.end()) {
			it = vars_.emplace(key, drake::symbolic::MakeVectorContinuousVariable(
				WidthFor(key), namer_(key))).first;
		}
		return it->second;
	}

	// Expression-cast counterpart of Vars() -- what every public
	// GraphOfConstraints accessor (agent_q, object_q, ...) actually returns
	// to formula-building callers.
	drake::VectorX<drake::symbolic::Expression> Get(const Key& key) const {
		return Vars(key).template cast<drake::symbolic::Expression>();
	}

	// True iff `free_vars` includes any component of any key currently in
	// this family (see this class's docstring for why "currently in" is
	// exhaustive despite lazy creation).
	bool ReferencesAny(const drake::symbolic::Variables& free_vars) const {
		for (const auto& [key, vars] : vars_)
			for (int j = 0; j < vars.size(); ++j)
				if (free_vars.include(vars[j])) return true;
		return false;
	}

	// Every key whose placeholder appears in free_vars -- used to detect
	// which assignable var(s)/link(s) a formula references
	// (add_constraint/add_edge_constraint's routing logic).
	std::vector<Key> KeysReferencedBy(const drake::symbolic::Variables& free_vars) const {
		std::vector<Key> out;
		for (const auto& [key, vars] : vars_) {
			for (int j = 0; j < vars.size(); ++j) {
				if (free_vars.include(vars[j])) { out.push_back(key); break; }
			}
		}
		return out;
	}

	// Populates sub[Vars(i)[j]] = row_value(i, j) for i in [0, n), j in
	// [0, width()) -- Key must be int. Used to build a MILP Substitution
	// against a node/edge's decision-variable row.
	template <typename RowFn>
	void SubstituteRange(drake::symbolic::Substitution* sub, int n, RowFn&& row_value) const {
		for (int i = 0; i < n; ++i) {
			const auto& vars = Vars(i);
			for (int j = 0; j < vars.size(); ++j)
				(*sub)[vars[j]] = row_value(i, j);
		}
	}

	// Environment counterpart of SubstituteRange, for the double-valued
	// runtime evaluators (EvaluateSymbolicNodeConstraint/EdgeConstraint).
	template <typename RowFn>
	void InsertRange(drake::symbolic::Environment* env, int n, RowFn&& row_value) const {
		for (int i = 0; i < n; ++i) {
			const auto& vars = Vars(i);
			for (int j = 0; j < vars.size(); ++j)
				env->insert(vars[j], row_value(i, j));
		}
	}

private:
	// `if constexpr` (not SFINAE/specialization) discards the untaken
	// branch's instantiation entirely, so `widths_->at(key)` never has to
	// type-check for a non-int Key (agent_link_pos/agent_link_rot, keyed by
	// pair<int,string>) -- those always take the constant-`width_` branch
	// and never construct a per-key family in the first place. A negative
	// stored width (see GraphOfConstraints's `_var_widths`) means "not yet
	// resolvable" -- e.g. an assignable variable whose current candidate
	// agents don't share a width -- and throws a clear, actionable error
	// rather than silently returning garbage.
	int WidthFor(const Key& key) const {
		if constexpr (std::is_same_v<Key, int>) {
			if (!widths_) return width_;
			const int w = widths_->at(key);
			if (w < 0) {
				throw std::runtime_error(
					"PlaceholderVarFamily: " + namer_(key) + " has no resolvable width -- "
					"its candidate agents don't share a config width; narrow it to a "
					"uniform-width subset via add_variable_constraint first.");
			}
			return w;
		} else {
			return width_;
		}
	}

	int width_ = 0;
	const std::vector<int>* widths_ = nullptr;
	std::function<std::string(const Key&)> namer_;
	mutable std::map<Key, drake::VectorX<drake::symbolic::Variable>> vars_;
};

struct GraphOfConstraints {

	const std::vector<CubicConfigurationSpline::Spec> _robot_specs;
	const std::vector<std::string> _robot_names;
	std::vector<RobotKind> _robot_kinds;
	const std::vector<CubicConfigurationSpline::Spec> _object_specs;
	const std::vector<std::string> _object_names;
	Graph<py::object> structure;
	std::map<int, std::vector<int>> node_to_phis_map;
	std::map<std::pair<int, int>, std::vector<int>> edge_to_phis_map;
	std::map<std::pair<int, int>, double> edge_to_min_tau_map;
	std::set<int> unpassable_nodes;

	// Optional human-readable names for nodes (e.g. "approach", "pick_up"),
	// keyed by node id. Purely cosmetic bookkeeping -- consulted only by
	// get_node_name/logging, never by the solvers. A node with no entry here
	// falls back to its numeric id (see get_node_name).
	std::map<int, std::string> node_names;

	// Node Phi maps
	std::map<int, int> phi_to_variable_map;
	std::map<int, int> _phi_to_static_assignment_map;
	std::map<int, struct DeferredOp> ops;
	std::map<int, struct SymbolicNodeConstraint> symbolic_ops;
	std::map<int, std::vector<std::tuple<std::string, std::string, std::string>>> _grasp_change_map;
	std::map<int, std::vector<std::pair<std::string, std::string>>> _assignable_grasp_change_map;

	// Edge phi maps
	std::map<int, int> edge_phi_to_variable_map;
	std::map<int, struct DeferredEdgeOp> edge_ops;
	std::map<int, struct SymbolicEdgeConstraint> symbolic_edge_ops;
	std::map<int, int> _edge_phi_to_static_assignment_map;

	// Populated by add_edge_constraint(..., live=true) -- see that method's
	// docstring. Edge-only: a node constraint straddles exactly one node's
	// activity state (it can only reference that node's own agent_q/object_q
	// placeholders -- the DRAKE_DEMAND in add_edge_constraint is exactly what
	// stops a node-scoped formula from reaching into another node's row) and
	// is, by construction, already known-satisfied the moment its node
	// passes (that's the actual passing criterion, checked by
	// GraphOfConstraintsMPC._solve_for_timing before removing it from
	// remaining_phases) -- so live/frozen has no effect on a node
	// constraint's own already-certified residual. It only matters for an
	// edge constraint, whose `u` side can be a passed node while `v` is still
	// being solved for.
	std::set<int> live_edge_phis;

	// Var phi map
	std::map<int, struct DeferredVarOp> var_ops;

	// Hold registry -- see HoldDeclaration.
	std::map<int, struct HoldDeclaration> hold_ops;

	// Python-registered forward-kinematics overrides, keyed by
	// (agent_id, link_name) -- see set_robot_fk/link_pose.
	std::map<std::pair<int, std::string>, py::function> robot_fk_registry;

	// backtracking map
	std::map<int, std::vector<int>> backtrack_map;

	// Assignment-commit registry. `commit_trigger_node_to_var` is a build-time
	// declaration (see add_variable_commit): completing `node` should pin
	// variable `var`'s resolved agent for as long as anything downstream still
	// references it. `committed_assignments` is the runtime counterpart --
	// which variables are *currently* pinned, and to which agent -- mutated by
	// GraphOfConstraintsMPC as nodes complete (commit_variable_assignment) or
	// get reopened by backtracking (clear_variable_commitment). Kept here
	// (rather than threaded through GraphWaypointMPC::Solve's shared
	// interface) so MILPWaypointMPC can read it directly without forcing a
	// signature change onto the duck-typed EvolutionaryWaypointSolver, which
	// does not consume it.
	std::map<int, int> commit_trigger_node_to_var;
	std::map<int, int> committed_assignments;

	// Rest
	int num_phis, num_edge_phis, num_var_phis, num_holds;
	int num_variables, _num_total_assignables;
	int num_agents, num_objects, total_dim;

	// Per-entity CUMULATIVE column offset into the flat ambient state row
	// (`x`/MILP's `W`) -- `_agent_col_offsets[ag]`/`_object_col_offsets[ob]`
	// is where agent `ag`'s/object `ob`'s own `robot_ambient_dim(ag)`/
	// `object_ambient_dim(ob)`-wide slice starts. Agents/objects need not
	// share a width with each other. CSR-style (size `num_agents+1`/
	// `num_objects+1`): the trailing entry is the total width of that
	// block, so `agent_col_offset(num_agents)` doubles as "where the object
	// block starts" (valid even when num_objects==0) and
	// `object_col_offset(num_objects) == total_dim`. Built once in the
	// constructor from `_robot_specs`/`_object_specs`.
	std::vector<int> _agent_col_offsets, _object_col_offsets;

	// Per-entity width -- `_agent_widths[ag] == robot_ambient_dim(ag)`,
	// `_object_widths[ob] == object_ambient_dim(ob)` (redundant with the
	// offset tables above, which encode the same thing as differences of
	// consecutive entries, but each accessor has its own direct consumer:
	// offsets for column arithmetic, these for _agent_q's/_object_q's own
	// per-key PlaceholderVarFamily width). Built once, alongside the offset
	// tables, in the constructor.
	//
	// `_var_widths[var]` is DIFFERENT in kind: variables are created one at
	// a time over the graph's lifetime (add_variable()), not fixed at
	// construction, so this vector grows via push_back as they are; a
	// negative entry means "not yet resolvable" (see PlaceholderVarFamily::
	// WidthFor's own comment) -- var `var`'s candidate agents (every agent,
	// by default, until add_variable_constraint narrows them) don't
	// currently share a width. Resolved eagerly in add_variable() when
	// every agent happens to share one width (the common case, unchanged
	// from before this refactor); re-resolved (and validated -- throws if
	// the narrowed candidate set STILL disagrees) in add_variable_constraint.
	std::vector<int> _agent_widths, _object_widths, _var_widths;

	// A fresh variable's implied candidate set is every agent (until/unless
	// add_variable_constraint narrows it) -- computed ONCE in the
	// constructor (agent widths never change after construction) rather
	// than re-scanning all `num_agents` on every add_variable() call.
	// Negative when agents don't all share a width.
	int _default_var_width;

	// Ambient Cartesian workspace dimensionality (2 or 3) that
	// agent_link_pos/link_pose's REGISTERED-fk_fn path operate in.
	// Defaults to 3 for full backward compatibility. Sizes
	// agent_link_pos's placeholder width, so a 2-workspace graph can
	// compare it against a 2-wide object_q. Also consulted by link_pose's
	// built-in RobotKind fallback (PoseFromRow, utils.hpp -- kPointMass/
	// kPosYaw are workspace_dim-aware; kPosQuat/kPosRotMat demand
	// workspace_dim == 3) and by the point_position wrapper (which
	// truncates PointPosFromRow's always-3D result to workspace_dim) -- so
	// the runtime hold-drift check (GraphOfConstraintsMPC, which reads both
	// link_pose and point_position together) works for a 2-workspace graph
	// as long as workspace_dim is actually set to 2 (it does NOT follow
	// any robot's/object's own width automatically).
	int workspace_dim;

	// Required for big-M computation
	Eigen::VectorXd _global_x_lb;
	Eigen::VectorXd _global_x_ub;

	// Placeholder families for the unified symbolic constraint API -- see
	// PlaceholderVarFamily's own docstring for why these are lazy
	// PlaceholderVarFamily instances rather than 10 separate hand-rolled
	// members. `dim`/`non_robot_dim`/`workspace_dim` aren't known until the
	// constructor BODY computes them (they depend on robot_specs/
	// object_specs), so these are default-constructed here and assigned
	// their real width/namer in the constructor body, not this
	// declaration -- see graph_of_constraints.cpp.
	//
	// _agent_q/_object_q/_var_agent_q: node-scope (and "along the edge"
	// edge-scope) placeholders -- what agent_q(k)/object_q(k)/
	// var_agent_q(var) return.
	PlaceholderVarFamily<int> _agent_q;
	PlaceholderVarFamily<int> _object_q;
	PlaceholderVarFamily<int> _var_agent_q;

	// _param: runtime-editable scalar placeholders (width 1, one Variable
	// per declared id) -- what param(id) returns, and what set_param(id, .)
	// updates. Unlike agent_q/object_q/var_agent_q (which stand for a
	// decision variable a solve() call optimizes over), a param is a
	// caller-supplied CONSTANT: both compilers substitute it with the
	// current entry of _param_values at solve/evaluate time, the same way
	// they already substitute x0 into a depot/stationarity row, so a
	// generator can reference a value it doesn't know yet at add_constraint
	// time (e.g. derived from live sensor/env state) and correct it later
	// via set_param without re-authoring the constraint's Formula. See
	// add_param/param/set_param below, and formula_compiler.py's/spec.py's
	// param_map (evolutionary side: threaded as a genuine jax runtime
	// argument, not a value baked into the compiled/jitted closure, so
	// set_param never forces a retrace).
	PlaceholderVarFamily<int> _param;
	Eigen::VectorXd _param_values;

	// _agent_link_pos: world-position placeholder (workspace_dim-wide) for
	// a registered (agent_id, link_name)'s forward kinematics -- see
	// agent_link_pos() and set_robot_fk/link_pose above.
	PlaceholderVarFamily<std::pair<int, std::string>> _agent_link_pos;

	// _agent_link_rot: world-rotation placeholder for the same
	// (agent_id, link_name) forward kinematics, flattened row-major
	// (workspace_dim*workspace_dim-wide -- entry (i, j) of the
	// workspace_dim x workspace_dim rotation matrix sits at index
	// i*workspace_dim + j, matching numpy/jax's default 'C'-order
	// flatten()/reshape()) -- see agent_link_rot() below.
	PlaceholderVarFamily<std::pair<int, std::string>> _agent_link_rot;

	// _agent_q_u/_v, _object_q_u/_v: the u (start) and v (end) side of a
	// *relational* edge constraint (see add_edge_constraint) -- a formula
	// built from these is a single relation coupling both endpoints.
	// Distinct from _agent_q/_object_q, which are reserved for node
	// constraints and for "along the edge" edge constraints: an invariant
	// applied independently at each node the edge might span, not a
	// relation between two specific endpoints.
	PlaceholderVarFamily<int> _agent_q_u;
	PlaceholderVarFamily<int> _object_q_u;
	PlaceholderVarFamily<int> _agent_q_v;
	PlaceholderVarFamily<int> _object_q_v;

	// u/v-side counterparts of _var_agent_q -- the *relational* analogue
	// of var_agent_q(), letting a two-sided edge formula reference "whichever
	// agent this assignable variable resolves to" independently on each
	// side (e.g. v_var_agent_q(var) - u_var_agent_q(var), the assignable
	// analogue of v_agent_q(k) - u_agent_q(k)). Unlike MILP -- which has no
	// way to express this at all, since drake::symbolic::Formula has no
	// dynamic-indexing primitive and must instead enumerate every candidate
	// agent and big-M gate each one (see milp_waypoint_mpc.cpp's
	// AddHoldRigidityAssignable/AddHoldRigidityAssignableGated) -- the JAX
	// evolutionary solver resolves these placeholders with a genuinely
	// dynamic (per-individual) jax.lax.dynamic_slice keyed off the
	// GA-searched assignment (see spec.py's _make_row_resolver), so no
	// enumeration is needed there. Consequently MILPWaypointMPC does NOT
	// support compiling a relational edge formula that references these --
	// symbolic_constraint_compiler.cpp's CompileSymbolicEdgeConstraint has no
	// substitution entry for them and will fail (unsubstituted free
	// variables) if one ever reaches it.
	PlaceholderVarFamily<int> _var_agent_q_u;
	PlaceholderVarFamily<int> _var_agent_q_v;

	// Symbolic variable per assignable variable, used to write conditional
	// edge ordering formulas (e.g. r0_sym == r1_sym means same agent assigned).
	std::vector<drake::symbolic::Variable> _assignment_sym_vars;
	std::map<drake::symbolic::Variable::Id, int> _sym_id_to_variable_id;

	// Free binary symbolic variables for conditional ordering formulas.
	std::vector<drake::symbolic::Variable> _binary_cond_sym_vars;
	std::map<drake::symbolic::Variable::Id, int> _sym_id_to_binary_cond_id;

	// Maps DAG edge (u,v) -> Formula that must hold for the ordering t(u)≤t(v)
	// to be enforced. Edges absent from this map get hard ordering constraints.
	std::map<std::pair<int,int>, drake::symbolic::Formula> _conditional_ordering_map;

	// Returns the symbolic variable representing the assignment index of variable r.
	const drake::symbolic::Variable& assignment_sym(int r) const {
		return _assignment_sym_vars.at(r);
	}

	// Creates a new free binary symbolic variable for use in conditional ordering formulas.
	drake::symbolic::Variable add_binary_cond_var() {
		int id = _binary_cond_sym_vars.size();
		drake::symbolic::Variable v("bv_" + std::to_string(id));
		_binary_cond_sym_vars.push_back(v);
		_sym_id_to_binary_cond_id[v.get_id()] = id;
		return v;
	}

	// Adds a conditional ordering edge: t(u)≤t(v) is enforced iff formula f holds.
	// Does NOT add to the structure graph — conditional edges are invisible to BFS/routing.
	void add_conditional_edge_ordering(int u, int v, const drake::symbolic::Formula& f) {
		_conditional_ordering_map[{u, v}] = f;
	}

	// Constructor
	GraphOfConstraints(const std::vector<CubicConfigurationSpline::Spec>& robot_specs,
			   const std::vector<CubicConfigurationSpline::Spec>& object_specs,
			   double global_x_lb,
			   double global_x_ub,
			   const std::vector<std::string>& robot_names = {},
			   const std::vector<std::string>& object_names = {},
			   int workspace_dim = 3);

	int add_variable();

	RobotKind robot_kind(int ag) const { return _robot_kinds.at(ag); }

	ConstraintDegree robot_rigidity_constraint_degree(int ag) const {
		switch (robot_kind(ag)) {
		case RobotKind::kPointMass:   return ConstraintDegree::kLinear;
		case RobotKind::kPosYaw:      return ConstraintDegree::kGeneral;
		case RobotKind::kPosRotMat:   return ConstraintDegree::kQuadratic;
		case RobotKind::kPosQuat:     return ConstraintDegree::kGeneral;
		case RobotKind::kArticulated: return ConstraintDegree::kGeneral;
		}
	}

	bool robot_is_free_body(int ag) const {
		const auto k = robot_kind(ag);
		return k == RobotKind::kPointMass || k == RobotKind::kPosYaw ||
		       k == RobotKind::kPosQuat   || k == RobotKind::kPosRotMat;
	}

	bool robot_is_pos_yaw(int ag) const     { return robot_kind(ag) == RobotKind::kPosYaw; }
	bool robot_is_pos_quat(int ag) const    { return robot_kind(ag) == RobotKind::kPosQuat; }
	bool robot_is_pos_rot_mat(int ag) const { return robot_kind(ag) == RobotKind::kPosRotMat; }
	bool robot_is_point_mass(int ag) const  { return robot_kind(ag) == RobotKind::kPointMass; }

	int robot_ambient_dim(int ag) const;

	int robot_tangent_dim(int ag) const;

	int object_ambient_dim(int ob) const;

	// Per-column mask over a full waypoint/state row (length `total_dim`):
	// entry `c` is 1 iff some constraint attached to `node` references
	// ambient column `c` -- i.e. that component of some agent's/object's
	// config is pinned at `node`. Covers node constraints (`symbolic_ops`)
	// and symbolic edge constraints incident to `node` -- both the "along
	// the edge" invariant form (plain agent_q/object_q/var_agent_q,
	// applied at the endpoint) and the relational u_/v_ form (u-side refs
	// bind node u, v-side refs bind node v). `var_assignments` (a waypoint
	// solver's `view_var_assignments()`, indexed by var id, -1 ==
	// unassigned) resolves var_agent_q(var) references to a concrete
	// agent's columns; a reference to a var that is unassigned or out of
	// range for the passed vector is skipped (so an empty vector simply
	// ignores every assignable-constraint reference). "References" is
	// literal Formula membership -- an inequality or a one-sided bound
	// counts, not only an equality. A legacy non-symbolic DeferredOp is
	// not introspectable and contributes nothing.
	//
	// The spline builder uses this to decide, per block, whether `node` is
	// a genuine knot for that block or a pass-through the interpolation
	// should bridge (every column of the block reading back 0 == that
	// block was left free at this node).
	Eigen::VectorXi constrained_columns(
		int node, const Eigen::VectorXi& var_assignments = Eigen::VectorXi()) const;

	// See `_agent_col_offsets`/`_object_col_offsets`'s own doc comment.
	// `ag`/`ob` may equal num_agents/num_objects (one past the last real
	// entry) to get that block's total width.
	int agent_col_offset(int ag) const { return _agent_col_offsets.at(ag); }
	int object_col_offset(int ob) const { return _object_col_offsets.at(ob); }

	// Python-registered forward-kinematics override for a single
	// (agent_id, link_name), consulted by link_pose before falling back to
	// the built-in RobotKind dispatch. `fk_fn` is a Python callable
	// `(q_agent: np.ndarray) -> (position: np.ndarray(workspace_dim,),
	// rotation: np.ndarray(workspace_dim, workspace_dim))`, where q_agent
	// is that agent's own dim-sized config slice (not the full state row).
	// position/rotation may be 2D or 3D (matching this graph's
	// workspace_dim) -- link_pose returns them at whatever size fk_fn
	// actually produces, and agent_link_pos's placeholder is sized to
	// workspace_dim to match.
	//
	// `fk_fn` is consulted by TWO independent callers, so it should be
	// written using jax.numpy internally rather than plain numpy: (1)
	// link_pose (below), called from C++ via pybind with a concrete
	// numpy.ndarray -- runs eagerly, works fine for a jax.numpy-only body
	// since jnp ops accept numpy input transparently; and (2) the JAX
	// evolutionary solver (src/goc_mpc/evolutionary_waypoint_solver/spec.py),
	// which reads the raw callable straight out of robot_fk_registry (see
	// the pybind-exposed property of the same name) and calls it DIRECTLY
	// in Python -- bypassing this C++ boundary entirely -- with a JAX
	// tracer during jax.jit/vmap tracing, so it can back an
	// agent_link_pos(agent_id, link_name) constraint placeholder (see
	// below) that the evolutionary solver actually searches/solves
	// against. A function using only jnp ops (no data-dependent branching,
	// no numpy-only calls) is valid for both call sites; a plain
	// numpy-only fk_fn works for (1) but will fail if ever referenced by
	// an agent_link_pos constraint (2), since that path traces it under
	// jax.jit with an abstract input.
	void set_robot_fk(int agent_id, const std::string& link_name, py::function fk_fn);

	// Forward kinematics for `link_name` on robot `agent_id`, from a full
	// state row `x` (total_dim, same agents-then-objects layout as
	// GraphOfConstraintsMPC's x). Looks up a Python-registered override
	// (set_robot_fk) for (agent_id, link_name) first; if none is
	// registered, falls back to the built-in per-robot-kind pose (see
	// utils.hpp's PoseFromRow, which this wraps) -- `link_name` is
	// otherwise unused by that fallback, since none of the built-in
	// RobotKinds model more than one link. Exposes the same "what pose does
	// this robot's state row represent" computation used by C++ constraint
	// builders (e.g. add_robot_pos_linear_eq) to non-C++-constraint
	// consumers, so there is exactly one place this logic lives. Throws for
	// kArticulated with no registered override: true joint-chain FK isn't
	// supported by the built-in dispatch.
	//
	// Returns a pose sized to this graph's workspace_dim (2 or 3 -- see
	// that member's doc comment): position is workspace_dim-long, rotation
	// is workspace_dim x workspace_dim. A registered fk_fn's result is
	// checked against workspace_dim (raises clearly on mismatch, see the
	// .cpp). The built-in RobotKind fallback follows workspace_dim for
	// kPointMass/kPosYaw (PoseFromRow_PointMass/PoseFromRow_PosYaw,
	// utils.hpp); kPosQuat/kPosRotMat are inherently 3D rotation
	// representations and throw if workspace_dim != 3.
	std::pair<Eigen::VectorXd, Eigen::MatrixXd> link_pose(int agent_id, const Eigen::VectorXd& x,
							      const std::string& link_name = "ee") const;

	// World position of object/point `point_id` from a full state row `x`,
	// sized to workspace_dim (matching link_pose's position sizing --
	// GraphOfConstraintsMPC's runtime hold-drift check, the only caller
	// that needs this Python-facing wrapper rather than the internal
	// always-3D PointPosFromRow directly, combines the two:
	// `R_we.T @ (point_position(...) - p_we)`). See utils.hpp's
	// PointPosFromRow, which this wraps and truncates to workspace_dim.
	Eigen::VectorXd point_position(int point_id, const Eigen::VectorXd& x) const;

	Graph<py::object> get_structure() const { return structure; }

	// Determines, for each agent, the ordered sequence of nodes it visits,
	// and the cross-agent timing interactions (shared nodes -> EQUAL,
	// cross-agent structural edges -> LESS_THAN) needed to keep multiple
	// agents' schedules correctly coupled. Node ownership is resolved per
	// phi (assignable-var resolution, then legacy static grasp assignment,
	// then -- for a plain literal agent_q(i)/object_q formula that goes
	// through neither -- which agent_q_vars the formula actually
	// references), NOT assumed from the node alone; see get_agent_paths's
	// definition for why "no phi at this node resolves to a specific
	// agent" (a pure object-only node) is the only case that still falls
	// back to "every agent". `t_by_node` (the waypoint MPC's resolved
	// per-node arrival-time estimate) orders each agent's own node list;
	// pass an empty vector to fall back to BFS/topological order (e.g.
	// before any waypoint solve has produced timings yet).
	std::tuple<std::vector<std::optional<int>>,
		   std::vector<std::vector<int>>,
		   std::vector<struct AgentInteraction>> get_agent_paths(
			   const std::vector<int>& remaining_vertices,
			   const Eigen::VectorXi& assignments,
			   const Eigen::VectorXd& t_by_node) const;

	std::map<std::pair<int, int>, int> get_next_edge_phis(const std::vector<int> completed_vertices) const;

	// Holds (see HoldDeclaration/hold_ops) currently in progress: those whose
	// u_node (pick-up) is NOT in remaining_vertices (i.e. already completed)
	// while their v_node (release) IS -- so whatever they declared held
	// should still be rigidly attached to the holding robot right now.
	// Unlike get_next_edge_phis (which walks structure's actual DAG edges
	// via incoming_cut_edges), a hold's (u_node, v_node) pair need not be a
	// literal graph edge -- there's typically at least one interior node
	// (e.g. an approach-to-release waypoint) scheduled between them -- so
	// this is a plain membership check against hold_ops rather than a graph
	// traversal.
	std::map<int, HoldDeclaration> get_current_holds(const std::vector<int>& remaining_vertices) const;

	// Reindexes agent_interactions' agent_i_depth/agent_j_depth -- indices
	// into a per-agent node-id sequence, used to slice `time_deltas_list[i].
	// head(depth+1)` when enforcing a cross-agent LESS_THAN/EQUAL timing
	// constraint -- from get_agent_paths' own (real-node-only) agent_nodes
	// lists to indices into a caller-supplied sequence instead. Needed when
	// a caller expands each agent's real-node sequence into a DENSER one
	// (e.g. with traced interior waypoints spliced in between consecutive
	// real nodes, id == -1) before building the timing QP: `depth` then has
	// to mean "row in the dense sequence", not "index into the sparse
	// real-node list", or a cross-agent constraint would sum too few
	// segments and under-constrain the timing. `agent_node_ids[i]`: one
	// entry per decision-variable row for agent i, holding either the real
	// graph node id at that row or -1 for a synthetic row. Also used by
	// get_agent_paths itself to resolve its own (trivially "dense" ==
	// "sparse") depths, so there's exactly one implementation of this
	// lookup.
	static std::vector<AgentInteraction> reindex_agent_interactions(
		std::vector<AgentInteraction> agent_interactions,
		const std::vector<std::vector<int>>& agent_node_ids);

	const std::map<int, DeferredVarOp>& get_var_ops() const {
		return var_ops;
	}

	std::vector<int> get_phi_ids(int node) const;

	bool evaluate_phi(int phi_id,
			  const Eigen::VectorXd& x,
			  const Eigen::VectorXi& assignments,
			  double tol) const;

	bool evaluate_edge_phi(int phi_id,
			       const Eigen::VectorXd& x,
			       const Eigen::VectorXi& var_assignments,
			       double tol) const;

	int get_edge_phi_agent(int phi_id, const Eigen::VectorXi& var_assignments) const;

	void add_backtrack_links(int edge_id, std::vector<int> backtrack_nodes);
	void add_manual_backtrack_links(int edge_id, std::vector<int> backtrack_nodes);

	// Declares that completing `node` should pin variable `var`'s resolved
	// agent (see commit_trigger_node_to_var). Build-time only -- does not
	// itself commit anything; GraphOfConstraintsMPC calls
	// commit_variable_assignment once `node` actually completes.
	void add_variable_commit(int var, int node);

	// Runtime pin state (see committed_assignments). `commit_variable_
	// assignment` is idempotent-ish: it overwrites any existing pin for
	// `var`, so re-arming after a fresh commit trigger always reflects the
	// latest resolution. `clear_variable_commitment` is a no-op if `var`
	// isn't currently pinned.
	void commit_variable_assignment(int var, int agent);
	void clear_variable_commitment(int var);

	// If `node` is a registered commit trigger (see add_variable_commit),
	// returns the variable it commits; otherwise std::nullopt.
	std::optional<int> get_commit_trigger_var(int node) const;

	// Grasp change util
	void add_grasp_change(int phi_id, std::string command, int robot_id, int cube_id);
	void add_assignable_grasp_change(int phi_id, std::string command, int cube_id);
	std::vector<std::tuple<std::string, std::string, std::string>> get_grasp_changes(int k, Eigen::VectorXi assignments) const;

	// Unpassable node util
	void make_node_unpassable(int k);

	// Node creation, thin wrappers around structure.add_node()/add_nodes(n)
	// that also register `name`/`names` in node_names (see below) in the
	// same call, so callers don't need a separate set_node_name round-trip.
	// Naming is still optional here -- callers that don't pass a name can
	// keep calling structure.add_node()/add_nodes(n) directly, or call these
	// and name the node(s) later via set_node_name/set_node_names.
	int add_node(std::optional<std::string> name = std::nullopt);
	std::vector<int> add_nodes(int n, std::optional<std::vector<std::string>> names = std::nullopt);

	// Node naming util (see node_names above).
	void set_node_name(int k, const std::string& name);
	void set_node_names(const std::vector<int>& ks, const std::vector<std::string>& names);
	// Returns node_names.at(k) if set, otherwise k's numeric id as a string
	// (e.g. "3") so callers can print a node's name unconditionally without
	// having to special-case unnamed graphs.
	std::string get_node_name(int k) const;
	
	// Adding Constraints
	int add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
	int add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
	int add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c = 0.0);

	int add_robots_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_robots_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	int add_robot_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_robot_linear_ineq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	int add_robot_pos_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_robot_quat_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_assignable_robot_quat_linear_eq(int k, int var, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	int add_point_linear_eq(int k, int point_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	int add_point_linear_ineq(int k, int point_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	int add_assignable_linear_eq(int k, int var, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	int add_robot_above_cube_constraint(int k,
					    int robot_id,
					    int cube_id,
					    double delta_z,
					    double x_offset = 0.0,
					    double y_offset = 0.0);

	int add_assignable_robot_to_point_displacement_constraint(int k,
								  int var,
								  int point_id,
								  const Eigen::Vector3d& disp);

	int add_robot_to_point_displacement_constraint(int k,
						       int robot_id,
						       int point_id,
						       const Eigen::VectorXd& disp,
						       double tol = 0.0);
	int add_robot_to_point_alignment_constraint(int k,
						    int robot_id,
						    int point_id,
						    const Eigen::Vector3d& ee_ray_body,
						    // optional for roll disambiguation:
						    std::optional<Eigen::Vector3d> u_body_opt = std::nullopt,         // u_b (must be ⟂ ee_ray_body)
						    std::optional<Eigen::Vector3d> roll_ref_world = std::nullopt,     // t (any, not necessarily ⟂ d)
						    bool roll_ref_flat = false,
						    bool require_positive_pointing = true,
						    double eps_d = 0.05, double tau_tperp = 0.05);

	int add_point_to_point_displacement_constraint(int k,
						       int point_a,
						       int point_b,
						       Eigen::Vector3d& disp,
						       double tol = 0.05);

	int add_point_to_point_alignment_constraint(int k,
						    int point_a,
						    int point_b,
						    const Eigen::Vector3d& dir_W);

	// Edge Constraints

	// DEPRECATED: superseded by add_hold below, which is now the canonical
	// way to declare a hold. This still adds a proximity check (and
	// dual-populates hold_ops for backward compat) but MILP's Constraint 14
	// (rigid-hold + stationary-object dynamics) is moving to read hold_ops
	// directly instead of edge_ops[phi].cubes + the static/assignable
	// edge-phi maps -- prefer add_hold/add_assignable_hold in new code.
	int add_robot_holding_cube_constraint(int u,
					      int v,
					      int robot_id,
					      int cube_id,
					      double holding_distance_max = 0.1,
					      bool use_l2 = false);

	int add_edge_point_to_point_displacement_constraint(int u,
							    int v,
							    int point_a,
							    int point_b,
							    Eigen::Vector3d& disp,
							    Eigen::Vector3d& tol);

	int add_robot_relative_rotation_constraint(int u,
						   int v,
						   int robot_id,
						   Eigen::Quaternion<double>& quat);

	int add_robot_relative_displacement_constraint(int u,
						       int v,
						       int robot_id,
						       Eigen::Vector3d& disp);

	int add_robot_holding_points_constraint(int u,
						int v,
						int robot_id,
						int point_ids,
						double holding_distance_max = 0.1);

	// Assignable Edge Constraints

	int add_edge_assignable_robot_to_point_displacement_constraint(int u,
								       int v,
								       int var,
								       int point_id,
								       Eigen::Vector3d& disp,
								       Eigen::Vector3d& tol);

	// DEPRECATED: superseded by add_assignable_hold below -- see
	// add_robot_holding_cube_constraint's deprecation note above.
	int add_assignable_robot_holding_point_constraint(int u,
							  int v,
							  int var,
							  int point_id,
							  double holding_distance_max = 0.1,
							  bool use_l2 = false);

	// Canonical hold-declaration API (see HoldDeclaration) -- the primary,
	// Python-exposed way to declare that `held_point_ids` are rigidly held
	// by a robot over edge (u -> v), either a statically assigned robot
	// (add_hold) or one resolved later via an assignable variable
	// (add_assignable_hold). Pure bookkeeping today (no proximity/rigidity
	// constraint is added yet); MILPWaypointMPC's Constraint 14 and the JAX
	// evolutionary solver are both moving to derive their rigid-hold /
	// stationary-object dynamics constraints from this registry directly.
	// add_assignable_hold also auto-registers `u` as a commit trigger for
	// `var` (see add_variable_commit) -- once pick-up completes, the
	// routing solve can no longer reassign the hold's holder mid-grasp.
	int add_hold(int u, int v, int robot_ag, std::vector<int> held_point_ids);
	int add_assignable_hold(int u, int v, int var, std::vector<int> held_point_ids);

	// Timing (Edge) Constraints

	void add_edge_min_tau_constraint(int u,
					 int v,
					 double minimum_time_delta);

	// Variable Constraints

	// 'var' can only be assigned to {robot_ids}
	int add_variable_constraint(int var,
				    std::set<int> robot_ids);

	// var1 != var2
	int add_variable_ineq_constraint(int var1,
					 int var2);

	// Symbolic unified constraint API
	drake::VectorX<drake::symbolic::Expression> agent_q(int agent_id) const;
	drake::VectorX<drake::symbolic::Expression> object_q(int object_id) const;
	drake::VectorX<drake::symbolic::Expression> var_agent_q(int var);
	// World-position placeholder (workspace_dim-vector -- see that
	// member's doc comment) for a registered (agent_id, link_name)'s
	// forward kinematics -- see set_robot_fk/link_pose above. Lazily
	// creates a fresh workspace_dim-wide placeholder Variable vector per
	// distinct (agent_id, link_name), mirroring var_agent_q's
	// own lazy-creation pattern. Usable anywhere agent_q/object_q are
	// (add_constraint, or an "along the edge" add_edge_constraint) -- NOT
	// as a u_/v_-prefixed relational edge placeholder, since it's tied to
	// one fixed agent at authoring time, not a two-sided relation.
	// Compilable only by the JAX evolutionary solver (which resolves it
	// by calling the registered fk_fn directly, in Python, against
	// robot_fk_registry -- see set_robot_fk's doc comment); MILPWaypointMPC
	// raises a clear error if a formula referencing this placeholder ever
	// reaches its compiler (symbolic_constraint_compiler.cpp), since
	// drake::symbolic::Formula has no way to represent arbitrary FK.
	drake::VectorX<drake::symbolic::Expression> agent_link_pos(int agent_id, const std::string& link_name);
	// Rotation counterpart of agent_link_pos, above -- same registered
	// fk_fn (its rotation half), same lazy-creation/scope/MILP-compilation
	// story, flattened row-major into a workspace_dim*workspace_dim-vector
	// (see _agent_link_rot's doc comment for the index convention). A
	// target rotation matrix R (workspace_dim x workspace_dim numpy array)
	// can therefore be compared directly via
	// eq(graph.agent_link_rot(agent_id, link_name), R.flatten()) -- numpy's
	// default flatten() order matches this placeholder's layout exactly.
	drake::VectorX<drake::symbolic::Expression> agent_link_rot(int agent_id, const std::string& link_name);

	// Declares a new runtime-editable scalar parameter, initialized to
	// `initial_value`, and returns its id (0, 1, 2, ... in declaration
	// order). Use param(id) to reference it inside a Formula passed to
	// add_constraint/add_edge_constraint; use set_param(id, .) later (any
	// time before the next solve/evaluate_phi call that needs the new
	// value) to update it in place -- see _param's doc comment for why this
	// is cheap on both solvers.
	int add_param(double initial_value = 0.0);
	// Scalar placeholder Expression for parameter `id` -- usable anywhere
	// agent_q/object_q are (add_constraint, or an "along the edge"
	// add_edge_constraint).
	drake::symbolic::Expression param(int id) const;
	// Overwrites parameter `id`'s current value. Does NOT touch the
	// Formula(s) that reference it -- those were fixed at add_constraint
	// time -- only the value both compilers substitute in for it going
	// forward.
	void set_param(int id, double value);
	const Eigen::VectorXd& view_param_values() const { return _param_values; }
	int num_params() const { return static_cast<int>(_param_values.size()); }

	int add_constraint(int node, const drake::symbolic::Formula& f);
	int add_assignable_constraint(int node, int var, const drake::symbolic::Formula& f);

	drake::VectorX<drake::symbolic::Expression> u_agent_q(int agent_id) const;
	drake::VectorX<drake::symbolic::Expression> u_object_q(int object_id) const;
	drake::VectorX<drake::symbolic::Expression> v_agent_q(int agent_id) const;
	drake::VectorX<drake::symbolic::Expression> v_object_q(int object_id) const;
	// See _var_agent_q_u/_v's docstring for what these mean and their
	// MILP-compilation limitation.
	drake::VectorX<drake::symbolic::Expression> u_var_agent_q(int var);
	drake::VectorX<drake::symbolic::Expression> v_var_agent_q(int var);

	// `live`: once this edge's `u` node passes (leaves a receding-horizon
	// waypoint solver's remaining_vertices), should the constraint's `u` side
	// keep reading the REAL, closed-loop state at that node (live_edge_phis
	// above), or the solver's last FROZEN value from while the node was
	// still active (the default)? The canonical case is a rigid-transport
	// edge tying a released/grasped object's displacement to its carrying
	// agent's: pyrobosim's actual grasp model (Robot._attach_object)
	// collapses agent and object to the exact same pose the instant a grasp
	// happens, zero standoff, not whatever nominal offset the plan
	// approached the grasp with -- so that edge must read live, or it keeps
	// propagating the (by-then-fictional) planned offset forward to every
	// later node forever, and a node constraint pinning the object's final
	// position downstream (e.g. a place node) can never converge within
	// tolerance. Most edge constraints want the opposite (frozen):
	// re-reading noisy live state for something that's just a planned
	// intermediate waypoint would make the residual jitter with tracking
	// error instead of holding the plan's own resolved value. Not consulted
	// by MILPWaypointMPC (which has no live/frozen distinction -- see
	// EvolutionaryWaypointSolver's module docstring); only meaningful for
	// the evolutionary waypoint solver today.
	int add_edge_constraint(int u, int v, const drake::symbolic::Formula& f, bool live = false);

private:

	template <typename EF, typename F>
	int _add_op(DeferredOpKind kind, int node, EF&& eval_f, F&& f) {
		const int id = num_phis++;
		node_to_phis_map[node].push_back(id);
		ops[id] = DeferredOp{kind, id, node, std::forward<EF>(eval_f), std::forward<F>(f)};
		return id;
	}

	template <typename EF, typename F>
	int _add_assignable_op(DeferredOpKind kind, int node, int var, EF&& eval_f, F&& f) {
		const int id = num_phis++;
		node_to_phis_map[node].push_back(id);
		phi_to_variable_map[id] = var;
		ops[id] = DeferredOp{kind, id, node, std::forward<EF>(eval_f), std::forward<F>(f)};
		return id;
	}

	template <typename EF, typename WF, typename SF>
	int _add_edge_op(DeferredOpKind kind, int u, int v, std::set<int> cubes, EF&& eval_f, WF&& wp_f, SF&& sp_f) {
		const int id = num_edge_phis++;
		edge_to_phis_map[std::make_pair(u, v)].push_back(id);
		edge_ops[id] = DeferredEdgeOp{kind, id, u, v, cubes,
			std::forward<EF>(eval_f), std::forward<WF>(wp_f), std::forward<SF>(sp_f)};
		return id;
	}

	template <typename EF, typename WF, typename SF>
	int _add_assignable_edge_op(DeferredOpKind kind, int u, int v, int var, std::set<int> cubes, EF&& eval_f, WF&& wp_f, SF&& sp_f) {
		const int id = num_edge_phis++;
		edge_to_phis_map[std::make_pair(u, v)].push_back(id);
		edge_phi_to_variable_map[id] = var;
		edge_ops[id] = DeferredEdgeOp{kind, id, u, v, cubes,
			std::forward<EF>(eval_f), std::forward<WF>(wp_f), std::forward<SF>(sp_f)};
		return id;
	}

	template <typename F>
	int _add_var_op(DeferredOpKind kind, F&& f) {
		const int id = num_var_phis++;
		var_ops[id] = DeferredVarOp{kind, id, std::forward<F>(f)};
		return id;
	}

	int _add_symbolic_op(int node, const drake::symbolic::Formula& f) {
		const int id = num_phis++;
		node_to_phis_map[node].push_back(id);
		std::vector<BlockResidualGroup> groups;
		std::vector<drake::symbolic::Formula> ungrouped;
		PopulateBlockResidualGroups(*this, f, &groups, &ungrouped);
		symbolic_ops[id] = SymbolicNodeConstraint{
			id, node, std::nullopt, {}, f, std::move(groups), std::move(ungrouped)};
		return id;
	}

	int _add_symbolic_assignable_op(int node, int var, const drake::symbolic::Formula& f) {
		const int id = num_phis++;
		node_to_phis_map[node].push_back(id);
		phi_to_variable_map[id] = var;
		std::vector<BlockResidualGroup> groups;
		std::vector<drake::symbolic::Formula> ungrouped;
		PopulateBlockResidualGroups(*this, f, &groups, &ungrouped);
		symbolic_ops[id] = SymbolicNodeConstraint{
			id, node, var, {}, f, std::move(groups), std::move(ungrouped)};
		return id;
	}

	int _add_symbolic_multi_var_op(int node, const std::vector<int>& var_ids, const drake::symbolic::Formula& f) {
		const int id = num_phis++;
		node_to_phis_map[node].push_back(id);
		std::vector<BlockResidualGroup> groups;
		std::vector<drake::symbolic::Formula> ungrouped;
		PopulateBlockResidualGroups(*this, f, &groups, &ungrouped);
		symbolic_ops[id] = SymbolicNodeConstraint{
			id, node, std::nullopt, var_ids, f, std::move(groups), std::move(ungrouped)};
		return id;
	}

	int _add_symbolic_edge_op(int u, int v, const drake::symbolic::Formula& f,
				  bool along_edge, bool live = false, std::optional<int> var_id = std::nullopt) {
		const int id = num_edge_phis++;
		edge_to_phis_map[std::make_pair(u, v)].push_back(id);
		if (var_id.has_value()) edge_phi_to_variable_map[id] = var_id.value();
		std::vector<BlockResidualGroup> groups;
		std::vector<drake::symbolic::Formula> ungrouped;
		PopulateBlockResidualGroups(*this, f, &groups, &ungrouped);
		symbolic_edge_ops[id] = SymbolicEdgeConstraint{
			id, u, v, f, along_edge, var_id, std::move(groups), std::move(ungrouped)};
		if (live) live_edge_phis.insert(id);
		return id;
	}
};

/*
 * Subgraph
 */

struct SubgraphOfConstraints {
	InducedSubgraphView<py::object> structure;
	std::map<int, int> _variable_to_subgraph_variable_id; // unique ids for variables relevant to subgraph.
	std::map<int, DeferredOp> _subgraph_ops;
	std::map<int, DeferredEdgeOp> _subgraph_edge_ops;
	std::map<int, SymbolicNodeConstraint> _subgraph_symbolic_ops;
	std::map<int, SymbolicEdgeConstraint> _subgraph_symbolic_edge_ops;
	std::map<int, HoldDeclaration> _subgraph_hold_ops;

	SubgraphOfConstraints(GraphOfConstraints *graph, const std::vector<int>& vertices) :
		structure(graph->structure, vertices) {

		// std::map<int, int> phi_to_subgraph_node_id;
		// std::map<int, int> phi_to_subgraph_assignable_id;
		// std::map<int, int> subgraph_assignable_id_to_phi;

		int num_subgraph_variables = 0;

		for (int u : vertices) {
			// if there is/are phi associated with v
			if (graph->node_to_phis_map.contains(u)) {
				for (int phi_id : graph->node_to_phis_map.at(u)) {
					if (graph->ops.contains(phi_id)) {
						_subgraph_ops[phi_id] = graph->ops.at(phi_id);
					} else if (graph->symbolic_ops.contains(phi_id)) {
						_subgraph_symbolic_ops[phi_id] = graph->symbolic_ops.at(phi_id);
					}

					// Record the mapping from phi id to subgraph node and assignable var idxs.
					if (graph->phi_to_variable_map.contains(phi_id)) {
						const int variable_id = graph->phi_to_variable_map.at(phi_id);

						if (!_variable_to_subgraph_variable_id.contains(variable_id)) {
							_variable_to_subgraph_variable_id[variable_id] = num_subgraph_variables++;
						}

					}
				}
			}
		}

		for (const auto& edge : graph->structure.edges()) {
			int u = edge.u;
			int v = edge.e->to;
			if (structure.contains_node(u) || structure.contains_node(v)) {
				std::pair<int, int> e = std::make_pair(u, v);
				if (graph->edge_to_phis_map.contains(e)) {
					// Store the relevant edge ops so they can be applied
					for (int edge_phi_id : graph->edge_to_phis_map.at(e)) {
						if (graph->edge_ops.contains(edge_phi_id)) {
							_subgraph_edge_ops[edge_phi_id] = graph->edge_ops.at(edge_phi_id);
						} else if (graph->symbolic_edge_ops.contains(edge_phi_id)) {
							_subgraph_symbolic_edge_ops[edge_phi_id] = graph->symbolic_edge_ops.at(edge_phi_id);
						}
					}
				}
			}
		}

		// Hold registry entries (see HoldDeclaration) whose edge touches this
		// subgraph -- unlike edge_ops above, each hold already carries its own
		// u_node/v_node directly, so no edge_to_phis_map indirection is needed.
		for (const auto& [hold_id, hold] : graph->hold_ops) {
			if (structure.contains_node(hold.u_node) || structure.contains_node(hold.v_node)) {
				_subgraph_hold_ops[hold_id] = hold;
			}
		}
	}

	const std::map<int, DeferredOp>& get_subgraph_ops() const {
		return _subgraph_ops;
	}

	const std::map<int, DeferredEdgeOp>& get_subgraph_edge_ops() const {
		return _subgraph_edge_ops;
	}

	const std::map<int, SymbolicNodeConstraint>& get_subgraph_symbolic_ops() const {
		return _subgraph_symbolic_ops;
	}

	const std::map<int, SymbolicEdgeConstraint>& get_subgraph_symbolic_edge_ops() const {
		return _subgraph_symbolic_edge_ops;
	}

	const std::map<int, HoldDeclaration>& get_subgraph_hold_ops() const {
		return _subgraph_hold_ops;
	}

	int num_nodes() const {
		return structure.num_nodes();
	}

	int num_variables() const {
		return _variable_to_subgraph_variable_id.size();
	}

	int subgraph_id(int u) const {
		return structure.subgraph_id(u);
	}

	int subgraph_variable_id(int var) const {
		if (!_variable_to_subgraph_variable_id.contains(var)) { return -1; }
		return _variable_to_subgraph_variable_id.at(var);
	}
};
