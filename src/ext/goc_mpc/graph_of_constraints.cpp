#include "graph_of_constraints.hpp"
#include "symbolic_constraint_compiler.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <numeric>
#include <map>
#include <optional>
#include <tuple>


using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;

// Constructor
GraphOfConstraints::GraphOfConstraints(
		const std::vector<CubicConfigurationSpline::Spec>& robot_specs,
		const std::vector<CubicConfigurationSpline::Spec>& object_specs,
		double global_x_lb,
		double global_x_ub,
		const std::vector<std::string>& robot_names,
		const std::vector<std::string>& object_names,
		int workspace_dim)
	: _robot_specs(robot_specs),
	  _robot_names(robot_names),
	  _object_specs(object_specs),
	  _object_names(object_names),
	  num_phis(0),
	  num_edge_phis(0),
	  num_var_phis(0),
	  num_holds(0),
	  num_variables(0),
	  _num_total_assignables(0),
	  num_agents(robot_specs.size()),
	  num_objects(object_specs.size()),
	  dim(0),
	  non_robot_dim(0),
	  workspace_dim(workspace_dim) {

	if (!robot_names.empty() && robot_names.size() != robot_specs.size())
		throw std::runtime_error("robot_names size must match robot_specs size.");
	if (workspace_dim != 2 && workspace_dim != 3)
		throw std::runtime_error("workspace_dim must be 2 or 3.");
	if (!object_names.empty() && object_names.size() != object_specs.size())
		throw std::runtime_error("object_names size must match object_specs size.");

	for (const auto& spec : robot_specs) {
		const bool has_quat    = std::any_of(spec.begin(), spec.end(), [](const auto& b) {
			return b.type == CubicConfigurationSpline::Block::Type::SO3Quat; });
		const bool has_rot_mat = std::any_of(spec.begin(), spec.end(), [](const auto& b) {
			return b.type == CubicConfigurationSpline::Block::Type::SO3Mat; });
		const bool has_torus   = std::any_of(spec.begin(), spec.end(), [](const auto& b) {
			return b.type == CubicConfigurationSpline::Block::Type::Torus; });
		const bool all_eucl    = std::all_of(spec.begin(), spec.end(), [](const auto& b) {
			return b.type == CubicConfigurationSpline::Block::Type::R; });

		if (has_quat)         _robot_kinds.push_back(RobotKind::kPosQuat);
		else if (has_rot_mat) _robot_kinds.push_back(RobotKind::kPosRotMat);
		else if (has_torus)   _robot_kinds.push_back(RobotKind::kPosYaw);
		else if (all_eucl)    _robot_kinds.push_back(RobotKind::kPointMass);
		else                  _robot_kinds.push_back(RobotKind::kArticulated);
	}

	for (const auto& spec : robot_specs) {
		int robot_qdim = 0;
		for (const auto& b : spec) robot_qdim += b.size;
		if (dim == 0) {
			dim = robot_qdim;
		} else if (dim != robot_qdim) {
			throw std::runtime_error("Only supporting robots with the same dimension.");
		}
	}

	for (const auto& spec : object_specs) {
		int obj_dim = 0;
		for (const auto& b : spec) obj_dim += b.size;
		if (non_robot_dim == 0) {
			non_robot_dim = obj_dim;
		} else if (non_robot_dim != obj_dim) {
			throw std::runtime_error("Only supporting objects with the same dimension.");
		}
	}

	total_dim = num_agents * dim + num_objects * non_robot_dim;

	_global_x_lb = Eigen::VectorXd::Constant(total_dim, global_x_lb);
	_global_x_ub = Eigen::VectorXd::Constant(total_dim, global_x_ub);

	// Placeholder families -- see PlaceholderVarFamily's docstring for why
	// these are assigned here (constructor body) rather than in the
	// initializer list: their width depends on dim/non_robot_dim/
	// workspace_dim, none of which are known until the computations above
	// run. Every family is lazily populated regardless -- this assignment
	// doesn't itself create any placeholder Variables, just records each
	// family's width/namer for when agent_q()/object_q()/etc. first do.
	_agent_q = PlaceholderVarFamily<int>(dim, [](const int& i) { return fmt::format("agent_{}_q", i); });
	_object_q = PlaceholderVarFamily<int>(non_robot_dim, [](const int& o) { return fmt::format("object_{}_q", o); });
	_var_agent_q = PlaceholderVarFamily<int>(dim, [](const int& v) { return fmt::format("var_{}_agent_q", v); });
	_param = PlaceholderVarFamily<int>(1, [](const int& p) { return fmt::format("param_{}", p); });
	_agent_link_pos = PlaceholderVarFamily<std::pair<int, std::string>>(
		workspace_dim, [](const std::pair<int, std::string>& k) {
			return fmt::format("agent_{}_link_{}_pos", k.first, k.second);
		});
	_agent_link_rot = PlaceholderVarFamily<std::pair<int, std::string>>(
		workspace_dim * workspace_dim, [](const std::pair<int, std::string>& k) {
			return fmt::format("agent_{}_link_{}_rot", k.first, k.second);
		});
	_agent_q_u = PlaceholderVarFamily<int>(dim, [](const int& i) { return fmt::format("agent_{}_q_u", i); });
	_agent_q_v = PlaceholderVarFamily<int>(dim, [](const int& i) { return fmt::format("agent_{}_q_v", i); });
	_object_q_u = PlaceholderVarFamily<int>(non_robot_dim, [](const int& o) { return fmt::format("object_{}_q_u", o); });
	_object_q_v = PlaceholderVarFamily<int>(non_robot_dim, [](const int& o) { return fmt::format("object_{}_q_v", o); });
	_var_agent_q_u = PlaceholderVarFamily<int>(dim, [](const int& v) { return fmt::format("var_{}_agent_q_u", v); });
	_var_agent_q_v = PlaceholderVarFamily<int>(dim, [](const int& v) { return fmt::format("var_{}_agent_q_v", v); });

	int offset = 0;
	for (int ag = 0; ag < num_agents; ++ag) {
		for (const auto& b : _robot_specs[ag]) {
			if (b.type == CubicConfigurationSpline::Block::Type::SO3Quat ||
			    b.type == CubicConfigurationSpline::Block::Type::SO3Mat) {
				for (int j = offset; j < offset + b.size; ++j) {
					_global_x_lb(j) = -1;
					_global_x_ub(j) = 1;
				}
			}
			offset += b.size;
		}
	}
}

// add variable
// add assignable phi which depends on a variable => phi_to_variable_map

// when subgraph
// go through nodes
// for each phi at each node
// record the mapping from variable to subgraph_variable id
// record the mapping from node to subgraph node id

// when constructing the problem
// pass in variable_to_subgraph_variable_id

// record the variables the relevant to each phi. associate in the "phi_to_subgraph_variable_id"

int GraphOfConstraints::add_variable()
{
	int id = num_variables++;
	drake::symbolic::Variable sym_var("r_" + std::to_string(id));
	_assignment_sym_vars.push_back(sym_var);
	_sym_id_to_variable_id[sym_var.get_id()] = id;
	return id;
}


int GraphOfConstraints::robot_ambient_dim(int ag) const {
	int d = 0;
	for (const auto& b : _robot_specs.at(ag)) d += b.size;
	return d;
}

int GraphOfConstraints::robot_tangent_dim(int ag) const {
	int d = 0;
	for (const auto& b : _robot_specs.at(ag)) {
		d += (b.type == CubicConfigurationSpline::Block::Type::SO3Quat ||
		      b.type == CubicConfigurationSpline::Block::Type::SO3Mat) ? 3 : b.size;
	}
	return d;
}

int GraphOfConstraints::object_ambient_dim(int ob) const {
	int d = 0;
	for (const auto& b : _object_specs.at(ob)) d += b.size;
	return d;
}

void GraphOfConstraints::set_robot_fk(int agent_id, const std::string& link_name, py::function fk_fn) {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	robot_fk_registry[{agent_id, link_name}] = std::move(fk_fn);
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> GraphOfConstraints::link_pose(int agent_id, const Eigen::VectorXd& x,
									  const std::string& link_name) const {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	DRAKE_DEMAND(x.size() == total_dim);

	auto it = robot_fk_registry.find({agent_id, link_name});
	if (it != robot_fk_registry.end()) {
		const Eigen::VectorXd q_agent = x.segment(agent_id * dim, dim);
		py::tuple result = it->second(q_agent);
		DRAKE_DEMAND(result.size() == 2);
		// Coerce through numpy.asarray explicitly rather than relying on
		// pybind11/eigen's implicit conversion: fk_fn may be written in
		// jax.numpy (see set_robot_fk's doc comment) and return jax.Array
		// values rather than plain numpy.ndarray -- asarray guarantees a
		// real numpy array (via jax.Array's __array__) before the Eigen
		// cast, regardless of pybind11's own implicit-conversion coverage.
		static const py::object asarray = py::module_::import("numpy").attr("asarray");
		Eigen::VectorXd position = asarray(result[0]).cast<Eigen::VectorXd>();
		Eigen::MatrixXd rotation = asarray(result[1]).cast<Eigen::MatrixXd>();
		// fk_fn is expected to return a pose in this graph's own
		// workspace_dim (2 or 3) -- checked here, at the one place every
		// fk_fn result passes through, so a mismatched registration fails
		// clearly instead of surfacing later as a confusing shape error
		// in whatever formula/expression consumes it.
		if (position.size() != workspace_dim || rotation.rows() != workspace_dim ||
		    rotation.cols() != workspace_dim) {
			throw std::runtime_error(fmt::format(
				"link_pose: fk_fn registered for (agent {}, link '{}') returned a "
				"position of size {} and a rotation of shape {}x{}, but this graph's "
				"workspace_dim is {} -- fk_fn must return (position: ({},), rotation: "
				"({}, {})).",
				agent_id, link_name, position.size(), rotation.rows(), rotation.cols(),
				workspace_dim, workspace_dim, workspace_dim, workspace_dim));
		}
		return {position, rotation};
	}

	return PoseFromRow(this, agent_id, "", x);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::agent_link_pos(int agent_id, const std::string& link_name) {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	return _agent_link_pos.Get({agent_id, link_name});
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::agent_link_rot(int agent_id, const std::string& link_name) {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	return _agent_link_rot.Get({agent_id, link_name});
}

Eigen::VectorXd GraphOfConstraints::point_position(int point_id, const Eigen::VectorXd& x) const {
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);
	DRAKE_DEMAND(x.size() == total_dim);
	Eigen::Vector3d p_WC = PointPosFromRow(this, point_id, x);
	return p_WC.head(workspace_dim);
}

namespace {

// Resolves the specific agent(s) that own phi_id, in priority order:
//   1. an assignable var (var_agent_q) with a resolved MILP assignment,
//   2. a legacy static grasp assignment (add_grasp_change),
//   3. (only when neither above applies) which of the graph's literal
//      agent_q_vars[i] the phi's own Formula actually references -- the
//      case for a plain add_constraint(node, eq(q0[...], ...)) formula,
//      which was never routed through the assignable machinery and so
//      has no entry in phi_to_variable_map at all.
// Returns an empty set when none of the above resolves anything (a pure
// object-only phi, or a legacy DeferredOp with no Formula to introspect
// and no assignment) -- callers must treat that as "this phi has no
// opinion about agent ownership", not "belongs to every agent": a node
// can have both an object-only phi (no opinion) and an agent-specific
// phi (a real opinion) at once, and the object-only one must not drown
// the real one out.
std::set<int> PhiOwningAgents(const GraphOfConstraints& graph, int phi_id,
                              const Eigen::VectorXi& assignments) {
	if (graph.phi_to_variable_map.contains(phi_id)) {
		const int a = assignments(phi_id);
		if (a != -1) return {a};
		return {};
	}
	if (graph._phi_to_static_assignment_map.contains(phi_id)) {
		return {graph._phi_to_static_assignment_map.at(phi_id)};
	}
	if (graph.symbolic_ops.contains(phi_id)) {
		const drake::symbolic::Variables free_vars =
			graph.symbolic_ops.at(phi_id).formula.GetFreeVariables();
		const std::vector<int> owners = graph._agent_q.KeysReferencedBy(free_vars);
		return std::set<int>(owners.begin(), owners.end());
	}
	return {};
}

// Edge counterpart of PhiOwningAgents, for resolving a node's ownership
// through an incident edge constraint when the node has no owning phi of
// its own (see EdgePhiOwningAgents' caller in assign_node below): a node
// can constrain only its object at the node itself, with the agent's
// position there pinned entirely by an edge constraint to a neighboring
// node (add_edge_constraint) -- that's real, specific ownership, not "no
// opinion".
//
// Node phi ids and edge phi ids are separate counters (num_phis vs.
// num_edge_phis, see graph_of_constraints.hpp), so an edge phi id can't be
// looked up in the node-indexed `assignments` vector `PhiOwningAgents`
// takes -- an assignable edge constraint (edge_phi_to_variable_map) would
// need its own resolved-assignment vector, which get_agent_paths doesn't
// currently receive. This only covers constraints actually routed through
// `_add_edge_op` (add_edge_constraint and friends); the canonical hold
// registry (add_hold/add_assignable_hold) bypasses that machinery entirely
// -- see HoldOwningAgents below for its ownership resolution.
std::set<int> EdgePhiOwningAgents(const GraphOfConstraints& graph, int edge_phi_id) {
	if (graph._edge_phi_to_static_assignment_map.contains(edge_phi_id)) {
		return {graph._edge_phi_to_static_assignment_map.at(edge_phi_id)};
	}
	if (graph.symbolic_edge_ops.contains(edge_phi_id)) {
		const drake::symbolic::Variables free_vars =
			graph.symbolic_edge_ops.at(edge_phi_id).formula.GetFreeVariables();
		// An edge formula references either the plain agent_q placeholders
		// (an "along the edge" invariant, SymbolicEdgeConstraint::along_edge
		// == true) or the relational u_agent_q/v_agent_q pair (a single
		// u<->v relation, along_edge == false) -- checking all three
		// families is simpler than branching on along_edge and correct
		// either way, since a given formula only ever references one of
		// these three (disjoint) placeholder sets.
		std::set<int> owners;
		for (int ag : graph._agent_q.KeysReferencedBy(free_vars)) owners.insert(ag);
		for (int ag : graph._agent_q_u.KeysReferencedBy(free_vars)) owners.insert(ag);
		for (int ag : graph._agent_q_v.KeysReferencedBy(free_vars)) owners.insert(ag);
		return owners;
	}
	return {};
}

// Hold-registry counterpart of EdgePhiOwningAgents: resolves a node's
// ownership through a rigid-carry hold (add_hold/add_assignable_hold)
// along the (u, v) edge, the ONLY ownership evidence a node like Place has
// by design -- its own node constraint pins just the held object (see
// pyrobosim_gymnasium's _place_add), leaving the robot's position there to
// come entirely from the transport edge back to Pick (_place_edge_add).
// HoldDeclaration is pure bookkeeping (hold_ops), never routed through
// _add_edge_op, so it has no phi_id and EdgePhiOwningAgents never sees it
// -- checked here as a separate tier instead.
//
// A statically-assigned hold (add_hold, robot_ag set) resolves
// immediately. An assignable hold (add_assignable_hold, var_id set)
// resolves through `var_to_agent` -- THIS solve's own variable assignment
// (see get_agent_paths, which builds it fresh from assignments/
// phi_to_variable_map every call), not `committed_assignments` (which only
// gets an entry once the hold's u_node -- Pick -- has actually COMPLETED
// at runtime; see add_variable_commit/GraphOfConstraintsMPC._maybe_commit).
// get_agent_paths runs every cycle to route the WHOLE remaining graph
// (_solve_for_timing), well before a not-yet-reached Pick ever completes --
// waiting on commit left every such Place node's ownership undecided until
// then, which fell through to the "assign to every agent" branch below:
// the OTHER agent's totally unconstrained column at that node got fed to
// the timing solver as a real waypoint, plus a spurious cross-agent EQUAL
// sync constraint got attached to a node the other agent has no actual
// business visiting (confirmed on po_goc_mpc's pick_place_task: stray
// waypoint circles at room corners nowhere near any surface, and the
// second robot deadlocked in place, never reaching its own Pick). The
// solver's own `assignments` already resolves every declared variable as
// soon as a solve succeeds -- no need to wait for the physical Pick.
// Returns {} if `var_to_agent` has no entry for the hold's var (defensive;
// same "no opinion yet" contract as EdgePhiOwningAgents/PhiOwningAgents).
std::set<int> HoldOwningAgents(const GraphOfConstraints& graph, int u, int v,
                               const std::map<int, int>& var_to_agent) {
	std::set<int> owners;
	for (const auto& [hold_id, hold] : graph.hold_ops) {
		if (hold.u_node != u || hold.v_node != v) continue;
		if (hold.robot_ag.has_value()) {
			owners.insert(*hold.robot_ag);
			continue;
		}
		if (!hold.var_id.has_value()) continue;  // defensive; HoldDeclaration always sets one
		const auto it = var_to_agent.find(*hold.var_id);
		if (it != var_to_agent.end()) owners.insert(it->second);
	}
	return owners;
}

// Flattens `f`'s top-level And-conjunction into its non-And leaf atoms
// (Eq/Leq/Geq/Lt/Gt/a bare bool-valued formula), appending them to *out* in
// encounter order. Recurses through nested And (And-of-And, however
// constructed) so every leaf ends up at the top level of *out* regardless of
// how deeply the original conjunction was nested.
void FlattenConjunction(const drake::symbolic::Formula& f,
                        std::vector<drake::symbolic::Formula>* out) {
	using drake::symbolic::FormulaKind;
	if (f.get_kind() == FormulaKind::And) {
		for (const auto& sub : drake::symbolic::get_operands(f)) {
			FlattenConjunction(sub, out);
		}
		return;
	}
	out->push_back(f);
}

// Which side of an edge formula (or neither, for a plain node formula) a
// resolved agent_id was found on -- keys BlockResidualGroup grouping so a
// u-side and v-side reference to the same (agent_id, block_index) never
// merge into one group.
enum class PlaceholderSide { kPlain, kU, kV };

struct ResolvedColumn {
	PlaceholderSide side;
	int agent_id;
	int column;  // offset within that agent's dim-wide row
};

// Resolves `var` against graph._agent_q/_agent_q_u/_agent_q_v ONLY, not
// _var_agent_q(_u/_v) -- an assignable hold/constraint's actual agent isn't
// known until MILP resolves it (or, for the evolutionary solver, until
// solve time), so a formula built from a var_agent_q placeholder can't be
// grouped by (agent_id, block_index) at add_constraint time. Same
// limitation _resolve_holds (evolutionary_waypoint_solver/spec.py) already
// has for assignable holds -- see PopulateBlockResidualGroups' own doc
// comment (graph_of_constraints.hpp). Returns std::nullopt if `var` isn't a
// component of any currently-created placeholder row in any of the three
// families (e.g. an object_q variable, or simply not one of these
// placeholders at all).
std::optional<ResolvedColumn> ResolveVariable(const GraphOfConstraints& graph,
                                              const drake::symbolic::Variable& var) {
	struct FamilyBinding {
		const PlaceholderVarFamily<int>* family;
		PlaceholderSide side;
	};
	const FamilyBinding families[] = {
		{&graph._agent_q,   PlaceholderSide::kPlain},
		{&graph._agent_q_u, PlaceholderSide::kU},
		{&graph._agent_q_v, PlaceholderSide::kV},
	};
	for (const auto& binding : families) {
		for (int agent_id = 0; agent_id < graph.num_agents; ++agent_id) {
			if (!binding.family->Contains(agent_id)) continue;
			const auto& row = binding.family->Vars(agent_id);
			for (int col = 0; col < row.size(); ++col) {
				if (row(col).equal_to(var)) {
					return ResolvedColumn{binding.side, agent_id, col};
				}
			}
		}
	}
	return std::nullopt;
}

}  // namespace

void PopulateBlockResidualGroups(
	const GraphOfConstraints& graph, const drake::symbolic::Formula& f,
	std::vector<BlockResidualGroup>* groups,
	std::vector<drake::symbolic::Formula>* ungrouped) {

	using drake::symbolic::Expression;
	using drake::symbolic::FormulaKind;

	std::vector<drake::symbolic::Formula> leaves;
	FlattenConjunction(f, &leaves);

	struct InProgressGroup {
		BlockResidualGroup group;
		std::vector<bool> filled;
		std::vector<int> leaf_indices;  // which `leaves` entries fed this group
	};
	std::map<std::tuple<int, int, int>, InProgressGroup> in_progress;
	std::vector<bool> leaf_grouped(leaves.size(), false);

	for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
		const drake::symbolic::Formula& leaf = leaves[i];
		if (leaf.get_kind() != FormulaKind::Eq) continue;

		const Expression lhs = get_lhs_expression(leaf);
		const Expression rhs = get_rhs_expression(leaf);

		// Whichever side (if either) is a single Variable -- the other side
		// is whatever pins that ambient component's value; only the
		// manifold-block SIDE needs to resolve to a placeholder.
		std::optional<drake::symbolic::Variable> candidate;
		if (is_variable(lhs)) candidate = get_variable(lhs);
		else if (is_variable(rhs)) candidate = get_variable(rhs);
		if (!candidate.has_value()) continue;

		const auto resolved = ResolveVariable(graph, *candidate);
		if (!resolved.has_value()) continue;

		const auto& spec = graph._robot_specs.at(resolved->agent_id);
		int block_index = -1, component = -1, block_offset = 0;
		for (int b = 0; b < static_cast<int>(spec.size()); ++b) {
			if (resolved->column < block_offset + spec[b].size) {
				block_index = b;
				component = resolved->column - block_offset;
				break;
			}
			block_offset += spec[b].size;
		}
		if (block_index < 0) continue;  // defensive; shouldn't happen
		const auto& block = spec[block_index];
		// R blocks are already correct under raw per-component subtraction
		// -- only Torus (wrap-around) and SO3Quat/SO3Mat (rotation) need the
		// whole-block treatment BlockPositionDelta gives them.
		if (block.type == CubicConfigurationSpline::Block::Type::R) continue;

		const auto key = std::make_tuple(static_cast<int>(resolved->side),
		                                 resolved->agent_id, block_index);
		auto it = in_progress.find(key);
		if (it == in_progress.end()) {
			InProgressGroup ip;
			ip.group.type = block.type;
			ip.group.agent_id = resolved->agent_id;
			ip.group.block_index = block_index;
			ip.group.components.resize(block.size);
			ip.filled.assign(block.size, false);
			it = in_progress.emplace(key, std::move(ip)).first;
		}
		auto& ip = it->second;
		if (component < 0 || component >= static_cast<int>(ip.filled.size()) ||
		    ip.filled[component]) {
			continue;  // out-of-range or duplicate pin -- leave this leaf ungrouped
		}
		ip.group.components[component] = std::make_pair(lhs, rhs);
		ip.filled[component] = true;
		ip.leaf_indices.push_back(i);
	}

	for (auto& [key, ip] : in_progress) {
		const bool complete = std::all_of(ip.filled.begin(), ip.filled.end(),
		                                  [](bool b) { return b; });
		if (!complete) continue;  // its leaves stay in *ungrouped, added below
		for (int idx : ip.leaf_indices) leaf_grouped[idx] = true;
		groups->push_back(std::move(ip.group));
	}

	for (int i = 0; i < static_cast<int>(leaves.size()); ++i) {
		if (!leaf_grouped[i]) ungrouped->push_back(leaves[i]);
	}
}

std::tuple<std::vector<std::optional<int>>,
	   std::vector<std::vector<int>>,
	   std::vector<struct AgentInteraction>> GraphOfConstraints::get_agent_paths(
		   const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXi& assignments,
		   const Eigen::VectorXd& t_by_node) const {
	const InducedSubgraphView<py::object> sg = InducedSubgraphView<py::object>(
		structure, remaining_vertices);

	// Unrestricted view (every node, not just remaining_vertices), used
	// ONLY for assign_node's edge-based ownership fallback below -- see
	// that fallback's own comment for why this needs to see edges to
	// already-completed neighbors too, unlike the BFS traversal/
	// interaction-detection logic further down, which stays scoped to
	// `sg` (only remaining, not-yet-completed nodes matter for ordering
	// and cross-agent synchronization going forward).
	std::vector<int> all_nodes(structure.num_nodes());
	std::iota(all_nodes.begin(), all_nodes.end(), 0);
	const InducedSubgraphView<py::object> full_sg = InducedSubgraphView<py::object>(
		structure, all_nodes);

	std::vector<std::vector<int>> agent_nodes(num_agents);
	std::map<int, std::set<int>> node_to_agents_map;
	std::vector<struct AgentInteraction> agent_interactions;

	// var_id -> its resolved agent, from THIS solve's own `assignments`
	// (indexed by phi_id -- see PhiOwningAgents) via phi_to_variable_map --
	// built once here, rather than re-derived per hold lookup, since
	// several phis can share one var_id (e.g. Pick's x/y/yaw equalities are
	// 3 separate phis all tied to the same Robot variable) and they always
	// agree once assigned. Feeds HoldOwningAgents below; see its own
	// docstring for why this replaces committed_assignments as that
	// function's source of truth.
	std::map<int, int> var_to_agent;
	for (const auto& [phi_id, var_id] : phi_to_variable_map) {
		const int a = assignments(phi_id);
		if (a != -1) var_to_agent[var_id] = a;
	}

	// Cross-agent-ness for an edge (or co-ownership of a single node) is
	// "no agent is common to both owner sets" -- not "some pair of agents
	// differs" (that's true of almost any two multi-owner sets and would
	// spuriously flag an edge two agents both legitimately continue
	// through). A LESS_THAN/EQUAL interaction is only needed for agent
	// pairs that don't already share continuity through their own,
	// already-tracked per-agent node list.
	auto add_interactions = [&](const std::set<int>& owners_u, const std::set<int>& owners_v,
	                            int node_u, int node_v, AgentInteraction::Type type) {
		for (int ag_i : owners_u) {
			for (int ag_j : owners_v) {
				if (type == AgentInteraction::Type::LESS_THAN && ag_i == ag_j) continue;
				if (type == AgentInteraction::Type::EQUAL && ag_j <= ag_i) continue;
				agent_interactions.emplace_back(ag_i, -1, ag_j, -1, node_u, node_v, type);
			}
		}
	};

	auto assign_node = [&](int node) -> const std::set<int>& {
		auto it = node_to_agents_map.find(node);
		if (it != node_to_agents_map.end()) return it->second;

		std::set<int> owners;
		if (node_to_phis_map.contains(node)) {
			for (int phi_id : node_to_phis_map.at(node)) {
				const std::set<int> phi_owners = PhiOwningAgents(*this, phi_id, assignments);
				owners.insert(phi_owners.begin(), phi_owners.end());
			}
		}
		// TODO: I think this shouldn't be applied ONLY when owners is
		// empty. This may need to be on an agent's path due to an edge
		// constraint even when some node constraint forces it to be
		// on another agent's path
		if (owners.empty()) {
			// No NODE phi reveals agent-specific ownership -- check this
			// node's incident edges before giving up: an agent's position
			// at this node may be pinned entirely by an edge constraint to
			// a neighboring node (add_edge_constraint) rather than by
			// anything at the node itself, which is real, specific
			// ownership through the edge, not "no opinion". E.g. Place's
			// own node constraint is object-only by design (the robot's
			// position there comes purely from the transport edge back to
			// Pick -- see pyrobosim_gymnasium's _place_add/_place_edge_add),
			// so Place's only ownership evidence lives on that edge.
			//
			// Uses `full_sg` (every node), NOT `sg` (remaining_vertices
			// only): node completion in this codebase is one-directional --
			// the only way a node re-enters `remaining_vertices` is
			// backtracking, which explicitly re-adds it (goc_mpc.py's
			// `_backtrack`), so a genuinely no-longer-relevant edge would
			// already show up in `sg` again by the time it matters. Using
			// `sg` here instead made a node's edge-based ownership evidence
			// vanish the moment its predecessor completed and dropped out
			// of `remaining_vertices` -- e.g. Place losing sight of the
			// Pick->Place transport edge that's its ONLY ownership evidence
			// right after Pick completes -- which fell through to the
			// "genuinely agent-agnostic" branch below and spuriously
			// assigned Place to every agent, injecting a bogus cross-agent
			// EQUAL timing-sync constraint between agents that were never
			// meant to interact (confirmed: this is what was producing
			// GraphTimingMPC's intermittent Ipopt "Error in step
			// computation" failures on po_goc_mpc's pick_place_task
			// experiment -- the spurious EQUAL constraint tied one agent's
			// well-scaled remaining-path costs to another agent's
			// already-arrived, near-zero ones in the same shared QP).
			for (const auto& e : full_sg.neighbors(node)) {
				const std::set<int> hold_owners = HoldOwningAgents(*this, node, e.to, var_to_agent);
				owners.insert(hold_owners.begin(), hold_owners.end());
				const auto phis_it = edge_to_phis_map.find({node, e.to});
				if (phis_it == edge_to_phis_map.end()) continue;
				for (int edge_phi_id : phis_it->second) {
					const std::set<int> edge_owners = EdgePhiOwningAgents(*this, edge_phi_id);
					owners.insert(edge_owners.begin(), edge_owners.end());
				}
			}
			for (const auto& in : full_sg.incoming_neighbors(node)) {
				const std::set<int> hold_owners = HoldOwningAgents(*this, in.from, node, var_to_agent);
				owners.insert(hold_owners.begin(), hold_owners.end());
				const auto phis_it = edge_to_phis_map.find({in.from, node});
				if (phis_it == edge_to_phis_map.end()) continue;
				for (int edge_phi_id : phis_it->second) {
					const std::set<int> edge_owners = EdgePhiOwningAgents(*this, edge_phi_id);
					owners.insert(edge_owners.begin(), edge_owners.end());
				}
			}
		}
		if (owners.empty()) {
			// Still nothing, even from incident edges -- a genuinely
			// agent-agnostic node: fall back to every agent, same as when
			// this function had no per-phi resolution at all.
			for (int ag = 0; ag < num_agents; ++ag) owners.insert(ag);
		}

		for (int ag : owners) agent_nodes[ag].push_back(node);

		// A node co-owned by >1 agent is a real synchronization point
		// (e.g. a handoff node pinning both agents at once): both must
		// reach it at the same time.
		if (owners.size() > 1) add_interactions(owners, owners, node, node, AgentInteraction::Type::EQUAL);

		return node_to_agents_map.emplace(node, std::move(owners)).first->second;
	};

	std::vector<std::optional<int>> parents = sg.bfs_visit_from_sources(
		[&](int node, int /*depth*/, std::optional<int> parent) {
			const std::set<int>& owners_v = assign_node(node);
			if (parent.has_value()) {
				const std::set<int>& owners_u = assign_node(*parent);
				add_interactions(owners_u, owners_v, *parent, node, AgentInteraction::Type::LESS_THAN);
			}
		},
		[&](int u, int /*u_depth*/, int v, int /*v_depth*/) {
			// Both endpoints have necessarily already been discovered (and
			// so already own-resolved) by the time a non-tree edge is seen.
			const std::set<int>& owners_u = assign_node(u);
			const std::set<int>& owners_v = assign_node(v);
			add_interactions(owners_u, owners_v, u, v, AgentInteraction::Type::LESS_THAN);
		});

	// Order each agent's own nodes by the waypoint MPC's resolved arrival
	// time when available; otherwise keep BFS/topological discovery order
	// (e.g. before any waypoint solve has produced timings yet).
	if (t_by_node.size() > 0) {
		for (auto& nodes : agent_nodes) {
			std::stable_sort(nodes.begin(), nodes.end(), [&](int a, int b) {
				return t_by_node(a) < t_by_node(b);
			});
		}
	}

	// Depths are resolved last, against each agent's FINAL (sorted) node
	// list, rather than at discovery time -- discovery order need not
	// match t_by_node order. get_agent_paths' own agent_nodes lists are
	// themselves a (trivial) "dense" node-id sequence -- one real id per
	// row, nothing synthetic -- so this is the same lookup a caller doing
	// its own dense expansion needs, just reused here instead of
	// duplicated.
	agent_interactions = GraphOfConstraints::reindex_agent_interactions(
		std::move(agent_interactions), agent_nodes);

	return std::make_tuple(parents, agent_nodes, agent_interactions);
}

std::vector<AgentInteraction> GraphOfConstraints::reindex_agent_interactions(
		std::vector<AgentInteraction> agent_interactions,
		const std::vector<std::vector<int>>& agent_node_ids) {
	for (auto& intr : agent_interactions) {
		const auto& ni = agent_node_ids.at(intr.agent_i);
		auto it = std::find(ni.begin(), ni.end(), intr.node_u);
		if (it != ni.end()) intr.agent_i_depth = static_cast<int>(std::distance(ni.begin(), it));
		const auto& nj = agent_node_ids.at(intr.agent_j);
		auto jt = std::find(nj.begin(), nj.end(), intr.node_v);
		if (jt != nj.end()) intr.agent_j_depth = static_cast<int>(std::distance(nj.begin(), jt));
	}
	return agent_interactions;
}

std::map<std::pair<int, int>, int> GraphOfConstraints::get_next_edge_phis(const std::vector<int> remaining_vertices) const {
	std::map<std::pair<int, int>, int> e_to_phi_map;

	for (const auto& e : structure.incoming_cut_edges(remaining_vertices)) {
		if (this->edge_to_phis_map.contains(e)) {
			for (int edge_phi_id : this->edge_to_phis_map.at(e)) {
				e_to_phi_map[e] = edge_phi_id;
			}
		}
	}

	return e_to_phi_map;
}

std::map<int, HoldDeclaration> GraphOfConstraints::get_current_holds(const std::vector<int>& remaining_vertices) const {
	std::set<int> remaining(remaining_vertices.begin(), remaining_vertices.end());

	std::map<int, HoldDeclaration> current;
	for (const auto& [hold_id, hold] : hold_ops) {
		if (!remaining.contains(hold.u_node) && remaining.contains(hold.v_node)) {
			current[hold_id] = hold;
		}
	}

	return current;
}

std::vector<int> GraphOfConstraints::get_phi_ids(int node) const {
	// TODO: Maybe expand if nodes in the future support multiple phi ids (probably will).
	if (node_to_phis_map.contains(node)) {
		return node_to_phis_map.at(node);
	}
	return std::vector<int>();
}

bool GraphOfConstraints::evaluate_phi(int phi_id,
                                      const Eigen::VectorXd& x,
                                      const Eigen::VectorXi& assignments,
                                      double tol) const {
	if (ops.contains(phi_id)) {
		const DeferredOp& op = ops.at(phi_id);
		double v = op.eval(x, assignments(phi_id));
		std::cout << "violation: " << v << std::endl;
		return v < tol;
	} else if (symbolic_ops.contains(phi_id)) {
		double v = EvaluateSymbolicNodeConstraint(*this, symbolic_ops.at(phi_id), x, assignments(phi_id));
		return v < tol;
	}
	return true;
}

bool GraphOfConstraints::evaluate_edge_phi(int phi_id,
					   const Eigen::VectorXd& x,
					   const Eigen::VectorXi& var_assignments,
					   double tol) const {
	if (edge_ops.contains(phi_id)) {
		const DeferredEdgeOp& op = edge_ops.at(phi_id);
		double v = op.eval(x, var_assignments);
		return v < tol;
	} else if (symbolic_edge_ops.contains(phi_id)) {
		double v = EvaluateSymbolicEdgeConstraint(*this, symbolic_edge_ops.at(phi_id), x, var_assignments);
		return v < tol;
	}
	return true;
}

int GraphOfConstraints::get_edge_phi_agent(int phi_id, const Eigen::VectorXi& var_assignments) const {
	if (_edge_phi_to_static_assignment_map.contains(phi_id)) {
		return _edge_phi_to_static_assignment_map.at(phi_id);
	} else if (edge_phi_to_variable_map.contains(phi_id)) {
		int var = edge_phi_to_variable_map.at(phi_id);
		return var_assignments(var);
	}
	return -1;
}

void GraphOfConstraints::add_backtrack_links(int edge_id, std::vector<int> backtrack_nodes) {
	// If edge_id isn't in the map yet, [] automatically creates an empty vector for it.
	auto& existing_nodes = backtrack_map[edge_id];
	std::vector<int> new_nodes{};

	for (int node : backtrack_nodes) {
		auto descendants = structure.dfs(node);
		new_nodes.insert(new_nodes.end(), descendants.begin(), descendants.end());
	}

	existing_nodes.insert(existing_nodes.end(), new_nodes.begin(), new_nodes.end());
	std::sort(existing_nodes.begin(), existing_nodes.end());
	existing_nodes.erase(std::unique(existing_nodes.begin(), existing_nodes.end()), existing_nodes.end());
}

void GraphOfConstraints::add_manual_backtrack_links(int edge_id, std::vector<int> backtrack_nodes) {
	// If edge_id isn't in the map yet, [] automatically creates an empty vector for it.
	auto& existing_nodes = backtrack_map[edge_id];
	existing_nodes.insert(existing_nodes.end(), backtrack_nodes.begin(), backtrack_nodes.end());
	std::sort(existing_nodes.begin(), existing_nodes.end());
	existing_nodes.erase(std::unique(existing_nodes.begin(), existing_nodes.end()), existing_nodes.end());
}

void GraphOfConstraints::add_variable_commit(int var, int node) {
	commit_trigger_node_to_var[node] = var;
}

void GraphOfConstraints::commit_variable_assignment(int var, int agent) {
	committed_assignments[var] = agent;
}

void GraphOfConstraints::clear_variable_commitment(int var) {
	committed_assignments.erase(var);
}

std::optional<int> GraphOfConstraints::get_commit_trigger_var(int node) const {
	auto it = commit_trigger_node_to_var.find(node);
	if (it == commit_trigger_node_to_var.end()) return std::nullopt;
	return it->second;
}

// Grasp util

void GraphOfConstraints::add_grasp_change(int phi_id,
					  std::string command,
					  int robot_id,
					  int cube_id) {
	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;


	std::string robot_model_name = _robot_names.at(robot_id);
	std::string cube_model_name = _object_names.at(cube_id);
	_grasp_change_map[phi_id].emplace_back(command, robot_model_name, cube_model_name);
}


// this should be used on an existing assignable phi
void GraphOfConstraints::add_assignable_grasp_change(int phi_id,
						     std::string command,
						     int cube_id) {
	std::string cube_model_name = _object_names.at(cube_id);
	_assignable_grasp_change_map[phi_id].emplace_back(command, cube_model_name);
}

std::vector<std::tuple<std::string, std::string, std::string>> GraphOfConstraints::get_grasp_changes(int k, Eigen::VectorXi assignments) const {
	std::vector<std::tuple<std::string, std::string, std::string>> changes;

	for (int phi_id : get_phi_ids(k)) {
		if (_grasp_change_map.contains(phi_id)) {
			for (const auto& change : _grasp_change_map.at(phi_id)) {
				changes.push_back(change);
			}
		}

		if (_assignable_grasp_change_map.contains(phi_id)) {
			// if an assignable grasp change was added to this phi,
			// it should be assigned at this point.
			int robot_id = assignments(phi_id);
			if (robot_id == -1) {
				throw std::runtime_error(fmt::format("Somehow constraint {} at node {} was not assigned.", phi_id, k));
			} else {
				const std::string& robot_model_name = _robot_names.at(robot_id);
				for (const auto& assignable_change : _assignable_grasp_change_map.at(phi_id)) {
					const std::string& command = assignable_change.first;
					const std::string& cube_model_name = assignable_change.second;
					changes.push_back(std::make_tuple(command, robot_model_name, cube_model_name));
				}
			}
		}
	}

	return changes;
}

void GraphOfConstraints::make_node_unpassable(int k) {
	unpassable_nodes.insert(k);
}

int GraphOfConstraints::add_node(std::optional<std::string> name) {
	const int id = structure.add_node();
	if (name.has_value()) node_names[id] = std::move(name.value());
	return id;
}

std::vector<int> GraphOfConstraints::add_nodes(int n, std::optional<std::vector<std::string>> names) {
	std::vector<int> ids = structure.add_nodes(n);
	if (names.has_value()) {
		if (names->size() != ids.size())
			throw std::runtime_error("add_nodes: names must be the same size as n.");
		for (size_t i = 0; i < ids.size(); ++i) node_names[ids[i]] = (*names)[i];
	}
	return ids;
}

void GraphOfConstraints::set_node_name(int k, const std::string& name) {
	node_names[k] = name;
}

void GraphOfConstraints::set_node_names(const std::vector<int>& ks, const std::vector<std::string>& names) {
	if (ks.size() != names.size())
		throw std::runtime_error("set_node_names: ks and names must be the same size.");
	for (size_t i = 0; i < ks.size(); ++i) node_names[ks[i]] = names[i];
}

std::string GraphOfConstraints::get_node_name(int k) const {
	auto it = node_names.find(k);
	return it != node_names.end() ? it->second : std::to_string(k);
}

///////////////////////////////////////////////////////////////////////////////
//                                   HOLDS                                   //
///////////////////////////////////////////////////////////////////////////////

int GraphOfConstraints::add_hold(int u, int v, int robot_ag, std::vector<int> held_point_ids) {
	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(robot_ag >= 0 && robot_ag < num_agents);

	const int id = num_holds++;
	hold_ops[id] = HoldDeclaration{id, u, v, held_point_ids, robot_ag, std::nullopt};
	return id;
}

int GraphOfConstraints::add_assignable_hold(int u, int v, int var, std::vector<int> held_point_ids) {
	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);

	const int id = num_holds++;
	hold_ops[id] = HoldDeclaration{id, u, v, held_point_ids, std::nullopt, var};

	// Once `u` (pick-up) completes, `var` is physically committed -- the
	// routing solve must not reassign the hold's holder mid-grasp. Reuses
	// the same commit mechanism add_variable_commit exposes to plans
	// directly (see commit_trigger_node_to_var/committed_assignments and
	// Constraint 8b in milp_waypoint_mpc.cpp), so every assignable hold gets
	// this protection automatically rather than needing every call site to
	// declare it by hand.
	add_variable_commit(var, u);

	return id;
}

///////////////////////////////////////////////////////////////////////////////
//                         TIMING (EDGE) CONSTRAINTS                         //
///////////////////////////////////////////////////////////////////////////////

void GraphOfConstraints::add_edge_min_tau_constraint(int u,
						     int v,
						     double minimum_time_delta) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(minimum_time_delta >= 0);

	edge_to_min_tau_map[std::make_pair(u, v)] = minimum_time_delta;
}

///////////////////////////////////////////////////////////////////////////////
//                            VARIABLE CONSTRAINTS                           //
///////////////////////////////////////////////////////////////////////////////

int GraphOfConstraints::add_variable_constraint(
	int var,
	std::set<int> robot_ids) {

	DRAKE_DEMAND(var >= 0 && var < num_variables);
	for (int robot_id : robot_ids) {
		DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	}

 	return _add_var_op(
		DeferredOpKind::kLinearEq,
		[=, this](auto& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int phi_id,
			  const auto& X,
			  const auto & Assignments) {

			// Get the variable row for this variable.
			const int variable_k = subgraph.subgraph_variable_id(var);
			if (variable_k < 0) return;

			// For every robot (i) we want to constrain the Assignments to
			// something like [0, 0, 0, 1, 1, 0, 1] where the 1 entries are the
			// robots that are allowed
			for (int i = 0; i < num_agents; i++) {
				// If i is not in robot_ids (not allowed)
				if (robot_ids.find(i) == robot_ids.end()) {
					// Make sure it is constrained to NOT be assigned
					const auto s = Assignments(variable_k, i);
					prog.AddLinearEqualityConstraint(s, 0);
				}
			}
		});
}

int GraphOfConstraints::add_variable_ineq_constraint(
	int var1,
	int var2) {

	DRAKE_DEMAND(var1 >= 0 && var1 < num_variables);
	DRAKE_DEMAND(var2 >= 0 && var2 < num_variables);

 	return _add_var_op(
		DeferredOpKind::kLinearIneq,
		[=, this](auto& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int phi_id,
			  const auto& X,
			  const auto& Assignments) {

			const int variable1_k = subgraph.subgraph_variable_id(var1);
			const int variable2_k = subgraph.subgraph_variable_id(var2);

			if (variable1_k >= 0 && variable2_k >= 0) {
				for (int i = 0; i < num_agents; i++) {
					const auto s = Assignments(variable1_k, i) + Assignments(variable2_k, i);
					// 1 <= binary v1 + binary v2 <= 1 implies both
					// cannot be zero and both cannot be one.
					prog.AddLinearConstraint(s, 1, 1);
				}
			}
		});
}

// Symbolic unified constraint API

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::agent_q(int agent_id) const {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	return _agent_q.Get(agent_id);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::var_agent_q(int var) {
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	return _var_agent_q.Get(var);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::object_q(int object_id) const {
	DRAKE_DEMAND(object_id >= 0 && object_id < num_objects);
	return _object_q.Get(object_id);
}

int GraphOfConstraints::add_param(double initial_value) {
	const int id = static_cast<int>(_param_values.size());
	_param_values.conservativeResize(id + 1);
	_param_values(id) = initial_value;
	_param.Vars(id);  // eager creation -- mirrors add_binary_cond_var
	return id;
}

drake::symbolic::Expression GraphOfConstraints::param(int id) const {
	DRAKE_DEMAND(id >= 0 && id < _param_values.size());
	return _param.Get(id)[0];
}

void GraphOfConstraints::set_param(int id, double value) {
	DRAKE_DEMAND(id >= 0 && id < _param_values.size());
	_param_values(id) = value;
}

// For unified edge constraint API (relational form)

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::u_agent_q(int agent_id) const {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	return _agent_q_u.Get(agent_id);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::u_object_q(int object_id) const {
	DRAKE_DEMAND(object_id >= 0 && object_id < num_objects);
	return _object_q_u.Get(object_id);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::v_agent_q(int agent_id) const {
	DRAKE_DEMAND(agent_id >= 0 && agent_id < num_agents);
	return _agent_q_v.Get(agent_id);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::v_object_q(int object_id) const {
	DRAKE_DEMAND(object_id >= 0 && object_id < num_objects);
	return _object_q_v.Get(object_id);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::u_var_agent_q(int var) {
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	return _var_agent_q_u.Get(var);
}

drake::VectorX<drake::symbolic::Expression>
GraphOfConstraints::v_var_agent_q(int var) {
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	return _var_agent_q_v.Get(var);
}


int GraphOfConstraints::add_constraint(int node, const drake::symbolic::Formula& f) {
	// Detect variable-agent placeholders in the formula.
	const drake::symbolic::Variables free_vars = f.GetFreeVariables();
	const std::vector<int> involved_var_ids = _var_agent_q.KeysReferencedBy(free_vars);

	if (involved_var_ids.size() == 1) {
		return add_assignable_constraint(node, involved_var_ids[0], f);
	}

	if (involved_var_ids.size() > 1) {
		// Multi-variable disjunction: compiled at build time by enumerating
		// all n_agents^k combos and emitting Or (see CompileSymbolicNodeConstraint).
		return _add_symbolic_multi_var_op(node, involved_var_ids, f);
	}

	// No variable-agent placeholders — plain node constraint.
	return _add_symbolic_op(node, f);
}

int GraphOfConstraints::add_edge_constraint(int u, int v, const drake::symbolic::Formula& f, bool live) {
	const drake::symbolic::Variables free_vars = f.GetFreeVariables();

	// u_var_agent_q/v_var_agent_q are relational-only (see their docstring):
	// a formula referencing either is inherently a two-sided relation, same
	// as _agent_q_u/_v -- included here so the mixing DRAKE_DEMAND below
	// still catches an erroneous formula combining them with the plain
	// "along the edge" placeholder set. Note this doesn't extend
	// symbolic_constraint_compiler.cpp's substitution -- MILP can't compile
	// a relational formula referencing these regardless of how it's
	// classified here (see _var_agent_q_u/_v's docstring).
	const bool has_relational = _agent_q_u.ReferencesAny(free_vars) || _object_q_u.ReferencesAny(free_vars) ||
				    _agent_q_v.ReferencesAny(free_vars) || _object_q_v.ReferencesAny(free_vars) ||
				    _var_agent_q_u.ReferencesAny(free_vars) || _var_agent_q_v.ReferencesAny(free_vars);

	const std::vector<int> involved_var_ids = _var_agent_q.KeysReferencedBy(free_vars);
	// agent_link_pos/agent_link_rot have no u_/v_ relational counterpart
	// (see their doc comments -- tied to one fixed agent+link at authoring
	// time, not a two-sided relation), so a formula referencing either is
	// always classified as "along the edge", same as a plain
	// agent_q/object_q reference.
	const bool has_plain = _agent_q.ReferencesAny(free_vars) || _object_q.ReferencesAny(free_vars) ||
			       !involved_var_ids.empty() || _agent_link_pos.ReferencesAny(free_vars) ||
			       _agent_link_rot.ReferencesAny(free_vars);

	DRAKE_DEMAND(!(has_relational && has_plain));  // can't mix u_/v_ relational placeholders
							// with plain "along the edge" placeholders
							// in the same edge formula

	if (involved_var_ids.size() > 1) {
		throw std::runtime_error(
			"add_edge_constraint: an \"along the edge\" formula referencing more "
			"than one distinct var_agent_q(var) isn't supported");
	}

	if (involved_var_ids.size() == 1) {
		_num_total_assignables++;
		return _add_symbolic_edge_op(u, v, f, /*along_edge=*/true, live, involved_var_ids[0]);
	}

	return _add_symbolic_edge_op(u, v, f, /*along_edge=*/has_plain, live);
}

int GraphOfConstraints::add_assignable_constraint(
	int node, int var, const drake::symbolic::Formula& f) {

	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(_var_agent_q.Contains(var));

	_num_total_assignables++;
	return _add_symbolic_assignable_op(node, var, f);
}
