#include "graph_of_constraints.hpp"
#include "symbolic_constraint_compiler.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <numeric>


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

Eigen::Vector3d GraphOfConstraints::point_position(int point_id, const Eigen::VectorXd& x) const {
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);
	DRAKE_DEMAND(x.size() == total_dim);
	return CubePosFromRow(this, point_id, x);
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
// currently receive. No current constraint generator gives a node ITS
// ONLY ownership through an assignable edge constraint, so that priority
// tier is left unresolved here (falls through to {}) rather than guessing
// -- a real gap if that ever changes, not a silent miscompute today.
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

}  // namespace

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
				const auto phis_it = edge_to_phis_map.find({node, e.to});
				if (phis_it == edge_to_phis_map.end()) continue;
				for (int edge_phi_id : phis_it->second) {
					const std::set<int> edge_owners = EdgePhiOwningAgents(*this, edge_phi_id);
					owners.insert(edge_owners.begin(), edge_owners.end());
				}
			}
			for (const auto& in : full_sg.incoming_neighbors(node)) {
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

// Joint-Agent Constraint Adders (typed)

// lb <= x <= ub on node k
int GraphOfConstraints::add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	return _add_op(DeferredOpKind::kBoundingBox, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const unsigned int node_k = subgraph.subgraph_id(k);

			       VectorXDecisionVariable joint_config_k(num_agents * dim);
			       for (int ag = 0; ag < num_agents; ++ag) {
				       joint_config_k << X.row(node_k).segment(ag * dim, dim);;
			       }

			       prog.AddBoundingBoxConstraint(lb, ub, joint_config_k);
		       });
}

// Ax = b on node k
int GraphOfConstraints::add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	return _add_op(DeferredOpKind::kLinearEq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable config_k = X.row(node_k);
			       auto beq = prog.AddLinearEqualityConstraint(A, b, config_k);
		       });
}

// lb <= A x <= ub on node k
int GraphOfConstraints::add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	return _add_op(DeferredOpKind::kLinearIneq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable config_k = X.row(node_k);
			       auto constraint = prog.AddLinearConstraint(A, lb, ub, config_k);
		       });
}

// 0.5 x'Qx + b'x + c on node k
int GraphOfConstraints::add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c) {
	return _add_op(DeferredOpKind::kQuadraticCost, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable config_k = X.row(node_k);
			       auto constraint = prog.AddQuadraticCost(Q, b, c, config_k);
		       });
}


// Ax = b on node k
int GraphOfConstraints::add_robots_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	return _add_op(DeferredOpKind::kLinearEq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable agents_config_k = X.row(node_k).segment(0, num_agents * dim);
			       auto beq = prog.AddLinearEqualityConstraint(A, b, agents_config_k);
		       });
}

// lb <= A x <= ub on node k
int GraphOfConstraints::add_robots_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	return _add_op(DeferredOpKind::kLinearIneq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable agents_config_k = X.row(node_k).segment(0, num_agents * dim);
			       auto constraint = prog.AddLinearConstraint(A, lb, ub, agents_config_k);
		       });
}

// Ax = b on node k
int GraphOfConstraints::add_robot_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     return 0.0;
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto& Assignments) {

				     const int node_k = subgraph.subgraph_id(k);
				     VectorXDecisionVariable agent_config_k = X.row(node_k).segment(robot_id*dim, dim);
				     auto beq = prog.AddLinearEqualityConstraint(A, b, agent_config_k);
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

// lb <= A x <= ub on node k
int GraphOfConstraints::add_robot_linear_ineq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	int phi_id = _add_op(DeferredOpKind::kLinearIneq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       return 0.0;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto& Assignments) {
			       const int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable agent_config_k = X.row(node_k).segment(robot_id*dim, dim);
			       auto constraint = prog.AddLinearConstraint(A, lb, ub, agent_config_k);
		       });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_point_linear_eq(
	int k, int point_id,
	const Eigen::MatrixXd& A,
	const Eigen::VectorXd& b) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	// Expect A is (m x 3) if the point is 3D, and b is (m).
	DRAKE_DEMAND(A.cols() == 3);
	DRAKE_DEMAND(b.size() == A.rows());

	const int objs_start  = num_agents * dim;
	const int point_start = objs_start + point_id * non_robot_dim;

	int phi_id = _add_op(
		DeferredOpKind::kLinearEq, k,
		// ---- Evaluation: max absolute residual (0 means satisfied) ----
		[=, this](const Eigen::VectorXd& x, const int... /*unused*/) {
			const Eigen::Vector3d point_config_k = x.segment(point_start, 3);
			const Eigen::VectorXd r = A * point_config_k - b;  // residual
			return r.lpNorm<Eigen::Infinity>();                // max |residual|
		},
		// ---- Definition in Drake ----
		[=, this](auto& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int /*phi_id*/,
			  const auto& X,
			  const auto& /*unused*/) {
			const int node_k = subgraph.subgraph_id(k);
			VectorXDecisionVariable point_config_k =
				X.row(node_k).segment(point_start, 3);

			// Enforces A * point_config_k == b
			prog.AddLinearEqualityConstraint(A, b, point_config_k)
				.evaluator()->set_description(fmt::format("point {} linear constraint", point_id));
		});

	return phi_id;
}

// int GraphOfConstraints::add_point_linear_cost(
// 	int k, int point_id,
// 	const Eigen::MatrixXd& A,
// 	const Eigen::VectorXd& b) {

// 	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
// 	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

// 	// Expect A is (m x 3) if the point is 3D, and b is (m).
// 	DRAKE_DEMAND(A.cols() == 3);
// 	DRAKE_DEMAND(b.size() == A.rows());

// 	const int objs_start  = num_agents * dim;
// 	const int point_start = objs_start + point_id * non_robot_dim;

// 	int phi_id = _add_op(
// 		DeferredOpKind::kLinearEq, k,
// 		// ---- Evaluation: max absolute residual (0 means satisfied) ----
// 		[=, this](const Eigen::VectorXd& x, const int... /*unused*/) {
// 			const Eigen::Vector3d point_config_k = x.segment(point_start, 3);
// 			const Eigen::VectorXd r = A * point_config_k - b;  // residual
// 			return r.lpNorm<1>();                // max |residual|
// 		},
// 		// ---- Definition in Drake ----
// 		[=, this](auto& prog,
// 			  const SubgraphOfConstraints& subgraph,
// 			  const int /*phi_id*/,
// 			  const auto& X,
// 			  const auto& /*unused*/) {
// 			const int node_k = subgraph.subgraph_id(k);
// 			VectorXDecisionVariable point_config_k =
// 				X.row(node_k).segment(point_start, 3);

// 			// Enforces A * point_config_k == b
// 			prog.AddLinearEqualityConstraint(A, b, point_config_k)
// 				.evaluator()->set_description(fmt::v8::format("point {} linear constraint", point_id));
// 		});

// 	return phi_id;
// }

int GraphOfConstraints::add_point_linear_ineq(
	int k, int point_id,
	const Eigen::MatrixXd& A,
	const Eigen::VectorXd& lb,
	const Eigen::VectorXd& ub) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	// A is (m x 3), lb/ub are (m)
	DRAKE_DEMAND(A.cols() == 3);
	DRAKE_DEMAND(lb.size() == A.rows());
	DRAKE_DEMAND(ub.size() == A.rows());

	const int objs_start  = num_agents * dim;
	const int point_start = objs_start + point_id * non_robot_dim;

	int phi_id = _add_op(
		DeferredOpKind::kLinearIneq, k,
		// ---- Evaluation: returns max violation (0 if satisfied) ----
		[=, this](const Eigen::VectorXd& x, const int... /*unused*/) {
			const Eigen::Vector3d point_config_k = x.segment(point_start, 3);

			const Eigen::ArrayXd ax  = (A * point_config_k).array();
			const Eigen::ArrayXd v1  = (lb.array() - ax).max(0.0);   // lb - Ax > 0 ⇒ lower-bound violation
			const Eigen::ArrayXd v2  = (ax - ub.array()).max(0.0);   // Ax - ub > 0 ⇒ upper-bound violation
			const Eigen::ArrayXd vio = v1.max(v2);                   // per-row violation
			return vio.matrix().lpNorm<Eigen::Infinity>();           // max violation
		},
		// ---- Definition in Drake ----
		[=, this](auto& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int /*phi_id*/,
			  const auto& X,
			  const auto& /*unused*/) {
			const int node_k = subgraph.subgraph_id(k);
			VectorXDecisionVariable point_config_k =
				X.row(node_k).segment(point_start, 3);

			// Imposes lb ≤ A * point_config_k ≤ ub elementwise
			prog.AddLinearConstraint(A, lb, ub, point_config_k);
		});

	return phi_id;
}

int GraphOfConstraints::add_robot_pos_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(A.cols() == 3);
	DRAKE_DEMAND(b.size() == A.rows());

	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     return 0.0;
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto& Assignments) {
				     const int node_k = subgraph.subgraph_id(k);
				     Eigen::Matrix<Expression, Eigen::Dynamic, 1> row = X.row(node_k);
				     auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", row);
				     prog.AddLinearEqualityConstraint(A*b == p_WR);
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_robot_quat_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(A.cols() == 4);
	DRAKE_DEMAND(b.size() == A.rows());

	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     return 0.0;
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto& Assignments) {
				     const int node_k = subgraph.subgraph_id(k);
				     VectorXDecisionVariable agent_quat_k = X.row(node_k).segment(robot_id*dim + 3, 4);
				     prog.AddLinearEqualityConstraint(A, b, agent_quat_k)
					     .evaluator()->set_description(fmt::format("robot {} quaternion constraint", robot_id));
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_assignable_robot_quat_linear_eq(int k, int var, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(A.cols() == 4);
	DRAKE_DEMAND(b.size() == A.rows());

	// ----- Build per-row big-M from bounds on quaternion entries -----
	// Assume the quaternion block is contiguous: [robot_id*dim + 3 ... + 6]
	// We conservatively bound |a^T q - b| ≤ sum_t |a_t| * max(|lb_t|, |ub_t|) + |b|
	// Use the bounds from robot 0 (or take a max across robots if your bounds differ).
	Eigen::VectorXd M(A.rows());
	{
		const int base0 = /* robot 0 */ 0 * dim + 3;
		Eigen::Array4d max_abs_q;
		for (int t = 0; t < 4; ++t) {
			const double lb = _global_x_lb(base0 + t);
			const double ub = _global_x_ub(base0 + t);
			max_abs_q(t) = std::max(std::abs(lb), std::abs(ub));
		}
		for (int j = 0; j < A.rows(); ++j) {
			double row_bound = 0.0;
			for (int t = 0; t < 4; ++t) {
				row_bound += std::abs(A(j, t)) * max_abs_q(t);
			}
			M(j) = row_bound + std::abs(b(j));
			// Small safety inflation (optional):
			M(j) *= 1.01;
		}
	}

	_num_total_assignables++;

	return _add_assignable_op(DeferredOpKind::kAgentLinearEq, k, var,
				  [=, this](const Eigen::VectorXd& x,
					    const int robot_id) {
					  const int q_start = robot_id * dim + 3;
					  Eigen::Vector4d q = x.segment(q_start, 4);
					  /* TODO: FIX THIS CONSTRAINT SO THAT
					   * IT IS A PROPER ORIENTATION
					   * CONSTRAINT. FOR NOW I'M JUST
					   * ASSUMING A IS IDENTITY MATRIX AND B
					   * IS THE TARGET CONSTRAINT. */
					  Eigen::Vector4d target_q = b;
					  return 1 - std::abs(q.dot(target_q));
				  },
				  [=, this](auto& prog,
					    const SubgraphOfConstraints& subgraph,
					    const int phi_id,
					    const auto& X,
					    const auto& Assignments) {

					  const int node_k  = subgraph.subgraph_id(k);
					  const int variable_k = subgraph.subgraph_variable_id(var);
					  if (variable_k < 0) return;
					  const double neg_inf = -std::numeric_limits<double>::infinity();

					  for (int i = 0; i < num_agents; ++i) {
						  const auto s = Assignments(variable_k, i);        // binary 0/1
						  const int q_start = i * dim + 3;

						  // For each row j of A: e_j = A_j * q_i - b_j
						  for (int j = 0; j < A.rows(); ++j) {
							  Expression e = -b(j);
							  // Add linear combination of the 4 quaternion vars
							  for (int t = 0; t < 4; ++t) {
								  const int col = q_start + t;
								  if (A(j, t) != 0.0) {
									  e += A(j, t) * X(node_k, col);
								  }
							  }

							  // -M_j (1 - s) ≤ e ≤ M_j (1 - s)
							  // Upper:   e - M_j*(1 - s) ≤ 0  => e + M_j*s - M_j ≤ 0
							  prog.AddLinearConstraint(e + M(j) * s - M(j), neg_inf, 0.0);
							  // Lower:  -e - M_j*(1 - s) ≤ 0  => -e + M_j*s - M_j ≤ 0
							  prog.AddLinearConstraint(-e + M(j) * s - M(j), neg_inf, 0.0);
						  }
					  }
				  });
}

// Single-Agent Constraint Adders (typed)
// Note: these copy the numpy array's passed to them, but they're called
// once so it's fine.

// Compute max / min of c^T x over box lb<=x<=ub.
inline std::pair<double,double> max_min_ct_x_over_box(const Eigen::RowVectorXd& c,
						      const Eigen::VectorXd& lb,
						      const Eigen::VectorXd& ub) {
	DRAKE_DEMAND(c.size() == lb.size());
	double maxv = 0.0, minv = 0.0;
	for (int j = 0; j < c.size(); ++j) {
		// if c[j] is positive, maxv is maximized when x = up[j] and
		// minimized when = lb[j]. If negative, its the opposite.
		if (c[j] >= 0) { maxv += c[j] * ub[j]; minv += c[j] * lb[j]; }
		else           { maxv += c[j] * lb[j]; minv += c[j] * ub[j]; }
	}
	return {maxv, minv};
}

// A_i x_i = b on node k for some agent i
// Enforce: A * x_{k,i} = b for the unique agent i with A_(var,i) = 1.
// A.rows() == b.size(), A.cols() == d_
int GraphOfConstraints::add_assignable_linear_eq(int k,
						 int var,
						 const Eigen::MatrixXd& A,
						 const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(A.cols() == dim);
	DRAKE_DEMAND(b.size() == A.rows());

	// record an increase in the total number of assignables. (could be removed).
	_num_total_assignables++;

	return _add_assignable_op(DeferredOpKind::kAgentLinearEq, k, var,
				  [=, this](const Eigen::VectorXd& x,
					    const int robot_id) {
					  const int robot_start = robot_id * dim;
					  Eigen::VectorXd robot_q = x.segment(robot_start, dim);
					  return ((A * robot_q) - b).lpNorm<Eigen::Infinity>();
				  },
				  [=, this](auto& prog,
					    const SubgraphOfConstraints& subgraph,
					    const int phi_id,
					    const auto& X,
					    const auto& Assignments) {

					  const int node_k = subgraph.subgraph_id(k);
					  const int variable_k = subgraph.subgraph_variable_id(var);
					  if (variable_k < 0) return;

					  for (int i = 0; i < num_agents; ++i) {
						  // Variables [ x_{k,i} ; s ] with s = A(variable_k, i)
						  VectorXDecisionVariable vars(dim + 1);
						  for (int j = 0; j < dim; ++j) vars[j] = X(node_k, i*dim + j);
						  vars[dim] = Assignments(variable_k, i);

						  auto _agent_x_lb = _global_x_lb.segment(i*dim, dim);
						  auto _agent_x_ub = _global_x_ub.segment(i*dim, dim);

						  for (int r = 0; r < A.rows(); ++r) {
							  const Eigen::RowVectorXd c = A.row(r);
							  const auto [max_cx, min_cx] = max_min_ct_x_over_box(
								  c,
								  _agent_x_lb,
								  _agent_x_ub);

							  const double rhs = b[r];
							  // Pick M so that when s = 0 the constraint is loose:
							  const double M_up = std::max(0.0, max_cx - rhs);  // for c^T x <= rhs
							  const double M_lo = std::max(0.0, rhs - min_cx); // for c^T x >= rhs

							  // Encode using constant bounds (move M*(1-s) to LHS):
							  //  c^T x - M(1-s) <= rhs    ⇔  c^T x + M s <= rhs + M
							  // -c^T x - M(1-s) <= -rhs   ⇔ -c^T x + M s <= -rhs + M
							  Eigen::RowVectorXd a_up(dim + 1);
							  a_up.head(dim) = c;    a_up[dim] = M_up;
							  const double b_up = rhs + M_up;

							  Eigen::RowVectorXd a_lo(dim + 1);
							  a_lo.head(dim) = -c;   a_lo[dim] = M_lo;
							  const double b_lo = -rhs + M_lo;

							  const double ninf = -std::numeric_limits<double>::infinity();

							  auto upper = prog.AddLinearConstraint(a_up, ninf, b_up, vars);
							  auto lower = prog.AddLinearConstraint(a_lo, ninf, b_lo, vars);
						  }
					  }
				  });
}

int GraphOfConstraints::add_robot_above_cube_constraint(
	int k,
	int robot_id, // std::string robot_model_name,
	int cube_id, // std::string cube_model_name,
	double delta_z,
	double x_offset,
	double y_offset) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	// DRAKE_DEMAND(agent_i >= 0 && agent_i < num_agents);
	// DRAKE_DEMAND(cube_i >= 0 && cube_i < num_objects);
	// If you track num_objects, you can also check cube_i bounds here.

	int phi_id = _add_op(DeferredOpKind::kNonlinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {

				     auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", x);
				     auto p_WC = CubePosFromRow(this, cube_id, x);

				     Eigen::Vector3d g;
				     g << (p_WR(0) - p_WC(0) - x_offset),
					     (p_WR(1) - p_WC(1) - y_offset),
					     (p_WR(2) - p_WC(2) - delta_z);

				     double violation = 0.0;
				     for (int i = 0; i < 3; ++i) {
					     violation = std::max(violation, std::abs(g(i)));
				     }
				     return violation;
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto& Assignments) {

				     const int node_k = subgraph.subgraph_id(k);

				     // Convert X[row] decision variables to Expressions.
				     Eigen::VectorX<Expression> q_all(total_dim);
				     for (int j = 0; j < total_dim; ++j) {
					     q_all(j) = Expression(X(node_k, j));
				     }

				     auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", q_all);
				     auto p_WC = CubePosFromRow(this, cube_id, q_all);

				     Eigen::Vector3<Expression> g;
				     g << (p_WR(0) - p_WC(0) - x_offset),
					     (p_WR(1) - p_WC(1) - y_offset),
					     (p_WR(2) - p_WC(2) - delta_z);

				     prog.AddConstraint(g, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
			     }
		);

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}


int GraphOfConstraints::add_assignable_robot_to_point_displacement_constraint(
	int k,
	int var,
	int point_id,
	const Eigen::Vector3d& disp) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int objs_start  = num_agents * dim;                         // start of object coords
	const int point_start = objs_start + point_id * non_robot_dim;    // start of this point's coords

	// M = 2 * half_extent + |disp|
	Eigen::Vector3d M;
	for (int ax = 0; ax < 3; ++ax) {
		const double half_extent =
			(_global_x_ub(ax) - _global_x_lb(ax)) * 0.5;
		M(ax) = 2.0 * half_extent + std::abs(disp(ax));
	}

	_num_total_assignables++;

	return _add_assignable_op(
		DeferredOpKind::kLinearEq, k, var,
		[=, this](const Eigen::VectorXd& x, const int robot_id) {
			const int robot_start = robot_id * dim;
			Eigen::Vector3d p_WE = x.segment(robot_start, 3);
			Eigen::Vector3d p_WP = x.segment(point_start, 3);
			Eigen::Vector3d r = (p_WP - p_WE) - disp;
			return r.lpNorm<Eigen::Infinity>();
		},
		// ---- builder: add gated equalities with big-M ----
		[=, this](auto& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int /*phi_id*/,
			  const auto& X,                 // decision matrix for X
			  const auto& Assignments) {     // binary assignment matrix A

			const int node_k = subgraph.subgraph_id(k);
			const int variable_k = subgraph.subgraph_variable_id(var);
			if (variable_k < 0) return;

			const double neg_inf = -std::numeric_limits<double>::infinity();

			for (int i = 0; i < num_agents; ++i) {
				const auto s = Assignments(variable_k, i);
				const int robot_start = i * dim;

				for (int ax = 0; ax < 3; ++ax) {
					const drake::symbolic::Expression e =
						X(node_k, point_start + ax)   // point position component
						- X(node_k, robot_start + ax)   // robot position component
						- disp(ax);

					// e <= M*(1 - s)  <=>  e + M*s - M <= 0
					prog.AddLinearConstraint(e + M(ax) * s - M(ax), neg_inf, 0.0);

					// -e <= M*(1 - s) <=> -e + M*s - M <= 0
					prog.AddLinearConstraint(-e + M(ax) * s - M(ax), neg_inf, 0.0);
				}
			}
		});
}

int GraphOfConstraints::add_robot_to_point_displacement_constraint(
	int k,
	int robot_id,
	int point_id,
	const Eigen::VectorXd& disp,
	double tol) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int k_slice = std::min(robot_ambient_dim(robot_id), object_ambient_dim(point_id));
	DRAKE_DEMAND(disp.size() == k_slice);

	const int robot_start = robot_id * dim;
	const int objs_start = num_agents * dim;
	const int point_start = objs_start + point_id * non_robot_dim;

	Eigen::VectorXd lb = Eigen::VectorXd::Constant(k_slice, -tol);
	Eigen::VectorXd ub = Eigen::VectorXd::Constant(k_slice,  tol);

	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     Eigen::VectorXd p_WE = x.segment(robot_start, k_slice);
				     Eigen::VectorXd p_WP = x.segment(point_start, k_slice);
				     Eigen::VectorXd r  = (p_WP - p_WE) - disp;
				     return r.lpNorm<Eigen::Infinity>();
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto& Assignments) {
				     const unsigned int node_k = subgraph.subgraph_id(k);
				     VectorXDecisionVariable row = X.row(node_k);

				     VectorXDecisionVariable p_WE = row.segment(robot_start, k_slice);
				     VectorXDecisionVariable p_WP = row.segment(point_start, k_slice);

				     prog.AddLinearConstraint((p_WP - p_WE) - disp, lb, ub);
			     }
		);

	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_robot_to_point_displacement_cost(
	int k,
	int robot_id,
	int point_id,
	Eigen::Vector3d& disp) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int robot_start = robot_id * dim;
	const int objs_start = num_agents * dim;
	const int point_start = objs_start + point_id * non_robot_dim;

	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     Eigen::Vector3d p_WE = x.segment(robot_start, 3);
				     Eigen::Vector3d p_WP = x.segment(point_start, 3);
				     Eigen::Vector3d r  = (p_WP - p_WE) - disp;   // want r == 0
				     return r.squaredNorm();
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto&... /*unused*/) {
				     const unsigned int node_k = subgraph.subgraph_id(k);
				     VectorXDecisionVariable row = X.row(node_k);

				     VectorXDecisionVariable p_WE = row.segment(robot_start, 3);
				     VectorXDecisionVariable p_WP = row.segment(point_start, 3);

				     prog.AddQuadraticCost(((p_WP - p_WE) - disp).squaredNorm());
			     }
		);

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_robot_to_point_alignment_constraint(
	int k, int robot_id, int point_id, const Eigen::Vector3d& ee_ray_body,
	// optional for roll disambiguation:
	std::optional<Eigen::Vector3d> u_body_opt,         // u_b (must be ⟂ ee_ray_body)
	std::optional<Eigen::Vector3d> roll_ref_world,     // t (any, not necessarily ⟂ d)
	bool roll_ref_flat,
	bool require_positive_pointing,
	double eps_d, double tau_tperp) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int robot_start = robot_id * dim;
	const int objs_start  = num_agents * dim;

	int phi_id = _add_op(DeferredOpKind::kNonlinearEq, k,
			     [=, this](const Eigen::VectorXd& x, const int...) {
				     using Eigen::Vector3d;
				     using Eigen::Matrix3d;

				     // --- Small helpers (hinge and squared hinge) ---
				     auto hinge = [](double a){ return std::max(0.0, a); };
				     auto sqhinge = [&](double a){ const double h = hinge(a); return h*h; };

				     // --- Extract pose & point from the numeric state ---
				     auto [p_WE, R_WE] = PoseFromRow(this, robot_id, "ee_link", x);
				     auto p_WC = CubePosFromRow(this, point_id, x);

				     // --- Build r, d ---
				     const Vector3d r = R_WE * ee_ray_body;    // body ray in world
				     const Vector3d d = p_WC - p_WE;           // displacement to target

				     double residual = 0.0;

				     // (1) Point-at: r × d = 0  → use squared norm
				     const Vector3d rc = r.cross(d);

				     residual += rc.squaredNorm();

				     // (1b) Optional positive-facing: r·d >= 0  → penalize only if negative
				     if (require_positive_pointing) {
					     residual += sqhinge(-r.dot(d));  // (max(0, -r·d))^2
				     }

				     // (1c) Degeneracy guard: ||d||^2 >= eps_d^2 → penalize only if below threshold
				     const double d2 = d.squaredNorm();
				     if (d2 < eps_d*eps_d) {
					     residual += (eps_d*eps_d - d2) * (eps_d*eps_d - d2);
				     }

				     // (2) Roll disambiguation branch A: world roll reference vector
				     if (roll_ref_world && u_body_opt) {
					     const Eigen::Vector3d& t   = *roll_ref_world;
					     const Eigen::Vector3d& u_b = *u_body_opt;  // u_b ⟂ ee_ray_body guaranteed by caller
					     const Vector3d u = R_WE * u_b;

					     // Projection t_perp = t - (t·d)/(d·d) d  (robust to t not ⟂ d)
					     // Guard d·d near zero already handled above
					     Vector3d t_perp = t;
					     if (d2 > 0.0) {
						     const double t_dot_d = t.dot(d);
						     t_perp -= (t_dot_d / d2) * d;
					     }
					     // Enforce u × t_perp = 0
					     const Vector3d cx = u.cross(t_perp);
					     residual += cx.squaredNorm();

					     // Optional stabilizer u·d = 0
					     residual += (u.dot(d)) * (u.dot(d));

					     // Optional guard ||t_perp|| >= tau_tperp
					     const double tperp2 = t_perp.squaredNorm();
					     if (tperp2 < tau_tperp * tau_tperp) {
						     const double viol = (tau_tperp * tau_tperp - tperp2);
						     residual += viol * viol;
					     }
				     }
				     // (2) Roll disambiguation branch B: "flat" (z=0 plane) for u
				     else if (roll_ref_flat && u_body_opt) {
					     const Eigen::Vector3d& u_b = *u_body_opt;
					     const Vector3d u = R_WE * u_b;

					     // Mirror the constraint u(2) ∈ [-tol, tol] with a squared hinge on excess
					     const double tol = 1e-2;  // keep in sync with builder
					     residual += sqhinge(std::abs(u.z()) - tol);
				     }

				     return residual;
			     },
			     [=, this](auto& prog, const SubgraphOfConstraints& subgraph, const int /*phi_id*/,
				       const auto& X, const auto& Assignments) {
				     const unsigned int node_k = subgraph.subgraph_id(k);
				     Eigen::Matrix<Expression, Eigen::Dynamic, 1> row = X.row(node_k);

				     auto [p_WE, R_WE] = PoseFromRow(this, robot_id, "ee_link", row);
				     auto p_WC = CubePosFromRow(this, point_id, row);

				     // r = R * v_b, d = P - E
				     Eigen::Matrix<Expression,3,1> r = R_WE * ee_ray_body;
				     Eigen::Matrix<Expression,3,1> d = p_WC - p_WE;

				     // (1) Point-at: r × d = 0
				     auto rc = r.cross(d);
				     for (int i = 0; i < 3; ++i) prog.AddConstraint(rc(i) == 0)
									 .evaluator()->set_description("pointing at constraint");

				     // (1b) Optional: positive facing
				     if (require_positive_pointing) prog.AddConstraint(r.dot(d) >= 0)
									    .evaluator()->set_description("positive pointing");
			       
				     // (1c) Degeneracy guard: ||d|| >= eps_d
				     Expression d2 = d.dot(d);
				     prog.AddConstraint(d2 >= eps_d*eps_d)
					     .evaluator()->set_description("degeneracy guard");

				     // (2) Optional roll disambiguation
				     if (roll_ref_world && u_body_opt) {
					     const Eigen::Vector3d& t = *roll_ref_world;
					     const Eigen::Vector3d& u_b = *u_body_opt;  // caller ensures u_b ⟂ ee_ray_body
					     Eigen::Matrix<Expression,3,1> u = R_WE * u_b;

					     // Either projection-based:
					     Expression t_dot_d = t(0)*d(0) + t(1)*d(1) + t(2)*d(2);
					     Expression d_dot_d = d2;
					     Eigen::Matrix<Expression,3,1> t_perp;
					     t_perp << t(0) - (t_dot_d / d_dot_d) * d(0),
						     t(1) - (t_dot_d / d_dot_d) * d(1),
						     t(2) - (t_dot_d / d_dot_d) * d(2);

					     auto cx = u.cross(t_perp);
					     for (int i = 0; i < 3; ++i) prog.AddConstraint(cx(i) == 0);
					     prog.AddConstraint(u.dot(d) == 0);                // optional stabilizer
					     Expression tperp2 = t_perp.dot(t_perp);
					     prog.AddConstraint(tperp2 >= tau_tperp*tau_tperp); // optional guard
				     } else if (roll_ref_flat && u_body_opt) {
					     const double tol = 1e-2;
					     const Eigen::Vector3d& u_b = *u_body_opt;
					     Eigen::Matrix<Expression,3,1> u = R_WE * u_b;

					     prog.AddQuadraticConstraint(u(2), -tol, tol)
						     .evaluator()->set_description("flat roll constraint");;
				     }
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_robot_to_point_alignment_cost(
	int k, int robot_id, int point_id,
	const Eigen::Vector3d& ee_ray_body,                 // v_b
	std::optional<Eigen::Vector3d> u_body_opt,          // u_b (must be ⟂ v_b if provided)
	std::optional<Eigen::Vector3d> roll_ref_world,      // t (world)
	bool roll_ref_flat,                                  // use flat alternative if no t
	bool require_positive_pointing,                      // prefer r·d > 0
	// --- weights & small constants (defaults are gentle) ---
	double w_point    /*=1.0*/,
	double w_roll     /*=0.1*/,
	double w_flat     /*=0.05*/,
	double w_guard    /*=0.0*/,      // set >0 if you want to discourage tiny ||d||
	double w_u_stab   /*=0.01*/,     // small stabilizer for u·d ≈ 0 in roll mode
	double eps        /*=1e-10*/,    // denom regularizer
	double eps_d      /*=1e-3*/) {   // scale for guard

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int robot_start = robot_id * dim;
	const int objs_start  = num_agents * dim;

	// Optional: numeric evaluation for logging / debugging (returns cost value)
	auto numeric_eval = [=, this](const Eigen::VectorXd& x, const int...) {
		using Eigen::Vector3d; using Eigen::Matrix3d;

		// Pose at node k
		auto [p_WE, R_WE] = PoseFromRow(this, robot_id, "ee_link", x);
		auto p_WC = CubePosFromRow(this, point_id, x);

		const Vector3d r = R_WE * ee_ray_body;       // body ray in world
		const Vector3d d = p_WC - p_WE;
		const double d2 = d.squaredNorm();
		const double r_dot_d = r.dot(d);

		double J = 0.0;

		// Pointing cost
		if (require_positive_pointing) {
			const double d_norm = std::sqrt(d2 + eps);
			const double val = 1.0 - r_dot_d / d_norm;
			J += w_point * (val*val);
		} else {
			J += w_point * (1.0 - (r_dot_d*r_dot_d) / (d2 + eps));
		}

		// Roll disambiguation against world t
		if (roll_ref_world && u_body_opt) {
			const Vector3d& t   = *roll_ref_world;
			const Vector3d& u_b = *u_body_opt;
			const Vector3d u = R_WE * u_b;

			Vector3d t_perp = t - (t.dot(d) / (d2 + eps)) * d;
			const double tperp2 = t_perp.squaredNorm();
			const double u_dot_tperp = u.dot(t_perp);
			J += w_roll * (1.0 - (u_dot_tperp*u_dot_tperp) / (tperp2 + eps));

			// small stabilizer to keep u ⟂ d (helps when t≈d)
			J += w_u_stab * std::pow(u.dot(d), 2);
		}
		// Flat alternative: penalize u_z (smooth)
		else if (roll_ref_flat && u_body_opt) {
			const Vector3d& u_b = *u_body_opt;
			const Vector3d u = R_WE * u_b;
			J += w_flat * (u.z() * u.z());
		}

		// Optional soft guard against d≈0 (bounded, smooth; often can be 0)
		if (w_guard > 0.0) {
			const double s2 = eps_d*eps_d;
			J += w_guard * (s2 / (d2 + s2));
		}

		return J;
	};

	// Symbolic builder: adds costs to the Drake program
	auto builder = [=, this](auto& prog, const SubgraphOfConstraints& subgraph, const int /*phi_id*/,
				 const auto& X, const auto&...) {
		using drake::symbolic::Expression;
		const unsigned int node_k = subgraph.subgraph_id(k);
		Eigen::Matrix<Expression, Eigen::Dynamic, 1> row = X.row(node_k);

		auto [p_WE, R_WE] = PoseFromRow(this, robot_id, "ee_link", row);
		auto p_WC = CubePosFromRow(this, point_id, row);

		const Eigen::Matrix<Expression,3,1> r = R_WE * ee_ray_body;
		const Eigen::Matrix<Expression,3,1> d = p_WC - p_WE;

		Expression d2 = d.dot(d);
		Expression r_dot_d = r.dot(d);

		// Pointing cost
		if (require_positive_pointing) {
			Expression d_norm = drake::symbolic::sqrt(d2 + eps);
			Expression val = 1.0 - r_dot_d / d_norm;
			prog.AddCost(w_point * drake::symbolic::pow(val, 2.0));
		} else {
			prog.AddCost(w_point * (1.0 - (r_dot_d * r_dot_d) / (d2 + eps)));
		}

		// Roll disambiguation vs world t (projection)
		if (roll_ref_world && u_body_opt) {
			const Eigen::Vector3d& t   = *roll_ref_world;
			const Eigen::Vector3d& u_b = *u_body_opt;
			Eigen::Matrix<Expression,3,1> u = R_WE * u_b;

			Expression t_dot_d = t(0)*d(0) + t(1)*d(1) + t(2)*d(2);
			Eigen::Matrix<Expression,3,1> t_perp;
			t_perp << t(0) - (t_dot_d / (d2 + eps)) * d(0),
				t(1) - (t_dot_d / (d2 + eps)) * d(1),
				t(2) - (t_dot_d / (d2 + eps)) * d(2);

			Expression tperp2 = t_perp.dot(t_perp);
			Expression u_dot_tperp = u(0)*t_perp(0) + u(1)*t_perp(1) + u(2)*t_perp(2);
			prog.AddCost(w_roll * (1.0 - (u_dot_tperp * u_dot_tperp) / (tperp2 + eps)));

			// small stabilizer u·d
			prog.AddCost(w_u_stab * drake::symbolic::pow(u.dot(d), 2.0));
		}
		// Flat alternative
		else if (roll_ref_flat && u_body_opt) {
			const Eigen::Vector3d& u_b = *u_body_opt;
			Eigen::Matrix<Expression,3,1> u = R_WE * u_b;
			prog.AddCost(w_flat * (u(2) * u(2)));
		}

		// Optional guard against ||d||→0 (bounded, smooth)
		if (w_guard > 0.0) {
			Expression s2 = eps_d * eps_d;
			prog.AddCost(w_guard * (s2 / (d2 + s2)));
		}

		// Note: keep ONLY unit-quaternion and joint/box constraints hard elsewhere.
		// Do NOT add any equalities for alignment here.
	};

	// If your op system has a "cost" kind, use it; otherwise use whatever bucket you
	// use for soft terms. If unavailable, you can also directly call `builder` on
	// the active program instead of registering.
	int phi_id = _add_op(DeferredOpKind::kNonlinearCost, k, numeric_eval, builder);

	_phi_to_static_assignment_map[phi_id] = robot_id;
	return phi_id;
}

int GraphOfConstraints::add_point_to_point_displacement_constraint(
	int k,
	int point_a,
	int point_b,
	Eigen::Vector3d& disp,
	double tol) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(point_a >= 0 && point_a < num_objects);
	DRAKE_DEMAND(point_b >= 0 && point_b < num_objects);

	const int objs_start = num_agents * dim;
	const int startA = objs_start + point_a * non_robot_dim;
	const int startB = objs_start + point_b * non_robot_dim;

	Eigen::VectorXd lb = Eigen::VectorXd::Constant(3, -tol);
	Eigen::VectorXd ub = Eigen::VectorXd::Constant(3,  tol);

	return _add_op(DeferredOpKind::kLinearEq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       Eigen::Vector3d pA = x.segment(startA, 3);
			       Eigen::Vector3d pB = x.segment(startB, 3);
			       Eigen::Vector3d r  = (pB - pA) - disp;   // want r == 0
			       return r.lpNorm<Eigen::Infinity>() - tol;
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&... /*unused*/) {
			       const unsigned int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable row = X.row(node_k);

			       VectorXDecisionVariable pA = row.segment(startA, 3);
			       VectorXDecisionVariable pB = row.segment(startB, 3);

			       // residual = (pB - pA) - disp
			       if (tol == 0.0) {
				       // Enforce pB - pA = disp  (3 scalar equalities)
				       prog.AddLinearEqualityConstraint(pB - pA, disp);
			       } else {
				       prog.AddLinearConstraint((pB - pA) - disp, lb, ub);
			       }
		       }
		);
}

int GraphOfConstraints::add_point_to_point_displacement_cost(
	int k,
	int point_a,
	int point_b,
	Eigen::Vector3d& disp) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(point_a >= 0 && point_a < num_objects);
	DRAKE_DEMAND(point_b >= 0 && point_b < num_objects);

	const int objs_start = num_agents * dim;
	const int startA = objs_start + point_a * non_robot_dim;
	const int startB = objs_start + point_b * non_robot_dim;

	return _add_op(DeferredOpKind::kLinearEq, k,
		       [=, this](const Eigen::VectorXd& x,
				 const int... /*unused*/) {
			       Eigen::Vector3d pA = x.segment(startA, 3);
			       Eigen::Vector3d pB = x.segment(startB, 3);
			       Eigen::Vector3d r  = (pB - pA) - disp;   // want r == 0
			       return r.squaredNorm();
		       },
		       [=, this](auto& prog,
				 const SubgraphOfConstraints& subgraph,
				 const int phi_id,
				 const auto& X,
				 const auto&... /*unused*/) {
			       const unsigned int node_k = subgraph.subgraph_id(k);
			       VectorXDecisionVariable row = X.row(node_k);

			       VectorXDecisionVariable pA = row.segment(startA, 3);
			       VectorXDecisionVariable pB = row.segment(startB, 3);

			       prog.AddQuadraticCost(((pB - pA) - disp).squaredNorm());
		       }
		);
}

int GraphOfConstraints::add_point_to_point_alignment_constraint(
	int k,
	int point_a,
	int point_b,
	const Eigen::Vector3d& dir_W) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(point_a >= 0 && point_a < num_objects);
	DRAKE_DEMAND(point_b >= 0 && point_b < num_objects);
	DRAKE_DEMAND(dir_W.norm() > 1e-12 && "Alignment direction must be nonzero.");

	const int objs_start = num_agents * dim;
	const int startA = objs_start + point_a * non_robot_dim;
	const int startB = objs_start + point_b * non_robot_dim;

	// Build an orthonormal basis {u1,u2,û} with û = dir/||dir||.
	const Eigen::Vector3d uhat = dir_W.normalized();

	// Pick a helper axis not (near-)parallel to û for stable cross product.
	Eigen::Vector3d a;
	if (std::abs(uhat.x()) <= std::abs(uhat.y()) && std::abs(uhat.x()) <= std::abs(uhat.z()))
		a = Eigen::Vector3d::UnitX();
	else if (std::abs(uhat.y()) <= std::abs(uhat.x()) && std::abs(uhat.y()) <= std::abs(uhat.z()))
		a = Eigen::Vector3d::UnitY();
	else
		a = Eigen::Vector3d::UnitZ();

	const Eigen::Vector3d u1 = (uhat.cross(a)).normalized();
	const Eigen::Vector3d u2 =  uhat.cross(u1);  // already unit, orthogonal to u1

	return _add_op(DeferredOpKind::kNonlinearEq, k,
		       [=, this](const Eigen::VectorXd& x, const int... /*unused*/) {
			       const Eigen::Vector3d pA = x.segment(startA, 3);
			       const Eigen::Vector3d pB = x.segment(startB, 3);
			       const Eigen::Vector3d d  = (pB - pA);
			       Eigen::Vector2d r;
			       r << u1.dot(d), u2.dot(d);
			       return r.norm();
		       }, [=, this](auto& prog,
				    const SubgraphOfConstraints& subgraph,
				    const int /*phi_id*/,
				    const auto& X,
				    const auto&... /*unused*/) {
			       const int sg_k = subgraph.subgraph_id(k);
			       Eigen::RowVectorX<Expression> row = X.row(sg_k).template cast<Expression>();

			       const Eigen::Matrix<Expression,3,1> pA = row.segment(startA, 3).transpose();
			       const Eigen::Matrix<Expression,3,1> pB = row.segment(startB, 3).transpose();
			       const Eigen::Matrix<Expression,3,1> d  = (pB - pA);

			       prog.AddLinearEqualityConstraint(u1.transpose().cast<Expression>() * d, 0.0);
			       prog.AddLinearEqualityConstraint(u2.transpose().cast<Expression>() * d, 0.0);

			       // OPTIONAL (if you want to forbid the opposite direction and enforce same orientation):
			       //    uhatᵀ (pB - pA) ≥ 0   (also linear)
			       // prog.AddLinearConstraint(uhat.transpose().cast<Expression>() * d, 0.0,
			       //                          std::numeric_limits<double>::infinity());
		       });
}

///////////////////////////////////////////////////////////////////////////////
//                              EDGE CONSTRAINTS                             //
///////////////////////////////////////////////////////////////////////////////

// Shared by add_robot_holding_cube_constraint's endpoint (u/v) and interior
// (any node scheduled between u and v) applications, so both go through the
// same box-proximity logic. `gate` == nullptr means "always active" (the
// endpoint case); non-null relaxes the box via big-M so it only binds when
// `*gate` == 1 (the interior/betweenness case).
static void AddBoxProximityConstraint(
	drake::solvers::MathematicalProgram& prog,
	GraphOfConstraints* graph,
	int robot_id,
	int point_id,
	double d,
	double M_prox,
	const drake::solvers::MatrixXDecisionVariable& X,
	int graph_row,
	const drake::symbolic::Variable* gate) {

	Eigen::VectorX<Expression> q = X.row(graph_row);

	auto [p_WR, R_WR] = PoseFromRow(graph, robot_id, "ee_link", q);
	auto p_WC = CubePosFromRow(graph, point_id, q);

	const Eigen::Vector3<Expression> dp = p_WR - p_WC;

	const double kInf = std::numeric_limits<double>::infinity();

	// Box: |dx| <= d, |dy| <= d, |dz| <= d  (no squares, no quadratic)
	for (int i = 0; i < 3; ++i) {
		if (gate == nullptr) {
			prog.AddConstraint(dp(i), -d, d);
		} else {
			const Expression slack = M_prox * (1.0 - Expression(*gate));
			prog.AddLinearConstraint(dp(i) - d - slack, -kInf, 0.0);
			prog.AddLinearConstraint(-dp(i) - d - slack, -kInf, 0.0);
		}
	}
}

int GraphOfConstraints::add_robot_holding_cube_constraint(
	int u,
	int v,
	int robot_id,
	int point_id,
	double holding_distance_max,
	bool use_l2) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	// If you track num_objects, you can also check cube_i bounds here.

	// Single-coordinate positions are each within the global bounds, so the
	// magnitude of their difference (dp(i) above) is bounded by the global
	// range; double it for a safety margin.
	const double M_prox = 2.0 * (_global_x_ub - _global_x_lb).maxCoeff();

	int edge_phi_id = _add_edge_op(DeferredOpKind::kNonlinearEq, u, v, std::set<int>({point_id}),
			    [=, this](const Eigen::VectorXd& x,
				      const Eigen::VectorXi&/*unused*/) {
				    auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", x);
				    auto p_WC = CubePosFromRow(this, point_id, x);

				    Eigen::Vector3d r = (p_WC - p_WR);

				    double violation = 0.0;
				    if (use_l2) {
					    violation = r.lpNorm<2>() - holding_distance_max;
				    } else {
					    violation = r.lpNorm<Eigen::Infinity>() - holding_distance_max;
				    }

				    if (violation > 0) {
					    std::cout << "holding constraint violation: " << violation << std::endl;
					    std::cout << "robot id: " << robot_id << std::endl;
					    std::cout << "point id: " << point_id << std::endl;
					    std::cout << "p_WC: " << p_WC << std::endl;
					    std::cout << "p_WR: " << p_WR << std::endl;
					    std::cout << "r: " << r << std::endl;
				    }
				    return violation;
			    },
			    [=, this](drake::solvers::MathematicalProgram& prog,
				      const SubgraphOfConstraints& subgraph,
				      const int phi_id,
				      const drake::solvers::MatrixXDecisionVariable& X,
				      const drake::solvers::MatrixXDecisionVariable& /*unused*/,
				      const Eigen::VectorXd& x_u) {

				    const double d = holding_distance_max;

				    if (subgraph.structure.contains_node(u)) {
					    AddBoxProximityConstraint(prog, this, robot_id, point_id, d, M_prox,
								       X, subgraph.structure.subgraph_id(u), nullptr);
				    }
				    if (subgraph.structure.contains_node(v)) {
					    AddBoxProximityConstraint(prog, this, robot_id, point_id, d, M_prox,
								       X, subgraph.structure.subgraph_id(v), nullptr);
				    }
			    },
			    [](drake::solvers::MathematicalProgram& prog,
			       const int phi_id,
			       const Eigen::VectorXi& var_assignments,
			       const drake::solvers::MatrixXDecisionVariable& Xi) {
				    // std::cout << "adding edge op for short path" << std::endl;
			    });

	// record that this constraint is statically assigned to this robot.
	_edge_phi_to_static_assignment_map[edge_phi_id] = robot_id;

	// The holding invariant is an independent per-node check (not a relation
	// coupling u and v together), so it should also hold at any other node
	// that ends up scheduled between u and v in the solved route — see
	// Constraint 13 in graph_waypoint_mpc.cpp for where this gets invoked.
	edge_ops[edge_phi_id].interior_builder =
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& /*subgraph*/,
			  const int /*phi_id*/,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/,
			  int sg_w,
			  const drake::symbolic::Variable& gate) {
			AddBoxProximityConstraint(prog, this, robot_id, point_id, holding_distance_max,
						   M_prox, X, sg_w, &gate);
		};

	// DEPRECATED path: dual-populate the canonical hold registry (see
	// HoldDeclaration) for backward compat -- new code should call
	// add_hold directly instead of this function.
	add_hold(u, v, robot_id, {point_id});

	return edge_phi_id;
}

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

int GraphOfConstraints::add_edge_point_to_point_displacement_constraint(
	int u,
	int v,
	int point_a,
	int point_b,
	Eigen::Vector3d& disp,
	Eigen::Vector3d& tol) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(point_a >= 0 && point_a < num_objects);
	DRAKE_DEMAND(point_b >= 0 && point_b < num_objects);

	int edge_phi_id = _add_edge_op(DeferredOpKind::kLinearEq, u, v, std::set<int>({}),
			    [=, this](const Eigen::VectorXd& x,
				      const Eigen::VectorXi&/*unused*/) {
				    auto p_WC_a = CubePosFromRow(this, point_a, x);
				    auto p_WC_b = CubePosFromRow(this, point_b, x);
				    Eigen::Vector3d r  = (p_WC_b - p_WC_a) - disp;   // want r == 0
				    Eigen::Vector3d err = r.cwiseAbs() - tol;
				    auto violation = err.maxCoeff();
				    if (violation > 0) {
					    std::cout << "arranged constraint violation: " << violation << std::endl;
					    std::cout << "point_a id: " << point_a << std::endl;
					    std::cout << "point_b id: " << point_b << std::endl;
					    std::cout << "p_WC_a: " << p_WC_a << std::endl;
					    std::cout << "p_WC_b: " << p_WC_b << std::endl;
					    std::cout << "actual disp: " << (p_WC_b - p_WC_a) << std::endl;
					    std::cout << "r: " << r << std::endl;
				    }
				    return violation;
			    },
			    [=, this](drake::solvers::MathematicalProgram& prog,
				      const SubgraphOfConstraints& subgraph,
				      const int phi_id,
				      const drake::solvers::MatrixXDecisionVariable& X,
				      const drake::solvers::MatrixXDecisionVariable& /*unused*/,
				      const Eigen::VectorXd& x_u) {
				    return;
			    },
			    [](drake::solvers::MathematicalProgram& prog,
			       const int phi_id,
			       const Eigen::VectorXi& var_assignments,
			       const drake::solvers::MatrixXDecisionVariable& Xi) {
				    return;
			    });

	// record that this constraint is statically assigned to this robot.
	// _edge_phi_to_static_assignment_map[edge_phi_id] = robot_id;

	return edge_phi_id;
}

int GraphOfConstraints::add_robot_relative_rotation_constraint(
	int u,
	int v,
	int robot_id,
	Eigen::Quaternion<double>& quat) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);

	const int robot_start = robot_id * dim;

	// Normalize and precompute the constant relative rotation matrix.
	const Eigen::Quaternion<double> qrel = quat.normalized();
	const double wr = qrel.w(), xr = qrel.x(), yr = qrel.y(), zr = qrel.z();

	int edge_phi_id = _add_edge_op(
		DeferredOpKind::kNonlinearEq, u, v, std::set<int>{},
		// ---------- Evaluation: always satisfied. no backtracking ----------
		[=, this](const Eigen::VectorXd& x, const Eigen::VectorXi& /*unused*/) {
			return 0.0;
		},
		// ---------- Add constraints to Drake ----------
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int /*phi_id*/,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/,
			  const Eigen::VectorXd& x_u) {
			const unsigned int sg_u = subgraph.subgraph_id(u);
			const unsigned int sg_v = subgraph.subgraph_id(v);

			if (sg_u == -1 && sg_v != -1) {
				// When x_u is passed, it is in x_u
				Eigen::RowVectorXd row_u = x_u;
				Eigen::RowVectorX<Expression> row_v = AsExprRow(X.row(sg_v));

				Eigen::Vector4d q_u = row_u.segment(robot_start + 3, 4);
				Eigen::Vector4<Expression> q_v = row_v.segment(robot_start + 3, 4);

				const double wr = qrel.w(), xr = qrel.x(), yr = qrel.y(), zr = qrel.z();

				// Compose: q_expected = q_u cross q_rel  (body-fixed)
				Eigen::Matrix<double,4,1> qexp;
				qexp << q_u(0)*wr - q_u(1)*xr - q_u(2)*yr - q_u(3)*zr,
					q_u(0)*xr + q_u(1)*wr + q_u(2)*zr - q_u(3)*yr,
					q_u(0)*yr - q_u(1)*zr + q_u(2)*wr + q_u(3)*xr,
					q_u(0)*zr + q_u(1)*yr - q_u(2)*xr + q_u(3)*wr;

				// Enforce q_v == qexp (elementwise), with hemisphere fix:
				// dot(q_v, qexp) >= 0 to avoid the -q ambiguity.
				Expression dot = q_v(0)*qexp(0) + q_v(1)*qexp(1) + q_v(2)*qexp(2) + q_v(3)*qexp(3);
				prog.AddConstraint(dot >= 0.0);
				for (int i=0; i<4; ++i) {
					prog.AddConstraint(q_v(i) - qexp(i) == 0);
				}
			} else if (sg_u != -1 && sg_v != -1) {
				Eigen::RowVectorX<Expression> row_u = AsExprRow(X.row(sg_u));
				Eigen::RowVectorX<Expression> row_v = AsExprRow(X.row(sg_v));

				Eigen::Vector4<Expression> q_u, q_v;
				q_u = row_u.segment(robot_start + 3, 4);
				q_v = row_v.segment(robot_start + 3, 4);

				const double wr = qrel.w(), xr = qrel.x(), yr = qrel.y(), zr = qrel.z();

				// Compose: q_expected = q_u cross q_rel  (body-fixed)
				Eigen::Matrix<Expression,4,1> qexp;
				qexp << q_u(0)*wr - q_u(1)*xr - q_u(2)*yr - q_u(3)*zr,
					q_u(0)*xr + q_u(1)*wr + q_u(2)*zr - q_u(3)*yr,
					q_u(0)*yr - q_u(1)*zr + q_u(2)*wr + q_u(3)*xr,
					q_u(0)*zr + q_u(1)*yr - q_u(2)*xr + q_u(3)*wr;

				// Enforce q_v == qexp (elementwise), with hemisphere fix:
				// dot(q_v, qexp) >= 0 to avoid the -q ambiguity.
				Expression dot = q_v(0)*qexp(0) + q_v(1)*qexp(1) + q_v(2)*qexp(2) + q_v(3)*qexp(3);
				prog.AddConstraint(dot >= 0.0);
				for (int i=0; i<4; ++i) {
					prog.AddConstraint(q_v(i) - qexp(i) == 0);
				}
			}
		},
		// Short-path variant (unused)
		[](drake::solvers::MathematicalProgram&, const int, const Eigen::VectorXi&,
		   const drake::solvers::MatrixXDecisionVariable&) { return; });

	// Statically assigned to this robot.
	_edge_phi_to_static_assignment_map[edge_phi_id] = robot_id;
	return edge_phi_id;
}

int GraphOfConstraints::add_robot_relative_displacement_constraint(
	int u,
	int v,
	int robot_id,
	Eigen::Vector3d& disp) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);

	const int robot_start = robot_id * dim;

	int edge_phi_id = _add_edge_op(
		DeferredOpKind::kLinearEq, u, v, std::set<int>{},
		// ---------- Evaluation: always satisfied. no backtracking ----------
		[=, this](const Eigen::VectorXd& x, const Eigen::VectorXi& /*unused*/) {
			return 0.0;
		},
		// ---------- Add constraints to Drake ----------
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int /*phi_id*/,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/,
			  const Eigen::VectorXd& x_u) {
			const unsigned int sg_u = subgraph.subgraph_id(u);
			const unsigned int sg_v = subgraph.subgraph_id(v);

			if (sg_u == -1 && sg_v != -1) {
				// When x_u is passed, it is in x_u
				// std::cout << "HERE ADDING CONSTRAINT RELATIVE TO:\n" << x_u << std::endl;

				Eigen::RowVectorXd row_u = x_u;
				Eigen::RowVectorX<Expression> row_v = AsExprRow(X.row(sg_v));

				Eigen::Vector3d p_WE_u = row_u.segment(robot_start, 3);
				Eigen::Vector3<Expression> p_WE_v = row_v.segment(robot_start, 3);

				prog.AddLinearEqualityConstraint(p_WE_v - p_WE_u, disp);
			} else if (sg_u != -1 && sg_v != -1) {
				Eigen::RowVectorX<Expression> row_u = AsExprRow(X.row(sg_u));
				Eigen::RowVectorX<Expression> row_v = AsExprRow(X.row(sg_v));

				Eigen::Vector3<Expression> p_WE_u = row_u.segment(robot_start, 3);
				Eigen::Vector3<Expression> p_WE_v = row_v.segment(robot_start, 3);

				prog.AddLinearEqualityConstraint(p_WE_v - p_WE_u, disp);
			}
		},
		// Short-path variant (unused)
		[](drake::solvers::MathematicalProgram&, const int, const Eigen::VectorXi&,
		   const drake::solvers::MatrixXDecisionVariable&) { return; });

	// Statically assigned to this robot.
	_edge_phi_to_static_assignment_map[edge_phi_id] = robot_id;
	return edge_phi_id;
}

int GraphOfConstraints::add_edge_assignable_robot_to_point_displacement_constraint(
	int u,
	int v,
	int var,
	int point_id,
	Eigen::Vector3d& disp,
	Eigen::Vector3d& tol) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	return _add_assignable_edge_op(
		DeferredOpKind::kAgentLinearEq, u, v, var, std::set<int>(),
		[=, this](const Eigen::VectorXd& x,
			  const Eigen::VectorXi& assignments) {
			const int robot_id = assignments(var);
			// If robot isn't assigned for this constraint now,
			// assume its violated.
			if (robot_id == -1) { return 99.0; }

			auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", x);
			auto p_WC = CubePosFromRow(this, point_id, x);
			Eigen::Vector3d r  = (p_WC - p_WR) - disp;   // want r == 0
			Eigen::Vector3d err = r.cwiseAbs() - tol;
			return err.maxCoeff();
		},
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int phi_id,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/,
			  const Eigen::VectorXd& x_u) {
			return;
		},
		[](drake::solvers::MathematicalProgram& prog,
		   const int phi_id,
		   const Eigen::VectorXi& var_assignments,
		   const drake::solvers::MatrixXDecisionVariable& Xi) {
			return;
		});
}

int GraphOfConstraints::add_assignable_robot_holding_point_constraint(
	int u,
	int v,
	int var,
	int point_id,
	double holding_distance_max,
	bool use_l2) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int edge_phi_id = _add_assignable_edge_op(
		DeferredOpKind::kAgentLinearEq, u, v, var, std::set<int>({point_id}),
		[=, this](const Eigen::VectorXd& x,
			  const Eigen::VectorXi& assignments) {
			const int robot_id = assignments(var);
			// If robot isn't assigned for this constraint now,
			// assume its violated.
			if (robot_id == -1) { return 99.0; }

			auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", x);
			auto p_WC = CubePosFromRow(this, point_id, x);

			Eigen::Vector3d r = (p_WC - p_WR);

			double violation = 0.0;
			if (use_l2) {
				violation = r.lpNorm<2>() - holding_distance_max;
			} else {
				violation = r.lpNorm<Eigen::Infinity>() - holding_distance_max;
			}

			if (violation > 0) {
				std::cout << "holding constraint violation: " << violation << std::endl;
				std::cout << "robot id: " << robot_id << std::endl;
				std::cout << "point id: " << point_id << std::endl;
				std::cout << "p_WC: " << p_WC << std::endl;
				std::cout << "p_WR: " << p_WR << std::endl;
				std::cout << "use_l2: " << use_l2 << std::endl;
				std::cout << "r: " << r << std::endl;
			}

			return violation;
		},
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int phi_id,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/,
			  const Eigen::VectorXd& x_u) {
			return;
		},
		[](drake::solvers::MathematicalProgram& prog,
		   const int phi_id,
		   const Eigen::VectorXi& var_assignments,
		   const drake::solvers::MatrixXDecisionVariable& Xi) {
			return;
		});

	// DEPRECATED path: dual-populate the canonical hold registry (see
	// HoldDeclaration) for backward compat -- new code should call
	// add_assignable_hold directly instead of this function.
	add_assignable_hold(u, v, var, {point_id});

	return edge_phi_id;
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
