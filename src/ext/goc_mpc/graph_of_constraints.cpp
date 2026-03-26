#include "graph_of_constraints.hpp"
#include "../utils.hpp"


using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;
using drake::math::RigidTransform;
using drake::math::RotationMatrix;

// Constructor
GraphOfConstraints::GraphOfConstraints(const std::vector<std::string> robots,
				       const std::vector<std::string> objects,
				       double global_x_lb,
				       double global_x_ub)
	: _robot_names(robots),
	  _object_names(objects),
	  num_phis(0),
	  num_edge_phis(0),
	  num_var_phis(0),
	  num_variables(0),
	  _num_total_assignables(0),
	  num_agents(robots.size()),
	  num_objects(objects.size()),
	  dim(0),
	  non_robot_dim(0) {

	for (const std::string& s : robots) {
		int robot_qdim;
		if (s.find("pos_quat") != std::string::npos) {
			robot_qdim = 3 + 4;
		} else if (s.find("pos_rot_mat") != std::string::npos) {
			robot_qdim = 3 + 9;
		} else if (s.find("point_mass") != std::string::npos) {
			robot_qdim = 3;
		} else {
			throw std::runtime_error("Only supporting 'point_mass', 'pos_quat', and 'pos_rot_mat' robots.");
		}

		if (dim == 0) {
			dim = robot_qdim;
		} else if (dim != robot_qdim) {
			throw std::runtime_error("Only supporting robots with the same dimension.");
		}
	}

	// Only supporting other 3D points in state space.
	non_robot_dim = 3;

	total_dim = num_agents * dim + num_objects * non_robot_dim;

	_global_x_lb = Eigen::VectorXd::Constant(total_dim, global_x_lb);
	_global_x_ub = Eigen::VectorXd::Constant(total_dim, global_x_ub);

	int i = 0;
	for (int ag = 0; ag < num_agents; ++ag) {
		if (robot_is_pos_quat(ag)) {
			for (int j = i+3; j < i+3+4; ++j) {
				_global_x_lb(j) = -1;
				_global_x_ub(j) = 1;
			}
		} else if (robot_is_pos_rot_mat(ag)) {
			for (int j = i+3; j < i+3+9; ++j) {
				_global_x_lb(j) = -1;
				_global_x_ub(j) = 1;
			}
		}
		i += dim;
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
	int next_variable_id = num_variables;
	return num_variables++;
}

bool GraphOfConstraints::robot_is_free_body(int ag) const {
	return robot_is_pos_quat(ag) || robot_is_pos_rot_mat(ag) || robot_is_point_mass(ag);
}

bool GraphOfConstraints::robot_is_pos_quat(int ag) const {
	return _robot_names.at(ag).find("pos_quat") != std::string::npos;
}

bool GraphOfConstraints::robot_is_pos_rot_mat(int ag) const {
	return _robot_names.at(ag).find("pos_rot_mat") != std::string::npos;
}

bool GraphOfConstraints::robot_is_point_mass(int ag) const {
	return _robot_names.at(ag).find("point_mass") != std::string::npos;
}

std::tuple<std::vector<std::optional<int>>,
	   std::vector<std::vector<int>>,
	   std::vector<struct AgentInteraction>> GraphOfConstraints::get_agent_paths(
		   const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXi& assignments) const {
	const InducedSubgraphView<py::object> sg = InducedSubgraphView<py::object>(
		structure, remaining_vertices);

	// This function introduces the idea now that every node has exactly one phi. That seems pretty reasonable.

	std::vector<std::vector<int>> agent_nodes(num_agents);
	std::map<int, std::vector<std::pair<int, int>>> node_to_agent_and_depth_pairs_map;
	std::vector<std::pair<int, int>> cross_agent_edges;
	std::vector<struct AgentInteraction> agent_interactions;

	std::vector<std::optional<int>> parents = sg.bfs_visit_from_sources(
		[this, &assignments, &agent_nodes, &agent_interactions, &node_to_agent_and_depth_pairs_map, &cross_agent_edges]
		(int node, int depth, std::optional<int> parent) {
			// std::cout << "processing " << node << std::endl;

			if (node_to_phis_map.contains(node)) {
				std::set<int> assignments_for_node;

				for (int phi_id : node_to_phis_map.at(node)) {
					int assignment = -1;

					assignment = assignments(phi_id);

					// std::cout << phi_id << " belonging to " << node << " is dynamically assigned to " << assignment << std::endl;

					if (_phi_to_static_assignment_map.contains(phi_id) && assignment != -1) {
						// std::cout << "_phi_to_static_assignment_map for " << phi_id << " gives " << _phi_to_static_assignment_map.at(phi_id) << " but assignment = " << assignment << std::endl;
						throw std::runtime_error("conflicting assignment");
					} else if (_phi_to_static_assignment_map.contains(phi_id)) {
						assignment = _phi_to_static_assignment_map.at(phi_id);
						// std::cout << phi_id << " belonging to " << node << " is statically assigned to " << assignment << std::endl;
					}

					// TODO: Clean this up for the case where there
					// might be multiple phi's on a single node and
					// we should add the nodes multiple times.
					if (assignment == -1) {
						// std::cout << "adding node for all by default" << std::endl;
						for (int ag = 0; ag < num_agents; ++ag) {
							assignments_for_node.insert(ag);
						}
					} else {
						assignments_for_node.insert(assignment);
					}
				}

				for (int assignment : assignments_for_node) {
					// std::cout << "assignment for node[" << node << "]: " << assignment << std::endl;
					int depth = agent_nodes[assignment].size();
					agent_nodes[assignment].push_back(node);
					node_to_agent_and_depth_pairs_map[node].emplace_back(assignment, depth);
				}
			}

			// if a node is shared in the agent paths it is an equality agent interaction
			int num_pairs = node_to_agent_and_depth_pairs_map[node].size();
 			if (num_pairs > 1) {
				for (int i = 0; i < num_pairs; ++i) {
					auto& [ag_i, depth_i] = node_to_agent_and_depth_pairs_map[node][i];
					for (int j = i; j < num_agents; ++j) {
						auto& [ag_j, depth_j] = node_to_agent_and_depth_pairs_map[node][j];
						agent_interactions.emplace_back(
							ag_i, depth_i, ag_j, depth_j, node, node, AgentInteraction::Type::EQUAL);
					}
				}
			}

			// if there is an edge in the bfs tree from one agent's
			// path to another agent's path, that is a cross agent
			// edge and will eventually be a less than interaction.
			if (parent.has_value()) {
				bool is_cross_agent = false;
				for (const auto& [ag_u, _] : node_to_agent_and_depth_pairs_map.at(*parent)) {
					for (const auto& [ag_v, _] : node_to_agent_and_depth_pairs_map.at(node)) {
						if (ag_u != ag_v) {
							is_cross_agent = true;
							break;
						}
					}
				}

				if (is_cross_agent) {
					cross_agent_edges.emplace_back(*parent, node);
				}
			}
		},
		[this, &agent_interactions, &node_to_agent_and_depth_pairs_map, &cross_agent_edges]
		(int u, int u_depth, int v, int v_depth) {
			// std::cout << "adding cross agent edge " << u << "->" << v << std::endl;
			cross_agent_edges.emplace_back(u, v);
		});

	for (auto& [u, v] : cross_agent_edges) {
		std::vector<std::pair<int, int>> u_agent_and_depth_pairs = node_to_agent_and_depth_pairs_map[u];
		std::vector<std::pair<int, int>> v_agent_and_depth_pairs = node_to_agent_and_depth_pairs_map[v];
		for (auto& [ag_i, depth_i] : u_agent_and_depth_pairs) {
			for (auto& [ag_j, depth_j] : v_agent_and_depth_pairs) {
				agent_interactions.emplace_back(
					ag_i, depth_i, ag_j, depth_j, u, v, AgentInteraction::Type::LESS_THAN);
			}
		}
	}

	return std::make_tuple(parents, agent_nodes, agent_interactions);
}

std::map<int, struct DeferredEdgeOp> GraphOfConstraints::get_next_edge_ops(const std::vector<int> remaining_vertices) const {
	std::map<int, struct DeferredEdgeOp> e_ops;

	for (const auto& e : structure.incoming_cut_edges(remaining_vertices)) {
		if (this->edge_to_phis_map.contains(e)) {
			for (int edge_phi_id : this->edge_to_phis_map.at(e)) {
				e_ops[edge_phi_id] = this->edge_ops.at(edge_phi_id);
			}
		}
	}

	return e_ops;
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
	if (!ops.contains(phi_id)) {
		return true;
	} else {
		const DeferredOp& op = ops.at(phi_id);
		double v = op.eval(x, assignments(phi_id));
		std::cout << "violation: " << v << std::endl;
		return v < tol;
	}
}

bool GraphOfConstraints::evaluate_edge_phi(int phi_id,
					   const Eigen::VectorXd& x,
					   const Eigen::VectorXi& var_assignments,
					   double tol) const {
	if (!edge_ops.contains(phi_id)) {
		return true;
	} else {
		const DeferredEdgeOp& op = edge_ops.at(phi_id);
		double v = op.eval(x, var_assignments);
		return v < tol;
	}
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
	backtrack_map[edge_id] = backtrack_nodes;
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
				       const auto&) {

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
				 const auto&) {
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
				       const auto&) {
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
				       const auto&) {
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

					  const int node_k     = subgraph.subgraph_id(k);
					  const int variable_k = subgraph.subgraph_variable_id(var);
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

					  for (int i = 0; i < num_agents; ++i) {
						  // Variables [ x_{k,i} ; s ] with s = A(variable_k, i)
						  VectorXDecisionVariable vars(dim + 1);
						  for (int j = 0; j < dim; ++j) vars[j] = X(node_k, i*dim + j);
						  vars[dim] = Assignments(variable_k, i);   // <-- use A as selector

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
				       const auto&... /*unused*/) {

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

			const int node_k     = subgraph.subgraph_id(k);
			const int variable_k = subgraph.subgraph_variable_id(var);

			const double neg_inf = -std::numeric_limits<double>::infinity();

			for (int i = 0; i < num_agents; ++i) {
				const auto s = Assignments(variable_k, i);   // binary: 0/1
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
	Eigen::Vector3d& disp,
	double tol) {

	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(robot_id >= 0 && robot_id < num_agents);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	const int robot_start = robot_id * dim;
	const int objs_start = num_agents * dim;
	const int point_start = objs_start + point_id * non_robot_dim;

	Eigen::VectorXd lb = Eigen::VectorXd::Constant(3, -tol);
	Eigen::VectorXd ub = Eigen::VectorXd::Constant(3,  tol);

	int phi_id = _add_op(DeferredOpKind::kLinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {
				     Eigen::Vector3d p_WE = x.segment(robot_start, 3);
				     Eigen::Vector3d p_WP = x.segment(point_start, 3);
				     Eigen::Vector3d r  = (p_WP - p_WE) - disp;   // want r == 0
				     return r.lpNorm<Eigen::Infinity>();
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

				     // Enforce pB - pA = disp  (3 scalar equalities)
				     prog.AddLinearConstraint((p_WP - p_WE) - disp, lb, ub);

				     // prog.AddLinearEqualityConstraint(p_WP - p_WE, disp);
			     }
		);

	// record that this constraint is statically assigned to this robot.
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
				       const auto& X, const auto&...) {
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
				    std::cout << "holding constraint violation: " << violation << std::endl;
				    return violation;
			    },
			    [=, this](drake::solvers::MathematicalProgram& prog,
				      const SubgraphOfConstraints& subgraph,
				      const int phi_id,
				      const drake::solvers::MatrixXDecisionVariable& X,
				      const drake::solvers::MatrixXDecisionVariable& /*unused*/,
				      const Eigen::VectorXd& x_u) {

				    const double d = holding_distance_max;
				    auto add_box_proximity = [&](int graph_row) {
					    Eigen::VectorX<Expression> q = X.row(graph_row);

					    auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", q);
					    auto p_WC = CubePosFromRow(this, point_id, q);

					    const Eigen::Vector3<Expression> dp = p_WR - p_WC;

					    // Box: |dx| <= d, |dy| <= d, |dz| <= d  (no squares, no quadratic)
					    for (int i = 0; i < 3; ++i) {
						    auto c_i = prog.AddConstraint(dp(i), -d, d);
					    }
				    };

				    if (subgraph.structure.contains_node(u)) {
					    add_box_proximity(subgraph.structure.subgraph_id(u));
				    }
				    if (subgraph.structure.contains_node(v)) {
					    add_box_proximity(subgraph.structure.subgraph_id(v));
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

	return edge_phi_id;
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

	return _add_assignable_edge_op(
		DeferredOpKind::kAgentLinearEq, u, v, var, std::set<int>({point_id}),
		[=, this](const Eigen::VectorXd& x,
			  const Eigen::VectorXi& assignments) {

			const int robot_id = assignments(var);

			auto [p_WR, R_WR] = PoseFromRow(this, robot_id, "ee_link", x);
			auto p_WC = CubePosFromRow(this, point_id, x);

			Eigen::Vector3d r = (p_WC - p_WR);

			double violation = 0.0;
			if (use_l2) {
				violation = r.lpNorm<2>() - holding_distance_max;
			} else {
				violation = r.lpNorm<Eigen::Infinity>() - holding_distance_max;
			}
			std::cout << "holding constraint violation: " << violation << std::endl;
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

			// Get the variable we want to constrain
			const int variable_k = subgraph.subgraph_variable_id(var);

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

			// Get the variable we want to constrain
			const int variable1_k = subgraph.subgraph_variable_id(var1);
			const int variable2_k = subgraph.subgraph_variable_id(var2);

			if (variable1_k != -1 && variable2_k != -1) {
				for (int i = 0; i < num_agents; i++) {
					const auto s = Assignments(variable1_k, i) + Assignments(variable2_k, i);
					// 1 <= binary v1 + binary v2 <= 1 implies both
					// cannot be zero and both cannot be one.
					prog.AddLinearConstraint(s, 1, 1);
				}
			}
		});
}
