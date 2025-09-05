#include "graph_of_constraints.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;
using drake::math::RigidTransform;
using drake::multibody::RigidBody;

// Constructor
GraphOfConstraints::GraphOfConstraints(MultibodyPlant<Expression>& plant,
				       const std::vector<std::string> robots,
				       const std::vector<std::string> objects,
				       double global_x_lb,
				       double global_x_ub)
	: _plant(&plant),
	  _robot_names(robots),
	  _object_names(objects),
	  num_phis(0),
	  num_edge_phis(0),
	  num_variables(0),
	  _num_total_assignables(0),
	  num_agents(robots.size()),
	  num_objects(objects.size()),
	  dim(0),
	  non_robot_dim(0) {

	for (const std::string& s : robots) {
		ModelInstanceIndex robot = _plant->GetModelInstanceByName(s);

		int robot_qdim;
		if (s.find("free_body") != std::string::npos) {
			// special case because I want quaternion state space.
			robot_qdim = 7;
		} else {
			robot_qdim = _plant->num_actuated_dofs(robot);
		}

		if (dim == 0) {
			dim = robot_qdim;
		} else if (dim != robot_qdim) {
			throw std::runtime_error("Only supporting robots with the same dimension.");
		}
	}

	for (const std::string& s : objects) {
		ModelInstanceIndex obj = _plant->GetModelInstanceByName(s);
		/* silly check because these should be 3dof points, but whatever */
		int obj_qdim = _plant->num_positions(obj);
		int obj_qddim = _plant->num_velocities(obj);
		if (non_robot_dim == 0 && obj_qdim == obj_qddim) {
			non_robot_dim = obj_qdim;
		} else if (non_robot_dim != obj_qdim) {
			throw std::runtime_error("Only supporting objects with the same dimension.");
		}
	}

	total_dim = num_agents * dim + num_objects * non_robot_dim;

	_global_x_lb = Eigen::VectorXd::Constant(total_dim, global_x_lb);
	_global_x_ub = Eigen::VectorXd::Constant(total_dim, global_x_ub);
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
	return _robot_names.at(ag).find("free_body") != std::string::npos;
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
			if (node_to_phis_map.contains(node)) {
				std::set<int> assignments_for_node;

				for (int phi_id : node_to_phis_map.at(node)) {
					int assignment = -1;

					assignment = assignments(phi_id);

					// std::cout << phi_id << " belonging to " << node << " is dynamically assigned to " << assignment << std::endl;

					if (_phi_to_static_assignment_map.contains(phi_id) && assignment != -1) {
						std::cout << "_phi_to_static_assignment_map for " << phi_id << " gives " << _phi_to_static_assignment_map.at(phi_id) << " but assignment = " << assignment << std::endl;
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
				throw std::runtime_error("Somehow constraint was not assigned.");
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
int GraphOfConstraints::add_agents_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
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
int GraphOfConstraints::add_agents_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
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
int GraphOfConstraints::add_agent_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
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
int GraphOfConstraints::add_agent_linear_ineq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
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

int GraphOfConstraints::add_agent_pos_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
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
				     VectorXDecisionVariable agent_pos_k = X.row(node_k).segment(robot_id*dim, 3);
				     auto beq = prog.AddLinearEqualityConstraint(A, b, agent_pos_k);
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
}

int GraphOfConstraints::add_agent_quat_linear_eq(int k, int robot_id, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
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
				     auto beq = prog.AddLinearEqualityConstraint(A, b, agent_quat_k);
			     });

	// record that this constraint is statically assigned to this robot.
	_phi_to_static_assignment_map[phi_id] = robot_id;

	return phi_id;
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
				
						  for (int r = 0; r < A.rows(); ++r) {
							  const Eigen::RowVectorXd c = A.row(r);
							  const auto [max_cx, min_cx] = max_min_ct_x_over_box(
								  c,
								  _global_x_lb,
								  _global_x_ub);

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

template <typename T>
void GraphOfConstraints::set_configuration(
	std::unique_ptr<drake::systems::Context<T>>& context,
	const Eigen::VectorX<T>& q_all) const {

	using drake::multibody::JointIndex;
	using drake::multibody::ModelInstanceIndex;
	using drake::math::RollPitchYaw;

	// VectorX<T> q = plant->GetPositions(*context);

	// for (ModelInstanceIndex i : plant ->
	int i = 0;
	for (std::string r_name : _robot_names) {
		const auto& mi = _plant->GetModelInstanceByName(r_name);

		if (r_name.find("free_body") != std::string::npos) {


			// Eigen::Vector3<T> p_W;
			// p_W << q_all.segment(i, 3);
			// i += 3;

			// const T w = q_all(i + 0);
			// const T x = q_all(i + 1);
			// const T y = q_all(i + 2);
			// const T z = q_all(i + 3);
			// i += 4;

			// // Normalize without branches (smooth except at norm2 = 0).
			// const T norm2 = w*w + x*x + y*y + z*z;
			// // Strongly recommend: add equality constraint norm2 == 1 in the program.
			// const T s  = T(1) / norm2;   // 1 / |q|^2
			// const T s2 = T(2) * s;       // 2 / |q|^2

			// Eigen::Matrix<T,3,3> Rm;
			// Rm(0,0) = T(1) - s2*(y*y + z*z);
			// Rm(0,1) =       s2*(x*y - w*z);
			// Rm(0,2) =       s2*(x*z + w*y);

			// Rm(1,0) =       s2*(x*y + w*z);
			// Rm(1,1) = T(1) - s2*(x*x + z*z);
			// Rm(1,2) =       s2*(y*z - w*x);

			// Rm(2,0) =       s2*(x*z - w*y);
			// Rm(2,1) =       s2*(y*z + w*x);
			// Rm(2,2) = T(1) - s2*(x*x + y*y);

			// drake::math::RotationMatrix<T> R_WB(Rm);
			// const drake::math::RigidTransform<T> X_WB(R_WB, p_W);

			// const auto& body = _plant->GetBodyByName("ee_link", mi);
			// _plant->SetFreeBodyPose(context.get(), body, X_WB);

			Eigen::Vector3<T> p_W;
			p_W << q_all.segment(i, 3);
			i+=3;

			// w, x, y, z
			Eigen::Quaternion<T> q_W(q_all(i), q_all(i+1), q_all(i+2), q_all(i+3));
			i+=4;

			// Pick the actual free base body in this model instance.
			const auto& body = _plant->GetBodyByName("ee_link", mi);

			const RigidTransform<T> X_WB(q_W, p_W);
			_plant->SetFreeBodyPose(context.get(), body, X_WB);
		} else {
			const std::vector<JointIndex> joint_indices = _plant->GetActuatedJointIndices(mi);
			for (JointIndex j : joint_indices) {
				const auto& joint = _plant->get_joint(j);
				DRAKE_DEMAND(joint.num_positions() == 1);  // Only supporting 1-dof joints
				// Set position for this joint in context.
				joint.SetPositions(context.get(), q_all.segment(i, 1));
				i++;
			}
		}
	}

	DRAKE_DEMAND(i == num_agents * dim);

	for (std::string o_name : _object_names) {
		const auto& mi = _plant->GetModelInstanceByName(o_name);
		_plant->SetPositions(context.get(), mi, q_all.segment(i, non_robot_dim));
		i += non_robot_dim;
	}

	DRAKE_DEMAND(i == total_dim);
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

	const std::string& robot_model_name = _robot_names.at(robot_id);
	const std::string& cube_model_name = _object_names.at(cube_id);

	const ModelInstanceIndex robot_mi = _plant->GetModelInstanceByName(robot_model_name);
	const ModelInstanceIndex cube_mi  = _plant->GetModelInstanceByName(cube_model_name);

	int phi_id = _add_op(DeferredOpKind::kNonlinearEq, k,
			     [=, this](const Eigen::VectorXd& x,
				       const int... /*unused*/) {

				     using drake::math::RigidTransform;
				     using drake::symbolic::Expression;
				     using drake::symbolic::Evaluate;

				     const auto& robot_body = _plant->GetBodyByName("ee_link", robot_mi);
				     const auto& cube_body  = _plant->GetBodyByName("cb_body",  cube_mi);

				     Eigen::VectorX<Expression> q_all = x.cast<Expression>();

				     auto context = _plant->CreateDefaultContext();
				     set_configuration(context, q_all);

				     const RigidTransform<Expression> X_WR =
					     _plant->EvalBodyPoseInWorld(*context, robot_body);
				     const RigidTransform<Expression> X_WC =
					     _plant->EvalBodyPoseInWorld(*context, cube_body);

				     // g(q) = [x_r - x_c, y_r - y_c, z_r - z_c - Δz] = 0
				     Eigen::Vector3<Expression> g;
				     g << (X_WR.translation().x() - X_WC.translation().x() - x_offset),
					     (X_WR.translation().y() - X_WC.translation().y() - y_offset),
					     (X_WR.translation().z() - X_WC.translation().z() - delta_z);

				     // dp_expr has no free symbols (only constants), so Evaluate(...) -> double works.
				     double violation = 0.0;
				     for (int i = 0; i < 3; ++i) {
					     const double gi = g[i].Evaluate();
					     violation = std::max(violation, std::abs(gi));
				     }
				     return violation;
			     },
			     [=, this](auto& prog,
				       const SubgraphOfConstraints& subgraph,
				       const int phi_id,
				       const auto& X,
				       const auto&... /*unused*/) {

				     using drake::multibody::JointIndex;
				     using drake::systems::Context;

				     const int node_k = subgraph.subgraph_id(k);

				     // Convert X[row] decision variables to Expressions.
				     Eigen::VectorX<Expression> q_all(total_dim);
				     for (int j = 0; j < total_dim; ++j) {
					     q_all(j) = Expression(X(node_k, j));
				     }

				     // Context<Expression> with these positions.
				     auto context = _plant->CreateDefaultContext();
				     set_configuration(context, q_all);

				     // World poses of each model's body.
				     const auto& robot_body = _plant->GetBodyByName("ee_link", robot_mi);
				     const auto& cube_body  = _plant->GetBodyByName("cb_body", cube_mi);
				     const RigidTransform<Expression> X_WR =
					     _plant->EvalBodyPoseInWorld(*context, robot_body);
				     const RigidTransform<Expression> X_WC =
					     _plant->EvalBodyPoseInWorld(*context, cube_body);

				     // g(q) = [x_r - x_c, y_r - y_c, z_r - z_c - Δz] = 0
				     Eigen::Vector3<Expression> g;
				     g << (X_WR.translation().x() - X_WC.translation().x() - x_offset),
					     (X_WR.translation().y() - X_WC.translation().y() - y_offset),
					     (X_WR.translation().z() - X_WC.translation().z() - delta_z);

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
				     return r.norm();
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
				     prog.AddLinearEqualityConstraint(p_WP - p_WE, disp);
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
				     Vector3d p_WE;
				     Matrix3d R_WE;
				     {
					     // robot pose at node k
					     const int robot_offset = robot_start;  // already captured from outer scope
					     PoseFromRow_FreeBody<double>(x, robot_offset, &p_WE, &R_WE);
				     }

				     const Vector3d p_WP = x.segment(objs_start + point_id * non_robot_dim, 3);

				     // --- Build r, d ---
				     const Vector3d r = R_WE * ee_ray_body;    // body ray in world
				     const Vector3d d = p_WP - p_WE;           // displacement to target

				     std::cout << "r:\n" << r << std::endl;

				     std::cout << "d:\n" << d << std::endl;

				     double residual = 0.0;

				     // (1) Point-at: r × d = 0  → use squared norm
				     const Vector3d rc = r.cross(d);

				     std::cout << "rc:\n" << rc << std::endl;

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
				     Eigen::RowVectorX<Expression> row = AsExprRow(X.row(node_k));

				     Eigen::Matrix<Expression,3,1> p_WE;
				     Eigen::Matrix<Expression,3,3> R_WE;
				     PoseFromRow_FreeBody<Expression>(row, robot_start, &p_WE, &R_WE);

				     const Eigen::Matrix<Expression,3,1> p_WP =
					     PointWorldFromRow(row, objs_start, non_robot_dim, point_id);

				     // r = R * v_b, d = P - E
				     Eigen::Matrix<Expression,3,1> r = R_WE * ee_ray_body;
				     Eigen::Matrix<Expression,3,1> d = p_WP - p_WE;

				     // (1) Point-at: r × d = 0
				     auto rc = r.cross(d);
				     for (int i = 0; i < 3; ++i) prog.AddConstraint(rc(i) == 0);

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

					     prog.AddQuadraticConstraint(u(2), -tol, tol);
				     }
			     });

	// record that this constraint is statically assigned to this robot.
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
			       std::cout << "pA: " << pA << std::endl;
			       std::cout << "pB: " << pB << std::endl;
			       std::cout << "r: " << r << std::endl;
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
			       prog.AddLinearConstraint((pB - pA) - disp, lb, ub);

			       // Enforce pB - pA = disp  (3 scalar equalities)
			       // prog.AddLinearEqualityConstraint(pB - pA, disp);
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

// EDGE-CONSTRAINTS

int GraphOfConstraints::add_robot_holding_cube_constraint(
	int u,
	int v,
	int robot_id,
	int cube_id,
	double holding_distance_max) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	// If you track num_objects, you can also check cube_i bounds here.

	const std::string& robot_model_name = _robot_names.at(robot_id);
	const std::string& cube_model_name = _object_names.at(cube_id);

	const ModelInstanceIndex robot_mi = _plant->GetModelInstanceByName(robot_model_name);
	const ModelInstanceIndex cube_mi  = _plant->GetModelInstanceByName(cube_model_name);

	int edge_phi_id = _add_edge_op(DeferredOpKind::kNonlinearEq, u, v, std::set<int>({cube_id}),
			    [=, this](const Eigen::VectorXd& x,
				      const Eigen::VectorXi&/*unused*/) {
				    using drake::math::RigidTransform;
				    using drake::symbolic::Expression;
				    using drake::symbolic::Evaluate;

				    const auto& robot_body = _plant->GetBodyByName("ee_link", robot_mi);
				    const auto& cube_body  = _plant->GetBodyByName("cb_body",  cube_mi);

				    Eigen::VectorX<Expression> q_all = x.cast<Expression>();

				    auto context = _plant->CreateDefaultContext();
				    set_configuration(context, q_all);

				    const RigidTransform<Expression> X_WR =
					    _plant->EvalBodyPoseInWorld(*context, robot_body);
				    const RigidTransform<Expression> X_WC =
					    _plant->EvalBodyPoseInWorld(*context, cube_body);

				    const Eigen::Vector3<Expression> dp_expr =
					    X_WR.translation() - X_WC.translation();

				    // dp_expr has no free symbols (only constants), so Evaluate(...) -> double works.
				    double violation = 0.0;
				    for (int i = 0; i < 3; ++i) {
					    const double dpi = dp_expr[i].Evaluate();
					    violation = std::max(violation, std::abs(dpi) - holding_distance_max);
				    }
				    return violation;
			    },
			    [=, this](drake::solvers::MathematicalProgram& prog,
				      const SubgraphOfConstraints& subgraph,
				      const int phi_id,
				      const drake::solvers::MatrixXDecisionVariable& X,
				      const drake::solvers::MatrixXDecisionVariable& /*unused*/) {
				    using drake::math::RigidTransform;
				    using drake::symbolic::Expression;

				    const auto& robot_body = _plant->GetBodyByName("ee_link", robot_mi);
				    const auto& cube_body  = _plant->GetBodyByName("cb_body",  cube_mi);

				    Eigen::VectorX<Expression> q_all(total_dim);

				    const double d = holding_distance_max;
				    auto add_box_proximity = [&](int graph_row) {
					    for (int j = 0; j < total_dim; ++j) q_all(j) = Expression(X(graph_row, j));

					    auto context = _plant->CreateDefaultContext();
					    set_configuration(context, q_all);

					    const RigidTransform<Expression> X_WR =
						    _plant->EvalBodyPoseInWorld(*context, robot_body);
					    const RigidTransform<Expression> X_WC =
						    _plant->EvalBodyPoseInWorld(*context, cube_body);

					    const Eigen::Vector3<Expression> dp = X_WR.translation() - X_WC.translation();

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

int GraphOfConstraints::add_assignable_robot_holding_point_constraint(
	int u,
	int v,
	int var,
	int point_id,
	double holding_distance_max) {

	DRAKE_DEMAND(u >= 0 && u < structure.num_nodes());
	DRAKE_DEMAND(v >= 0 && v < structure.num_nodes());
	DRAKE_DEMAND(var >= 0 && var < num_variables);
	DRAKE_DEMAND(point_id >= 0 && point_id < num_objects);

	return _add_assignable_edge_op(
		DeferredOpKind::kAgentLinearEq, u, v, var, std::set<int>({point_id}),
		[=, this](const Eigen::VectorXd& x,
			  const Eigen::VectorXi&/*unused*/) {
			return 0;
		},
		[=, this](drake::solvers::MathematicalProgram& prog,
			  const SubgraphOfConstraints& subgraph,
			  const int phi_id,
			  const drake::solvers::MatrixXDecisionVariable& X,
			  const drake::solvers::MatrixXDecisionVariable& /*unused*/) {
			return;
		},
		[](drake::solvers::MathematicalProgram& prog,
		   const int phi_id,
		   const Eigen::VectorXi& var_assignments,
		   const drake::solvers::MatrixXDecisionVariable& Xi) {
			return;
		});
}
