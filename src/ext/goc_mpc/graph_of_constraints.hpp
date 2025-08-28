#pragma once

#include <iostream>

#include <drake/common/symbolic/expression.h>
#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>
#include <drake/math/quaternion.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "../graphs.hpp"
#include "../utils.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::symbolic::Expression;
using drake::multibody::MultibodyPlant;
using drake::multibody::ModelInstanceIndex;
using namespace pybind11::literals;
namespace py = pybind11;


struct SubgraphOfConstraints;

enum class DeferredOpKind {
	kLinearEq,
	kLinearIneq,
	kBoundingBox,
	kQuadraticCost,
	kNonlinearEq,
	kOther,
	// MultiAgent
	kAgentLinearEq,
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
			   const drake::solvers::MatrixXDecisionVariable&)> waypoint_builder;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const int,
			   const Eigen::VectorXi&,
			   const drake::solvers::MatrixXDecisionVariable&)> short_path_builder;
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

struct GraphOfConstraints {

	std::shared_ptr<MultibodyPlant<Expression>> _plant;
	const std::vector<std::string> _robot_names;
	const std::vector<std::string> _object_names;
	Graph<py::object> structure;
	std::map<int, int> node_to_phi_map;
	std::map<std::pair<int, int>, int> edge_to_phi_map;

	// Node Phi maps
	std::map<int, int> phi_to_variable_map;
	std::map<int, int> _phi_to_static_assignment_map;
	std::map<int, struct DeferredOp> ops;
	std::map<int, std::tuple<std::string, std::string, std::string>> _grasp_change_map;
	std::map<int, std::pair<std::string, std::string>> _assignable_grasp_change_map;

	// Edge phi maps
	// std::map<int, int> edge_phi_to_variable_map;
	std::map<int, struct DeferredEdgeOp> edge_ops;

	// Rest
	int num_phis, num_edge_phis, num_variables, _num_total_assignables;
	int num_agents, num_objects, dim, non_robot_dim, total_dim;
	
	// Required for big-M computation
	Eigen::VectorXd _global_x_lb;
	Eigen::VectorXd _global_x_ub;

	// Constructor
	// GraphOfConstraints(unsigned int num_agents, unsigned int dim,
	// 		   const Eigen::VectorXd& global_x_lb,
	// 		   const Eigen::VectorXd& global_x_ub);

	GraphOfConstraints(MultibodyPlant<Expression>& plant,
			   const std::vector<std::string> robots,
			   const std::vector<std::string> objects,
			   double global_x_lb,
			   double global_x_ub);

	int add_variable();

	Graph<py::object> get_structure() const { return structure; }

	std::tuple<std::vector<std::optional<int>>,
		   std::vector<std::vector<int>>,
		   std::vector<struct AgentInteraction>> get_agent_paths(
			   const std::vector<int>& remaining_vertices,
			   const Eigen::VectorXi& assignments) const;

	std::map<int, struct DeferredEdgeOp> get_next_edge_ops(const std::vector<int> completed_vertices) const;

	std::vector<int> get_phi_ids(int node) const;

	bool evaluate_phi(int phi_id,
			  const Eigen::VectorXd& x,
			  const Eigen::VectorXi& assignments,
			  double tol) const;

	bool evaluate_edge_phi(int phi_id,
			       const Eigen::VectorXd& x,
			       const Eigen::VectorXi& var_assignments,
			       double tol) const;

	// Grasp change util
	void add_grasp_change(int phi_id, std::string command, int robot_id, int cube_id);
	void add_assignable_grasp_change(int phi_id, std::string command, int cube_id);
	std::vector<std::tuple<std::string, std::string, std::string>> get_grasp_changes(int k, Eigen::VectorXi assignments) const;
	
	// Plain Constraint Adders (typed)
	// Note: these copy the numpy array's passed to them, but they're called
	// once so it's fine.

	// lb <= x <= ub on node k
	int add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
	// Ax = b on node k
	int add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	// lb <= A x <= ub on node k
	int add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);
	// 0.5 x'Qx + b'x + c on node k
	int add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c = 0.0);

	// Ax = b on node k
	int add_agents_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);
	// lb <= A x <= ub on node k
	int add_agents_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	// Multi-Agent Constraint Adders (typed)

	// Ax_i = b on node k for some agent i
	int add_assignable_linear_eq(int k, int var, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	// Temp Plant-Based Constraint Adders

	// Sort of hard-coded
	int add_robot_above_cube_constraint(int k,
					    int robot_id,
					    int cube_id,
					    double delta_z);

	// EDGE CONSTRAINTS

	int add_robot_holding_cube_constraint(int u,
					      int v,
					      int robot_id,
					      int cube_id,
					      double holding_distance_max = 0.1);



private:
	template <typename T>
	void _set_configuration(
		std::unique_ptr<drake::systems::Context<T>>& context,
		Eigen::VectorX<T>& q_all);

	template <typename EF, typename F>
	int _add_op(DeferredOpKind kind, int node, EF&& eval_f, F&& f) {
		const int id = num_phis++;
		node_to_phi_map[node] = id;
		ops[id] = DeferredOp{kind, id, node, std::forward<EF>(eval_f), std::forward<F>(f)};
		return id;
	}

	template <typename EF, typename F>
	int _add_assignable_op(DeferredOpKind kind, int node, int var, EF&& eval_f, F&& f) {
		const int id = num_phis++;
		node_to_phi_map[node] = id;
		phi_to_variable_map[id] = var;
		ops[id] = DeferredOp{kind, id, node, std::forward<EF>(eval_f), std::forward<F>(f)};
		return id;
	}

	template <typename EF, typename WF, typename SF>
	int _add_edge_op(DeferredOpKind kind, int u, int v, std::set<int> cubes, EF&& eval_f, WF&& wp_f, SF&& sp_f) {
		const int id = num_edge_phis++;
		edge_to_phi_map[std::make_pair(u, v)] = id;
		edge_ops[id] = DeferredEdgeOp{kind, id, u, v, cubes,
			std::forward<EF>(eval_f), std::forward<WF>(wp_f), std::forward<SF>(sp_f)};
		return id;
	}

	// template <typename F>
	// int _add_assignable_edge_op(DeferredOpKind kind, int u, int v, int var, F&& f) {
	// 	const int id = num_edge_phis++;
	// 	edge_to_phi_map[std::make_pair(u, v)] = id;
	// 	phi_to_variable_map[id] = var;
	// 	edge_ops[id] = DeferredEdgeOp{kind, id, u, v, std::forward<F>(f)};
	// 	return id;
	// }
};

/*
 * Subgraph
 */

struct SubgraphOfConstraints {
	InducedSubgraphView<py::object> structure;
	std::map<int, int> _variable_to_subgraph_variable_id; // unique ids for variables relevant to subgraph.
	std::map<int, DeferredOp> _subgraph_ops;
	std::map<int, DeferredEdgeOp> _subgraph_edge_ops;

	SubgraphOfConstraints(GraphOfConstraints *graph, const std::vector<int>& vertices) :
		structure(graph->structure, vertices) {

		// std::map<int, int> phi_to_subgraph_node_id;
		// std::map<int, int> phi_to_subgraph_assignable_id;
		// std::map<int, int> subgraph_assignable_id_to_phi;

		int num_subgraph_variables = 0;
		
		for (int u : vertices) {
			// if there is/are phi associated with v
			if (graph->node_to_phi_map.contains(u)) {
				// Store the relevant ops so they can be applied
				const int phi_id = graph->node_to_phi_map.at(u);
				_subgraph_ops[phi_id] = graph->ops.at(phi_id);

				// Record the mapping from phi id to subgraph node and assignable var idxs.
				if (graph->phi_to_variable_map.contains(phi_id)) {
					const int variable_id = graph->phi_to_variable_map.at(phi_id);
					
					if (!_variable_to_subgraph_variable_id.contains(variable_id)) {
						_variable_to_subgraph_variable_id[variable_id] = num_subgraph_variables++;
					}
					
				}
			}
		}

		for (const auto& edge : graph->structure.edges()) {
			int u = edge.u;
			int v = edge.e->to;
			if (structure.contains_node(u) || structure.contains_node(v)) {
				std::pair<int, int> e = std::make_pair(u, v);
				if (graph->edge_to_phi_map.contains(e)) {
					// Store the relevant edge ops so they can be applied
					const int edge_phi_id = graph->edge_to_phi_map.at(e);
					_subgraph_edge_ops[edge_phi_id] = graph->edge_ops.at(edge_phi_id);
				}
			}
		}
	}

	const std::map<int, DeferredOp>& get_subgraph_ops() const {
		return _subgraph_ops;
	}

	const std::map<int, DeferredEdgeOp>& get_subgraph_edge_ops() const {
		return _subgraph_edge_ops;
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
