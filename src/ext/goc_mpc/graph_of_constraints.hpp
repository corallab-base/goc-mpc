#pragma once

#include <iostream>

#include <fmt/format.h>

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
	kNonlinearCost,
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
			   const drake::solvers::MatrixXDecisionVariable&,
			   const Eigen::VectorXd&)> waypoint_builder;
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
	std::map<int, std::vector<int>> node_to_phis_map;
	std::map<std::pair<int, int>, std::vector<int>> edge_to_phis_map;
	std::set<int> unpassable_nodes;

	// Node Phi maps
	std::map<int, int> phi_to_variable_map;
	std::map<int, int> _phi_to_static_assignment_map;
	std::map<int, struct DeferredOp> ops;
	std::map<int, std::vector<std::tuple<std::string, std::string, std::string>>> _grasp_change_map;
	std::map<int, std::vector<std::pair<std::string, std::string>>> _assignable_grasp_change_map;

	// Edge phi maps
	std::map<int, int> edge_phi_to_variable_map;
	std::map<int, struct DeferredEdgeOp> edge_ops;
	std::map<int, int> _edge_phi_to_static_assignment_map;

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

	bool robot_is_free_body(int ag) const;

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

	int get_edge_phi_agent(int phi_id, const Eigen::VectorXi& var_assignments) const;

	// Grasp change util
	void add_grasp_change(int phi_id, std::string command, int robot_id, int cube_id);
	void add_assignable_grasp_change(int phi_id, std::string command, int cube_id);
	std::vector<std::tuple<std::string, std::string, std::string>> get_grasp_changes(int k, Eigen::VectorXi assignments) const;

	// Unpassable node util
	void make_node_unpassable(int k);
	
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
						       Eigen::Vector3d& disp,
						       double tol = 0.0);
	int add_robot_to_point_displacement_cost(int k,
						 int robot_id,
						 int point_id,
						 Eigen::Vector3d& disp);

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

	int add_robot_to_point_alignment_cost(int k, int robot_id, int point_id,
					      const Eigen::Vector3d& ee_ray_body,
					      std::optional<Eigen::Vector3d> u_body_opt,
					      std::optional<Eigen::Vector3d> roll_ref_world,
					      bool roll_ref_flat,
					      bool require_positive_pointing,
					      double w_point=1.0,
					      double w_roll=0.1,
					      double w_flat=0.05,
					      double w_guard=0.0,
					      double w_u_stab=0.01,
					      double eps=1e-10,
					      double eps_d=1e-3);

	int add_point_to_point_displacement_constraint(int k,
						       int point_a,
						       int point_b,
						       Eigen::Vector3d& disp,
						       double tol = 0.05);
	int add_point_to_point_displacement_cost(int k,
						 int point_a,
						 int point_b,
						 Eigen::Vector3d& disp);

	int add_point_to_point_alignment_constraint(int k,
						    int point_a,
						    int point_b,
						    const Eigen::Vector3d& dir_W);

	// Edge Constraints

	int add_robot_holding_cube_constraint(int u,
					      int v,
					      int robot_id,
					      int cube_id,
					      double holding_distance_max = 0.1,
					      bool use_l2 = false);

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

	int add_assignable_robot_holding_point_constraint(int u,
							  int v,
							  int var,
							  int point_id,
							  double holding_distance_max = 0.1);

	template <typename T>
	void set_configuration(
		std::unique_ptr<drake::systems::Context<T>>& context,
		const Eigen::VectorX<T>& q_all) const;

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
			if (graph->node_to_phis_map.contains(u)) {
				for (int phi_id : graph->node_to_phis_map.at(u)) {
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
		}

		for (const auto& edge : graph->structure.edges()) {
			int u = edge.u;
			int v = edge.e->to;
			if (structure.contains_node(u) || structure.contains_node(v)) {
				std::pair<int, int> e = std::make_pair(u, v);
				if (graph->edge_to_phis_map.contains(e)) {
					// Store the relevant edge ops so they can be applied
					for (int edge_phi_id : graph->edge_to_phis_map.at(e)) {
						_subgraph_edge_ops[edge_phi_id] = graph->edge_ops.at(edge_phi_id);
					}
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
