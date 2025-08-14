#pragma once

#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include "drake/solvers/solve.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphWaypointProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable Assignments;
	drake::solvers::MatrixXDecisionVariable X;

	GraphWaypointProblem()
		: prog(nullptr) {}

	GraphWaypointProblem(const GraphWaypointProblem&) = delete;
	GraphWaypointProblem& operator=(const GraphWaypointProblem&) = delete;

	GraphWaypointProblem(GraphWaypointProblem&&) = default;
	GraphWaypointProblem& operator=(GraphWaypointProblem&&) = default;
};

GraphWaypointProblem build_graph_waypoint_problem(const py::array_t<unsigned int>& graph,
						  int num_assignables, int num_agents,
						  int num_nodes, int dim);

enum class DeferredOpKind {
	kLinearEq,
	kLinearIneq,
	kBoundingBox,
	kQuadraticCost,
	kNonlinearConstraint,
	kOther,
	// MultiAgent
	kAgentLinearEq,
};

struct DeferredOp {
	DeferredOpKind kind;
	unsigned int id;
	unsigned int node;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&)> builder;
};

struct GraphWaypointMPC {
	// Inputs: _graph (adjacency matrix) encoding ordering constraints.
	// _m number of agents
	// _z number of assignments
	Eigen::MatrixXi _graph;
	unsigned int _num_phis;
	unsigned int _num_agents, _num_total_assignables;
	unsigned int _num_total_nodes, _dim;

	// Necessary maps for properly evaluating original set of phi's (defined
	// over _graph) with the subgraph.
	std::map<unsigned int, unsigned int> _graph_to_phi_map;
	std::map<unsigned int, unsigned int> _phi_to_subgraph_node_id;
	std::map<unsigned int, unsigned int> _phi_to_subgraph_assignable_id;

	// Required for big-M computation
	Eigen::VectorXd _global_x_lb;
	Eigen::VectorXd _global_x_ub;

	// Phase management
//	std::set<int> _completed_phases;
// 	py::array_t<unsigned int> back_tracking_table;
// 	bool never_done = false;

	// Constructor
	GraphWaypointMPC(const Eigen::MatrixXi& graph,
			 unsigned int num_agents,
			 unsigned int dim,
			 const Eigen::VectorXd& global_x_lb,
			 const Eigen::VectorXd& global_x_ub);

	// Core solve routine
	std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXi>> solve(
		const std::set<unsigned int>& remaining_vertices,
		const Eigen::VectorXd& x0);

	// Plain Constraint Adders (typed)
	// Note: these copy the numpy array's passed to them, but they're called
	// once so it's fine.

	// lb <= x <= ub on node k
	void add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	// Ax = b on node k
	void add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	// lb <= A x <= ub on node k
	void add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	// 0.5 x'Qx + b'x + c on node k
	void add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c = 0.0);

	// Multi-Agent Constraint Adders (typed)

	// Ax_i = b on node k for some agent i
	void add_assignable_linear_eq(int node_k,
				      const Eigen::MatrixXd& A,
				      const Eigen::VectorXd& b);

private:
	template <typename F>
	void _add_op(DeferredOpKind kind, unsigned int node, F&& f) {
		const unsigned int id = _num_phis++;
		_graph_to_phi_map[node] = id;
		_ops[id] = DeferredOp{kind, id, node, std::forward<F>(f)};
	}

	std::map<unsigned int, struct DeferredOp> _ops;
};
