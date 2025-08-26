#pragma once

#include <iostream>
#include <algorithm>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include "drake/solvers/solve.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphWaypointProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::unique_ptr<SubgraphOfConstraints> subgraph;
	drake::solvers::MatrixXDecisionVariable Assignments;
	drake::solvers::MatrixXDecisionVariable X;

	GraphWaypointProblem()
		: prog(nullptr),
		  subgraph(nullptr) {}

	GraphWaypointProblem(const GraphWaypointProblem&) = delete;
	GraphWaypointProblem& operator=(const GraphWaypointProblem&) = delete;

	GraphWaypointProblem(GraphWaypointProblem&&) = default;
	GraphWaypointProblem& operator=(GraphWaypointProblem&&) = default;
};

GraphWaypointProblem build_graph_waypoint_problem(
	const GraphOfConstraints& graph,
	const std::vector<int>& remaining_vertices,
	Eigen::VectorXd x0);

struct GraphWaypointMPC {
	// reference to graph of constraints object.
	GraphOfConstraints* _graph;

	// persistent output buffers;
	// _waypoints is (_graph.num_nodes, _graph.num_agents * _graph.dim)
	Eigen::MatrixXd _waypoints;
	// _assignments is (_graph.num_phis,)
	Eigen::VectorXi _assignments;
	// _var_assignments is (_graph.num_variables,)
	Eigen::VectorXi _var_assignments;

	// Constructor
	GraphWaypointMPC(GraphOfConstraints& graph);

	// std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXi>>

	// Core solve routine, based on the remaining vertices, computes a
	// subgraph of graph of constraints, solves for the optimal agent
	// assignment in that graph based on some heuristics, as well as the
	// sequence of positions for the agents to satisfy the constraints.
	bool solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0);

	const Eigen::MatrixXd &view_waypoints() { return _waypoints; }
	const Eigen::VectorXi &view_assignments() { return _assignments; }
	const Eigen::VectorXi &view_var_assignments() { return _var_assignments; }
};
