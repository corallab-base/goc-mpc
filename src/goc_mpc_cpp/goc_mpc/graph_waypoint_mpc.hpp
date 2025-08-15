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

#include "graph_of_constraints.hpp"
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

GraphWaypointProblem build_graph_waypoint_problem(
	const GraphOfConstraints& graph,
	const std::vector<size_t>& remaining_vertices);

struct GraphWaypointMPC {
	// Inputs: _graph (adjacency matrix) encoding ordering constraints.
	// _m number of agents
	// _z number of assignments
	const GraphOfConstraints& _graph;

	// Constructor
	GraphWaypointMPC(const GraphOfConstraints& graph);

	// Core solve routine, based on the remaining vertices, computes a
	// subgraph of graph of constraints, solves for the optimal agent
	// assignment in that graph based on some heuristics, as well as the
	// sequence of positions for the agents to satisfy the constraints.
	std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXi>> solve(
		const std::vector<size_t>& remaining_vertices,
		const Eigen::VectorXd& x0);
};
