#pragma once

#include <iostream>
#include <algorithm>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solve.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphWaypointProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable A;
	drake::solvers::MatrixXDecisionVariable X;

	GraphWaypointProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	GraphWaypointProblem(const GraphWaypointProblem&) = delete;
	GraphWaypointProblem& operator=(const GraphWaypointProblem&) = delete;

	GraphWaypointProblem(GraphWaypointProblem&&) = default;
	GraphWaypointProblem& operator=(GraphWaypointProblem&&) = default;
};


GraphWaypointProblem build_graph_waypoint_problem(
	const py::array_t<unsigned int>& graph,
	int z, int m, int n, int d);
