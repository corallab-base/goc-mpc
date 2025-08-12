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


struct GraphOrderingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable p;

	GraphOrderingProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	GraphOrderingProblem(const GraphOrderingProblem&) = delete;
	GraphOrderingProblem& operator=(const GraphOrderingProblem&) = delete;

	GraphOrderingProblem(GraphOrderingProblem&&) = default;
	GraphOrderingProblem& operator=(GraphOrderingProblem&&) = default;
};


GraphOrderingProblem build_graph_ordering_problem(
	const py::array_t<double>& waypoints,
	const py::array_t<unsigned int>& graph,
	const py::array_t<double>& x0_np,
	const py::array_t<double>& v0_np);
