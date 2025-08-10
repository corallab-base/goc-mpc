#pragma once

#include <iostream>
#include <algorithm>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solve.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "timing_mpc_problem.hpp"
#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct TimingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable v;
	drake::solvers::VectorXDecisionVariable time_deltas;

	TimingProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	TimingProblem(const TimingProblem&) = delete;
	TimingProblem& operator=(const TimingProblem&) = delete;

	TimingProblem(TimingProblem&&) = default;
	TimingProblem& operator=(TimingProblem&&) = default;
};


TimingProblem build_timing_problem(
	const py::array_t<double>& waypoints,
	const py::array_t<double>& x0_np,
	const py::array_t<double>& v0_np,
	double time_cost,
	double ctrl_cost,
	bool opt_time_deltas,
	bool opt_last_vel,
	double max_vel,
	double max_acc,
	double max_jer,
	bool acc_cont,
	double time_cost2);
