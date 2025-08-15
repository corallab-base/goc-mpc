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

#include "../splines.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct ShortPathProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable xi;
	drake::solvers::MatrixXDecisionVariable v;

	ShortPathProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	ShortPathProblem(const ShortPathProblem&) = delete;
	ShortPathProblem& operator=(const ShortPathProblem&) = delete;

	ShortPathProblem(ShortPathProblem&&) = default;
	ShortPathProblem& operator=(ShortPathProblem&&) = default;
};


ShortPathProblem build_short_path_problem(
	const py::array_t<double>& ref_points_np,
	const py::array_t<double>& ref_velocities_np,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double tau);


struct GraphShortPathMPC {
	// Inputs, number of steps, dimension, reference traj
	unsigned int _num_steps, _dim;
	double _time_per_step;
	py::array_t<double> _times;

	// Outputs
	py::array_t<double> _points;
	py::array_t<double> _vels;

	// Constructor
	GraphShortPathMPC(unsigned int num_steps,
			  unsigned int dim,
			  double time_per_step);

	// Core solve routine
	int solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& v0, const CubicSpline& reference);

	py::array_t<double> get_points() const;
	py::array_t<double> get_times() const;
	py::array_t<double> get_vels() const;

};
