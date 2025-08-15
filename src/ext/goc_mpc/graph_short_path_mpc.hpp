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
	const Eigen::VectorXd& ref_points,
	const Eigen::VectorXd& ref_velocities,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double tau);


struct GraphShortPathMPC {
	// Inputs, number of steps, dimension, reference traj
	unsigned int _num_steps, _dim;
	double _time_per_step;
	Eigen::VectorXd _times;

	// Outputs
	Eigen::MatrixXd _points;
	Eigen::MatrixXd _vels;

	// Constructor
	GraphShortPathMPC(unsigned int num_steps,
			  unsigned int dim,
			  double time_per_step);

	// Core solve routine
	int solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& v0, const CubicSpline& reference);

	Eigen::MatrixXd get_points() const;
	Eigen::MatrixXd get_vels() const;
	Eigen::VectorXd get_times() const;
};
