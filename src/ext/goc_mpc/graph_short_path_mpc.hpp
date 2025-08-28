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
#include "../configuration_spline.hpp"
#include "../splines.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct ShortPathProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable Xi;
	drake::solvers::MatrixXDecisionVariable V;

	ShortPathProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	ShortPathProblem(const ShortPathProblem&) = delete;
	ShortPathProblem& operator=(const ShortPathProblem&) = delete;

	ShortPathProblem(ShortPathProblem&&) = default;
	ShortPathProblem& operator=(ShortPathProblem&&) = default;
};


ShortPathProblem build_short_path_problem(
	const GraphOfConstraints* graph,
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const Eigen::VectorXi& var_assignments,
	const std::vector<int> remaining_vertices,
	double tau);


struct GraphShortPathMPC {
	// Inputs: graph, number of steps, dimension, reference traj
	const GraphOfConstraints* _graph;
	unsigned int _num_steps, _num_agents, _dim;
	double _time_per_step;
	Eigen::VectorXd _times;

	// Outputs
	Eigen::MatrixXd _points;
	Eigen::MatrixXd _vels;

	// Constructor
	GraphShortPathMPC(const GraphOfConstraints& graph,
			  unsigned int num_steps,
			  unsigned int num_agents,
			  unsigned int dim,
			  double time_per_step);

	// Core solve routine
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const Eigen::VectorXi& var_assignments,
		   const std::vector<int>& remaining_vertices,
		   const std::vector<CubicConfigurationSpline>& references);

	const Eigen::MatrixXd &view_points() { return _points; }
	const Eigen::MatrixXd &view_vels() { return _vels; }
	const Eigen::VectorXd &view_times() { return _times; }
};
