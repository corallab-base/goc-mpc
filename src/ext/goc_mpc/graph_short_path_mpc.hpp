#pragma once

#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "obstacle_set.hpp"
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

	// Obstacle geometry, stored BY POINTER (not an owned copy) -- mirrors
	// `_graph` above. The caller (see ObstacleSet's own doc comment) keeps
	// the real ObstacleSet alive and can keep registering/updating
	// obstacles on it over the episode; every solve() call reads whatever
	// is currently in it, not a construction-time snapshot. Not yet read
	// anywhere in build_short_path_problem -- a later stage adds the
	// cost/constraint that consumes it.
	const ObstacleSet* _obstacles;

	// Outputs
	Eigen::MatrixXd _points;
	Eigen::MatrixXd _vels;

	// Recording Metrics
	drake::SteadyTimer _timer;
	double _last_solve_time;

	// Constructor. No default value for `obstacles` -- storing a pointer to
	// a default-constructed temporary argument would dangle the instant the
	// constructor call's full expression ends; callers must pass a real,
	// long-lived ObstacleSet (an empty one is fine).
	GraphShortPathMPC(const GraphOfConstraints& graph,
			  unsigned int num_steps,
			  unsigned int num_agents,
			  unsigned int dim,
			  double time_per_step,
			  const ObstacleSet& obstacles);

	// Core solve routine
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const Eigen::VectorXi& var_assignments,
		   const std::vector<int>& remaining_vertices,
		   const std::vector<CubicConfigurationSpline>& references);

	const Eigen::MatrixXd &view_points() { return _points; }
	const Eigen::MatrixXd &view_vels() { return _vels; }
	const Eigen::VectorXd &view_times() { return _times; }
	const ObstacleSet &view_obstacles() { return *_obstacles; }
	const double get_last_solve_time() { return _last_solve_time; }
};
