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

#include "../sec_mpc/timing_mpc_problem.hpp"
#include "../splines.hpp"
#include "../utils.hpp"

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
	const Eigen::MatrixXd& waypoints,
	const Eigen::MatrixXi& graph,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0);

struct GraphTimingMPC {
	// Inputs, waypoints and graph (adjacency matrix) encoding ordering constraints.
	// Eigen::MatrixXd _waypoints;
	// Eigen::MatrixXi _graph;
	unsigned int _num_agents, _dim;

	// Outputs
	// py::array_t<unsigned int> _ordering;
	// py::array_t<double> _time_deltas;
	// py::array_t<double> _vels;

	// Optimization parameters
	double _time_cost;
	double _ctrl_cost;

	// Phase management
	// std::set<int> _completed_phases;
	// Eigen::VectorXd _in_degrees; // in-degrees of remaining active phases
	// py::array_t<unsigned int> back_tracking_table;
	// bool never_done = false;

	// Constructor
	GraphTimingMPC(
		unsigned int num_agents, unsigned int dim,
		double time_cost = 1e0,
		double ctrl_cost = 1e0);

	// Core solve routine
	std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> solve(
		const Eigen::MatrixXi& graph,
		const Eigen::VectorXd& x0,
		const Eigen::VectorXd& v0,
		const Eigen::MatrixXd& waypoints,
		const Eigen::VectorXi& assignments);

	// Phase tracking
	// double current_minimum_time_delta() const;
	// bool done() const;

	// Safe indexing and accessors
	// py::array_t<unsigned int> get_ordering() const;
	// py::array_t<double> get_waypoints() const;
	// py::array_t<double> get_time_deltas() const;
	// py::array_t<double> get_times() const;
	// py::array_t<double> get_vels() const;

	// State updates
	// bool set_progressed_time(double time_delta, double time_delta_cutoff);
	// void set_updated_waypoints(const py::array_t<double>& _waypoints,
	// 			   bool set_next_waypoint_tangent);
	// void update_backtrack();
	// void update_set_phase(unsigned int phase_to);

	// Spline generator
	// void fill_cubic_spline(CubicSpline& S,
	// 		       const py::array_t<double>& x0,
	// 		       const py::array_t<double>& v0) const;
};
