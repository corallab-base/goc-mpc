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

#include "graph_ordering_mpc_problem.hpp"
#include "../sec_mpc/timing_mpc_problem.hpp"
#include "../splines.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphTimingMPC {
	// Inputs, waypoints and graph (adjacency matrix) encoding ordering constraints.
	py::array_t<double> _waypoints;
	py::array_t<unsigned int> _graph;

	// Outputs
	py::array_t<unsigned int> _ordering;
	py::array_t<double> _time_deltas;
	py::array_t<double> _vels;

	// Optimization parameters
	double _time_cost;
	double _ctrl_cost;

	// Phase management
	std::set<int> _completed_phases;
// 	py::array_t<unsigned int> back_tracking_table;

	unsigned int _n, _d;

// 	bool never_done = false;

	// Constructor
	GraphTimingMPC(const py::array_t<double>& waypoints,
		       const py::array_t<double>& graph,
		       double time_cost = 1e0,
		       double ctrl_cost = 1e0);

	// Core solve routine
	int solve(const py::array_t<double>& x0,
		  const py::array_t<double>& v0,
		  int verbose = 1);

	// Phase tracking
	bool done() const;

	// Safe indexing and accessors
	py::array_t<unsigned int> get_ordering() const;
	py::array_t<double> get_waypoints() const;
// 	py::array_t<double> get_time_deltas() const;
	py::array_t<double> get_times() const;
	py::array_t<double> get_vels() const;

// 	// State updates
// 	// bool set_progressed_time(double gap, double tau_cutoff = 0.0);
// 	// void set_updated_waypoints(const py::array_t<double>& _waypoints,
// 	// 			   bool set_next_waypoint_tangent);

// 	void update_backtrack();
// 	void update_set_phase(unsigned int phase_to);

	// Spline generator
	void fill_cubic_spline(CubicSpline& S,
			       const py::array_t<double>& x0,
			       const py::array_t<double>& v0) const;

private:
	py::array_t<double> _remaining_waypoints() const;
	py::array_t<double> _remaining_graph() const;
};
