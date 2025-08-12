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

#include "graph_waypoint_mpc_problem.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphWaypointMPC {
	// Inputs: _graph (adjacency matrix) encoding ordering constraints.
	// _m number of agents
	// _z number of assignments
	py::array_t<unsigned int> _graph;
	unsigned int _m, _z;
	unsigned int _n, _d;

	// Outputs
	py::array_t<unsigned int> _assignments;
	py::array_t<double> _waypoints;

	// Phase management
//	std::set<int> _completed_phases;
// 	py::array_t<unsigned int> back_tracking_table;
// 	bool never_done = false;

	// Constructor
	GraphWaypointMPC(const py::array_t<double>& graph,
			 unsigned int m,
			 unsigned int z,
			 unsigned int d);

	// Core solve routine
	int solve();

	// Safe indexing and accessors
	py::array_t<double> get_waypoints() const;
};
