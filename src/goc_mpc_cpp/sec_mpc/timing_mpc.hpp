/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include "drake/solvers/solve.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "timing_mpc_problem.hpp"
#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


// A wrapper of TimingOpt optimize the timing (and vels) along given waypoints,
// and progressing / backtracking the phase
struct TimingMPC {
	// Inputs
	py::array_t<double> waypoints;
	py::array_t<double> tangents;

	// Outputs
	py::array_t<double> time_deltas;
	py::array_t<double> vels;

	// Optimization parameters
	double time_cost;
	double ctrl_cost;

	// Tangent options
	bool use_next_waypoint_tangent = true;

	// Phase management
	unsigned int phase = 0;
	py::array_t<unsigned int> back_tracking_table;

	unsigned int _n, _d;

	bool never_done = false;

	// Constructor
	TimingMPC(const py::array_t<double>& _waypoints,
		  double _timeCost = 1e0,
		  double _ctrlCost = 1e0);

	// Core solve routine
	int solve(const py::array_t<double>& x0,
		  const py::array_t<double>& v0,
		  int verbose = 1);

	// Phase tracking
	unsigned int n_phases() const { return waypoints.shape(0); }
	bool done() const { return phase >= n_phases(); }

	// Safe indexing and accessors
	py::array_t<double> get_waypoints() const;
	py::array_t<double> get_time_deltas() const;
	py::array_t<double> get_times() const;
	py::array_t<double> get_vels() const;

	// State updates
	// bool set_progressed_time(double gap, double tau_cutoff = 0.0);
	// void set_updated_waypoints(const py::array_t<double>& _waypoints,
	// 			   bool set_next_waypoint_tangent);

	void update_backtrack();
	void update_set_phase(unsigned int phase_to);

	// Spline generator
	void fill_cubic_spline(CubicSpline& S,
			       const py::array_t<double>& x0,
			       const py::array_t<double>& v0) const;

private:
	py::array_t<double> _remaining_time_deltas() const;
};
