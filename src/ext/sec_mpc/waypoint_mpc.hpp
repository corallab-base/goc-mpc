/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

// #include <drake/mathematical_program.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

struct WaypointMPC {
	py::array_t<double> qHome;
	uint steps=0;

	//result
	py::array_t<double> path;
	py::array_t<double> tau;
	bool feasible = false;

	WaypointMPC(const py::array_t<double>& qHome);

	void solve(int verbose);
};
