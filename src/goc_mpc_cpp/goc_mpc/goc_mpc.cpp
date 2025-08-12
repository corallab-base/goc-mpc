#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_waypoint_mpc.hpp"
#include "graph_timing_mpc.hpp"

namespace py = pybind11;

/*
 * PYBIND11 MODULE
 */

void init_submodule_goc_mpc(py::module_& m) {
        py::module_ goc_mpc = m.def_submodule("goc_mpc", "GoC-MPC module.");
        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init<const py::array_t<double> &, const py::array_t<double> &, double, double>())
		.def("solve", &GraphTimingMPC::solve)
		.def("get_ordering", &GraphTimingMPC::get_ordering)
		.def("get_waypoints", &GraphTimingMPC::get_waypoints)
		.def("get_vels", &GraphTimingMPC::get_vels)
		.def("get_times", &GraphTimingMPC::get_times)
		.def("fill_cubic_spline", &GraphTimingMPC::fill_cubic_spline);
        py::class_<GraphWaypointMPC>(goc_mpc, "GraphWaypointMPC")
                .def(py::init<const py::array_t<double>&, unsigned int, unsigned int, unsigned int>())
		.def("solve", &GraphWaypointMPC::solve)
		.def("get_waypoints", &GraphWaypointMPC::get_waypoints);
}
