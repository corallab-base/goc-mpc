#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "waypoint_mpc.hpp"
#include "timing_mpc.hpp"

namespace py = pybind11;

/*
 * PYBIND11 MODULE
 */

void init_submodule_sec_mpc(py::module_& m) {
        py::module_ sec_mpc = m.def_submodule("sec_mpc", "SeC-MPC module.");
        py::class_<WaypointMPC>(sec_mpc, "WaypointMPC")
                .def(py::init<const py::array_t<double> &>());
        py::class_<TimingMPC>(sec_mpc, "TimingMPC")
                .def(py::init<const py::array_t<double> &, double, double>())
		.def("solve", &TimingMPC::solve)
		.def("get_waypoints", &TimingMPC::get_waypoints)
		.def("get_vels", &TimingMPC::get_vels)
		.def("get_times", &TimingMPC::get_times)
		.def("get_time_deltas", &TimingMPC::get_time_deltas)
		.def("fill_cubic_spline", &TimingMPC::fill_cubic_spline);
}
