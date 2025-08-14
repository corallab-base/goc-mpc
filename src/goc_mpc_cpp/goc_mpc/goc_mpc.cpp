#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "graph_waypoint_mpc.hpp"
#include "graph_timing_mpc.hpp"
#include "graph_short_path_mpc.hpp"

namespace py = pybind11;

/*
 * PYBIND11 MODULE
 */

void init_submodule_goc_mpc(py::module_& m) {
        py::module_ goc_mpc = m.def_submodule("goc_mpc", "GoC-MPC module.");
        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init<unsigned int, unsigned int, double, double>())
		.def("solve", &GraphTimingMPC::solve);

	// .def("get_ordering", &GraphTimingMPC::get_ordering)
	// .def("get_waypoints", &GraphTimingMPC::get_waypoints)
	// .def("get_vels", &GraphTimingMPC::get_vels)
	// .def("get_times", &GraphTimingMPC::get_times)
	// .def("fill_cubic_spline", &GraphTimingMPC::fill_cubic_spline);

        py::class_<GraphWaypointMPC>(goc_mpc, "GraphWaypointMPC")
                .def(py::init<const Eigen::MatrixXi&, unsigned int, unsigned int, const Eigen::VectorXd&, const Eigen::VectorXd&>())
		.def("solve", &GraphWaypointMPC::solve)
		.def("add_linear_eq", &GraphWaypointMPC::add_linear_eq)
		.def("add_assignable_linear_eq", &GraphWaypointMPC::add_assignable_linear_eq);

        py::class_<GraphShortPathMPC>(goc_mpc, "GraphShortPathMPC")
                .def(py::init<unsigned int, unsigned int, double>(),
		     py::arg("steps"), py::arg("dim"), py::arg("time_per_step"))
		.def("solve", &GraphShortPathMPC::solve)
		.def("get_points", &GraphShortPathMPC::get_points)
		.def("get_vels", &GraphShortPathMPC::get_vels)
		.def("get_times", &GraphShortPathMPC::get_times);
}
