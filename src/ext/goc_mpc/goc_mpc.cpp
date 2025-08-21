#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>

#include "graph_of_constraints.hpp"
#include "graph_waypoint_mpc.hpp"
#include "graph_timing_mpc.hpp"
#include "graph_short_path_mpc.hpp"

using drake::symbolic::Expression;
using drake::multibody::MultibodyPlant;
namespace py = pybind11;

/*
 * PYBIND11 MODULE
 */

void init_submodule_goc_mpc(py::module_& m) {
        py::module_ goc_mpc = m.def_submodule("goc_mpc", "GoC-MPC module.");

	py::class_<GraphOfConstraints>(goc_mpc, "GraphOfConstraints")
		.def(py::init<const MultibodyPlant<Expression>*,
		     const std::vector<std::string>,
		     const std::vector<std::string>,
		     const Eigen::VectorXd&,
		     const Eigen::VectorXd&>())
		.def_readonly("structure", &GraphOfConstraints::structure)
		.def_readonly("num_agents", &GraphOfConstraints::num_agents)
		.def_readonly("num_objects", &GraphOfConstraints::num_objects)
		.def_readonly("dim", &GraphOfConstraints::dim)
		.def_readonly("non_robot_dim", &GraphOfConstraints::non_robot_dim)
		.def_readonly("total_dim", &GraphOfConstraints::total_dim)
		.def("add_variable", &GraphOfConstraints::add_variable)
		.def("add_grasp_change", &GraphOfConstraints::add_grasp_change)
		.def("add_assignable_grasp_change", &GraphOfConstraints::add_assignable_grasp_change)
		.def("get_grasp_changes", &GraphOfConstraints::get_grasp_changes)
		.def("get_phi_ids", &GraphOfConstraints::get_phi_ids)
		.def("evaluate_phi", &GraphOfConstraints::evaluate_phi)
		.def("add_linear_eq", &GraphOfConstraints::add_linear_eq)
		.def("add_assignable_linear_eq", &GraphOfConstraints::add_assignable_linear_eq)
		.def("add_robot_above_cube_constraint", &GraphOfConstraints::add_robot_above_cube_constraint);

        py::class_<GraphWaypointMPC>(goc_mpc, "GraphWaypointMPC")
                .def(py::init<GraphOfConstraints&>())
		.def("solve", &GraphWaypointMPC::solve)
		.def("view_waypoints", &GraphWaypointMPC::view_waypoints, py::return_value_policy::reference_internal)
		.def("view_assignments", &GraphWaypointMPC::view_assignments, py::return_value_policy::reference_internal);

        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init<const GraphOfConstraints&, double, double, double, double, double>())
		.def("solve", &GraphTimingMPC::solve)
		.def("get_agent_spline_length", &GraphTimingMPC::get_agent_spline_length)
		.def("get_agent_spline_nodes", &GraphTimingMPC::get_agent_spline_nodes)
		.def("set_progressed_time", &GraphTimingMPC::set_progressed_time)
		.def("fill_cubic_splines", &GraphTimingMPC::fill_cubic_splines);

        py::class_<GraphShortPathMPC>(goc_mpc, "GraphShortPathMPC")
                .def(py::init<unsigned int, unsigned int, unsigned int, double>(),
		     py::arg("num_steps"), py::arg("num_agents"), py::arg("dim"), py::arg("time_per_step"))
		.def("solve", &GraphShortPathMPC::solve)
		.def("view_points", &GraphShortPathMPC::view_points, py::return_value_policy::reference_internal)
		.def("view_vels", &GraphShortPathMPC::view_vels, py::return_value_policy::reference_internal)
		.def("view_times", &GraphShortPathMPC::view_times, py::return_value_policy::reference_internal);
}
