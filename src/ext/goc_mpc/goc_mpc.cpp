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

	py::class_<DeferredEdgeOp>(goc_mpc, "DeferredEdgeOp")
		.def_readonly("u_node", &DeferredEdgeOp::u_node)
		.def_readonly("v_node", &DeferredEdgeOp::v_node);


	py::class_<GraphOfConstraints>(goc_mpc, "GraphOfConstraints")
		.def(py::init<MultibodyPlant<Expression>&,
		     const std::vector<std::string>,
		     const std::vector<std::string>,
		     double,
		     double>(),
		     py::keep_alive<1, 2>(),
		     py::arg("plant"),
		     py::arg("robot_names"),
		     py::arg("cube_names"),
		     py::arg("state_lower_bound"),
		     py::arg("state_upper_bound"))
		.def_readonly("structure", &GraphOfConstraints::structure)
		.def_readonly("num_agents", &GraphOfConstraints::num_agents)
		.def_readonly("num_objects", &GraphOfConstraints::num_objects)
		.def_readonly("dim", &GraphOfConstraints::dim)
		.def_readonly("non_robot_dim", &GraphOfConstraints::non_robot_dim)
		.def_readonly("total_dim", &GraphOfConstraints::total_dim)
		.def_readonly("unpassable_nodes", &GraphOfConstraints::unpassable_nodes)
		.def("add_variable", &GraphOfConstraints::add_variable)
		.def("add_grasp_change", &GraphOfConstraints::add_grasp_change)
		.def("add_assignable_grasp_change", &GraphOfConstraints::add_assignable_grasp_change)
		.def("get_grasp_changes", &GraphOfConstraints::get_grasp_changes)
		.def("make_node_unpassable", &GraphOfConstraints::make_node_unpassable)
		.def("get_phi_ids", &GraphOfConstraints::get_phi_ids)
		.def("get_next_edge_ops", &GraphOfConstraints::get_next_edge_ops)
		.def("evaluate_phi", &GraphOfConstraints::evaluate_phi)
		.def("evaluate_edge_phi", &GraphOfConstraints::evaluate_edge_phi)
		.def("add_linear_eq", &GraphOfConstraints::add_linear_eq)
		.def("add_robots_linear_eq", &GraphOfConstraints::add_robots_linear_eq)
		.def("add_robot_linear_eq", &GraphOfConstraints::add_robot_linear_eq)
		.def("add_robot_pos_linear_eq", &GraphOfConstraints::add_robot_pos_linear_eq)
		.def("add_robot_quat_linear_eq", &GraphOfConstraints::add_robot_quat_linear_eq)
		.def("add_assignable_linear_eq", &GraphOfConstraints::add_assignable_linear_eq)
		.def("add_assignable_robot_to_point_displacement_constraint", &GraphOfConstraints::add_assignable_robot_to_point_displacement_constraint)
		.def("add_robot_to_point_displacement_constraint", &GraphOfConstraints::add_robot_to_point_displacement_constraint)
		.def("add_robot_to_point_displacement_cost", &GraphOfConstraints::add_robot_to_point_displacement_cost)
		.def("add_robot_to_point_alignment_constraint", &GraphOfConstraints::add_robot_to_point_alignment_constraint,
		     py::arg("k"),
		     py::arg("robot_id"),
		     py::arg("point_id"),
		     py::arg("ee_ray_body"),
		     // optional for roll disambiguation:
		     py::arg("u_body_opt") = std::nullopt,         // u_b (must be ⟂ ee_ray_body)
		     py::arg("roll_ref_world") = std::nullopt,     // t (any, not necessarily ⟂ d)
		     py::arg("roll_ref_flat") = false,
		     py::arg("require_positive_pointing") = true,
		     py::arg("eps_d") = 0.05,
		     py::arg("tau_tperp") = 0.05)

		.def("add_point_to_point_displacement_constraint", &GraphOfConstraints::add_point_to_point_displacement_constraint,
		     py::arg("k"),
		     py::arg("point_a"),
		     py::arg("point_b"),
		     py::arg("disp"),
		     py::arg("tol") = 0.05)
		.def("add_point_to_point_alignment_constraint", &GraphOfConstraints::add_point_to_point_alignment_constraint)
		.def("add_assignable_robot_holding_point_constraint", &GraphOfConstraints::add_assignable_robot_holding_point_constraint)
		.def("add_robot_holding_cube_constraint", &GraphOfConstraints::add_robot_holding_cube_constraint)
		.def("add_robot_relative_rotation_constraint", &GraphOfConstraints::add_robot_relative_rotation_constraint)
		.def("add_robot_above_cube_constraint", &GraphOfConstraints::add_robot_above_cube_constraint,
		     py::arg("k"),
		     py::arg("robot_id"),
		     py::arg("cube_id"),
		     py::arg("delta_z"),
		     py::arg("x_offset") = 0.0,
		     py::arg("y_offset") = 0.0);

        py::class_<GraphWaypointMPC>(goc_mpc, "GraphWaypointMPC")
                .def(py::init<GraphOfConstraints&, std::vector<CubicConfigurationSpline>>(),
		     py::keep_alive<1, 3>())
		.def("solve", &GraphWaypointMPC::solve)
		.def("view_waypoints", &GraphWaypointMPC::view_waypoints, py::return_value_policy::reference_internal)
		.def("view_assignments", &GraphWaypointMPC::view_assignments, py::return_value_policy::reference_internal)
		.def("view_var_assignments", &GraphWaypointMPC::view_var_assignments, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &GraphWaypointMPC::get_last_solve_time);

        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init<const GraphOfConstraints&, std::vector<CubicConfigurationSpline>, double, double, double, double, double>(),
		     py::keep_alive<1, 3>())
		.def("solve", &GraphTimingMPC::solve)
		.def("get_agent_spline_length", &GraphTimingMPC::get_agent_spline_length)
		.def("get_agent_spline_nodes", &GraphTimingMPC::get_agent_spline_nodes)
		.def("set_progressed_time", &GraphTimingMPC::set_progressed_time)
		.def("fill_cubic_splines", &GraphTimingMPC::fill_cubic_splines)
		.def("get_next_taus", &GraphTimingMPC::get_next_taus)
		.def("get_next_nodes", &GraphTimingMPC::get_next_nodes)
		.def("view_wps_list", &GraphTimingMPC::view_wps_list)
		.def("view_vs_list", &GraphTimingMPC::view_vs_list)
		.def("view_time_deltas_list", &GraphTimingMPC::view_time_deltas_list)
		.def("view_agent_nodes_list", &GraphTimingMPC::view_agent_nodes_list)
		.def("view_agent_spline_length_map", &GraphTimingMPC::view_agent_spline_length_map)
		.def("get_last_solve_time", &GraphTimingMPC::get_last_solve_time);

        py::class_<GraphShortPathMPC>(goc_mpc, "GraphShortPathMPC")
                .def(py::init<const GraphOfConstraints&, unsigned int, unsigned int, unsigned int, double>(),
		     py::arg("graph"), py::arg("num_steps"), py::arg("num_agents"), py::arg("dim"), py::arg("time_per_step"))
		.def("solve", &GraphShortPathMPC::solve)
		.def("view_points", &GraphShortPathMPC::view_points, py::return_value_policy::reference_internal)
		.def("view_vels", &GraphShortPathMPC::view_vels, py::return_value_policy::reference_internal)
		.def("view_times", &GraphShortPathMPC::view_times, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &GraphShortPathMPC::get_last_solve_time);
}
