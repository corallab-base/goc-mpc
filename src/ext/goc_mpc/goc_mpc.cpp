#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <drake/bindings/pydrake/symbolic_types_pybind.h>

#include "graph_of_constraints.hpp"
#include "graph_waypoint_mpc.hpp"
#include "milp_waypoint_mpc.hpp"
#include "graph_timing_mpc.hpp"
#include "graph_short_path_mpc.hpp"
#include "../grid_interpolant.hpp"

using drake::symbolic::Expression;
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
		.def(py::init<const std::vector<CubicConfigurationSpline::Spec>&,
		              const std::vector<CubicConfigurationSpline::Spec>&,
		              double,
		              double,
		              const std::vector<std::string>&,
		              const std::vector<std::string>&>(),
		     py::arg("robot_specs"),
		     py::arg("object_specs"),
		     py::arg("state_lower_bound"),
		     py::arg("state_upper_bound"),
		     py::arg("robot_names") = std::vector<std::string>{},
		     py::arg("object_names") = std::vector<std::string>{})
		.def_readonly("_robot_specs", &GraphOfConstraints::_robot_specs)
		.def_readonly("_object_specs", &GraphOfConstraints::_object_specs)
		.def_readonly("structure", &GraphOfConstraints::structure)
		.def_readonly("num_agents", &GraphOfConstraints::num_agents)
		.def_readonly("num_objects", &GraphOfConstraints::num_objects)
		.def_readonly("num_phis", &GraphOfConstraints::num_phis)
		.def_readonly("num_variables", &GraphOfConstraints::num_variables)
		.def_readonly("dim", &GraphOfConstraints::dim)
		.def_readonly("non_robot_dim", &GraphOfConstraints::non_robot_dim)
		.def_readonly("total_dim", &GraphOfConstraints::total_dim)
		.def_readonly("unpassable_nodes", &GraphOfConstraints::unpassable_nodes)
		.def_readonly("backtrack_map", &GraphOfConstraints::backtrack_map)
		.def_readonly("phi_to_variable_map", &GraphOfConstraints::phi_to_variable_map)
		.def_readonly("phi_to_static_assignment_map", &GraphOfConstraints::_phi_to_static_assignment_map)
		.def_readonly("node_to_phis_map", &GraphOfConstraints::node_to_phis_map)
		.def_readonly("edge_to_phis_map", &GraphOfConstraints::edge_to_phis_map)
		// Raw Formula records for the unified symbolic constraint API
		// (add_constraint / add_assignable_constraint / add_edge_constraint),
		// keyed by phi id — introspectable so non-MILP consumers (e.g. the
		// JAX evolutionary solver) can compile the same constraint
		// themselves instead of duplicating it by hand. Combine with
		// node_to_phis_map/edge_to_phis_map (which node/edge each phi
		// belongs to) and phi_to_variable_map (which var an assignable phi
		// is gated on, if any).
		.def_property_readonly("phi_to_formula_map", [](const GraphOfConstraints& self) {
			std::map<int, drake::symbolic::Formula> out;
			for (const auto& [id, rec] : self.symbolic_ops) out[id] = rec.formula;
			return out;
		})
		.def_property_readonly("edge_phi_to_formula_map", [](const GraphOfConstraints& self) {
			std::map<int, drake::symbolic::Formula> out;
			for (const auto& [id, rec] : self.symbolic_edge_ops) out[id] = rec.formula;
			return out;
		})
		// Which compiled form each edge_phi_to_formula_map entry is: True
		// for an "along the edge" formula (plain agent_q/object_q/
		// var_agent_q placeholders, an invariant applied at both endpoints),
		// False for a relational one (u_agent_q/v_agent_q etc., a single
		// relation coupling the two endpoints) -- see add_edge_constraint.
		.def_property_readonly("edge_phi_to_along_edge_map", [](const GraphOfConstraints& self) {
			std::map<int, bool> out;
			for (const auto& [id, rec] : self.symbolic_edge_ops) out[id] = rec.along_edge;
			return out;
		})
		.def_readonly("conditional_ordering_map", &GraphOfConstraints::_conditional_ordering_map)
		.def_readonly("binary_cond_sym_vars", &GraphOfConstraints::_binary_cond_sym_vars)
		.def("add_variable", &GraphOfConstraints::add_variable)
		.def("add_grasp_change", &GraphOfConstraints::add_grasp_change)
		.def("add_assignable_grasp_change", &GraphOfConstraints::add_assignable_grasp_change)
		.def("get_grasp_changes", &GraphOfConstraints::get_grasp_changes)
		.def("make_node_unpassable", &GraphOfConstraints::make_node_unpassable)
		.def("get_phi_ids", &GraphOfConstraints::get_phi_ids)
		.def("get_next_edge_phis", &GraphOfConstraints::get_next_edge_phis)
		.def("evaluate_phi", &GraphOfConstraints::evaluate_phi)
		.def("evaluate_edge_phi", &GraphOfConstraints::evaluate_edge_phi)
		.def("get_edge_phi_agent", &GraphOfConstraints::get_edge_phi_agent)
		.def("add_backtrack_links", &GraphOfConstraints::add_backtrack_links)
		.def("add_manual_backtrack_links", &GraphOfConstraints::add_manual_backtrack_links)
		.def("add_robot_to_point_displacement_cost", &GraphOfConstraints::add_robot_to_point_displacement_cost)
		.def("add_robot_to_point_alignment_cost", &GraphOfConstraints::add_robot_to_point_alignment_cost,
		     py::arg("k"),
		     py::arg("robot_id"),
		     py::arg("point_id"),
		     py::arg("ee_ray_body"),
		     py::arg("u_body_opt") = std::nullopt,
		     py::arg("roll_ref_world") = std::nullopt,
		     py::arg("roll_ref_flat") = false,
		     py::arg("require_positive_pointing") = true,
		     py::arg("w_point") = 1.0,
		     py::arg("w_roll") = 0.1,
		     py::arg("w_flat") = 0.05,
		     py::arg("w_guard") = 0.0,
		     py::arg("w_u_stab") = 0.01,
		     py::arg("eps") = 1e-10,
		     py::arg("eps_d") = 1e-3)
		.def("add_point_to_point_displacement_cost", &GraphOfConstraints::add_point_to_point_displacement_cost,
		     py::arg("k"),
		     py::arg("point_a"),
		     py::arg("point_b"),
		     py::arg("disp"))
		// EDGE TIMING CONSTRAINTS  ///////////////////////////////////
		.def("add_edge_min_tau_constraint", &GraphOfConstraints::add_edge_min_tau_constraint,
		     py::arg("u"),
		     py::arg("v"),
		     py::arg("minimum_time_delta"))
		// VARIABLE CONSTRAINTS ///////////////////////////////////////
		.def("add_variable_constraint", &GraphOfConstraints::add_variable_constraint)
		.def("add_variable_ineq_constraint", &GraphOfConstraints::add_variable_ineq_constraint)
		// SYMBOLIC UNIFIED CONSTRAINT API ////////////////////////////
		.def("object_q", &GraphOfConstraints::object_q, py::arg("object_q"))
		.def("agent_q", &GraphOfConstraints::agent_q, py::arg("agent_q"))
		.def("var_agent_q", &GraphOfConstraints::var_agent_q, py::arg("var"))
		.def("u_object_q", &GraphOfConstraints::object_q_u, py::arg("object_q"))
		.def("u_agent_q", &GraphOfConstraints::agent_q_u, py::arg("agent_q"))
		.def("v_object_q", &GraphOfConstraints::object_q_v, py::arg("object_q"))
		.def("v_agent_q", &GraphOfConstraints::agent_q_v, py::arg("agent_q"))
		// accept either a single Formula or a numpy array of Formulas (from
		// element-wise == on object arrays) and reduce with conjunction.
		.def("add_constraint", [](GraphOfConstraints& self, int node,
		                          py::object formula_obj) -> int {
			drake::symbolic::Formula f;
			try {
				f = py::cast<drake::symbolic::Formula>(formula_obj);
			} catch (const py::cast_error&) {
				bool first = true;
				for (auto h : formula_obj) {
					auto fh = py::cast<drake::symbolic::Formula>(h);
					if (first) { f = fh; first = false; }
					else        f = f && fh;
				}
			}
			return self.add_constraint(node, f);
		}, py::arg("node"), py::arg("formula"))
		.def("add_edge_constraint", [](GraphOfConstraints& self, int u, int v,
					       py::object formula_obj) -> int {
			drake::symbolic::Formula f;
			try {
				f = py::cast<drake::symbolic::Formula>(formula_obj);
			} catch (const py::cast_error&) {
				bool first = true;
				for (auto h : formula_obj) {
					auto fh = py::cast<drake::symbolic::Formula>(h);
					if (first) { f = fh; first = false; }
					else        f = f && fh;
				}
			}
			return self.add_edge_constraint(u, v, f);
		}, py::arg("u"), py::arg("v"), py::arg("formula"))
		// CONDITIONAL ORDERING API ////////////////////////////
		.def("assignment_sym", &GraphOfConstraints::assignment_sym, py::arg("var"))
		.def("add_binary_cond_var", &GraphOfConstraints::add_binary_cond_var)
		.def("add_edge", [](GraphOfConstraints& self, int u, int v, py::object cond) {
			if (py::isinstance<py::bool_>(cond) || cond.is(py::none())) {
				self.structure.add_edge(u, v, cond);
			} else {
				drake::symbolic::Formula f = drake::symbolic::Formula::True();
				try {
					f = py::cast<drake::symbolic::Formula>(cond);
				} catch (const py::cast_error&) {
					bool first = true;
					for (auto h : cond) {
						auto fh = py::cast<drake::symbolic::Formula>(h);
						if (first) { f = fh; first = false; }
						else        f = f && fh;
					}
				}
				self.add_conditional_edge_ordering(u, v, f);
			}
		}, py::arg("u"), py::arg("v"), py::arg("cond") = py::cast(true));

	py::enum_<WaypointSolver>(goc_mpc, "WaypointSolver")
		.value("kGurobi", WaypointSolver::kGurobi)
		.value("kMosek",  WaypointSolver::kMosek)
		.value("kIPOPT",  WaypointSolver::kIPOPT)
		.export_values();

	py::enum_<WaypointObjective>(goc_mpc, "WaypointObjective")
		.value("kMinMaxL1",      WaypointObjective::kMinMaxL1)
		.value("kAvgL1",         WaypointObjective::kAvgL1)
		.value("kMinMaxL2",      WaypointObjective::kMinMaxL2)
		.value("kAvgL2",         WaypointObjective::kAvgL2)
		.value("kMinMaxGeodesic", WaypointObjective::kMinMaxGeodesic)
		.value("kAvgGeodesic",    WaypointObjective::kAvgGeodesic)
		.export_values();

	py::class_<EdgeCostFunctor, std::shared_ptr<EdgeCostFunctor>>(goc_mpc, "EdgeCostFunctor");

	py::class_<RegularGridInterpolant, EdgeCostFunctor,
		   std::shared_ptr<RegularGridInterpolant>>(goc_mpc, "RegularGridInterpolant")
		.def(py::init<Eigen::VectorXd, Eigen::VectorXd, std::vector<int>, Eigen::VectorXd>(),
		     py::arg("origin"),
		     py::arg("spacing"),
		     py::arg("shape"),
		     py::arg("values"))
		.def("dim", &RegularGridInterpolant::dim)
		.def("interpolate", &RegularGridInterpolant::Interpolate<double>, py::arg("query"));

	py::class_<GraphWaypointMPC, std::shared_ptr<GraphWaypointMPC>>(goc_mpc, "GraphWaypointMPC")
		.def("solve", &GraphWaypointMPC::Solve)
		.def("evaluate_edge_cost", &GraphWaypointMPC::EvaluateEdgeCost,
		     py::arg("agent"), py::arg("w_a"), py::arg("w_b"))
		.def("view_waypoints", &GraphWaypointMPC::view_waypoints, py::return_value_policy::reference_internal)
		.def("view_assignments", &GraphWaypointMPC::view_assignments, py::return_value_policy::reference_internal)
		.def("view_var_assignments", &GraphWaypointMPC::view_var_assignments, py::return_value_policy::reference_internal)
		.def("view_t_by_node", &GraphWaypointMPC::view_t_by_node, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &GraphWaypointMPC::get_last_solve_time);

        py::class_<MILPWaypointMPC, GraphWaypointMPC, std::shared_ptr<MILPWaypointMPC>>(goc_mpc, "MILPWaypointMPC")
                .def(py::init<GraphOfConstraints&,
			      std::vector<CubicConfigurationSpline>,
			      WaypointSolver,
			      bool,
			      WaypointObjective,
			      std::shared_ptr<EdgeCostFunctor>>(),
		     py::keep_alive<1, 2>(),
		     py::arg("graph"),
		     py::arg("splines"),
		     py::arg("solver")            = WaypointSolver::kGurobi,
		     py::arg("enforce_rigidity")  = false,
		     py::arg("objective")         = WaypointObjective::kMinMaxL1,
		     py::arg("edge_cost_fn")      = std::shared_ptr<EdgeCostFunctor>())
		.def("view_t", &MILPWaypointMPC::view_t, py::return_value_policy::reference_internal)
		.def("view_Z", &MILPWaypointMPC::view_Z, py::return_value_policy::reference_internal);

        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init<const GraphOfConstraints&, std::vector<CubicConfigurationSpline>, double, double, double, double, double, double, double, double>(),
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
