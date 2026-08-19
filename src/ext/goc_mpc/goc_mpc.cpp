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
#include "admm_short_path_mpc.hpp"
#include "sqp_short_path_mpc.hpp"

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

	// Canonical hold registry record -- see HoldDeclaration's own docstring
	// (graph_of_constraints.hpp). Exactly one of robot_ag/var_id is set.
	py::class_<HoldDeclaration>(goc_mpc, "HoldDeclaration")
		.def_readonly("id", &HoldDeclaration::id)
		.def_readonly("u_node", &HoldDeclaration::u_node)
		.def_readonly("v_node", &HoldDeclaration::v_node)
		.def_readonly("held_point_ids", &HoldDeclaration::held_point_ids)
		.def_readonly("robot_ag", &HoldDeclaration::robot_ag)
		.def_readonly("var_id", &HoldDeclaration::var_id);

	py::class_<AgentInteraction> agent_interaction(goc_mpc, "AgentInteraction");
	agent_interaction
		.def(py::init<int, int, int, int, int, int, AgentInteraction::Type>(),
		     py::arg("agent_i"), py::arg("agent_i_depth"),
		     py::arg("agent_j"), py::arg("agent_j_depth"),
		     py::arg("node_u"), py::arg("node_v"), py::arg("type"))
		.def_readwrite("agent_i", &AgentInteraction::agent_i)
		.def_readwrite("agent_i_depth", &AgentInteraction::agent_i_depth)
		.def_readwrite("agent_j", &AgentInteraction::agent_j)
		.def_readwrite("agent_j_depth", &AgentInteraction::agent_j_depth)
		.def_readwrite("node_u", &AgentInteraction::node_u)
		.def_readwrite("node_v", &AgentInteraction::node_v)
		.def_readwrite("type", &AgentInteraction::type);
	py::enum_<AgentInteraction::Type>(agent_interaction, "Type")
		.value("LESS_THAN", AgentInteraction::Type::LESS_THAN)
		.value("EQUAL", AgentInteraction::Type::EQUAL)
		.export_values();

	py::class_<GraphOfConstraints>(goc_mpc, "GraphOfConstraints")
		.def(py::init<const std::vector<CubicConfigurationSpline::Spec>&,
		              const std::vector<CubicConfigurationSpline::Spec>&,
		              double,
		              double,
		              const std::vector<std::string>&,
		              const std::vector<std::string>&,
		              int>(),
		     py::arg("robot_specs"),
		     py::arg("object_specs"),
		     py::arg("state_lower_bound"),
		     py::arg("state_upper_bound"),
		     py::arg("robot_names") = std::vector<std::string>{},
		     py::arg("object_names") = std::vector<std::string>{},
		     // Ambient Cartesian workspace dimensionality (2 or 3) --
		     // see GraphOfConstraints::workspace_dim's doc comment.
		     py::arg("workspace_dim") = 3)
		.def_readonly("workspace_dim", &GraphOfConstraints::workspace_dim)
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
		// Edge phi ids registered via add_edge_constraint(..., live=True) --
		// see that method's docstring. EvolutionaryWaypointSolver reads this
		// directly off the graph instead of taking a separate live_phi_ids
		// constructor argument, so the live/frozen choice lives with the
		// constraint that defines it, not with whichever solver happens to
		// run it.
		.def_readonly("live_edge_phis", &GraphOfConstraints::live_edge_phis)
		// Canonical hold registry (see HoldDeclaration) -- populated by
		// add_robot_holding_cube_constraint/add_assignable_robot_holding_
		// point_constraint. Single source of truth for "which edges hold
		// which objects" for non-MILP consumers (e.g. the JAX evolutionary
		// solver), instead of each re-deriving it from the legacy
		// DeferredEdgeOp::cubes + static/assignable edge-phi maps.
		.def_readonly("hold_ops", &GraphOfConstraints::hold_ops)
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
		.def("set_robot_fk", &GraphOfConstraints::set_robot_fk,
		     py::arg("agent_id"), py::arg("link_name"), py::arg("fk_fn"))
		// Raw (agent_id, link_name) -> fk_fn registry (see set_robot_fk's
		// doc comment) -- exposed read-only so the JAX evolutionary
		// solver (src/goc_mpc/evolutionary_waypoint_solver/spec.py) can
		// call a registered fk_fn DIRECTLY in Python (with a JAX tracer,
		// under jax.jit/vmap tracing) to resolve an agent_link_pos(...)/
		// agent_link_rot(...) constraint placeholder, bypassing
		// link_pose's C++/pybind boundary entirely (which requires a
		// concrete Eigen::VectorXd and cannot accept a tracer).
		.def_readonly("robot_fk_registry", &GraphOfConstraints::robot_fk_registry)
		.def("link_pose", &GraphOfConstraints::link_pose,
		     py::arg("agent_id"), py::arg("x"), py::arg("link_name") = "ee")
		.def("point_position", &GraphOfConstraints::point_position,
		     py::arg("point_id"), py::arg("x"))
		.def("add_node", &GraphOfConstraints::add_node,
		     py::arg("name") = py::none())
		.def("add_nodes", &GraphOfConstraints::add_nodes,
		     py::arg("n"), py::arg("names") = py::none())
		.def("set_node_name", &GraphOfConstraints::set_node_name,
		     py::arg("k"), py::arg("name"))
		.def("set_node_names", &GraphOfConstraints::set_node_names,
		     py::arg("ks"), py::arg("names"))
		.def("get_node_name", &GraphOfConstraints::get_node_name,
		     py::arg("k"))
		.def_readonly("node_names", &GraphOfConstraints::node_names)
		.def("get_phi_ids", &GraphOfConstraints::get_phi_ids)
		.def("get_next_edge_phis", &GraphOfConstraints::get_next_edge_phis)
		.def("get_agent_paths", &GraphOfConstraints::get_agent_paths,
		     py::arg("remaining_vertices"), py::arg("assignments"), py::arg("t_by_node"))
		.def_static("reindex_agent_interactions", &GraphOfConstraints::reindex_agent_interactions,
		     py::arg("agent_interactions"), py::arg("agent_node_ids"))
		.def("evaluate_phi", &GraphOfConstraints::evaluate_phi)
		.def("evaluate_edge_phi", &GraphOfConstraints::evaluate_edge_phi)
		.def("get_edge_phi_agent", &GraphOfConstraints::get_edge_phi_agent)
		.def("add_backtrack_links", &GraphOfConstraints::add_backtrack_links)
		.def("add_manual_backtrack_links", &GraphOfConstraints::add_manual_backtrack_links)
		// ASSIGNMENT COMMIT  //////////////////////////////////////////
		// Declares that completing `node` pins variable `var`'s resolved
		// agent for as long as anything downstream still references it --
		// see commit_trigger_node_to_var/committed_assignments.
		// GraphOfConstraintsMPC drives the runtime pin/unpin via
		// commit_variable_assignment/clear_variable_commitment as `node`
		// completes or gets reopened by backtracking.
		.def("add_variable_commit", &GraphOfConstraints::add_variable_commit,
		     py::arg("var"), py::arg("node"))
		.def("commit_variable_assignment", &GraphOfConstraints::commit_variable_assignment,
		     py::arg("var"), py::arg("agent"))
		.def("clear_variable_commitment", &GraphOfConstraints::clear_variable_commitment,
		     py::arg("var"))
		.def("get_commit_trigger_var", &GraphOfConstraints::get_commit_trigger_var,
		     py::arg("node"))
		// HOLD DECLARATIONS  //////////////////////////////////////////
		// Canonical way to declare that held_point_ids are rigidly held by
		// a robot over edge (u -> v) -- see HoldDeclaration/hold_ops above.
		.def("add_hold", &GraphOfConstraints::add_hold,
		     py::arg("u"),
		     py::arg("v"),
		     py::arg("robot_ag"),
		     py::arg("held_point_ids"))
		.def("add_assignable_hold", &GraphOfConstraints::add_assignable_hold,
		     py::arg("u"),
		     py::arg("v"),
		     py::arg("var"),
		     py::arg("held_point_ids"))
		// Holds currently in progress given remaining_vertices -- see
		// get_current_holds's own doc comment.
		.def("get_current_holds", &GraphOfConstraints::get_current_holds,
		     py::arg("remaining_vertices"))
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
		// Runtime-editable scalar parameters -- see GraphOfConstraints::_param's
		// doc comment. add_param declares one (returning its id); param(id)
		// is the placeholder Expression to reference inside a Formula;
		// set_param overwrites its value in place, cheap on both solvers
		// (no Formula re-authoring, no jax retrace on the evolutionary side).
		.def("add_param", &GraphOfConstraints::add_param, py::arg("initial_value") = 0.0)
		.def("param", &GraphOfConstraints::param, py::arg("id"))
		.def("set_param", &GraphOfConstraints::set_param, py::arg("id"), py::arg("value"))
		.def("view_param_values", &GraphOfConstraints::view_param_values,
		     py::return_value_policy::reference_internal)
		.def("num_params", &GraphOfConstraints::num_params)
		.def("agent_link_pos", &GraphOfConstraints::agent_link_pos,
		     py::arg("agent_id"), py::arg("link_name"))
		.def("agent_link_rot", &GraphOfConstraints::agent_link_rot,
		     py::arg("agent_id"), py::arg("link_name"))
		.def("u_object_q", &GraphOfConstraints::u_object_q, py::arg("object_q"))
		.def("u_agent_q", &GraphOfConstraints::u_agent_q, py::arg("agent_q"))
		.def("v_object_q", &GraphOfConstraints::v_object_q, py::arg("object_q"))
		.def("v_agent_q", &GraphOfConstraints::v_agent_q, py::arg("agent_q"))
		.def("u_var_agent_q", &GraphOfConstraints::u_var_agent_q, py::arg("var"))
		.def("v_var_agent_q", &GraphOfConstraints::v_var_agent_q, py::arg("var"))
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
					       py::object formula_obj, bool live) -> int {
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
			return self.add_edge_constraint(u, v, f, live);
		}, py::arg("u"), py::arg("v"), py::arg("formula"), py::arg("live") = false)
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
		.export_values();

	py::class_<GraphWaypointMPC, std::shared_ptr<GraphWaypointMPC>>(goc_mpc, "GraphWaypointMPC")
		.def("solve", &GraphWaypointMPC::Solve)
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
			      WaypointObjective>(),
		     py::keep_alive<1, 2>(),
		     py::arg("graph"),
		     py::arg("splines"),
		     py::arg("solver")            = WaypointSolver::kGurobi,
		     py::arg("enforce_rigidity")  = false,
		     py::arg("objective")         = WaypointObjective::kMinMaxL1)
		.def("view_t", &MILPWaypointMPC::view_t, py::return_value_policy::reference_internal)
		.def("view_Z", &MILPWaypointMPC::view_Z, py::return_value_policy::reference_internal);

        py::class_<GraphTimingMPC>(goc_mpc, "GraphTimingMPC")
                .def(py::init([](const GraphOfConstraints& graph,
                                  std::vector<CubicConfigurationSpline> splines,
                                  double time_cost, double time_cost2, double acceleration_cost,
                                  double energy_cost, double arclength_cost,
                                  py::object max_vel, py::object max_acc, py::object max_jerk,
                                  int max_iterations, double initial_trust_radius,
                                  double max_trust_radius, double min_trust_radius,
                                  double grad_tol, double interaction_penalty_weight) {
                        // max_vel/max_acc/max_jerk: a bare float broadcasts to every block in
                        // splines[0]'s spec (back-compat with every existing caller, including
                        // goc-mpc's own raw examples that construct this with plain floats); a
                        // list[float] must have exactly one entry per block. Broadcasting lives
                        // here, at the pybind boundary, rather than in goc_mpc.py/
                        // traced_timing_mpc.py, since it's the only place that covers every
                        // caller. (max_vel/max_acc must still resolve to <= 0/empty -- an actual
                        // bound throws inside the constructor, see GraphTimingMPC's own doc
                        // comment for why.)
                        const size_t num_blocks = splines.at(0).block_offsets_.size();
                        auto broadcast_bound = [num_blocks](const py::object& val, const char* name) {
                                if (val.is_none()) return std::vector<double>();
                                if (py::isinstance<py::float_>(val) || py::isinstance<py::int_>(val)) {
                                        return std::vector<double>(num_blocks, val.cast<double>());
                                }
                                std::vector<double> v;
                                try {
                                        v = val.cast<std::vector<double>>();
                                } catch (const py::cast_error&) {
                                        throw std::runtime_error(
                                                std::string(name) + " must be a float (broadcast to every "
                                                "block) or a list[float] of length matching the spline's "
                                                "block count");
                                }
                                if (!v.empty() && v.size() != num_blocks) {
                                        throw std::runtime_error(
                                                std::string(name) + ": expected " + std::to_string(num_blocks)
                                                + " values (one per spec block), got " + std::to_string(v.size()));
                                }
                                return v;
                        };
                        return GraphTimingMPC(graph, std::move(splines), time_cost, time_cost2,
                                acceleration_cost, energy_cost, arclength_cost,
                                broadcast_bound(max_vel, "max_vel"),
                                broadcast_bound(max_acc, "max_acc"),
                                broadcast_bound(max_jerk, "max_jerk"),
                                max_iterations, initial_trust_radius, max_trust_radius,
                                min_trust_radius, grad_tol, interaction_penalty_weight);
                     }), py::keep_alive<1, 3>(),
                     py::arg("graph"), py::arg("splines"),
                     py::arg("time_cost") = 1e0, py::arg("time_cost2") = 0e0,
                     py::arg("acceleration_cost") = 1.0, py::arg("energy_cost") = 0.0,
                     py::arg("arclength_cost") = 0.0,
                     py::arg("max_vel") = py::float_(-1.0), py::arg("max_acc") = py::float_(-1.0),
                     py::arg("max_jerk") = py::float_(-1.0),
                     py::arg("max_iterations") = 50,
                     py::arg("initial_trust_radius") = 1.0,
                     py::arg("max_trust_radius") = 50.0,
                     py::arg("min_trust_radius") = 1e-6,
                     py::arg("grad_tol") = 1e-6,
                     py::arg("interaction_penalty_weight") = 1.0e3)
		.def("solve", &GraphTimingMPC::solve,
		     py::arg("x0"), py::arg("v0"), py::arg("remaining_vertices"),
		     py::arg("waypoints"), py::arg("assignments"),
		     py::arg("t_by_node") = Eigen::VectorXd())
		.def("solve_dense", &GraphTimingMPC::solve_dense,
		     py::arg("x0"), py::arg("v0"), py::arg("agent_dense_wps"),
		     py::arg("agent_dense_node_ids"), py::arg("agent_interactions"))
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
		.def("get_last_solve_time", &GraphTimingMPC::get_last_solve_time)
		.def("get_last_iterations", &GraphTimingMPC::get_last_iterations)
		.def("get_last_trust_radius", &GraphTimingMPC::get_last_trust_radius);

	// Extensible obstacle registry passed into GraphShortPathMPC -- see
	// obstacle_set.hpp's own doc comment for why this is one open-ended
	// class (kept alive by the Python caller, held by pointer on the C++
	// side) rather than a per-primitive-kind array parameter.
	py::enum_<ObstacleKind>(goc_mpc, "ObstacleKind")
		.value("kSphere", ObstacleKind::kSphere)
		.value("kBox", ObstacleKind::kBox);

	py::class_<Obstacle>(goc_mpc, "Obstacle")
		.def_readonly("kind", &Obstacle::kind)
		.def_readonly("params", &Obstacle::params)
		.def_readonly("margin", &Obstacle::margin);

	// Caller-supplied, backend-agnostic signed-distance grid for one
	// agent's own local vicinity -- see AgentSdfGrid's own doc comment
	// (obstacle_set.hpp) for the buffer layout/convention. Read-only from
	// Python: built via ObstacleSet.set_agent_sdf_grid below, this
	// binding only exists so a caller can read back what's registered
	// (mirrors Obstacle's own read-only binding just above).
	py::class_<AgentSdfGrid>(goc_mpc, "AgentSdfGrid")
		.def_readonly("origin", &AgentSdfGrid::origin)
		.def_readonly("resolution", &AgentSdfGrid::resolution)
		.def_readonly("shape", &AgentSdfGrid::shape)
		.def_readonly("values", &AgentSdfGrid::values)
		.def_readonly("gradient", &AgentSdfGrid::gradient)
		.def_readonly("margin", &AgentSdfGrid::margin);

	py::class_<ObstacleSet>(goc_mpc, "ObstacleSet")
		.def(py::init<>())
		.def("add_sphere", &ObstacleSet::add_sphere,
		     py::arg("center"), py::arg("radius"), py::arg("margin") = 0.0)
		.def("add_box", &ObstacleSet::add_box,
		     py::arg("center"), py::arg("half_extents"), py::arg("margin") = 0.0)
		.def("clear", &ObstacleSet::clear)
		.def("set_point_cloud", &ObstacleSet::set_point_cloud,
		     py::arg("points"), py::arg("margin") = 0.0)
		.def("query_point_cloud_radius", &ObstacleSet::query_point_cloud_radius,
		     py::arg("center"), py::arg("radius"))
		.def("point_cloud", &ObstacleSet::point_cloud, py::return_value_policy::reference_internal)
		.def("point_cloud_margin", &ObstacleSet::point_cloud_margin)
		.def("has_point_cloud", &ObstacleSet::has_point_cloud)
		.def("obstacles", &ObstacleSet::obstacles, py::return_value_policy::reference_internal)
		// SqpShortPathMPC's Stage-3 replacement: one local SDF grid PER
		// AGENT (not shared like every obstacle kind above), wholesale-
		// replaced each call like set_point_cloud -- see
		// ObstacleSet::set_agent_sdf_grid's own doc comment. `gradient`
		// defaults to empty (derive from `values`, the consistency-first
		// default); pass a non-empty array to hand over a backend-native
		// gradient field instead.
		.def("set_agent_sdf_grid", &ObstacleSet::set_agent_sdf_grid,
		     py::arg("agent"), py::arg("origin"), py::arg("resolution"), py::arg("shape"),
		     py::arg("values"), py::arg("gradient") = Eigen::VectorXd(), py::arg("margin") = 0.0)
		.def("clear_agent_sdf_grid", &ObstacleSet::clear_agent_sdf_grid, py::arg("agent"))
		.def("agent_sdf_grid", &ObstacleSet::agent_sdf_grid, py::arg("agent"),
		     py::return_value_policy::reference_internal);

        py::class_<GraphShortPathMPC>(goc_mpc, "GraphShortPathMPC")
                .def(py::init<const GraphOfConstraints&, unsigned int, unsigned int, unsigned int, double,
		     const ObstacleSet&, double, bool>(),
		     py::arg("graph"), py::arg("num_steps"), py::arg("num_agents"), py::arg("dim"),
		     py::arg("time_per_step"), py::arg("obstacles"), py::arg("obstacle_repulsion_weight") = 0.5,
		     py::arg("use_hard_constraints") = true,
		     // `graph` and `obstacles` are both stored by the C++ side as raw
		     // pointers (see GraphShortPathMPC::_graph/_obstacles) -- keep
		     // both Python arguments alive at least as long as this
		     // GraphShortPathMPC instance (pybind's own default `_graph`
		     // handling has always relied on the Python-side caller doing
		     // this too, e.g. GraphOfConstraintsMPC.self.graph; keep_alive
		     // makes the `obstacles` half of that contract explicit and
		     // safe even if a caller forgets to hold its own reference).
		     py::keep_alive<1, 2>(), py::keep_alive<1, 7>())
		.def("solve", &GraphShortPathMPC::solve)
		.def("view_points", &GraphShortPathMPC::view_points, py::return_value_policy::reference_internal)
		.def("view_vels", &GraphShortPathMPC::view_vels, py::return_value_policy::reference_internal)
		.def("view_times", &GraphShortPathMPC::view_times, py::return_value_policy::reference_internal)
		.def("view_obstacles", &GraphShortPathMPC::view_obstacles, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &GraphShortPathMPC::get_last_solve_time);

	py::class_<AdmmShortPathMPC>(goc_mpc, "AdmmShortPathMPC")
		.def(py::init<const GraphOfConstraints&, unsigned int, unsigned int, unsigned int, double,
		     const ObstacleSet&, double, unsigned int, double>(),
		     py::arg("graph"), py::arg("num_steps"), py::arg("num_agents"), py::arg("dim"),
		     py::arg("time_per_step"), py::arg("obstacles"), py::arg("rho") = 5.0,
		     py::arg("num_iterations") = 8, py::arg("point_cloud_query_margin") = 1.0,
		     // Same lifetime discipline as GraphShortPathMPC's binding above.
		     py::keep_alive<1, 2>(), py::keep_alive<1, 7>())
		.def("solve", &AdmmShortPathMPC::solve)
		.def("view_points", &AdmmShortPathMPC::view_points, py::return_value_policy::reference_internal)
		.def("view_vels", &AdmmShortPathMPC::view_vels, py::return_value_policy::reference_internal)
		.def("view_times", &AdmmShortPathMPC::view_times, py::return_value_policy::reference_internal)
		.def("view_obstacles", &AdmmShortPathMPC::view_obstacles, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &AdmmShortPathMPC::get_last_solve_time);

	py::class_<SqpShortPathMPC>(goc_mpc, "SqpShortPathMPC")
		.def(py::init<const GraphOfConstraints&, unsigned int, unsigned int, unsigned int, double,
		     const ObstacleSet&, Eigen::VectorXd, double, double, double, double, int, double, double,
		     double, double, double>(),
		     py::arg("graph"), py::arg("num_steps"), py::arg("num_agents"), py::arg("dim"),
		     py::arg("time_per_step"), py::arg("obstacles"),
		     py::arg("agent_radii") = Eigen::VectorXd(),
		     py::arg("tracking_weight") = 1.0, py::arg("velocity_tracking_weight") = 1.0,
		     py::arg("acceleration_weight") = 1.0,
		     py::arg("penalty_weight") = 1.0e3,
		     py::arg("max_iterations") = 30, py::arg("initial_trust_radius") = 0.5,
		     py::arg("max_trust_radius") = 5.0, py::arg("min_trust_radius") = 1.0e-6,
		     py::arg("grad_tol") = 1.0e-6, py::arg("constraint_prune_margin") = 1.0,
		     // Same lifetime discipline as GraphShortPathMPC/AdmmShortPathMPC's
		     // bindings above.
		     py::keep_alive<1, 2>(), py::keep_alive<1, 7>())
		.def("solve", &SqpShortPathMPC::solve)
		.def("view_points", &SqpShortPathMPC::view_points, py::return_value_policy::reference_internal)
		.def("view_vels", &SqpShortPathMPC::view_vels, py::return_value_policy::reference_internal)
		.def("view_times", &SqpShortPathMPC::view_times, py::return_value_policy::reference_internal)
		.def("view_obstacles", &SqpShortPathMPC::view_obstacles, py::return_value_policy::reference_internal)
		.def("get_last_solve_time", &SqpShortPathMPC::get_last_solve_time)
		.def("get_last_iterations", &SqpShortPathMPC::get_last_iterations)
		.def("get_last_trust_radius", &SqpShortPathMPC::get_last_trust_radius);
}
