#include "graph_waypoint_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


GraphWaypointProblem build_graph_waypoint_problem(
	GraphOfConstraints* graph,
	const std::vector<int>& remaining_vertices,
	Eigen::VectorXd x0) {

	const int num_agents = graph->num_agents;
	const int num_objects = graph->num_objects;

	const SubgraphOfConstraints subgraph(graph, remaining_vertices);

	using namespace drake::solvers;

	// Create program
	GraphWaypointProblem problem;
	problem.prog = std::make_unique<MathematicalProgram>();

	// record the subgraph
	problem.subgraph = std::make_unique<SubgraphOfConstraints>(subgraph);

	// a: binary assignment variables (z x m).
	// Drake exposes a direct API for binary matrices. :contentReference[oaicite:1]{index=1}
	MatrixXDecisionVariable Assignments = problem.prog->NewBinaryVariables(subgraph.num_variables(), num_agents, "Assignments");
	problem.Assignments = Assignments;

	// One-hot per row (each task gets exactly one agent): sum_k a(row,k) = 1.
	for (int i = 0; i < subgraph.num_variables(); ++i) {
		problem.prog->AddLinearEqualityConstraint(
			Eigen::RowVectorXd::Ones(num_agents),
			1.0, Assignments.row(i));
	}

	const int robot_dim = graph->dim;
	const int objs_start = graph->num_agents * graph->dim;
	const int non_robot_dim = graph->non_robot_dim;

	// x: continuous configuration variables (n x m*d+o).
	MatrixXDecisionVariable X = problem.prog->NewContinuousVariables(subgraph.num_nodes(), num_agents * robot_dim + num_objects * non_robot_dim, "X");
	problem.X = X;

	//
	// OBJECTIVE FUNCTION
	//

	const auto& next_edge_ops = graph->get_next_edge_ops(remaining_vertices);
	std::set<int> possibly_manipulated_cubes_for_initial_layer;
	for (const auto& [_, op] : next_edge_ops) {
		possibly_manipulated_cubes_for_initial_layer.insert(
			op.cubes.begin(), op.cubes.end());
	}

	for (auto v : subgraph.structure.sources()) {
		int sg_v = subgraph.subgraph_id(v);

		// First, costs to minimize across transitions from x0 to the source
		// nodes in the subgraph.
		for (int ag = 0; ag < num_agents; ++ag) {
			VectorX<Expression> diff = x0.segment(ag * robot_dim, robot_dim) - X.row(sg_v).segment(ag * robot_dim, robot_dim);
			Expression dist = diff.squaredNorm();
			problem.prog->AddQuadraticCost(dist);
		}

		// TODO: Acknowledge when they move with the robot

		// Second, for the source nodes, add CONSTRAINTS saything that a block
		// does not move from x0 unless it is moved.
		for (int obj = 0; obj < num_objects; ++obj) {
			const int start = objs_start + obj * non_robot_dim;

			// Grab the decision-variable segment (as Expressions)
			// Note: X.row(sg_v) is RowVector<Expression>, so transpose to column.
			Eigen::VectorX<drake::symbolic::Expression> X_seg =
				X.row(sg_v).segment(start, non_robot_dim).transpose();

			// Grab the constant target segment
			Eigen::VectorXd x0_seg = x0.segment(start, non_robot_dim);

			if (!possibly_manipulated_cubes_for_initial_layer.contains(obj)) {
				std::cout << "added for " << v << ", " << obj << std::endl;
				// Enforce X_seg == x0_seg (all objects not effected by an edge constraint)
				problem.prog->AddLinearEqualityConstraint(X_seg, x0_seg);
			}
		}
	}


	std::map<int, std::set<int>> possibly_manipulated_cubes_during_each_layer;
	const auto& layers = subgraph.structure.topological_layer_cut_snapshot(
		[&graph, &possibly_manipulated_cubes_during_each_layer]
		(int level_k, int u, int v) {
			// this callback is called for all u, v where u is in
			// the layers less than or equal to k (the current) to
			// any layer greater than k, before moving on to
			// processing layer k+1.  Therefore, is can be used to
			// accumulate all the possibly manipulated cubes before
			// nodes in layer k+1.
			if (graph->edge_to_phi_map.contains(std::make_pair(u, v))) {
				int edge_phi_id = graph->edge_to_phi_map.at(std::make_pair(u, v));
				DeferredEdgeOp& op = graph->edge_ops.at(edge_phi_id);
				possibly_manipulated_cubes_during_each_layer[level_k].insert(
					op.cubes.begin(), op.cubes.end());
			}
		});

	// Add inter-waypoint costs for all edges
	// You control the cost body via add_edge_cost
	// Iterate edges of the subgraph
	for (auto edge : subgraph.structure.edges()) {
		int u = edge.u;
		int sg_u = subgraph.subgraph_id(u);
		int v = edge.e->to;
		int sg_v = subgraph.subgraph_id(v);
		for (int ag = 0; ag < num_agents; ++ag) {
			VectorX<Expression> diff = X.row(sg_u).segment(ag * robot_dim, robot_dim) - X.row(sg_v).segment(ag * robot_dim, robot_dim);
			Expression dist = diff.squaredNorm();
			problem.prog->AddQuadraticCost(dist);
		}

		// TODO: Acknowledge when they move with the robot
		for (int obj = 0; obj < num_objects; ++obj) {
			const int start = objs_start + obj * non_robot_dim;

			// Grab the segment from node u
			Eigen::VectorX<drake::symbolic::Expression> X_seg_u =
				X.row(sg_u).segment(start, non_robot_dim).transpose();

			// Grab the segment from node v (as Expressions)
			// Note: X.row(sg_v) is RowVector<Expression>, so transpose to column.
			Eigen::VectorX<drake::symbolic::Expression> X_seg_v =
				X.row(sg_v).segment(start, non_robot_dim).transpose();

			// Only enforce stationary if there is no phi at this layer
			// that interferes with this cube.
			const int layer = layers.node_to_level.at(u);
			if (possibly_manipulated_cubes_during_each_layer.contains(layer)) {
				if (!possibly_manipulated_cubes_during_each_layer.at(layer).contains(obj)) {
					std::cout << "added for " << u << "->" << v << ", " << obj << std::endl;
					problem.prog->AddLinearEqualityConstraint(
						X_seg_u - X_seg_v, Eigen::VectorXd::Zero(non_robot_dim));
				}
			} else {
				std::cout << "added for " << u << "->" << v << ", " << obj << std::endl;
				problem.prog->AddLinearEqualityConstraint(
					X_seg_u - X_seg_v, Eigen::VectorXd::Zero(non_robot_dim));
			}

			// Enforce X_seg == x0_seg (all components)
			// problem.prog->AddLinearEqualityConstraint(X_seg_u - X_seg_v, Eigen::VectorXd::Zero(non_robot_dim));
		}
	}

	// Add constraints/costs from registry
	for (const auto& [phi_id, op] : subgraph.get_subgraph_ops()) {
		std::cout << "adding phi " << phi_id << std::endl;
		op.builder(*(problem.prog), subgraph, phi_id, X, Assignments);
	}

	// Add constraints/costs from edge registry
	for (const auto& [edge_phi_id, edge_op] : subgraph.get_subgraph_edge_ops()) {
		std::cout << "adding edge phi " << edge_phi_id << std::endl;
		edge_op.waypoint_builder(*(problem.prog), subgraph, edge_phi_id, X, Assignments);
	}

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(GraphOfConstraints& graph)
	: _graph(&graph) {
	// Allocate persistent output buffers.
	_waypoints = Eigen::MatrixXd::Zero(_graph->structure.num_nodes(), _graph->total_dim);
	_assignments = Eigen::VectorXi::Zero(_graph->num_phis);
	_var_assignments = Eigen::VectorXi::Zero(_graph->num_variables);
}


using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::IpoptSolverDetails;

inline void PrintSolverReport(const MathematicalProgram& prog,
                              const MathematicalProgramResult& result,
                              double tol = 1e-6) {
	using std::cout;
	using std::endl;

	cout << "=== Drake Solve() Report ===\n";
	cout << "Solver:           " << result.get_solver_id().name() << "\n";
	// to_string(SolutionResult) is supported; if your Drake is older, cast to int.
	cout << "SolutionResult:   " << to_string(result.get_solution_result()) << "\n";
	cout << "Success?          " << (result.is_success() ? "yes" : "no") << "\n";

	// Print optimal cost if available.
	if (std::isfinite(result.get_optimal_cost())) {
		cout << "Optimal cost:     " << result.get_optimal_cost() << "\n";
	}

	// Try to obtain a decision vector to evaluate constraints.
	Eigen::VectorXd x;
	bool have_x = false;
	try {
		x = result.GetSolution(prog.decision_variables());
		have_x = true;
	} catch (const std::exception&) {
		// Fall back to initial guess (may be zero) if solver did not return x.
		if (prog.decision_variables().size() > 0) {
			x = prog.initial_guess();
			have_x = true;
			cout << "(No returned solution vector; using initial guess for diagnostics.)\n";
		}
	}

	// Constraint violation scan.
	if (have_x) {
		auto print_binding_violation = [&](const auto& bindings, const char* kind) {
			for (const auto& b : bindings) {
				const auto& c = *b.evaluator();
				double tol = 0.05;
				if (!c.CheckSatisfied(x, tol)) {
					cout << "Violation [" << kind << "]: "
					     << " (constraint: " << c.get_description() << ")\n";
				}
			}
		};

		cout << "--- Constraint violations (inf-norm > " << tol << ") ---\n";
		print_binding_violation(prog.bounding_box_constraints(),   "BoundingBox");
		print_binding_violation(prog.linear_equality_constraints(),"LinEq");
		print_binding_violation(prog.linear_constraints(),         "LinIneq");
		print_binding_violation(prog.lorentz_cone_constraints(),   "Lorentz");
		print_binding_violation(prog.rotated_lorentz_cone_constraints(),"RotLorentz");
		print_binding_violation(prog.quadratic_constraints(),      "Quadratic");
		print_binding_violation(prog.exponential_cone_constraints(),"ExpCone");
		print_binding_violation(prog.generic_constraints(),        "Generic");
		// print_binding_violation(prog.polynomial_constraints(),     "Polynomial");
	} else {
		cout << "(No decision vector available to evaluate constraints.)\n";
	}

	// Try to print some solver-specific details (best-effort).
	try {
		const auto& sid = result.get_solver_id().name();
		if (sid == "Ipopt") {
			struct IpoptDetails {
				int status; int iterations; double objective; double dual_inf;
				double constr_viol; double comp_viol; double primal_step; double dual_step;
			};
			// If your Drake provides IpoptSolver::Details, use that exact type.
			// Example (adjust to your Drake version):
			// const auto& d = result.get_solver_details<IpoptSolver>();
			// cout << "Ipopt status: " << d.status << ", iters: " << d.iterations
			//      << ", constr_viol: " << d.constr_viol << "\n";
		} else if (sid == "OSQP") {
			// Example (adjust to your Drake version):
			// const auto& d = result.get_solver_details<OsqpSolver>();
			// cout << "OSQP status: " << d.status_val << ", iters: " << d.iter
			//      << ", obj: " << d.obj_val << "\n";
		} else if (sid == "Gurobi") {
			// const auto& d = result.get_solver_details<GurobiSolver>();
			// cout << "Gurobi status: " << d.optimization_status << ", iters: "
			//      << d.iteration_count << ", mip_gap: " << d.mip_relative_gap << "\n";
		}
	} catch (const std::exception& e) {
		cout << "(Could not print solver-specific details: " << e.what() << ")\n";
	}

	cout << "=== End Report ===\n";
}



bool GraphWaypointMPC::solve(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	// TODO: use x0
	GraphWaypointProblem problem = build_graph_waypoint_problem(_graph, remaining_vertices, x0);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);
	// auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		const int num_remaining_nodes = remaining_vertices.size();
		const int num_subgraph_assignables = problem.Assignments.rows();
		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;
		const int dim = _graph->dim;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				int variable_id = _graph->phi_to_variable_map.at(i);
				int subgraph_variable_id = problem.subgraph->subgraph_variable_id(variable_id);
				if (subgraph_variable_id != -1) {
					for (int j = 0; j < num_agents; ++j) {
						const double val = result.GetSolution(problem.Assignments(subgraph_variable_id, j));
						if (val > 0.5) {
							_assignments(i) = j;
							_var_assignments(variable_id) = j;
							break;
						}
					}
				} else {
					_assignments(i) = -1;
					_var_assignments(variable_id) = -1;
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			// Eigen::RowVectorXd row(num_agents * dim + );
			// for (int j = 0; j < num_agents; ++j) {
			// 	row.segment(j * dim, dim) = X_flat.row(i).segment(j * dim * num_agents + j);
			// }
			_waypoints.row(v) = X_flat.row(i);
		}
		return true;
	} else {
		// PrintSolverReport(*(problem.prog), result, 1e-6);
		return false;
	}
}

