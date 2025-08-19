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

	const SubgraphOfConstraints subgraph(graph, remaining_vertices);

	// const InducedSubgraphView<py::object> subgraph = InducedSubgraphView<py::object>(graph->structure, remaining_vertices);
	// std::map<int, int> phi_to_subgraph_node_id;
	// std::map<int, int> phi_to_subgraph_assignable_id;
	// std::map<int, int> subgraph_assignable_id_to_phi;
	// const int num_nodes = subgraph.num_nodes();

	// int num_subgraph_assignables = 0;
	// std::vector<DeferredOp> subgraph_ops;
	// for (int v : remaining_vertices) {
	// 	if (graph->node_to_phi_map.contains(v)) {
	// 		// Store the relevant ops so they can be applied
	// 		const int phi_id = graph->node_to_phi_map.at(v);

	// 		subgraph_ops.push_back(graph->ops.at(phi_id));

	// 		// Record the mapping from phi id to subgraph node and assignable var idxs.
	// 		phi_to_subgraph_node_id[phi_id] = subgraph.subgraph_id(v);
	// 		if (graph->ops.at(phi_id).kind == DeferredOpKind::kAgentLinearEq) {
	// 			subgraph_assignable_id_to_phi[num_subgraph_assignables] = phi_id;
	// 			phi_to_subgraph_assignable_id[phi_id] = num_subgraph_assignables++;
	// 		}
	// 	}
	// }

	using namespace drake::solvers;

	// Create program
	GraphWaypointProblem problem;
	problem.prog = std::make_unique<MathematicalProgram>();

	// record the subgraph
	problem.subgraph = std::make_unique<SubgraphOfConstraints>(subgraph);

	// record subgraph_assignable_id_to_phi because eventually we want to ascribe the assignments to specific phis
	// problem.subgraph_assignable_id_to_phi = subgraph_assignable_id_to_phi;
	// problem.phi_to_subgraph_assignable_id = phi_to_subgraph_assignable_id;

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

	// x: continuous configuration variables (n x d).
	MatrixXDecisionVariable X = problem.prog->NewContinuousVariables(subgraph.num_nodes() * num_agents, graph->dim, "X");
	problem.X = X;

	//
	// OBJECTIVE FUNCTION
	//

	// First, costs to minimize across transitions from x0 to the source
	// nodes in the subgraph.
	const int dim = graph->dim;
	for (auto v : subgraph.structure.sources()) {
		int sg_v = subgraph.subgraph_id(v);
		for (int ag = 0; ag < num_agents; ++ag) {
			const int node_v_ag_idx = sg_v * num_agents + ag;
			VectorX<Expression> diff = x0.segment(ag * dim, dim) - X.row(node_v_ag_idx);
			Expression dist = diff.squaredNorm();
			problem.prog->AddQuadraticCost(dist);
		}
	}

	// Add inter-waypoint costs for all edges
	// You control the cost body via add_edge_cost
	// Iterate edges of the subgraph
	for (auto edge : subgraph.structure.edges()) {
		int u = edge.u;
		int sg_u = subgraph.subgraph_id(u);
		int v = edge.e->to;
		int sg_v = subgraph.subgraph_id(v);
		for (int ag = 0; ag < num_agents; ++ag) {
			const int node_u_ag_idx = sg_u * num_agents + ag;
			const int node_v_ag_idx = sg_v * num_agents + ag;
			VectorX<Expression> diff = X.row(node_u_ag_idx) - X.row(node_v_ag_idx);
			Expression dist = diff.squaredNorm();
			problem.prog->AddQuadraticCost(dist);
		}
	}

	// clear graph of constraints' _constraints_per_phi
	graph->clear_constraints_per_phi();

	// Add constraints/costs from registry
	for (const auto& [phi_id, op] : subgraph.get_subgraph_ops()) {
		op.builder(*(problem.prog), subgraph, phi_id, problem.X, problem.Assignments);
	}

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(GraphOfConstraints& graph)
	: _graph(&graph) {
	// Allocate persistent output buffers.
	_waypoints = Eigen::MatrixXd::Zero(_graph->structure.num_nodes(), _graph->num_agents * _graph->dim);
	_assignments = Eigen::VectorXi::Zero(_graph->num_phis);
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
							break;
						}
					}
				} else {
					_assignments(i) = -1;
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			Eigen::RowVectorXd row(num_agents * dim);
			for (int j = 0; j < num_agents; ++j) {
				row.segment(j * dim, dim) = X_flat.row(i * num_agents + j);
			}
			_waypoints.row(i) = row;
		}

		return true;
	} else {
		return false;
	}
}

