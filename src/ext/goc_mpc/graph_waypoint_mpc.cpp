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
	const std::vector<size_t>& remaining_vertices,
	Eigen::VectorXd x0) {

	const int num_agents = graph->num_agents;

	const InducedSubgraphView<py::object> subgraph = InducedSubgraphView<py::object>(graph->structure, remaining_vertices);
	std::map<size_t, size_t> phi_to_subgraph_node_id;
	std::map<size_t, size_t> phi_to_subgraph_assignable_id;
	std::map<size_t, size_t> subgraph_assignable_id_to_phi;
	const int num_nodes = subgraph.num_nodes();

	size_t num_subgraph_assignables = 0;
	std::vector<DeferredOp> subgraph_ops;
	for (int v : remaining_vertices) {
		if (graph->phi_map.contains(v)) {
			// Store the relevant ops so they can be applied
			const size_t phi_id = graph->phi_map.at(v);

			subgraph_ops.push_back(graph->ops.at(phi_id));

			// Record the mapping from phi id to subgraph node and assignable var idxs.
			phi_to_subgraph_node_id[phi_id] = subgraph.subgraph_id(v);
			if (graph->ops.at(phi_id).kind == DeferredOpKind::kAgentLinearEq) {
				subgraph_assignable_id_to_phi[num_subgraph_assignables] = phi_id;
				phi_to_subgraph_assignable_id[phi_id] = num_subgraph_assignables++;
			}
		}
	}

	using namespace drake::solvers;

	// Create program
	GraphWaypointProblem problem;
	problem.prog = std::make_unique<MathematicalProgram>();

	// record the subgraph
	problem.subgraph = std::make_unique<InducedSubgraphView<py::object>>(subgraph);

	// record subgraph_assignable_id_to_phi because eventually we want to ascribe the assignments to specific phis
	problem.subgraph_assignable_id_to_phi = subgraph_assignable_id_to_phi;
	problem.phi_to_subgraph_assignable_id = phi_to_subgraph_assignable_id;

	// a: binary assignment variables (z x m).
	// Drake exposes a direct API for binary matrices. :contentReference[oaicite:1]{index=1}
	MatrixXDecisionVariable Assignments = problem.prog->NewBinaryVariables(num_subgraph_assignables, num_agents, "Assignments");
	problem.Assignments = Assignments;

	// One-hot per row (each task gets exactly one agent): sum_k a(row,k) = 1.
	for (int i = 0; i < num_subgraph_assignables; ++i) {
		problem.prog->AddLinearEqualityConstraint(
			Eigen::RowVectorXd::Ones(num_agents),
			1.0, Assignments.row(i));
	}

	// x: continuous configuration variables (n x d).
	MatrixXDecisionVariable X = problem.prog->NewContinuousVariables(num_nodes * num_agents, graph->dim, "X");
	problem.X = X;

	//
	// OBJECTIVE FUNCTION
	//

	// First, costs to minimize across transitions from x0 to the source
	// nodes in the subgraph.
	const size_t dim = graph->dim;
	for (auto v : subgraph.sources()) {
		size_t sg_v = subgraph.subgraph_id(v);
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
	for (auto edge : subgraph.edges()) {
		size_t u = edge.u;
		size_t sg_u = subgraph.subgraph_id(u);
		size_t v = edge.e->to;
		size_t sg_v = subgraph.subgraph_id(v);
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
	for (const DeferredOp op : subgraph_ops) {
		op.builder(*(problem.prog), problem.X, problem.Assignments, phi_to_subgraph_node_id, phi_to_subgraph_assignable_id);
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
	const std::vector<size_t>& remaining_vertices,
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
		const size_t num_phis = _graph->num_phis;
		const size_t num_agents = _graph->num_agents;
		const size_t dim = _graph->dim;

		for (size_t i = 0; i < num_phis; ++i) {
			if (problem.phi_to_subgraph_assignable_id.contains(i)) {
				const size_t subgraph_assignable_id = problem.subgraph_assignable_id_to_phi[i];
				for (int j = 0; j < num_agents; ++j) {
					const double val = result.GetSolution(problem.Assignments(subgraph_assignable_id, j));
					if (val > 0.5) {
						_assignments(i) = j;
						break;
					}
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		for (size_t v : remaining_vertices) {
			const size_t i = problem.subgraph->subgraph_id(v);
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

