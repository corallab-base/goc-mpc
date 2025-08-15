#include "graph_waypoint_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


GraphWaypointProblem build_graph_waypoint_problem(
	const GraphOfConstraints* graph,
	const std::vector<size_t>& remaining_vertices) {

	const int num_agents = graph->num_agents;

	const InducedSubgraphView<py::object> subgraph = InducedSubgraphView<py::object>(graph->structure, remaining_vertices);
	std::map<size_t, size_t> phi_to_subgraph_node_id;
	std::map<size_t, size_t> phi_to_subgraph_assignable_id;
	std::map<size_t, size_t> subgraph_assignable_id_to_phi;
	const int num_nodes = subgraph.num_nodes();

	std::cout << "here1" << std::endl;
	
	size_t num_subgraph_assignables = 0;
	std::vector<DeferredOp> subgraph_ops;
	for (int v : remaining_vertices) {
		std::cout << "v: " << v << std::endl;

		if (graph->phi_map.contains(v)) {
			// Store the relevant ops so they can be applied
			const size_t phi_id = graph->phi_map.at(v);

			std::cout << "phi_id: " << phi_id << std::endl;

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

	// record subgraph_assignable_id_to_phi because eventually we want to ascribe the assignments to specific phis
	problem.subgraph_assignable_id_to_phi = subgraph_assignable_id_to_phi;

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

	// Add constraints/costs from registry
	for (const DeferredOp op : subgraph_ops) {
		op.builder(*(problem.prog), problem.X, problem.Assignments, phi_to_subgraph_node_id, phi_to_subgraph_assignable_id);
	}

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(const GraphOfConstraints& graph)
	: _graph(&graph) {}

std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXi>> GraphWaypointMPC::solve(
	const std::vector<size_t>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	// TODO: use x0

	// unsigned int num_subgraph_assignables = 0;
	// Eigen::MatrixXi subgraph(num_remaining_nodes, num_remaining_nodes);
	// _phi_to_subgraph_node_id.clear();
	// _phi_to_subgraph_assignable_id.clear();

	// int i = 0;
	// for (auto i_it = remaining_vertices.begin(); i_it != remaining_vertices.end(); ++i_it) {
	// 	// Record the mapping from phi id to subgraph node and assignable var idxs.
	// 	const unsigned int phi_id = _graph_to_phi_map[*i_it];
	// 	const DeferredOp& op = _ops[phi_id];
	// 	_phi_to_subgraph_node_id[phi_id] = i;
	// 	if (op.kind == DeferredOpKind::kAgentLinearEq) {
	// 		_phi_to_subgraph_assignable_id[phi_id] = num_subgraph_assignables++;
	// 	}

	// 	// Compute subgraph row.
	// 	int j = 0;
	// 	for (auto j_it = remaining_vertices.begin(); j_it != remaining_vertices.end(); ++j_it) {
	// 		subgraph(i, j) = _graph(*i_it, *j_it);
	// 		++j;
	// 	}
	// 	++i;
	// }

	// std::cout << "Subgraph m:\n" << subgraph << std::endl;

	GraphWaypointProblem problem = build_graph_waypoint_problem(_graph, remaining_vertices);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);
	// auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		std::cout << "Success" << std::endl;

		const int num_remaining_nodes = remaining_vertices.size();
		const int num_subgraph_assignables = problem.Assignments.rows();
		const size_t num_phis = _graph->num_phis;
		const size_t num_agents = _graph->num_agents;
		const size_t dim = _graph->dim;

		Eigen::VectorXi assignments = Eigen::VectorXi::Constant(num_phis, -1);
		for (size_t i = 0; i < num_subgraph_assignables; ++i) {
			const size_t phi_id = problem.subgraph_assignable_id_to_phi[i];

			for (int j = 0; j < num_agents; ++j) {
				const double val = result.GetSolution(problem.Assignments(i, j));
				if (val > 0.5) {
					assignments(phi_id) = j;
					break;
				}
			}
		}

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		Eigen::MatrixXd X(num_remaining_nodes, num_agents * dim);
		for (int i = 0; i < num_remaining_nodes; ++i) {
			// take block of m rows and stack them horizontally
			Eigen::RowVectorXd row(num_agents * dim);
			for (int j = 0; j < num_agents; ++j) {
				row.segment(j * dim, dim) = X_flat.row(i * num_agents + j);
			}
			X.row(i) = row;
		}

		return std::make_pair(X, assignments);
	} else {
		std::cerr << "Optimization failed." << std::endl;
		return std::nullopt;
	}
}

