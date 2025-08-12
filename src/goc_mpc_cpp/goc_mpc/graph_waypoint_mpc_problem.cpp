#include "graph_waypoint_mpc_problem.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


GraphWaypointProblem build_graph_waypoint_problem(
	const py::array_t<unsigned int>& graph,
	int z, int m, int n, int d) {

	using namespace drake::solvers;

	// Create program
	GraphWaypointProblem problem;

	// a: binary assignment variables (z x m).
	// Drake exposes a direct API for binary matrices. :contentReference[oaicite:1]{index=1}
	MatrixXDecisionVariable A = problem.prog->NewBinaryVariables(z, m, "A");
	problem.A = A;

	// One-hot per row (each task gets exactly one agent): sum_k a(row,k) = 1.
	for (int i = 0; i < z; ++i) {
		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(n), 1.0, A.row(i));
	}

	// x: continuous configuration variables (n x d).
	MatrixXDecisionVariable X = problem.prog->NewContinuousVariables(n, d, "X");
	problem.X = X;

	//
	// OBJECTIVE FUNCTION
	//

	// Add inter-waypoint costs for edges (i, j) indicated by adj(i,j) ≠ 0.
	// You control the cost body via `add_edge_cost`.
	auto graph_u = graph.unchecked<2>();
	const Eigen::MatrixXd Q = Eigen::MatrixXd::Identity(n, n);
	const Eigen::VectorXd b = Eigen::VectorXd::Zero(n);
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			if (graph_u(i, j) != 0 && i != j) {
				VectorX<Expression> diff = X.row(i) - X.row(j);
				Expression dist = diff.squaredNorm();
				problem.prog->AddQuadraticCost(dist);
			}
		}
	}

	return std::move(problem);
}
