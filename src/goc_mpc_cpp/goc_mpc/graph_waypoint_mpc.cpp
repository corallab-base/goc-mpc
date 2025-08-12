#include "graph_waypoint_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(const py::array_t<double>& graph,
				   unsigned int m,
				   unsigned int z,
				   unsigned int d)
	: _graph(graph),
	  _m(m),
	  _z(z),
	  _d(d) {

	_n = graph.shape(0);

	// initialize output assignment array (z,)
	_assignments = py::array_t<unsigned int>(_z);
	auto assignments_mut = _assignments.mutable_unchecked<1>();
	for (ssize_t i = 0; i < _z; ++i) {
		assignments_mut(i) = 0;
	}

	// initialize output waypoints array (n, d)
	_waypoints = py::array_t<double>({_n, _d});
	auto waypoints_mut = _waypoints.mutable_unchecked<2>();
	for (size_t i = 0; i < _n; ++i) {
		for (size_t j = 0; j < _d; ++j) {
			waypoints_mut(i, j) = 0.0;
		}
	}
}

int GraphWaypointMPC::solve() {

	struct GraphWaypointProblem problem = build_graph_waypoint_problem(
		_graph, _z, _m, _n, _d);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);
	// auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		std::cout << "Success" << std::endl;

		auto assignments_mut = _assignments.mutable_unchecked<1>();
		for (int i = 0; i < _z; ++i) {
			for (int j = 0; j < _m; ++j) {
				const double val = result.GetSolution(problem.A(i, j));
				if (val > 0.5) {
					assignments_mut(i) = j;
					break;
				}
			}
		}

		const auto X = result.GetSolution(problem.X);
		auto waypoints_mut = _waypoints.mutable_unchecked<2>();
		for (int i = 0; i < _n; ++i) {
			for (int j = 0; j < _d; ++j) {
				waypoints_mut(i, j) = X(i, j);
			}
		}
	} else {
		std::cerr << "Optimization failed." << std::endl;
	}

	return 0;
}

// Safe indexing and accessors
py::array_t<double> GraphWaypointMPC::get_waypoints() const {
	return _waypoints;
}
