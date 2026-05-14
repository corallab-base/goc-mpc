#pragma once

#include <limits>
#include <iostream>
#include <algorithm>

#include <fmt/format.h>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/snopt_solver.h>
#include <drake/solvers/nlopt_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/solve.h>
#include <drake/solvers/solver_options.h>
#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


enum class WaypointSolver { kGurobi, kMosek, kIPOPT };
enum class WaypointObjective { kSquaredDistance, kL1 };


struct GraphWaypointProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::unique_ptr<SubgraphOfConstraints> subgraph;
	drake::solvers::MatrixXDecisionVariable Assignments;
	drake::solvers::MatrixXDecisionVariable X;
	std::unique_ptr<drake::solvers::Binding<drake::solvers::BoundingBoxConstraint>> A_bounds;

	GraphWaypointProblem()
		: prog(nullptr),
		  subgraph(nullptr),
		  A_bounds(nullptr) {}

	GraphWaypointProblem(const GraphWaypointProblem&) = delete;
	GraphWaypointProblem& operator=(const GraphWaypointProblem&) = delete;

	GraphWaypointProblem(GraphWaypointProblem&&) = default;
	GraphWaypointProblem& operator=(GraphWaypointProblem&&) = default;
};

GraphWaypointProblem BuildGraphWaypointProblem(
	GraphOfConstraints* graph,
	std::shared_ptr<std::vector<CubicConfigurationSpline>> splines,
	const std::vector<int>& remaining_vertices,
	Eigen::VectorXd x0,
	Eigen::MatrixXd previous_X,
	Eigen::VectorXi previous_var_assignments,
	bool enforce_rigidity,
	bool relax_binary_vars,
	WaypointObjective objective = WaypointObjective::kSquaredDistance);

struct GraphWaypointMPC {
	// reference to graph of constraints object.
	GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// persistent output buffers;
	// _waypoints is (_graph.num_nodes, _graph.num_agents * _graph.dim)
	Eigen::MatrixXd _waypoints;
	// _assignments is (_graph.num_phis,)
	Eigen::VectorXi _assignments;
	// _var_assignments is (_graph.num_variables,)
	Eigen::VectorXi _var_assignments;
	bool _first_cycle;

	// Solver configuration
	WaypointSolver _solver;
	bool _enforce_rigidity;
	WaypointObjective _objective;

	// Recording Metrics
	drake::SteadyTimer _timer;
	double _last_solve_time;

	// Constructor
	GraphWaypointMPC(GraphOfConstraints& graph,
			 std::vector<CubicConfigurationSpline> splines,
			 WaypointSolver solver = WaypointSolver::kGurobi,
			 bool enforce_rigidity = false,
			 WaypointObjective objective = WaypointObjective::kSquaredDistance);

	// Core solve routine, based on the remaining vertices, computes a
	// subgraph of graph of constraints, solves for the optimal agent
	// assignment in that graph based on some heuristics, as well as the
	// sequence of positions for the agents to satisfy the constraints.
	bool solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0);

	const Eigen::MatrixXd &view_waypoints() { return _waypoints; }
	const Eigen::VectorXi &view_assignments() { return _assignments; }
	const Eigen::VectorXi &view_var_assignments() { return _var_assignments; }
	const double get_last_solve_time() { return _last_solve_time; }

private:
	bool SolveWithMosek(const std::vector<int>& remaining_vertices,
			    const Eigen::VectorXd& x0);
	bool SolveWithGurobi(const std::vector<int>& remaining_vertices,
			     const Eigen::VectorXd& x0);
	bool SolveWithEnumerationAndIPOPT(const std::vector<int>& remaining_vertices,
					  const Eigen::VectorXd& x0);
};
