#pragma once

#include <limits>
#include <iostream>
#include <algorithm>
#include <memory>
#include <set>

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
#include "graph_waypoint_mpc.hpp"
#include "waypoint_common.hpp"
#include "../configuration_spline.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphWaypointProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::unique_ptr<SubgraphOfConstraints> subgraph;
	drake::solvers::MatrixXDecisionVariable Assignments;
	drake::solvers::MatrixXDecisionVariable W;
	std::unique_ptr<drake::solvers::Binding<drake::solvers::BoundingBoxConstraint>> A_bounds;

	// Routing MILP variables.
	// Routing graph N = {depot=0} ∪ task_nodes(1..n_V) ∪ {sink=n_V+1}, |N|=n_N.
	// Z[j*n_N^2 + a*n_N + b]: binary, agent j traverses arc (a,b).
	drake::solvers::VectorXDecisionVariable Z;
	// t[i]: shared timestamp at routing node i (depot fixed to 0).
	drake::solvers::VectorXDecisionVariable t;
	// e[j*n_N^2 + a*n_N + b]: realized arc travel cost (L1 distance when arc used).
	drake::solvers::VectorXDecisionVariable e;
	// L_ms[0]: makespan t(sink) (objective of the MinMax variants).
	drake::solvers::VectorXDecisionVariable L_ms;
	// Routing graph size (cached for solution extraction).
	int n_N{0};

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
	WaypointObjective objective = WaypointObjective::kMinMaxL1,
	SubtourElimination subtour = SubtourElimination::kNone,
	double length_weight = 0.0);

// MILP-formulation waypoint solver: builds one Drake MathematicalProgram
// (BuildGraphWaypointProblem, below) covering assignment, routing, and
// per-node configuration jointly, and hands it to Gurobi/Mosek (or, for
// kIPOPT, an enumeration+IPOPT fallback — currently disabled, see
// SolveWithEnumerationAndIPOPT).
struct MILPWaypointMPC : public GraphWaypointMPC {
	// Routing MILP outputs (set after each successful Gurobi/Mosek solve).
	// _t_routing is (n_N,): shared timestamps per routing node.
	Eigen::VectorXd _t_routing;
	// _Z_routing is (num_agents * n_N * n_N,): binary routing arcs.
	Eigen::VectorXd _Z_routing;

	// Solver configuration
	WaypointSolver _solver;
	bool _enforce_rigidity;
	SubtourElimination _subtour;
	double _length_weight;

	// Constructor
	MILPWaypointMPC(GraphOfConstraints& graph,
			std::vector<CubicConfigurationSpline> splines,
			WaypointSolver solver = WaypointSolver::kGurobi,
			bool enforce_rigidity = false,
			WaypointObjective objective = WaypointObjective::kMinMaxL1,
			SubtourElimination subtour = SubtourElimination::kNone,
			double length_weight = 0.0);

	// Core solve routine, based on the remaining vertices, computes a
	// subgraph of graph of constraints, solves for the optimal agent
	// assignment in that graph based on some heuristics, as well as the
	// sequence of positions for the agents to satisfy the constraints.
	bool Solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0) override;

	// MILP-only routing diagnostics — not part of the WaypointMPC contract.
	const Eigen::VectorXd &view_t() { return _t_routing; }
	const Eigen::VectorXd &view_Z() { return _Z_routing; }

private:
	bool SolveWith(const drake::solvers::SolverInterface& solver,
		       const std::vector<int>& remaining_vertices,
		       const Eigen::VectorXd& x0);
};
