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
#include "../configuration_spline.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


enum class WaypointSolver { kGurobi, kMosek, kIPOPT };


// Shared MILP-encoding helpers, also reused (unchanged) by
// evolutionary_waypoint_mpc.cpp: Assignments/cond_binary_vars remain real
// decision variables there too, just fed to a different eventual solving
// strategy, so these closures/compilers apply verbatim.

// L1 distance between two symbolic configuration expressions, summed over a
// spline's R blocks only, encoded via one slack variable per component plus
// two linear inequalities (no big-M, LP-representable).
drake::symbolic::Expression ComputeL1Distance(
	drake::solvers::MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<drake::symbolic::Expression>& x1,
	const VecX<drake::symbolic::Expression>& x2,
	const std::string& tag);

// L2 (Euclidean) distance between two symbolic configuration expressions,
// summed over a spline's R blocks only, one nonnegative scalar per block
// constrained via a Lorentz cone.
drake::symbolic::Expression ComputeL2Distance(
	drake::solvers::MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<drake::symbolic::Expression>& x1,
	const VecX<drake::symbolic::Expression>& x2,
	const std::string& tag);

// Compile b_same: binary MILP variable = 1 iff variables r0 and r1 are
// assigned to the same agent in the current (or previous) solve.
drake::symbolic::Variable CompileVarEq(
	drake::solvers::MathematicalProgram& prog,
	int r0, int r1,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments);

// Compile a conditional ordering formula to a binary MILP variable.
// Returns a variable b such that b=1 iff the formula holds.
// Recognised atoms: VarEq (r_sym_i == r_sym_j) and BinVarEq (bv_sym == 0/1).
// Compound formulas use Drake FormulaKind::And / Or / Not.
drake::symbolic::Variable CompileConditionToBinary(
	drake::solvers::MathematicalProgram& prog,
	const drake::symbolic::Formula& f,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments,
	const GraphOfConstraints* graph,
	const std::vector<drake::symbolic::Variable>& cond_binary_vars);

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: t(t_idx_w) weakly between t(t_idx_u) and t(t_idx_v)  ⟹  b = 1.
// One-directional by design: free to be 0 when the node isn't between, but
// never forced to 0 when it genuinely is between.
drake::symbolic::Variable GetOrAddBetweennessIndicator(
	drake::solvers::MathematicalProgram& prog,
	const drake::solvers::VectorXDecisionVariable& t_var,
	int t_idx_u, int t_idx_w, int t_idx_v,
	double M_time,
	std::map<std::tuple<int, int, int>, drake::symbolic::Variable>& cache);

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: [t(t_idx_a1), t(t_idx_b1)] overlaps [t(t_idx_a2), t(t_idx_b2)] ⟹ b=1.
// Same one-directional soundness contract as GetOrAddBetweennessIndicator.
drake::symbolic::Variable GetOrAddIntervalOverlapIndicator(
	drake::solvers::MathematicalProgram& prog,
	const drake::solvers::VectorXDecisionVariable& t_var,
	int t_idx_a1, int t_idx_b1, int t_idx_a2, int t_idx_b2,
	double M_time,
	std::map<std::tuple<int, int, int, int>, drake::symbolic::Variable>& cache);


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
	// L_ms[0]: makespan scalar (objective).
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
	WaypointObjective objective = WaypointObjective::kMinMaxL1);

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

	// remaining_vertices as of the previous Solve() call -- lets Solve()
	// detect a node's active->passed transition (see Solve()'s live-rebind
	// comment) instead of just its current active/passed state. Unset
	// before the first Solve() call, so no transition is possible there
	// (the initial remaining_vertices is the graph's starting condition,
	// not something that "just passed").
	std::set<int> _prev_remaining_set;
	bool _has_prev_remaining_set{false};

	// Constructor
	MILPWaypointMPC(GraphOfConstraints& graph,
			std::vector<CubicConfigurationSpline> splines,
			WaypointSolver solver = WaypointSolver::kGurobi,
			bool enforce_rigidity = false,
			WaypointObjective objective = WaypointObjective::kMinMaxL1);

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
	bool SolveWithMosek(const std::vector<int>& remaining_vertices,
			    const Eigen::VectorXd& x0);
	bool SolveWithGurobi(const std::vector<int>& remaining_vertices,
			     const Eigen::VectorXd& x0);
};
