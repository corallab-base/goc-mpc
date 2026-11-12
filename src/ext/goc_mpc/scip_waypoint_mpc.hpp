#pragma once

#include <limits>
#include <map>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <scip/type_scip.h>
#include <scip/type_var.h>

#include "gcs_waypoint_mpc.hpp"
#include "local_nlp_solver.hpp"

// The last SCIPWaypointMPC solve, as SCIP reported it.
// One branch-and-bound node, as SCIP's node events and the local NLP
// relaxation saw it.
struct SCIPNodeRecord {
	long long number = -1, parent = -1;
	int depth = 0;
	// Seconds into the solve when SCIP focused the node (NaN: never focused,
	// e.g. deleted after a better bound pruned it).
	double focus_time = std::numeric_limits<double>::quiet_NaN();
	// The branching bound changes from its parent ("name <= v" / "name >=
	// v"), by the exported program's variable names.
	std::vector<std::string> branchings;
	// Assignments(role, agent) at its last local solve's final point.
	std::vector<std::string> assignments;
	// Its local NLP solves (indices into local_log) and how the last ended:
	// "feasible", "infeasible: cut off (leaf)" or "infeasible: branched
	// (integers free)".
	std::vector<int> local_solves;
	std::string relaxation;
	double lower_bound = std::numeric_limits<double>::quiet_NaN();
	// SCIP's events on the node, in order: focused, feasible, infeasible,
	// branched, deleted.
	std::vector<std::string> events;
};

struct SCIPSolveStats {
	std::string status;
	long long nodes = 0;
	double primal_bound = std::numeric_limits<double>::quiet_NaN();
	double dual_bound = std::numeric_limits<double>::quiet_NaN();
	double gap = std::numeric_limits<double>::quiet_NaN();
	double solving_time = 0.0;
	// The solutions SCIP kept, by the time (seconds into the solve) each
	// was found: that time and its objective.
	std::vector<double> solution_times, solution_objectives;
	// Local NLP relaxations, one per node: how
	// many, how many missed SCIP's feasibility tolerance, and their total
	// wall time.
	int local_solves = 0, local_failures = 0;
	double local_solve_time = 0.0;
	// How the last failed one ended.
	std::string last_local_failure;
	// The solution is the least-violated failed leaf solve (no feasible one
	// was found), missing the nonlinear rows and cones by fallback_violation.
	bool fallback = false;
	double fallback_violation = 0.0;
	// Every local solve, in order.
	std::vector<LocalSolveRecord> local_log;
	// Every node SCIP created, in order of first event; and its leaf counts
	// (feasible, infeasible, pruned by the objective limit / bound).
	std::vector<SCIPNodeRecord> node_log;
	long long feasible_leaves = 0, infeasible_leaves = 0, objlim_leaves = 0;
};

// GCSWaypointMPC's problem (a MILP, or a MISOCP under the L2 objectives)
// solved by SCIP instead of Gurobi/Mosek. The problem is built exactly as
// GCSWaypointMPC builds it, handed to SCIP through BeginExternalSolve's
// ExternalMIP, and SCIP's solution is checked and stored by
// FinishExternalSolve, so every view_* is inherited unchanged.
//
// Each cone z = A x + b, z(0) >= |z(1:)|, becomes auxiliary variables tied to
// A x + b by linear rows plus one SOC nonlinear constraint over them.
//
// The solve is NLP-based branch-and-bound on the integer variables alone,
// whatever the problem holds: SCIP only checks the cones and nonlinear node
// constraints (e.g. FK pins), and each node's relaxation replaces SCIP's LP
// with its presolved problem at the node -- integers relaxed, the nonlinear
// constraints and cones kept -- solved locally (local_solver) from the
// initialization_oracle's guess (without one, the agents' current
// configurations), and treated as globally optimal: its value
// prunes, its fractional integers are branched on, and an integral one is a
// solution. A local
// solve that misses feasibility cuts off only a node whose integers are all
// fixed; otherwise SCIP branches without a bound from it. A cut-off leaf
// whose linear rows hold is kept outside SCIP as a candidate, and when SCIP
// finds no solution the least-violated one is returned
// (fallback_to_least_violated). Bounds and
// gap are then only as good as the local solves; SCIP's LP-based machinery
// (cuts, strong branching, LP heuristics) is off.
//
// Configure() is the hook for changing SCIP's internals: override it to
// include relaxators, primal heuristics, branching rules, node selectors or
// constraint handlers on the problem before it is solved.
struct SCIPWaypointMPC : public GCSWaypointMPC {
	using ParamValue = std::variant<bool, long long, double, std::string>;

	// Seconds per solve; infinity for none.
	double _time_limit;
	// Applied after the defaults (limits/gap from mip_gap, limits/time,
	// separating/aggregation/freq = -1, presolving/maxrestarts = 0,
	// separating/maxrounds = 1), each
	// converted to the parameter's own SCIP type.
	std::map<std::string, ParamValue> _params;
	// FinishExternalSolve's tolerance on SCIP's solution.
	double _feasibility_tol;
	bool _verbose;
	// When SCIP finds no solution, store the least-violated failed leaf
	// local solve (integers rounded; linear rows held, nonlinear rows and
	// cones missed by _stats.fallback_violation) and return true.
	bool _fallback_to_least_violated;
	// The local NLP solver for the node relaxations.
	LocalNLPSolver _local_solver;
	SCIPSolveStats _stats;

	SCIPWaypointMPC(GraphOfConstraints& graph,
			std::vector<CubicConfigurationSpline> splines,
			bool enforce_rigidity = false,
			WaypointObjective objective = WaypointObjective::kMinMaxL1,
			SubtourElimination subtour = SubtourElimination::kMCF,
			double length_weight = 0.0,
			double mip_gap = 1e-4,
			bool warm_start = true,
			std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models = {},
			double max_tangent_halfwidth = std::numeric_limits<double>::infinity(),
			double time_limit = std::numeric_limits<double>::infinity(),
			std::map<std::string, ParamValue> params = {},
			double feasibility_tol = 1e-5,
			bool verbose = false,
			bool fallback_to_least_violated = true,
			LocalNLPSolver local_solver = LocalNLPSolver::kPenalizedLM,
			std::shared_ptr<InitializationOracle> initialization_oracle = nullptr);

	bool Solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0) override;

	const SCIPSolveStats& view_stats() const { return _stats; }

protected:
	// x holds SCIP's variables in decision-variable order (mip's indexing).
	virtual void Configure(SCIP* scip, const std::vector<SCIP_VAR*>& x, const ExternalMIP& mip) {}
};
