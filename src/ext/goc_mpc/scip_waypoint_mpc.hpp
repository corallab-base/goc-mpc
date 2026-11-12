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
	// Local NLP relaxations, one per node (nonlinear constraints only): how
	// many, how many missed SCIP's feasibility tolerance, and their total
	// wall time.
	int local_solves = 0, local_failures = 0;
	double local_solve_time = 0.0;
	// How the last failed one ended.
	std::string last_local_failure;
	// Every local solve, in order.
	std::vector<LocalSolveRecord> local_log;
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
// Nonlinear node constraints (e.g. FK pins) are only locally optimized, as
// NLP-based branch-and-bound on the integer variables alone: SCIP only
// checks them, and each node's relaxation replaces SCIP's LP with its
// presolved problem at the node -- integers relaxed, the nonlinear
// constraints and cones kept -- solved locally (local_solver) from the
// initialization_oracle's guess (without one, the agents' current
// configurations) and treated as globally optimal: its value prunes, its fractional
// integers are branched on, and an integral one is a solution. A local
// solve that misses feasibility cuts off only a node whose integers are all
// fixed; otherwise SCIP branches without a bound from it. Bounds and
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
	// When the solve finds no solution, store the last failed local NLP
	// iterate (integers rounded) as the waypoints anyway, unchecked -- for
	// inspecting where the local solve got stuck. Solve still returns false.
	bool _keep_failed_local;
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
			bool keep_failed_local = false,
			LocalNLPSolver local_solver = LocalNLPSolver::kPenalizedLM,
			std::shared_ptr<InitializationOracle> initialization_oracle = nullptr);

	bool Solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0) override;

	const SCIPSolveStats& view_stats() const { return _stats; }

protected:
	// x holds SCIP's variables in decision-variable order (mip's indexing).
	virtual void Configure(SCIP* scip, const std::vector<SCIP_VAR*>& x, const ExternalMIP& mip) {}
};
