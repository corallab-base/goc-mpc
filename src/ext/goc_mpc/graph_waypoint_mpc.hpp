#pragma once

#include <memory>
#include <set>
#include <vector>

#include <drake/common/timer.h>
#include <drake/solvers/mathematical_program_result.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"

// Solver-agnostic waypoint objective: MinMax variants minimize the makespan
// (the time the last agent finishes, including any waiting forced by
// orderings), Avg variants minimize the average route cost across agents.
// L1/L2 select the per-edge distance metric.
enum class WaypointObjective { kMinMaxL1, kAvgL1, kMinMaxL2, kAvgL2 };

// Abstract base for a waypoint-phase solver: given the remaining graph
// vertices and the current state x0, produces per-node configurations
// (waypoints), per-constraint and per-variable agent assignments, and a
// per-node arrival-time estimate. Holds the bookkeeping that's genuinely
// shared across solver families (output buffers, warm-start state, timing,
// objective/edge-cost config); Solve() is the one hook each concrete solver
// family (MILP/Gurobi/Mosek today, an evolutionary solver later) implements
// completely differently.
class GraphWaypointMPC {
protected:
	// reference to graph of constraints object.
	GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// persistent output buffers;
	// _waypoints is (_graph.num_nodes, _graph.total_dim)
	Eigen::MatrixXd _waypoints;
	// _assignments is (_graph.num_phis,)
	Eigen::VectorXi _assignments;
	// _var_assignments is (_graph.num_variables,)
	Eigen::VectorXi _var_assignments;
	bool _first_cycle;
	// _t_by_node_id is (num_graph_nodes,): per-node arrival-time estimate.
	Eigen::VectorXd _t_by_node_id;

	WaypointObjective _objective;

	// Recording Metrics
	drake::SteadyTimer _timer;
	double _last_solve_time;

	// remaining_vertices as of the previous Solve() call -- lets BeginSolve()
	// detect a node's active->passed transition instead of just its current
	// active/passed state. Unset before the first Solve() call, so no
	// transition is possible there (the initial remaining_vertices is the
	// graph's starting condition, not something that "just passed").
	std::set<int> _prev_remaining_set;
	bool _has_prev_remaining_set{false};

	GraphWaypointMPC(GraphOfConstraints& graph,
			 std::vector<CubicConfigurationSpline> splines,
			 WaypointObjective objective);

	// Start of a Drake-backed Solve(): starts the timer, warm-starts
	// _waypoints on the first cycle, and live-rebinds the rows of nodes that
	// just passed to the real x0.
	void BeginSolve(const std::vector<int>& remaining_vertices,
			const Eigen::VectorXd& x0);

	// Writes _assignments/_var_assignments from a solved Assignments matrix
	// (one row per subgraph variable, one column per agent).
	void StoreAssignments(const SubgraphOfConstraints& subgraph,
			      const drake::solvers::MatrixXDecisionVariable& Assignments,
			      const drake::solvers::MathematicalProgramResult& result);

public:
	virtual ~GraphWaypointMPC() = default;

	// Core solve routine: based on the remaining vertices, computes the
	// waypoint configurations and agent assignments satisfying the graph's
	// constraints, however the concrete solver family chooses to do so.
	virtual bool Solve(const std::vector<int>& remaining_vertices,
			   const Eigen::VectorXd& x0) = 0;

	const Eigen::MatrixXd& view_waypoints() const { return _waypoints; }
	const Eigen::VectorXi& view_assignments() const { return _assignments; }
	const Eigen::VectorXi& view_var_assignments() const { return _var_assignments; }
	const Eigen::VectorXd& view_t_by_node() const { return _t_by_node_id; }
	double get_last_solve_time() const { return _last_solve_time; }
};
