#pragma once

#include <memory>
#include <vector>

#include <drake/common/timer.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"

// Solver-agnostic waypoint objective: MinMax variants minimize the makespan
// (the worst per-agent route cost), Avg variants minimize the average route
// cost across agents. L1/L2 select the per-edge distance metric.
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
	// _waypoints is (_graph.num_nodes, _graph.num_agents * _graph.dim)
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

	GraphWaypointMPC(GraphOfConstraints& graph,
			 std::vector<CubicConfigurationSpline> splines,
			 WaypointObjective objective);

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
