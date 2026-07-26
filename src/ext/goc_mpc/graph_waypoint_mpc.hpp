#pragma once

#include <memory>
#include <vector>

#include <drake/common/timer.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../edge_cost_functor.hpp"

// Solver-agnostic waypoint objective: MinMax variants minimize the makespan
// (the worst per-agent route cost), Avg variants minimize the average route
// cost across agents. L1/L2/Geodesic select the per-edge distance metric —
// Geodesic dispatches to an injected EdgeCostFunctor (see edge_cost_functor.hpp)
// rather than a closed-form norm.
enum class WaypointObjective { kMinMaxL1, kAvgL1, kMinMaxL2, kAvgL2, kMinMaxGeodesic, kAvgGeodesic };

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

	// Optional arbitrary edge cost (e.g. a precomputed arrival-time/geodesic
	// grid, see grid_interpolant.hpp). Solver-agnostic plumbing: not consumed
	// by GraphWaypointMPC's Gurobi/Mosek build (kMinMaxGeodesic/kAvgGeodesic
	// still throw there) — this is for a future custom solver.
	std::shared_ptr<EdgeCostFunctor> _edge_cost_fn;

	// Recording Metrics
	drake::SteadyTimer _timer;
	double _last_solve_time;

	GraphWaypointMPC(GraphOfConstraints& graph,
			 std::vector<CubicConfigurationSpline> splines,
			 WaypointObjective objective,
			 std::shared_ptr<EdgeCostFunctor> edge_cost_fn);

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

	// Evaluates the configured edge cost functor at (agent, w_a, w_b).
	// Throws if no functor was configured at construction.
	double EvaluateEdgeCost(int agent,
				const Eigen::VectorXd& w_a,
				const Eigen::VectorXd& w_b) const;
};
