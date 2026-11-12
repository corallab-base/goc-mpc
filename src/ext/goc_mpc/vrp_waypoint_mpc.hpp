#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "arc_travel_time.hpp"
#include "graph_waypoint_mpc.hpp"
#include "oracles/initialization_oracle.hpp"
#include "oracles/route_solver.hpp"

// Waypoint solver that takes its configurations from an initialization
// oracle and solves only the routing.
//
// Each solve asks the oracle for its guess over the remaining nodes (the
// previous guess as its warm start), then solves the routing MILP over the
// guess's feasible assignments (SolveRoutes): which assignment, and each
// agent's route through its nodes, with every configuration fixed at the
// oracle's. Arcs cost the configurations' L1/L2 distance, or under
// kMinMaxTravelTime/kAvgTravelTime agent j's ArcTravelTimeModel's travel
// time; the kMinMax* objectives minimize the longest route (plus
// length_weight times the total), the kAvg* ones the total.
//
// view_waypoints() reports each visiting agent's oracle configuration and
// every object's oracle state at each remaining node; an agent's row at a
// node it doesn't visit is interpolated per block in time between its
// neighbouring visits (holding after its last), as GCSWaypointMPC does.
// Solve() returns false when the oracle finds no feasible assignment or the
// MILP fails.
struct VRPWaypointMPC : public GraphWaypointMPC {
	std::shared_ptr<InitializationOracle> _initialization_oracle;
	// Per agent (travel-time objectives only).
	std::vector<std::shared_ptr<ArcTravelTimeModel>> _travel_time_models;
	double _length_weight;
	double _mip_gap;
	std::optional<InitialGuess> _initial_guess;
	std::optional<RouteSolution> _route_solution;
	std::vector<std::vector<int>> _routes;

	VRPWaypointMPC(GraphOfConstraints& graph, std::vector<CubicConfigurationSpline> splines,
		       std::shared_ptr<InitializationOracle> initialization_oracle,
		       WaypointObjective objective = WaypointObjective::kAvgL2,
		       std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models = {},
		       double length_weight = 0.0, double mip_gap = 1e-4);

	bool Solve(const std::vector<int>& remaining_vertices, const Eigen::VectorXd& x0) override;

	const std::vector<std::vector<int>>& view_routes() const { return _routes; }
	// The last solve's oracle guess and routing solution (null before any).
	const InitialGuess* view_initial_guess() const { return _initial_guess ? &*_initial_guess : nullptr; }
	const RouteSolution* view_route_solution() const { return _route_solution ? &*_route_solution : nullptr; }
};
