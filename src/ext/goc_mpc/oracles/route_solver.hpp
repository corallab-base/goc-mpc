#pragma once

#include <Eigen/Core>

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "initialization_oracle.hpp"

struct GraphOfConstraints;

// Routes over an initialization oracle's configurations, as a MILP.
//
// Each (node, agent) the oracle placed in a feasible assignment
// (guess.feasible) has one configuration; the MILP chooses one of those
// assignments (a selector binary each) and, per agent, a route from the
// depot (its x0 configuration) through exactly the nodes the assignment has
// it visit (arc binaries, flow conservation), ending at a free sink. A
// timestamp per node, shared by every agent visiting it, orders the routes:
// each used arc moves strictly forward in time, and the graph's orderings
// (hard, and conditional through the shared routing helpers) hold between
// node times. Each agent's route costs the sum of its arcs' costs (arcs
// into the sink cost nothing); the objective is the sum over agents, or with
// min_max their maximum plus length_weight times the sum. Solved with
// Drake's best available MILP solver (Gurobi when licensed).
struct RouteSolution {
	bool success = false;
	std::string solver;
	std::map<int, int> assignments;
	// Every node, by its timestamp; nodes several agents visit are one stop.
	std::vector<int> order;
	// Per agent, the graph nodes it visits, in order.
	std::vector<std::vector<int>> routes;
	// guess.nodes, and each visiting agent's configuration there (NaN
	// elsewhere), from the chosen assignment.
	std::vector<int> nodes;
	Eigen::MatrixXd X;
	// Sum of the agents' route costs, and each one's.
	double cost = 0.0;
	std::vector<double> agent_costs;
	// Per row of nodes, its timestamp (routing order: depot 0; consecutive
	// stops of a route at least 1 apart).
	std::vector<double> times;
	// The objective's value.
	double objective = 0.0;
	// Arc binaries in the MILP (one priced cost each).
	int arcs = 0;
	double time = 0.0;
	std::vector<std::string> log;
};

// An arc cost per row of (from, to), both (n x the agent's ambient dim).
using ArcCost = std::function<Eigen::VectorXd(int agent, const Eigen::MatrixXd& from, const Eigen::MatrixXd& to)>;

// The L2 (or, with l1, L1) distance between configurations.
ArcCost DistanceArcCost(bool l1 = false);

// x0: the current state (graph layout), the routes' start. mip_gap:
// relative gap at which the MILP stops.
RouteSolution SolveRoutes(const GraphOfConstraints& graph, const InitialGuess& guess, const Eigen::VectorXd& x0,
			  const ArcCost& arc_cost = DistanceArcCost(), bool min_max = false, double length_weight = 0.0,
			  double mip_gap = 1e-4);
