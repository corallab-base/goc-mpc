#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <map>
#include <string>
#include <utility>
#include <vector>

struct GraphOfConstraints;

// Initialization oracles: a starting point for a whole waypoint solve --
// assignments, routes, and every remaining node's agent configurations and
// object states -- computed from the graph's constraints rather than from
// the agents' current configurations.

// One inverse-kinematics query: configurations of `agent` within
// [lower, upper] that put its FK link `link` at world_T_link.
struct IkRequest {
	int agent = -1;
	std::string link;
	Eigen::Isometry3d world_T_link = Eigen::Isometry3d::Identity();
	Eigen::VectorXd reference, lower, upper;
};

// An agent's inverse kinematics, as an initialization oracle uses it.
class IkSolver {
public:
	virtual ~IkSolver() = default;

	// Configurations satisfying the request, best first; empty when none
	// was found.
	virtual std::vector<Eigen::VectorXd> Solve(const IkRequest& request) const = 0;

	// True when an empty Solve proves no configuration within the limits
	// reaches the pose, so the oracle may exclude the assignment that
	// asked for it.
	virtual bool complete() const { return false; }
};

struct InitialGuess;

// The IK solutions an oracle found for one agent's link pose at one node.
struct CandidateSet {
	int node = -1;
	int agent = -1;
	std::string link;
	Eigen::Isometry3d world_T_link = Eigen::Isometry3d::Identity();
	// The assignments (global variable id -> agent) that asked for this pose.
	std::vector<std::map<int, int>> assignments;
	// Every solution the IK solver gave, and the collision-free ones, nearest
	// the agent's x0 first.
	int ik_solutions = 0;
	std::vector<Eigen::VectorXd> configurations;
	// What each colliding solution hit.
	std::vector<std::string> rejected;
	// The index into configurations the assignment's guess took (-1: none).
	int chosen = -1;
};

// One assignment's guess, for an assignment whose guess violates nothing.
struct AssignmentGuess {
	// Global assignable variable id -> agent.
	std::map<int, int> assignments;
	// As InitialGuess::X and ::routes (routes in the hard orderings' order).
	Eigen::MatrixXd X;
	std::vector<std::vector<int>> routes;
};

struct InitializationRequest {
	const GraphOfConstraints* graph = nullptr;
	// The remaining graph nodes the solve covers.
	std::vector<int> nodes;
	// The current state: agents' then objects' columns (graph layout).
	Eigen::VectorXd x0;
	// Per global assignable variable id, the previous solve's agent (-1:
	// none); tried first.
	Eigen::VectorXi previous_assignments;
	// The previous solve's guess (warm start), or null.
	const InitialGuess* previous = nullptr;
};

// A starting point for every decision of the solve, in graph terms.
struct InitialGuess {
	// Every relation the oracle read holds at the guess.
	bool success = false;
	// X's rows: the request's nodes, in its order.
	std::vector<int> nodes;
	// nodes x graph total_dim: each visiting agent's configuration and each
	// object's state at the node; NaN in an agent's columns where it doesn't
	// visit.
	Eigen::MatrixXd X;
	// Global assignable variable id -> agent.
	std::map<int, int> assignments;
	// Per agent, the graph nodes it visits, in order.
	std::vector<std::vector<int>> routes;
	// Sets of (variable, agent) pairs no solution can hold together: each
	// was derived through exact relations only, and its IK is complete.
	std::vector<std::vector<std::pair<int, int>>> infeasible_assignments;
	// Every agent/node link pose the oracle solved IK for, across all the
	// assignments it tried; a pose two assignments solved alike appears once.
	std::vector<CandidateSet> candidates;
	// Every assignment the oracle tried whose guess violates nothing.
	std::vector<AssignmentGuess> feasible;
	// How each value was set, in order, for the returned assignment.
	std::vector<std::string> log;
	// Relations the guess violates, and constraints the oracle couldn't
	// read (only checked).
	std::vector<std::string> violations, unresolved;
	int assignments_tried = 0;
	double time = 0.0;
};

class InitializationOracle {
public:
	virtual ~InitializationOracle() = default;

	virtual InitialGuess Initialize(const InitializationRequest& request) const = 0;
};
