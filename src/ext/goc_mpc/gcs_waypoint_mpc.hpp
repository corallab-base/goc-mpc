#pragma once

#include <memory>
#include <utility>
#include <vector>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solver_interface.h>

#include "graph_of_constraints.hpp"
#include "graph_waypoint_mpc.hpp"
#include "waypoint_common.hpp"
#include "../configuration_spline.hpp"

// One agent's routing graph: one vertex per (task node, mode), where a mode
// is a role (subgraph variable row) the agent is a candidate for at that
// node, or kNoRole on a required visit. Arcs join the depot, vertices of
// distinct nodes, and the sink, after RoutingPresolve's arc pruning.
struct GCSModeGraph {
	static constexpr int kNoRole = -1;
	static constexpr int kDepot = -1;
	static constexpr int kSink = -2;

	struct Vertex {
		int sg;    // subgraph node id
		int mode;  // subgraph variable row, or kNoRole
	};
	std::vector<Vertex> vertices;
	// (tail, head): vertex indices, or kDepot (tail) / kSink (head).
	std::vector<std::pair<int, int>> arcs;
};

struct GCSWaypointProblem {
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::unique_ptr<SubgraphOfConstraints> subgraph;
	drake::solvers::MatrixXDecisionVariable Assignments;
	// Routing-node timestamps (RoutingPresolve's routing indices) and makespan.
	drake::solvers::VectorXDecisionVariable t;
	drake::solvers::VectorXDecisionVariable L;
	// Per agent j: mode graph, arc binaries Z[j] (one per arc), and vertex
	// position copies P[j][vertex].
	std::vector<GCSModeGraph> graphs;
	std::vector<drake::solvers::VectorXDecisionVariable> Z;
	std::vector<std::vector<drake::solvers::VectorXDecisionVariable>> P;
	// Per object o: (subgraph nodes x object width) positions.
	std::vector<drake::solvers::MatrixXDecisionVariable> O;

	GCSWaypointProblem() = default;
	GCSWaypointProblem(const GCSWaypointProblem&) = delete;
	GCSWaypointProblem& operator=(const GCSWaypointProblem&) = delete;
	GCSWaypointProblem(GCSWaypointProblem&&) = default;
	GCSWaypointProblem& operator=(GCSWaypointProblem&&) = default;
};

GCSWaypointProblem BuildGCSWaypointProblem(
	GraphOfConstraints* graph,
	std::shared_ptr<std::vector<CubicConfigurationSpline>> splines,
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0,
	const Eigen::MatrixXd& previous_X,
	const Eigen::VectorXi& previous_var_assignments,
	bool enforce_rigidity,
	WaypointObjective objective,
	SubtourElimination subtour,
	double length_weight = 0.0);

// Waypoint solver for the same problem as MILPWaypointMPC, sharing everything
// that doesn't depend on the encoding (waypoint_common.*), but encoding
// routing, role-gated constraints, and travel cost/time GCS-style: per-agent
// mode graphs whose vertex and arc copies lie in the perspectives of their
// convex sets, plus explicit subtour elimination -- no big-M on assignments
// or arc costs.
//
// There are no per-node configuration variables for an agent at a node it
// doesn't visit: a constraint on agent j at node v applies only if j visits
// v. The MILP's free W columns instead require some point to satisfy every
// row placed on an unvisited (agent, node) at once, which can couple other
// variables when several rows land on the same pair; the feasible set here
// is slightly larger in exactly those cases. view_waypoints() still reports
// every agent at every node: an unvisited agent's row is interpolated per
// block in time between its neighbouring visits (holding after its last).
struct GCSWaypointMPC : public GraphWaypointMPC {
	Eigen::VectorXd _t_routing;
	// Per agent: graph node ids visited, in order.
	std::vector<std::vector<int>> _routes;

	WaypointSolver _solver;
	bool _enforce_rigidity;
	SubtourElimination _subtour;
	double _length_weight;

	GCSWaypointMPC(GraphOfConstraints& graph,
		       std::vector<CubicConfigurationSpline> splines,
		       WaypointSolver solver = WaypointSolver::kGurobi,
		       bool enforce_rigidity = false,
		       WaypointObjective objective = WaypointObjective::kMinMaxL1,
		       SubtourElimination subtour = SubtourElimination::kMCF,
		       double length_weight = 0.0);

	bool Solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0) override;

	const Eigen::VectorXd& view_t() const { return _t_routing; }
	const std::vector<std::vector<int>>& view_routes() const { return _routes; }

private:
	bool SolveWith(const drake::solvers::SolverInterface& solver,
		       const std::vector<int>& remaining_vertices,
		       const Eigen::VectorXd& x0);
};
