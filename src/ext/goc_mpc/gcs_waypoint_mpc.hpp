#pragma once

#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/solver_interface.h>

#include "arc_travel_time.hpp"
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
	double length_weight = 0.0,
	const std::vector<std::shared_ptr<ArcTravelTimeModel>>& travel_time_models = {},
	double max_tangent_halfwidth = std::numeric_limits<double>::infinity());

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
//
// kMinMaxTravelTime/kAvgTravelTime replace each arc's L1/L2 distance with a
// tangent plane of agent j's ArcTravelTimeModel, taken once per solve at the
// Chebyshev centres of the arc's two vertex sets (x0 out of the depot), and
// applied in perspective to the arc copies -- linear, so the problem stays a
// MILP. Exact on pinned vertices; elsewhere it is the field's linearization
// over the whole set, and a vertex set wider than max_tangent_halfwidth in
// any of the model's columns raises.
struct GCSWaypointMPC : public GraphWaypointMPC {
	Eigen::VectorXd _t_routing;
	// Per agent: graph node ids visited, in order.
	std::vector<std::vector<int>> _routes;

	WaypointSolver _solver;
	bool _enforce_rigidity;
	SubtourElimination _subtour;
	double _length_weight;
	// Relative optimality gap at which the MIP solve stops.
	double _mip_gap;
	// Seed each solve with the previous solve's assignments and routes.
	bool _warm_start;
	// Per agent (travel-time objectives only).
	std::vector<std::shared_ptr<ArcTravelTimeModel>> _travel_time_models;
	double _max_tangent_halfwidth;
	// Per agent: (graph node id, global variable id or kNoRole) of each
	// vertex on the previous solve's route, in order.
	std::vector<std::vector<std::pair<int, int>>> _prev_route_modes;

	GCSWaypointMPC(GraphOfConstraints& graph,
		       std::vector<CubicConfigurationSpline> splines,
		       WaypointSolver solver = WaypointSolver::kGurobi,
		       bool enforce_rigidity = false,
		       WaypointObjective objective = WaypointObjective::kMinMaxL1,
		       SubtourElimination subtour = SubtourElimination::kMCF,
		       double length_weight = 0.0,
		       double mip_gap = 1e-4,
		       bool warm_start = true,
		       std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models = {},
		       double max_tangent_halfwidth = std::numeric_limits<double>::infinity());

	bool Solve(const std::vector<int>& remaining_vertices,
		   const Eigen::VectorXd& x0) override;

	const Eigen::VectorXd& view_t() const { return _t_routing; }
	const std::vector<std::vector<int>>& view_routes() const { return _routes; }

private:
	bool SolveWith(const drake::solvers::SolverInterface& solver,
		       const std::vector<int>& remaining_vertices,
		       const Eigen::VectorXd& x0);
	// Sets a MIP start on problem.prog from the previous solve: assignment
	// binaries, and per agent the arc binaries of its previous route
	// restricted to the vertices still in the problem. Continuous variables
	// are left for the solver to complete.
	void SetWarmStart(const GCSWaypointProblem& problem,
			  const std::vector<int>& remaining_vertices) const;
};
