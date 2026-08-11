#pragma once

#include <algorithm>
#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/nlopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../splines.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphOrderingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable p;

	GraphOrderingProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	GraphOrderingProblem(const GraphOrderingProblem&) = delete;
	GraphOrderingProblem& operator=(const GraphOrderingProblem&) = delete;

	GraphOrderingProblem(GraphOrderingProblem&&) = default;
	GraphOrderingProblem& operator=(GraphOrderingProblem&&) = default;
};

GraphOrderingProblem build_graph_ordering_problem(
	const Graph<py::object>& structure,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0);

struct GraphTimingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	std::vector<drake::solvers::MatrixXDecisionVariable> vs_list;
	std::vector<drake::solvers::VectorXDecisionVariable> time_deltas_list;

	GraphTimingProblem(int num_agents)
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()),
		  wps_list(num_agents),
		  vs_list(num_agents),
		  time_deltas_list(num_agents) {}

	GraphTimingProblem(const GraphTimingProblem&) = delete;
	GraphTimingProblem& operator=(const GraphTimingProblem&) = delete;

	GraphTimingProblem(GraphTimingProblem&&) = default;
	GraphTimingProblem& operator=(GraphTimingProblem&&) = default;
};

GraphTimingProblem build_graph_timing_problem(
	const Graph<py::object>& structure,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0);

// Per-agent QP pieces shared by both build_graph_timing_problem (positions
// fixed at graph-node targets) and build_dense_graph_timing_problem
// (positions fixed at a densely path-traced polyline): free velocity/
// time-delta decision variables through a sequence of fixed positions, plus
// the arclength/acceleration/energy/stability costs and max-vel/max-acc
// constraints on each resulting segment. See graph_timing_mpc.cpp.
struct AgentSegmentVars {
	drake::solvers::VectorXDecisionVariable time_deltas;
	drake::solvers::MatrixXDecisionVariable vs;
};

AgentSegmentVars add_agent_timing_segments(
	drake::solvers::MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const Eigen::MatrixXd& wps_i,
	const Eigen::VectorXd& x0_i,
	const Eigen::VectorXd& v0_i,
	int agent_index,
	double time_cost,
	double time_cost2,
	double acceleration_cost,
	double energy_cost,
	double arclength_cost,
	double stability_cost,
	const std::vector<double>& max_vel,
	const std::vector<double>& max_acc,
	const std::vector<double>& max_jerk);

// Adds a cross-agent LESS_THAN/EQUAL timing constraint (from
// GraphOfConstraints::get_agent_paths, depths already reindexed to whatever
// per-agent decision-variable sequence `time_deltas_list` actually holds --
// see GraphOfConstraints::reindex_agent_interactions) to `prog` for each of
// `agent_interactions`. Shared by build_graph_timing_problem (sparse,
// real-node-only) and build_dense_graph_timing_problem (dense) -- the
// constraint math itself (sum of taus up to a depth) doesn't care which.
void add_agent_interaction_constraints(
	drake::solvers::MathematicalProgram& prog,
	const std::vector<drake::solvers::VectorXDecisionVariable>& time_deltas_list,
	const std::vector<AgentInteraction>& agent_interactions,
	const std::map<std::pair<int, int>, double>& edge_to_min_tau_map);

// Dense-waypoint counterpart to build_graph_timing_problem: positions fixed
// at a densely path-traced polyline instead of graph-node targets, but
// (unlike GraphTimingMPC's old TracedGraphTimingMPC-backed path) DOES
// support AgentInteraction (cross-agent LESS_THAN/EQUAL) constraints. The
// caller (GraphTimingMPC::solve_dense) is expected to have already resolved
// graph ordering via GraphOfConstraints::get_agent_paths and expanded each
// agent's real-node sequence into a denser one (e.g. via path tracing)
// itself -- this function has no notion of the graph or of tracing, same
// separation of concerns as build_graph_timing_problem/get_agent_paths.
// `agent_dense_node_ids[i]`: one entry per row of `agent_dense_wps[i]`, the
// real graph node id at that row or -1 for a synthetic one -- becomes the
// returned problem's agent_nodes_list[i], propagated through to
// GraphTimingMPC::set_progressed_time/get_next_nodes/etc.
// `agent_interactions`' depths must already be indexed against
// `agent_dense_node_ids` (see GraphOfConstraints::reindex_agent_interactions),
// not the original sparse get_agent_paths output.
GraphTimingProblem build_dense_graph_timing_problem(
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<Eigen::MatrixXd>& agent_dense_wps,
	const std::vector<std::vector<int>>& agent_dense_node_ids,
	const std::vector<AgentInteraction>& agent_interactions,
	const std::map<std::pair<int, int>, double>& edge_to_min_tau_map,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double time_cost,
	double time_cost2,
	double acceleration_cost,
	double energy_cost,
	double arclength_cost,
	double stability_cost,
	const std::vector<double>& max_vel,
	const std::vector<double>& max_acc,
	const std::vector<double>& max_jerk);


struct GraphTimingMPC {
	// Input: reference to graph of constraints
	const GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// Persistent Output Buffers
	std::vector<Eigen::MatrixXd> _wps_list;
	std::vector<Eigen::MatrixXd> _vs_list;
	std::vector<Eigen::VectorXd> _time_deltas_list;
	std::vector<std::vector<int>> _agent_nodes_list;
	std::map<int, int> _agent_spline_length_map;

	// Optimization parameters
	double _time_cost;
	double _time_cost2;
	double _acceleration_cost;
	double _energy_cost;
	double _arclength_cost;
	// One entry per block in the spline's spec (block_offsets_ order); a
	// block's own entry <= 0 means unbounded for that block, matching the
	// pre-vector single-scalar sentinel semantics; an empty vector means
	// unbounded for every block (replaces the old bare -1.0 default). See
	// add_agent_timing_segments (graph_timing_mpc.cpp) for how each block
	// type turns its own bound into a constraint.
	std::vector<double> _max_vel;
	std::vector<double> _max_acc;
	std::vector<double> _max_jerk;
	// Convex alternative/complement to `_acceleration_cost`: penalizes
	// ||xJ - xJm1||^2 / tau^3 + ||vJ - vJm1||^2 / tau per segment, instead
	// of acceleration_cost's ||(xJ-xJm1) - 0.5*tau*(vJm1+vJ)||^2 / tau^3.
	// The difference is that this term's numerator doesn't depend on tau
	// (it's the raw squared endpoint gap, not a "coast at current velocity"
	// -corrected residual), so it can't develop the sign-indefinite cross
	// term (`-2*tau*(A.B)` from expanding the coast-corrected term) that
	// makes acceleration_cost's contribution non-convex whenever the
	// current velocity already points roughly toward the target -- see
	// po_goc_mpc.experiments.basic_fmm_experiment's spline-iterations
	// diagnostic, which found exactly this: two distinct local minima in
	// the per-cycle timing solve, with NLopt unreliably landing in either
	// one cycle to cycle. Every term this adds (linear-in-tau, positive
	// constant / tau^n, and quadratic-over-linear ||v||^2/tau, which stays
	// convex even when v is itself a decision variable -- the standard
	// convex "quadratic-over-linear" perspective function) is convex, so
	// combined with the already-convex arclength_cost and linear time_cost,
	// the whole per-cycle problem is convex when acceleration_cost/max_acc
	// are left off in favor of this term.
	double _stability_cost;

	// Phase management
	// std::set<int> _completed_phases;
	// Eigen::VectorXd _in_degrees; // in-degrees of remaining active phases
	// py::array_t<unsigned int> back_tracking_table;
	// bool never_done = false;

	// Recording Metrics
	drake::SteadyTimer _timer;
	double _last_solve_time;

	// Constructor
	// Defaults: only time_cost (linear "minimize total time") and
	// stability_cost are on. acceleration_cost and arclength_cost both
	// default to off (0.0), NOT on as they used to -- both contain terms
	// that are bilinear in (tau, velocity) before being squared/normed
	// (acceleration_cost's "coast-corrected" residual; arclength_cost's
	// per-quadrature-point sqrt(‖affine-in-tau-and-v‖^2 + eps)), which is
	// not convex in general, unlike stability_cost's own residual (see its
	// doc comment below) -- confirmed empirically as the source of real
	// Ipopt InfeasibleConstraints/IterationLimit failures AND (arclength_
	// cost specifically, via its near-unregularized sqrt(x^2+1e-8) at
	// near-zero velocity) severe per-cycle slowdowns (~9-10s/cycle) on
	// pick_place_task_experiment's branching, multi-segment traced paths --
	// stability_cost alone was both fully reliable and ~15-20x faster.
	// max_acc's hard bound (if a caller supplies one) uses the same
	// coast-corrected cubic-Hermite formula as acceleration_cost, so it has
	// the same non-convexity risk; prefer stability_cost over a hard bound
	// where possible.
	GraphTimingMPC(const GraphOfConstraints& graph,
		       std::vector<CubicConfigurationSpline> splines,
		       double time_cost = 1e0,
		       double time_cost2 = 0e0,
		       double acceleration_cost = 0.0,
		       double energy_cost = 0.0,
		       double arclength_cost = 0.0,
		       std::vector<double> max_vel = {},
		       std::vector<double> max_acc = {},
		       std::vector<double> max_jerk = {},
		       double stability_cost = 1.0);

	// Core solve routine
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const std::vector<int>& remaining_vertices,
		   const Eigen::MatrixXd& waypoints,
		   const Eigen::VectorXi& assignments,
		   const Eigen::VectorXd& t_by_node = Eigen::VectorXd());

	// Dense-waypoint counterpart to solve(): the caller has already
	// resolved graph ordering + cross-agent interactions itself (via
	// graph.get_agent_paths, on the SAME graph this instance was
	// constructed with) and expanded each agent's real-node sequence into
	// a denser one (e.g. via path tracing between consecutive real nodes),
	// reindexing `agent_interactions`' depths against that dense sequence
	// (GraphOfConstraints::reindex_agent_interactions) -- this just builds
	// and solves the resulting QP (build_dense_graph_timing_problem) and
	// stores the result the same way solve() does, so every other method
	// (fill_cubic_splines, set_progressed_time, get_next_nodes, ...) works
	// unchanged regardless of which one produced the current solution.
	// `agent_dense_node_ids[i]` becomes this instance's own
	// `_agent_nodes_list[i]` (real id per row, or -1 for synthetic) --
	// unlike solve()'s fixed-size (num_nodes-bounded) per-agent buffers,
	// this REPLACES `_wps_list[i]`/`_vs_list[i]`/`_time_deltas_list[i]`
	// wholesale each call, since a dense sequence's length isn't bounded by
	// the graph's own node count.
	bool solve_dense(const Eigen::VectorXd& x0,
			  const Eigen::VectorXd& v0,
			  const std::vector<Eigen::MatrixXd>& agent_dense_wps,
			  const std::vector<std::vector<int>>& agent_dense_node_ids,
			  const std::vector<AgentInteraction>& agent_interactions);

	int get_agent_spline_length(int agent) const;
	std::vector<int> get_agent_spline_nodes(int agent) const;

	std::set<int> set_progressed_time(double delta, double tau_cutoff);

	// Spline generator
	void fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				const Eigen::VectorXd& x0,
				const Eigen::VectorXd& v0) const;

	// Phase tracking
	// double current_minimum_time_delta() const;
	// bool done() const;

	// Safe indexing and accessors
	// py::array_t<unsigned int> get_ordering() const;
	// py::array_t<double> get_waypoints() const;
	// py::array_t<double> get_time_deltas() const;
	// py::array_t<double> get_times() const;
	const std::vector<double> get_next_taus() const;
	const std::vector<int> get_next_nodes() const;

	const std::vector<Eigen::MatrixXd> &view_wps_list() const { return _wps_list; }
	const std::vector<Eigen::MatrixXd> &view_vs_list() const { return _vs_list; }
	const std::vector<Eigen::VectorXd> &view_time_deltas_list() const { return _time_deltas_list; }
	const std::vector<std::vector<int>> &view_agent_nodes_list() const { return _agent_nodes_list; }
	const std::map<int, int> &view_agent_spline_length_map() const { return _agent_spline_length_map; }
	const double get_last_solve_time() { return _last_solve_time; }

	// State updates
	// bool set_progressed_time(double time_delta, double time_delta_cutoff);
	// void set_updated_waypoints(const py::array_t<double>& _waypoints,
	// 			   bool set_next_waypoint_tangent);
	// void update_backtrack();
	// void update_set_phase(unsigned int phase_to);


};
