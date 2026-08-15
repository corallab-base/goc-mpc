#pragma once

#include <map>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

// Trust-region Gauss-Newton SQP timing solver -- same problem as
// GnTimingMPC's (same gn_timing::ProblemLayout/AssembleObjective: per-agent
// time deltas `tau` and interior tangent-space velocities `v`, the
// coast-corrected min-acceleration cost `acceleration_cost * psi`, and the
// cross-agent LESS_THAN/EQUAL timing constraints), but solved as a sequence
// of small QPs (qpOASES::SQProblem, hot-started both across outer
// iterations and across MPC cycles) inside a hand-rolled trust-region loop,
// instead of one IPOPT solve.
//
// Why this exists alongside GnTimingMPC: IPOPT is a general-purpose
// interior-point method built for arbitrary nonlinear constraints, and its
// failure modes (RESTORATION_FAILURE, LOCAL_INFEASIBILITY, ...) are a hard
// stop even on a problem whose true feasible region is fine -- observed in
// practice on harder scenarios even with GnTimingMPC's exact Gauss-Newton
// Hessian (a better local curvature model doesn't make the underlying
// acceleration_cost objective convex; it's still the same non-convex
// coast-corrected residual GraphTimingMPC's own doc comment already names
// as a known culprit). This problem's constraints are all EXACTLY linear
// (the tau box plus the interaction rows -- see
// GraphOfConstraints::get_agent_paths), so "linearize the constraints"
// (the usual reason SQP needs a whole separate feasibility-restoration
// phase, and the usual reason THAT can fail) is a no-op here: every QP
// subproblem's feasible region is the REAL one, not a local approximation,
// every single iteration. Combined with a Gauss-Newton Hessian model that's
// PSD by construction and uniformly bounded on the tau box (D/V are LINEAR
// in v, so the Hessian's v-dependent blocks don't grow with v's actual
// value), every QP subproblem is solvable by construction -- so a failed
// step degrades to a smaller trust region (a detectable stall, recoverable
// by resetting) rather than IPOPT's outright refusal. This gives a genuine
// classical guarantee (Nocedal & Wright's trust-region convergence
// theorem): the iterates converge to a first-order stationary point (a
// local optimum, in the standard NLP sense) -- not a global-optimum
// guarantee (the objective is still non-convex), but a categorically
// stronger reliability property than IPOPT's for this problem shape.
//
// Same scope restrictions as GnTimingMPC (see that class's own doc
// comment): no max_vel/max_acc, no energy_cost/arclength_cost/
// stability_cost, SO3Mat throws, sparse (real-node-only) solve() only.
struct SqpTimingMPC {
	const GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// Persistent output buffers -- same shapes/semantics as GnTimingMPC's
	// own (and, in turn, GraphTimingMPC's).
	std::vector<Eigen::MatrixXd> _wps_list;
	std::vector<Eigen::MatrixXd> _vs_list;
	std::vector<Eigen::VectorXd> _time_deltas_list;
	std::vector<std::vector<int>> _agent_nodes_list;
	std::map<int, int> _agent_spline_length_map;

	double _time_cost;
	double _time_cost2;
	double _acceleration_cost;

	// Trust-region outer loop tuning. Defaults are conservative (small
	// initial radius, generous iteration budget) rather than tuned for
	// speed -- see solve()'s own comment on why correctness of the
	// accept/reject mechanics matters more here than shaving iterations.
	int _max_iterations;
	double _initial_trust_radius;
	double _max_trust_radius;
	// Below this radius the loop treats itself as stalled and stops (still
	// returning its best iterate so far) rather than spinning uselessly --
	// see solve()'s own comment.
	double _min_trust_radius;
	double _grad_tol;

	// Reused across solve() calls -- qpOASES::SQProblem's hot-start only
	// pays off if the SAME instance persists cycle-to-cycle (same reason
	// GraphTimingMPC/GnTimingMPC warm-start their own initial guess from
	// the previous cycle's converged solution). Held as a raw pointer with
	// manual lifetime management (constructed lazily on first solve(),
	// re-constructed whenever `n`/`m` change) since qpOASES::SQProblem
	// isn't move-constructible in a way that plays nicely with resizing.
	struct QpState;
	std::unique_ptr<QpState> _qp_state;

	drake::SteadyTimer _timer;
	double _last_solve_time = 0.0;
	// Diagnostics from the most recent solve(): how many outer iterations
	// it actually ran, and the trust radius it ended on (a radius pinned at
	// _min_trust_radius signals a stall -- see solve()'s own comment).
	int _last_iterations = 0;
	double _last_trust_radius = 0.0;

	SqpTimingMPC(const GraphOfConstraints& graph,
		     std::vector<CubicConfigurationSpline> splines,
		     double time_cost = 1e0,
		     double time_cost2 = 0e0,
		     double acceleration_cost = 1.0,
		     int max_iterations = 50,
		     double initial_trust_radius = 1.0,
		     double max_trust_radius = 50.0,
		     double min_trust_radius = 1e-6,
		     double grad_tol = 1e-6);
	~SqpTimingMPC();

	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const std::vector<int>& remaining_vertices,
		   const Eigen::MatrixXd& waypoints,
		   const Eigen::VectorXi& assignments,
		   const Eigen::VectorXd& t_by_node = Eigen::VectorXd());

	int get_agent_spline_length(int agent) const;
	std::vector<int> get_agent_spline_nodes(int agent) const;

	std::set<int> set_progressed_time(double delta, double tau_cutoff);

	void fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				const Eigen::VectorXd& x0,
				const Eigen::VectorXd& v0) const;

	const std::vector<double> get_next_taus() const;
	const std::vector<int> get_next_nodes() const;

	const std::vector<Eigen::MatrixXd> &view_wps_list() const { return _wps_list; }
	const std::vector<Eigen::MatrixXd> &view_vs_list() const { return _vs_list; }
	const std::vector<Eigen::VectorXd> &view_time_deltas_list() const { return _time_deltas_list; }
	const std::vector<std::vector<int>> &view_agent_nodes_list() const { return _agent_nodes_list; }
	const std::map<int, int> &view_agent_spline_length_map() const { return _agent_spline_length_map; }
	const double get_last_solve_time() { return _last_solve_time; }
	int get_last_iterations() const { return _last_iterations; }
	double get_last_trust_radius() const { return _last_trust_radius; }
};
