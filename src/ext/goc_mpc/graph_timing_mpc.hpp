#pragma once

#include <map>
#include <memory>
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

// Trust-region Gauss-Newton SQP timing solver: per-agent time deltas `tau`
// and interior tangent-space velocities `v`, the min-acceleration cost
// (acceleration_cost * psi, "psi" = compute_ctrl_cost's coast-corrected
// residual), and the cross-agent LESS_THAN/EQUAL timing constraints
// (GraphOfConstraints::get_agent_paths), solved as a sequence of small QPs
// (qpOASES::SQProblem, hot-started both across outer iterations and across
// MPC cycles) inside a trust-region loop -- gn_timing::AssembleObjective/
// BuildProblemLayout (timing_gn_layout.hpp) do the actual residual/
// Jacobian/Gauss-Newton-Hessian math.
//
// This replaces an earlier Drake-MathematicalProgram/IPOPT(L-BFGS-only)
// implementation under this same name, and a convex "stability_cost" proxy
// cost that implementation leaned on to avoid IPOPT's non-convexity-
// triggered failures (both since removed -- see git history): IPOPT is a
// general-purpose interior-point method built for arbitrary nonlinear
// constraints, and its failure modes (RESTORATION_FAILURE,
// LOCAL_INFEASIBILITY, MAXITER_EXCEEDED, ...) are a hard stop even on a
// problem whose true feasible region is fine. This problem's constraints
// are all EXACTLY linear (the tau box plus the interaction rows), so
// "linearize the constraints" (the usual reason SQP needs a whole separate
// feasibility-restoration phase, and the usual reason THAT can fail) is a
// no-op here: every QP subproblem's feasible region is the REAL one, not a
// local approximation, every single iteration. Combined with a Gauss-Newton
// Hessian model that's PSD by construction and uniformly bounded on the tau
// box (D/V are LINEAR in v, so the Hessian's v-dependent blocks don't grow
// with v's actual value), every QP subproblem is solvable by construction
// -- so a failed step degrades to a smaller trust region (a detectable,
// recoverable stall) rather than IPOPT's outright refusal.
//
// CORRECTION (2026-08-17): the "solvable by construction" claim above held
// for the tau box and the LESS_THAN/EQUAL cross-agent interaction rows only
// in the sense that the GLOBAL (un-trust-regioned) feasible set is always
// non-empty -- every row is exactly linear, never curved, so nothing here
// needs re-linearizing. It did NOT account for the TRUST-REGION-CLAMPED
// subproblem actually solved every iteration. An EQUAL row (two agents'
// cumulative time-to-a-shared-node forced equal -- exactly what
// `GraphOfConstraints::get_agent_paths`/`reindex_agent_interactions` emit
// for any node two agents both touch, e.g. a handoff) can have a CURRENT
// violation larger than any combination of moves within
// `[-trust_radius, trust_radius]` on the involved tau's can close, making
// that iteration's box-intersect-hyperplane genuinely empty -- qpOASES
// then correctly (not spuriously) reports infeasibility. Worse, for an
// EQUALITY row this trust-region shrink-and-retry loop's usual remedy
// backfires: a smaller box is a STRICTER subset, never easier to satisfy,
// so a stuck cycle burns every iteration down to min_trust_radius and
// commits whatever half-stalled iterate it has as the cycle's plan --
// confirmed, via a real 2-agent handoff scenario
// (po_goc_mpc's object_handoff_experiment.py under
// TracedTimingMPC/solve_dense), to visibly derail the commanded
// trajectory downstream, not just waste compute solving repeatedly.
//
// Fixed by slack-relaxing the interaction rows themselves -- the same
// exact-l1-penalty Sl1QP pattern GraphShortPathMPC uses for its obstacle/
// agent-pair rows (Nocedal & Wright Sec. 18.5, see that class's own doc
// comment): a LESS_THAN row gets one nonnegative slack; an EQUAL row
// (two-sided) gets a nonnegative plus/minus PAIR (`g_val = s_plus -
// s_minus`, the standard elastic-variable reformulation of an equality in
// an exact-penalty method); `_interaction_penalty_weight * (sum of every
// row's slack(s))` added to the QP objective, and the trust-region loop's
// accept/reject now runs on the merit function
// `phi(x) = f(x) + _interaction_penalty_weight * violation(x)` (mirroring
// GraphShortPathMPC's RunTrustRegionSqp) instead of the plain objective
// ratio. This restores "every QP subproblem feasible by construction" as
// an ACTUALLY true statement (dx=0, slack=|current violation| always
// satisfies every row, independent of trust_radius) -- the tau/v box and
// the smooth Gauss-Newton Hessian are unaffected, so behavior only changes
// in cycles where the interaction rows and the trust region would
// otherwise have conflicted, and reproduces the unrelaxed optimum whenever
// that's reachable within the trust region and
// `_interaction_penalty_weight` exceeds the active rows' true Lagrange
// multipliers (same caveat as any exact-penalty method -- see this
// parameter's own comment below). A stress test
// (25 randomized 3-agent scenarios with active cross-agent timing
// constraints) found the old IPOPT-based implementation failing on 24/25
// while this one succeeded on all 25, ~30x faster on average.
//
// This gives a genuine classical guarantee (Nocedal & Wright's trust-region
// convergence theorem, given the trust-region accept/reject mechanics below
// are implemented correctly -- which they are): the iterates converge to a
// first-order stationary point (a local optimum, in the standard NLP
// sense) -- not a global-optimum guarantee (acceleration_cost's objective
// is genuinely non-convex, its coast-corrected residual has a bilinear
// cross term in (tau, v)), but a categorically stronger reliability
// property than IPOPT's for this problem shape, and specifically the
// reason "stability_cost" (a convex-but-not-physically-meaningful proxy
// cost, dropping that same cross term to stay convex for IPOPT's sake) is
// no longer needed as a fallback -- the solver, not the objective, is what
// used to need convexity.
//
// Scope, narrower than the Drake/IPOPT implementation this replaces:
//   - cost: time_cost (linear) + time_cost2 (diagonal quadratic, both
//     exact, no GN approximation needed) + acceleration_cost*psi (GN-
//     Hessian-approximated -- see gn_timing::AssembleObjective's own doc
//     comment). No stability_cost/energy_cost/arclength_cost -- the
//     constructor throws if energy_cost/arclength_cost is requested
//     nonzero rather than silently dropping it (arclength_cost in
//     particular was already a known non-convexity/slowdown source in the
//     implementation this replaces); stability_cost isn't accepted at all
//     (not even as a throw-if-nonzero parameter) since it's a solved
//     problem, not an unsupported one -- see above.
//   - no max_vel/max_acc/max_jerk hard bounds -- the constructor throws if
//     max_vel/max_acc is requested with an actual (> 0) bound (the
//     unbounded sentinel <= 0 is accepted as a no-op, matching every
//     existing caller's default). Adding these back would mean linearizing
//     a genuinely nonlinear constraint every SQP iteration -- exactly the
//     kind of fragility this class exists to avoid for the interaction
//     constraints, and (for max_acc specifically) the same coast-corrected,
//     provably-non-convex formula acceleration_cost already has. max_jerk
//     was already an inert no-op in the implementation this replaces (its
//     handling was commented out) -- accepted and silently ignored here
//     too, matching that existing (do-nothing) behavior exactly, not a
//     regression.
//   - blocks: R, Torus, SO3Quat (same PositionDelta-based residual math,
//     reused verbatim via CubicConfigurationSpline::PositionDelta/
//     BlockPositionDelta<double>, evaluated ONCE since x's are fixed
//     constants at this phase -- so no autodiff is ever needed through the
//     wrap/quaternion-log branches themselves, only through the tau/v-
//     dependent D/V math built on top of that constant). SO3Mat throws,
//     matching BlockPositionDelta's own existing stub.
//   - constraint curvature (lambda-weighted Hessian of g) is never needed:
//     every g row here is exactly linear, so its Hessian is exactly zero,
//     not merely a Gauss-Newton approximation.
struct GraphTimingMPC {
	const GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// Persistent output buffers.
	std::vector<Eigen::MatrixXd> _wps_list;
	std::vector<Eigen::MatrixXd> _vs_list;
	std::vector<Eigen::VectorXd> _time_deltas_list;
	std::vector<std::vector<int>> _agent_nodes_list;
	std::map<int, int> _agent_spline_length_map;

	// The var-indexed assignment vector (var id -> resolved agent, -1 ==
	// unassigned) the last solve()/solve_dense() was handed -- kept for
	// fill_cubic_splines, which runs right after and asks the graph
	// (constrained_columns) which config blocks each node pins;
	// var_agent_q(...) references resolve through this. Empty only if the
	// caller passed nothing (a graph with no assignable-agent constraints).
	Eigen::VectorXi _last_var_assignments;

	// Per-agent (K+1) x num_blocks active-knot mask from the last
	// solve()/solve_dense() layout build (ProblemLayout::agent_block_active).
	// fill_cubic_splines feeds it straight to
	// CubicConfigurationSpline::set_block_active_mask -- so the output
	// spline's bridged knots are exactly the ones the timing solve merged
	// its BlockSegments across, with no second constrained_columns pass.
	// Empty entry == that agent had no active spline that cycle.
	std::vector<Eigen::MatrixXi> _agent_block_active_list;

	double _time_cost;
	double _time_cost2;
	double _acceleration_cost;

	// Weight on the interaction rows' slack relaxation (see this struct's
	// own doc comment's CORRECTION paragraph) -- large enough and a
	// relaxed-optimum solve reproduces the unrelaxed (hard-constrained)
	// one exactly whenever that's reachable within the current trust
	// region (standard exact-penalty behavior, Nocedal & Wright Sec. 18.5:
	// the threshold is the true Lagrange multiplier magnitude of whichever
	// interaction row is active at the optimum, not knowable a priori);
	// too small and a cross-agent LESS_THAN/EQUAL constraint can be
	// smoothly traded away against tracking/acceleration cost instead of
	// actually enforced. Default matches GraphShortPathMPC's own
	// `penalty_weight` default -- no case-specific tuning done yet.
	double _interaction_penalty_weight;

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

	// Reused across solve()/solve_dense() calls -- qpOASES::SQProblem's
	// hot-start only pays off if the SAME instance persists cycle-to-cycle.
	// Held as a raw pointer with manual lifetime management (constructed
	// lazily on first solve, re-constructed whenever `n`/`m` change) since
	// qpOASES::SQProblem isn't move-constructible in a way that plays
	// nicely with resizing.
	struct QpState;
	std::unique_ptr<QpState> _qp_state;

	drake::SteadyTimer _timer;
	double _last_solve_time = 0.0;
	// Diagnostics from the most recent solve()/solve_dense(): how many
	// outer iterations it actually ran, and the trust radius it ended on (a
	// radius pinned at _min_trust_radius signals a stall -- see solve()'s
	// own comment).
	int _last_iterations = 0;
	double _last_trust_radius = 0.0;

	// energy_cost/arclength_cost: accepted for constructor-signature
	// compatibility with every existing caller (which all default to 0.0),
	// but throws if either is requested nonzero -- see this struct's own
	// doc comment for why. max_vel/max_acc: same treatment, throws if
	// either carries an actual (> 0) bound (the unbounded sentinel <= 0 is
	// a no-op). max_jerk: accepted and silently ignored (matches the
	// pre-existing, already-inert behavior). No stability_cost parameter at
	// all -- see this struct's own doc comment.
	GraphTimingMPC(const GraphOfConstraints& graph,
		       std::vector<CubicConfigurationSpline> splines,
		       double time_cost = 1e0,
		       double time_cost2 = 0e0,
		       double acceleration_cost = 1.0,
		       double energy_cost = 0.0,
		       double arclength_cost = 0.0,
		       std::vector<double> max_vel = {},
		       std::vector<double> max_acc = {},
		       std::vector<double> max_jerk = {},
		       int max_iterations = 50,
		       double initial_trust_radius = 1.0,
		       double max_trust_radius = 50.0,
		       double min_trust_radius = 1e-6,
		       double grad_tol = 1e-6,
		       double interaction_penalty_weight = 1.0e3);
	~GraphTimingMPC();

	// Explicit, not implicit: declaring ~GraphTimingMPC() above (needed so
	// QpState can stay an incomplete pimpl-style forward declaration here,
	// only completed in graph_timing_mpc.cpp) suppresses the compiler's
	// implicit move constructor -- and pybind11's py::init(lambda) factory
	// form (goc_mpc.cpp, needed for the max_vel/max_acc broadcast
	// preprocessing that happens before construction) move-constructs the
	// lambda's returned-by-value temporary into its own storage, so it
	// needs one to exist. Defined (= default) in graph_timing_mpc.cpp,
	// where QpState is a complete type -- unique_ptr<QpState>'s move
	// constructor needs that. Copy stays implicitly deleted (unique_ptr
	// member), which is correct -- nothing here should ever be copied.
	GraphTimingMPC(GraphTimingMPC&&) noexcept;
	GraphTimingMPC& operator=(GraphTimingMPC&&) noexcept;

	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const std::vector<int>& remaining_vertices,
		   const Eigen::MatrixXd& waypoints,
		   const Eigen::VectorXi& var_assignments,
		   const Eigen::VectorXd& t_by_node = Eigen::VectorXd());

	// Dense-waypoint counterpart to solve(): the caller has already
	// resolved graph ordering + cross-agent interactions itself (via
	// graph.get_agent_paths, on the SAME graph this instance was
	// constructed with) and expanded each agent's real-node sequence into
	// a denser one (e.g. via path tracing between consecutive real nodes),
	// reindexing `agent_interactions`' depths against that dense sequence
	// (GraphOfConstraints::reindex_agent_interactions) -- this just builds
	// and solves the resulting problem (gn_timing::BuildDenseProblemLayout)
	// and stores the result the same way solve() does, so every other
	// method (fill_cubic_splines, set_progressed_time, get_next_nodes, ...)
	// works unchanged regardless of which one produced the current
	// solution. `agent_dense_node_ids[i]` becomes this instance's own
	// `_agent_nodes_list[i]` (real id per row, or -1 for synthetic) --
	// unlike solve()'s fixed-size (num_nodes-bounded) per-agent buffers,
	// this REPLACES `_wps_list[i]`/`_vs_list[i]`/`_time_deltas_list[i]`
	// wholesale each call, since a dense sequence's length isn't bounded by
	// the graph's own node count.
	bool solve_dense(const Eigen::VectorXd& x0,
			  const Eigen::VectorXd& v0,
			  const std::vector<Eigen::MatrixXd>& agent_dense_wps,
			  const std::vector<std::vector<int>>& agent_dense_node_ids,
			  const std::vector<AgentInteraction>& agent_interactions,
			  const Eigen::VectorXi& var_assignments = Eigen::VectorXi());

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
