#include "graph_short_path_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <Eigen/Sparse>
#include <proxsuite/proxqp/sparse/sparse.hpp>

#include "obstacle_projection.hpp"

using namespace sqp_short_path;

namespace {
namespace psp = proxsuite::proxqp;
using SpMat = Eigen::SparseMatrix<double, Eigen::ColMajor, long long>;
using Trip = Eigen::Triplet<double, long long>;
// proxqp treats any bound whose magnitude reaches helpers::infinite_bound
// (sqrt(DBL_MAX)) as +/-infinity -- the sentinel for a one-sided row.
const double kProxInf = proxsuite::helpers::infinite_bound<double>::value();
}  // namespace

// Sparse QP subproblem solver for the trust-region SQP outer loop (was
// qpOASES::SQProblem, dense). The smooth-cost Hessian is block-diagonal
// per agent (agents couple only through constraint ROWS, never H) and
// every constraint row is sparse (workspace_dim / 2*workspace_dim nonzeros
// + one slack column), so a sparse solve is the scaling lever as agent
// count grows -- see the v2 plan. proxqp's proximal method is also robust
// to the mildly-nonconvex, iterate-dependent H an SO3Quat block produces.
//
// Hot-started across outer iterations AND across solve() calls: proxqp's
// update() (values only, keeps the symbolic factorization) + WARM_START_
// WITH_PREVIOUS_RESULT while the H/C sparsity pattern is unchanged, falling
// back to a full init() when it isn't. The pattern only moves when an
// SO3Quat block's coupled Hessian entry crosses exactly zero, or a new
// solve() call's distance-pruned obstacle/pair set changes shape -- both
// rare -- so the common case is a cheap values-only update.
struct GraphShortPathMPC::QpState {
	psp::sparse::QP<double, long long> qp;
	int n = 0;
	int n_in = 0;
	bool initialized = false;
	// Sparsity-pattern signature (outer + inner index arrays concatenated)
	// of the H and C matrices last passed to init(); update() only takes
	// effect while these are unchanged.
	std::vector<long long> h_pattern;
	std::vector<long long> c_pattern;

	QpState(int n_, int n_in_) : qp(n_, 0, n_in_), n(n_), n_in(n_in_) {
		qp.settings.verbose = false;
		qp.settings.compute_timings = false;
		// 1e-7 absolute (proxqp's own default is 1e-5): loosening it to
		// 1e-6 was measurably WORSE -- the noisier QP steps made the
		// trust-region outer loop reject more and stop converging on the
		// harder multi-agent scenes. eps_rel 0 keeps it a pure absolute
		// criterion.
		qp.settings.eps_abs = 1.0e-7;
		qp.settings.eps_rel = 0.0;
	}
};

namespace {
// Compressed sparsity-pattern signature for the update()-vs-init() decision
// in RunTrustRegionSqp -- concatenated outerIndexPtr (outerSize+1 entries)
// and innerIndexPtr (nnz entries).
std::vector<long long> PatternKey(const SpMat& m) {
	std::vector<long long> key;
	const long long nnz = m.nonZeros();
	key.reserve(static_cast<size_t>(m.outerSize()) + 1 + static_cast<size_t>(nnz));
	for (int k = 0; k <= m.outerSize(); ++k) key.push_back(m.outerIndexPtr()[k]);
	for (long long k = 0; k < nnz; ++k) key.push_back(m.innerIndexPtr()[k]);
	return key;
}
}  // namespace

GraphShortPathMPC::GraphShortPathMPC(const GraphOfConstraints& graph,
				  unsigned int num_steps,
				  unsigned int num_agents,
				  double time_per_step,
				  const ObstacleSet& obstacles,
				  Eigen::VectorXd agent_radii,
				  double tracking_weight,
				  double velocity_tracking_weight,
				  double acceleration_weight,
				  double penalty_weight,
				  int max_iterations,
				  double initial_trust_radius,
				  double max_trust_radius,
				  double min_trust_radius,
				  double grad_tol,
				  double constraint_prune_margin)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _time_per_step(time_per_step),
	  _obstacles(&obstacles),
	  _agent_radii(agent_radii.size() == 0 ? Eigen::VectorXd::Zero(num_agents) : std::move(agent_radii)),
	  _smooth_cost_weights{tracking_weight, velocity_tracking_weight, acceleration_weight},
	  _penalty_weight(penalty_weight),
	  _max_iterations(max_iterations),
	  _initial_trust_radius(initial_trust_radius),
	  _max_trust_radius(max_trust_radius),
	  _min_trust_radius(min_trust_radius),
	  _grad_tol(grad_tol),
	  _constraint_prune_margin(constraint_prune_margin) {

	if (_agent_radii.size() != static_cast<int>(num_agents)) {
		throw std::runtime_error(
			"GraphShortPathMPC: agent_radii must be empty (defaults every agent to "
			"radius 0) or have exactly num_agents entries.");
	}

	_agent_shapes = BuildAgentShapes(graph, static_cast<int>(num_agents));
	// Agents are NOT required to share a tangent_dim/ambient_dim with each
	// other (Stage 5) -- each agent's own width, whatever it is, comes
	// straight from BuildAgentShapes. The one thing every agent still needs
	// is an ambient width of at least workspace_dim, since the fk fast path
	// (LinearizeObstacleConstraints et al.) reads that agent's leading
	// workspace_dim ambient columns as its world position unconditionally;
	// an agent that's too narrow for that would silently read past its own
	// slice into the next agent's columns (a corruption, not a crash --
	// this project's RelWithDebInfo build compiles out eigen_assert, see
	// feedback_eigen_row_col_no_assert in project memory), so this is
	// checked loudly here instead.
	for (unsigned int ag = 0; ag < num_agents; ++ag) {
		if (_agent_shapes[ag].ambient_dim() < graph.workspace_dim) {
			throw std::runtime_error(
				"GraphShortPathMPC: agent " + std::to_string(ag) + "'s ambient "
				"width is narrower than the graph's workspace_dim -- the fk fast "
				"path needs every agent's leading workspace_dim ambient columns "
				"to be its world position.");
		}
	}
	_axes = BuildAxisList(_agent_shapes);
	_agent_axis_offsets = BuildAgentAxisOffsets(_agent_shapes);
	_agent_ambient_offsets = BuildAgentAmbientOffsets(_agent_shapes);

	// Per-agent collision model: the trivial single-radius-0-point model
	// (exact old fk fast path) unless the caller registered a body via
	// graph.set_agent_collision_model (v2 plan Stage 3), in which case build
	// the real model here (a Drake MultibodyPlant for kArticulated).
	_agent_collision_models.reserve(num_agents);
	for (unsigned int ag = 0; ag < num_agents; ++ag) {
		const int wd = graph.workspace_dim;
		const int td = _agent_shapes[ag].tangent_dim();
		const int ad = _agent_shapes[ag].ambient_dim();
		const auto it = graph.agent_collision_specs.find(static_cast<int>(ag));
		if (it == graph.agent_collision_specs.end()) {
			_agent_collision_models.push_back(MakeTrivialCollisionModel(wd, td, ad));
		} else if (it->second.kind == AgentCollisionSpec::Kind::kArticulated) {
			_agent_collision_models.push_back(MakeDrakePlantCollisionModel(it->second, wd, td));
		} else {
			throw std::runtime_error(
				"GraphShortPathMPC: agent " + std::to_string(ag) + " registered a "
				"kRigidConstellation collision model, which is not implemented yet "
				"(v2 plan Stage 3b).");
		}
	}
	_smooth_hessian_normal = AssembleSmoothHessian(_agent_shapes, _agent_axis_offsets,
							static_cast<int>(num_steps), time_per_step,
							_smooth_cost_weights);

	// Xi.row(i) is offset by one tau from x0/v0's own time: Xi.row(0) is
	// treated as the state ONE tau after x0/v0 by the acceleration term
	// throughout this file, so the tracking reference must be sampled at
	// the same offset.
	_times = Eigen::VectorXd(_num_steps);
	for (unsigned int i = 0; i < _num_steps; ++i) {
		_times(i) = (i + 1) * _time_per_step;
	}
	// Total ambient/tangent width, summed across each agent's OWN width
	// (not `num_agents * ` a shared one) -- `_axes.size()` is exactly the
	// total tangent width (BuildAxisList flattens every agent's tangent
	// columns into one agent-major list), so only the ambient total needs
	// summing explicitly here.
	int total_ambient = 0;
	for (const auto& shape : _agent_shapes) total_ambient += shape.ambient_dim();
	_points = Eigen::MatrixXd::Zero(_num_steps, total_ambient);
	_vels = Eigen::MatrixXd::Zero(_num_steps, static_cast<int>(_axes.size()));
}

std::pair<Eigen::MatrixXd, Eigen::VectorXd> GraphShortPathMPC::eval_agent_collision_spheres(
		int agent_id, const Eigen::VectorXd& q_agent) const {
	const AgentCollisionModel& model = *_agent_collision_models.at(agent_id);
	if (q_agent.size() != model.ambient_dim()) {
		throw std::runtime_error(
			"GraphShortPathMPC::eval_agent_collision_spheres: q_agent has " +
			std::to_string(q_agent.size()) + " entries but agent " + std::to_string(agent_id) +
			"'s collision model expects " + std::to_string(model.ambient_dim()) + ".");
	}
	const std::vector<sqp_short_path::WorkspaceSphere> spheres = model.Eval(q_agent);
	const int wd = _graph->workspace_dim;
	Eigen::MatrixXd centers(static_cast<int>(spheres.size()), wd);
	Eigen::VectorXd radii(static_cast<int>(spheres.size()));
	for (int k = 0; k < static_cast<int>(spheres.size()); ++k) {
		centers.row(k) = spheres[k].center.head(wd).transpose();
		radii(k) = spheres[k].radius;
	}
	return {centers, radii};
}

std::vector<Eigen::MatrixXd> GraphShortPathMPC::eval_agent_collision_jacobians(
		int agent_id, const Eigen::VectorXd& q_agent) const {
	const AgentCollisionModel& model = *_agent_collision_models.at(agent_id);
	if (q_agent.size() != model.ambient_dim()) {
		throw std::runtime_error(
			"GraphShortPathMPC::eval_agent_collision_jacobians: q_agent has " +
			std::to_string(q_agent.size()) + " entries but agent " + std::to_string(agent_id) +
			"'s collision model expects " + std::to_string(model.ambient_dim()) + ".");
	}
	const std::vector<sqp_short_path::WorkspaceSphere> spheres = model.Eval(q_agent);
	std::vector<Eigen::MatrixXd> jacs;
	jacs.reserve(spheres.size());
	for (const sqp_short_path::WorkspaceSphere& s : spheres) jacs.push_back(s.jac);
	return jacs;
}

GraphShortPathMPC::~GraphShortPathMPC() = default;
GraphShortPathMPC::GraphShortPathMPC(GraphShortPathMPC&&) noexcept = default;
GraphShortPathMPC& GraphShortPathMPC::operator=(GraphShortPathMPC&&) noexcept = default;

namespace {

// Retracts every agent's every step by the smooth-cost step vector
// `dx_smooth` (length axes.size()*2*num_steps, same convention as
// sqp_short_path::IdxP/IdxV) -- positions via each agent's own
// CubicConfigurationSpline::Retract (R: plain add, Torus: wrap_pi(x+d)),
// velocities via plain addition (they live in a flat tangent space
// already, no manifold to respect).
void ApplyStep(const std::vector<CubicConfigurationSpline>& agent_shapes,
		const std::vector<int>& agent_axis_offsets, const std::vector<int>& agent_ambient_offsets,
		int num_steps, const Eigen::MatrixXd& points, const Eigen::MatrixXd& vels,
		const Eigen::VectorXd& dx_smooth,
		Eigen::MatrixXd* new_points, Eigen::MatrixXd* new_vels) {
	*new_points = points;
	*new_vels = vels;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		const int off = agent_axis_offsets[ag];
		const int dim = agent_shapes[ag].tangent_dim();
		const int ambient_dim = agent_shapes[ag].ambient_dim();
		for (int i = 0; i < num_steps; ++i) {
			Eigen::VectorXd dp(dim), dv(dim);
			for (int k = 0; k < dim; ++k) {
				dp(k) = dx_smooth(IdxP(off + k, i, num_steps));
				dv(k) = dx_smooth(IdxV(off + k, i, num_steps));
			}
			const Eigen::VectorXd p_old =
				points.row(i).segment(agent_ambient_offsets[ag], ambient_dim).transpose();
			const Eigen::VectorXd p_new = agent_shapes[ag].Retract<double>(p_old, dp);
			new_points->row(i).segment(agent_ambient_offsets[ag], ambient_dim) = p_new.transpose();
			new_vels->row(i).segment(off, dim) += dv.transpose();
		}
	}
}

double TotalSmoothCost(const std::vector<CubicConfigurationSpline>& agent_shapes,
			const std::vector<int>& agent_axis_offsets, const std::vector<int>& agent_ambient_offsets,
			int num_steps, double tau, const SmoothCostWeights& weights,
			const Eigen::VectorXd& x0, const Eigen::VectorXd& v0,
			const Eigen::MatrixXd& points, const Eigen::MatrixXd& vels,
			const Eigen::MatrixXd& ref_points, const Eigen::MatrixXd& ref_velocities) {
	double f = 0.0;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		const int dim = agent_shapes[ag].tangent_dim();
		const int ambient_dim = agent_shapes[ag].ambient_dim();
		f += EvaluateSmoothCost(
			agent_shapes[ag], num_steps, tau, weights,
			x0.segment(agent_ambient_offsets[ag], ambient_dim), v0.segment(agent_axis_offsets[ag], dim),
			points.block(0, agent_ambient_offsets[ag], num_steps, ambient_dim),
			vels.block(0, agent_axis_offsets[ag], num_steps, dim),
			ref_points.block(0, agent_ambient_offsets[ag], num_steps, ambient_dim),
			ref_velocities.block(0, agent_axis_offsets[ag], num_steps, dim));
	}
	return f;
}

// project_out's own "needs projecting" test (box: inside the
// margin-expanded axis-aligned box; sphere: within radius+margin) --
// duplicated here (not exposed by obstacle_projection.hpp) so
// EscapeAlongAxis below can check "did this actually clear the obstacle"
// with the exact same feasibility definition project_out itself uses.
bool IsInsideCandidate(const Eigen::VectorXd& p, const Candidate& c) {
	if (c.is_box) {
		const Eigen::VectorXd he = c.half_extents.array() + c.margin;
		return (p.array() >= (c.center - he).array()).all() &&
		       (p.array() <= (c.center + he).array()).all();
	}
	return (p - c.center).norm() < c.radius + c.margin;
}

// Last-resort escape for a point the alternating-projection rounds below
// leave oscillating between two (or more) candidates -- confirmed via a
// from-scratch reproduction (not a project_out bug) that this genuinely
// happens: two candidates whose "least-penetrated axis" (project_out's own
// per-candidate escape direction) is the SAME axis but in OPPOSITE signs
// walk a point back and forth between them forever, since project_out only
// ever looks at ONE candidate at a time. This tries every world axis/sign
// directly (not just each candidate's own preferred axis), computing the
// minimum push along that one axis that clears EVERY candidate
// simultaneously (valid because clearing a box/sphere via a single
// coordinate moving far enough past its projected extent is always
// sufficient, never just necessary -- see the per-candidate `required`
// derivation below), and takes whichever axis/sign needs the smallest such
// push. Runs ONLY on points still violated after the standard rounds, so
// it's zero-cost in the common (already-feasible) case.
Eigen::VectorXd EscapeAlongAxis(const Eigen::VectorXd& p, const std::vector<Candidate>& candidates,
				 int workspace_dim) {
	double best_delta = std::numeric_limits<double>::infinity();
	int best_axis = -1;
	double best_sign = 1.0;
	for (int k = 0; k < workspace_dim; ++k) {
		for (double sign : {1.0, -1.0}) {
			double delta = 0.0;
			for (const Candidate& c : candidates) {
				const double extent = c.is_box ? (c.half_extents(k) + c.margin) : (c.radius + c.margin);
				const double target_k = c.center(k) + sign * extent;
				delta = std::max(delta, std::max(0.0, sign * (target_k - p(k))));
			}
			if (delta < best_delta) {
				best_delta = delta;
				best_axis = k;
				best_sign = sign;
			}
		}
	}
	Eigen::VectorXd result = p;
	if (best_axis >= 0) {
		result(best_axis) += best_sign * best_delta;
	}
	return result;
}

// Final hard-feasibility safety net, using obstacle_projection.hpp's
// closed-form projection -- guarantees the RETURNED trajectory clears every
// registered obstacle regardless of whether the SQP loop above fully
// converged within `max_iterations` (design decision 7 in the project
// plan). This is the actual feasibility guarantee; the SQP loop's job is
// only to make this pass's corrections small. A single sweep isn't
// enough once obstacles are close enough to interact (confirmed
// empirically by stress testing), hence several repeated sweeps
// (Dykstra-style alternating
// projection) followed by EscapeAlongAxis for anything that's STILL
// violated (the two-obstacle oscillation case above -- confirmed to
// actually occur via a real stress-test trial, not just a theoretical
// worry). Fast-path-only, like everything below -- see
// ApplyAgentPairSafetyProjection's own comment for why, and for the
// intended future toggle once that stops being universally true.
// Closed-form-ish projection of `p` outside an AgentSdfGrid's own feasible
// region (`value >= 0`, i.e. already past `grid.margin`'s clearance) --
// mirrors project_out's role for sphere/box candidates, but there's no
// literal closed form for an arbitrary field, so this takes ONE Newton-style
// step along the (already unit-normalized-by-construction, see
// QueryAgentSdfGrid) ascending-value direction: `p + (-value) * grad_hat`,
// same "push exactly to the zero-level-set along the local gradient" idea
// SphereSdf's own closed-form projection reduces to near a sphere. Not
// exact for a genuinely curved field in one step (unlike a sphere), which
// is why this runs inside the SAME repeated-rounds loop as the sphere/box
// candidates below, not just once.
Eigen::VectorXd ProjectOutOfGrid(const Eigen::VectorXd& p, const AgentSdfGrid& grid, int workspace_dim) {
	const sqp_short_path::SdfSample sdf = sqp_short_path::QueryAgentSdfGrid(grid, p, workspace_dim);
	if (sdf.value >= 0.0) {
		return p;
	}
	const double norm = sdf.grad.norm();
	if (norm < 1.0e-9) {
		// Degenerate (flat/zero gradient, e.g. deep inside a saturated
		// region) -- same arbitrary-fixed-axis fallback as project_out's
		// own degenerate case.
		Eigen::VectorXd dir = Eigen::VectorXd::Zero(workspace_dim);
		dir(0) = 1.0;
		return p - sdf.value * dir;
	}
	return p - sdf.value * (sdf.grad / norm);
}

void ApplySafetyProjection(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
			    int workspace_dim, const ObstacleSet& obstacles, const AgentCollisionModels& models,
			    Eigen::MatrixXd* points) {
	constexpr int kSafetyPassRounds = 10;
	for (int ag = 0; ag < num_agents; ++ag) {
		// Non-trivial (rigid multi-sphere / articulated) agents have no
		// closed-form "project the config so fk(q) clears the obstacle"
		// (that's IK) -- their hard feasibility rests on the exact-penalty
		// QP rows plus (v2 plan Stage 3e, not yet implemented) a bounded
		// Gauss-Newton tangent-space projection. Skip them here rather
		// than corrupt their ambient columns with a point projection.
		if (!models[ag]->is_trivial()) continue;
		const AgentSdfGrid* grid = obstacles.agent_sdf_grid(ag);
		const Eigen::MatrixXd agent_workspace_traj =
			points->block(0, agent_ambient_offsets[ag], num_steps, workspace_dim);
		const std::vector<Candidate> candidates =
			gather_candidates(obstacles, agent_workspace_traj, workspace_dim, /*query_margin=*/1.0);
		if (candidates.empty() && !grid) continue;
		for (int round = 0; round < kSafetyPassRounds; ++round) {
			for (int i = 0; i < num_steps; ++i) {
				Eigen::VectorXd p = points->row(i).segment(agent_ambient_offsets[ag], workspace_dim).transpose();
				for (const Candidate& c : candidates) {
					p = project_out(p, c);
				}
				if (grid) {
					p = ProjectOutOfGrid(p, *grid, workspace_dim);
				}
				points->row(i).segment(agent_ambient_offsets[ag], workspace_dim) = p.transpose();
			}
		}
		for (int i = 0; i < num_steps; ++i) {
			Eigen::VectorXd p = points->row(i).segment(agent_ambient_offsets[ag], workspace_dim).transpose();
			bool any_violated = false;
			for (const Candidate& c : candidates) any_violated |= IsInsideCandidate(p, c);
			if (any_violated) {
				points->row(i).segment(agent_ambient_offsets[ag], workspace_dim) =
					EscapeAlongAxis(p, candidates, workspace_dim).transpose();
			}
			// Grid violations left after the rounds above aren't covered
			// by EscapeAlongAxis (built for closed-form box/sphere
			// per-axis extents, no equivalent for an arbitrary field --
			// see this function's own header comment) -- a few more
			// direct grid-only pushes instead, cheap and a no-op in the
			// common (already-feasible) case.
			if (grid) {
				Eigen::VectorXd pg = points->row(i).segment(agent_ambient_offsets[ag], workspace_dim).transpose();
				for (int extra = 0; extra < kSafetyPassRounds &&
					     sqp_short_path::QueryAgentSdfGrid(*grid, pg, workspace_dim).value < 0.0;
				     ++extra) {
					pg = ProjectOutOfGrid(pg, *grid, workspace_dim);
				}
				points->row(i).segment(agent_ambient_offsets[ag], workspace_dim) = pg.transpose();
			}
		}
	}
}

// Final hard-feasibility safety net for INTER-AGENT separation, mirroring
// ApplySafetyProjection above but BILATERAL: a violated pair is pushed
// apart symmetrically (each agent moves half the required separation)
// rather than one point being projected against a fixed obstacle. Same
// repeated-rounds rationale (a single sweep isn't enough once several
// pairs interact at once -- see ApplySafetyProjection's own comment).
//
// Fast-path-only, same as ApplySafetyProjection: both write directly into
// ambient position columns, which is only valid while fk(q)=q[:workspace_
// dim]. Neither is gated on that today because every agent IS fast-path
// today -- there's nothing yet to gate against. Once a general (non-
// selection) fk exists for some agent (out of scope for this solver, see
// the project plan), this closed-form projection has no equivalent for
// that agent (no closed form for "solve fk(q)=target_point", an IK
// problem) -- calling it would be wrong, not just imprecise. The intended
// fix at that point is a per-agent "is this agent's fk trivial" check
// (which the constraint-linearization code will ALSO need by then, to
// decide whether to chain through a real Jacobian -- see that discussion
// in the project plan) gating which agents these two passes touch, rather
// than a separate flag a caller has to keep in sync with which agents
// have a registered FK.
void ApplyAgentPairSafetyProjection(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
				     int workspace_dim, const Eigen::VectorXd& agent_radii,
				     const AgentCollisionModels& models, Eigen::MatrixXd* points) {
	if (num_agents < 2) {
		return;
	}
	constexpr int kSafetyPassRounds = 10;
	for (int round = 0; round < kSafetyPassRounds; ++round) {
		for (int i = 0; i < num_steps; ++i) {
			for (int ag_a = 0; ag_a < num_agents; ++ag_a) {
				for (int ag_b = ag_a + 1; ag_b < num_agents; ++ag_b) {
					// Bilateral point-push only makes sense when BOTH
					// agents are trivial point agents (see
					// ApplySafetyProjection's own gate); a non-trivial
					// body's separation is left to the QP rows / Stage 3e.
					if (!models[ag_a]->is_trivial() || !models[ag_b]->is_trivial()) continue;
					Eigen::VectorXd p_a = points->row(i).segment(agent_ambient_offsets[ag_a], workspace_dim).transpose();
					Eigen::VectorXd p_b = points->row(i).segment(agent_ambient_offsets[ag_b], workspace_dim).transpose();
					const double R = agent_radii(ag_a) + agent_radii(ag_b);
					const Eigen::VectorXd diff = p_b - p_a;
					const double d = diff.norm();
					if (d >= R) {
						continue;
					}
					Eigen::VectorXd dir;
					if (d < 1.0e-9) {
						dir = Eigen::VectorXd::Zero(workspace_dim);
						dir(0) = 1.0;
					} else {
						dir = diff / d;
					}
					const double push = 0.5 * (R - d);
					p_a -= push * dir;
					p_b += push * dir;
					points->row(i).segment(agent_ambient_offsets[ag_a], workspace_dim) = p_a.transpose();
					points->row(i).segment(agent_ambient_offsets[ag_b], workspace_dim) = p_b.transpose();
				}
			}
		}
	}
}

struct SqpResult {
	Eigen::MatrixXd points, vels;
	int iterations = 0;
	double trust_radius = 0.0;
};

SqpResult RunTrustRegionSqp(
		const std::vector<CubicConfigurationSpline>& agent_shapes,
		const std::vector<AxisLayout>& axes,
		const std::vector<int>& agent_axis_offsets,
		const std::vector<int>& agent_ambient_offsets,
		const AgentCollisionModels& models,
		const Eigen::MatrixXd& smooth_hessian_normal,
		const SmoothCostWeights& smooth_cost_weights,
		int num_steps, int num_agents, int workspace_dim, double tau,
		const Eigen::VectorXd& x0, const Eigen::VectorXd& v0,
		const Eigen::MatrixXd& ref_points, const Eigen::MatrixXd& ref_velocities,
		const ObstacleSet& obstacles,
		const std::vector<std::vector<ActiveObstacle>>& per_agent_obstacles,
		const std::vector<ActivePair>& active_pairs,
		const std::vector<ActiveGrid>& active_grids,
		const Eigen::VectorXd& agent_radii,
		double penalty_weight, int max_iterations,
		double initial_trust_radius, double max_trust_radius, double min_trust_radius, double grad_tol,
		Eigen::MatrixXd points, Eigen::MatrixXd vels,
		std::unique_ptr<GraphShortPathMPC::QpState>* qp_state) {

	const int per_axis = 2 * num_steps;
	const int n_smooth = static_cast<int>(axes.size()) * per_axis;
	// Fixed for the whole solve() call (design decision 6): row COUNT never
	// changes mid-call, only each row's coefficients/value do (re-linearized
	// every outer iteration). `m` sums the per-(agent,obstacle)/pair/grid
	// active-step ranges from the distance pruning (PruneObstaclesByDistance
	// etc.), which happens once before this function runs
	// (GraphShortPathMPC::solve()) -- same "computed once per solve() call"
	// discipline as everything else.
	// Each active (agent, obstacle) / pair / (agent, grid) contributes one
	// row PER BODY SPHERE (per pair of spheres for the inter-agent case) --
	// num_spheres() == 1 for every trivial agent, so this reduces to the
	// old per-step counts when nothing has a registered collision model.
	int obstacle_rows = 0;
	for (int ag = 0; ag < num_agents; ++ag) {
		const int ks = models[ag]->num_spheres();
		for (const ActiveObstacle& ao : per_agent_obstacles[ag])
			obstacle_rows += ks * (ao.step_hi - ao.step_lo);
	}
	int pair_rows = 0;
	for (const ActivePair& ap : active_pairs)
		pair_rows += models[ap.ag_a]->num_spheres() * models[ap.ag_b]->num_spheres() *
			     (ap.step_hi - ap.step_lo);
	int grid_rows = 0;
	for (int ag = 0; ag < num_agents; ++ag)
		if (active_grids[ag].grid)
			grid_rows += models[ag]->num_spheres() *
				     (active_grids[ag].step_hi - active_grids[ag].step_lo);
	const int m = obstacle_rows + pair_rows + grid_rows;
	// QP decision vector z = [dx_smooth (n_smooth) | slack (m)].
	const int n = n_smooth + m;
	// proxqp inequality rows: `m` slack-relaxed penalty rows (a^T dx + s >=
	// -c(x)), then the trust-region box on dx_smooth (n_smooth identity
	// rows, bounds move each outer iteration), then the slack lower bounds
	// s >= 0 (m identity rows). proxqp's sparse backend has no separate box
	// facility, so every bound is an explicit row -- each a single +/-1, so
	// the extra rows barely touch the sparsity.
	const int n_in = m + n_smooth + m;

	if (!*qp_state || (*qp_state)->n != n || (*qp_state)->n_in != n_in) {
		*qp_state = std::make_unique<GraphShortPathMPC::QpState>(n, n_in);
	}
	GraphShortPathMPC::QpState& st = **qp_state;

	double f_current = TotalSmoothCost(agent_shapes, agent_axis_offsets, agent_ambient_offsets, num_steps, tau,
					    smooth_cost_weights, x0, v0, points, vels, ref_points, ref_velocities);
	double violation_current =
		EvaluateObstacleViolation(num_steps, num_agents, agent_ambient_offsets, workspace_dim, points,
					   per_agent_obstacles, models) +
		EvaluateAgentPairViolation(num_steps, agent_ambient_offsets, workspace_dim, points, agent_radii,
					    active_pairs, models) +
		EvaluateAgentSdfGridViolation(num_steps, num_agents, agent_ambient_offsets, workspace_dim, points,
					       active_grids, models);
	double phi_current = f_current + penalty_weight * violation_current;

	double trust_radius = initial_trust_radius;
	int iter = 0;

	for (; iter < max_iterations; ++iter) {
		if (iter > 0 && trust_radius <= min_trust_radius) break;

		// H_smooth_total/g_smooth: the smooth-cost Hessian/RHS evaluated at
		// the CURRENT (points, vels). Starts from `smooth_hessian_normal`
		// (a COPY -- R/Torus's iteration-CONSTANT contribution, see
		// AssembleSmoothHessian's own comment) and adds each SO3Quat
		// block's own coupled, iteration-DEPENDENT contribution on top
		// (AccumulateSO3QuatBlock) -- R/Torus-only agents pay only the cost
		// of that one copy (H_smooth_total == smooth_hessian_normal
		// unchanged), not a behavior change from before Stage 4.
		Eigen::MatrixXd H_smooth_total = smooth_hessian_normal;
		Eigen::VectorXd g_smooth = Eigen::VectorXd::Zero(n_smooth);
		for (int ag = 0; ag < num_agents; ++ag) {
			const int off = agent_axis_offsets[ag];
			const int dim = agent_shapes[ag].tangent_dim();
			const int ambient_dim = agent_shapes[ag].ambient_dim();
			const Eigen::VectorXd x0_agent = x0.segment(agent_ambient_offsets[ag], ambient_dim);
			const Eigen::VectorXd v0_agent = v0.segment(off, dim);
			const Eigen::MatrixXd points_agent = points.block(0, agent_ambient_offsets[ag], num_steps, ambient_dim);
			const Eigen::MatrixXd vels_agent = vels.block(0, off, num_steps, dim);
			const Eigen::MatrixXd ref_points_agent =
				ref_points.block(0, agent_ambient_offsets[ag], num_steps, ambient_dim);
			const Eigen::MatrixXd ref_velocities_agent =
				ref_velocities.block(0, off, num_steps, dim);
			// R/Torus tangent columns keep the existing per-axis-scalar
			// treatment (BuildAxisRhs, iterate-dependent RHS only -- H
			// already placed by AssembleSmoothHessian, unchanged);
			// SO3Quat blocks get the coupled treatment on top of the same
			// H_smooth_total/g_smooth (AccumulateSO3QuatBlock) -- see
			// sqp_short_path_layout.hpp's own doc comment for why these two
			// need to differ.
			for (const auto& boff : agent_shapes[ag].block_offsets_) {
				if (boff.type == CubicConfigurationSpline::Block::Type::SO3Quat) {
					AccumulateSO3QuatBlock(
						SO3QuatBlock{ag, boff, off + boff.tangent_offset},
						agent_shapes[ag], num_steps, tau, smooth_cost_weights,
						x0_agent, v0_agent, points_agent, vels_agent,
						ref_points_agent, ref_velocities_agent,
						&H_smooth_total, &g_smooth);
					continue;
				}
				for (int k = boff.tangent_offset; k < boff.tangent_offset + boff.tangent_size; ++k) {
					const Eigen::VectorXd g_axis = BuildAxisRhs(
						AxisLayout{ag, k}, agent_shapes[ag], num_steps, tau, smooth_cost_weights,
						x0_agent, v0_agent, points_agent, vels_agent,
						ref_points_agent, ref_velocities_agent);
					g_smooth.segment((off + k) * per_axis, per_axis) = g_axis;
				}
			}
		}
		const Eigen::VectorXd grad_smooth = -2.0 * g_smooth;

		std::vector<ConstraintRow> rows = LinearizeObstacleConstraints(
			agent_axis_offsets, num_steps, num_agents, agent_ambient_offsets, workspace_dim, points,
			per_agent_obstacles, models);
		std::vector<ConstraintRow> pair_rows = LinearizeAgentPairConstraints(
			agent_axis_offsets, num_steps, agent_ambient_offsets, workspace_dim, points, agent_radii,
			active_pairs, models);
		rows.insert(rows.end(), std::make_move_iterator(pair_rows.begin()),
			    std::make_move_iterator(pair_rows.end()));
		std::vector<ConstraintRow> grid_row_list = LinearizeAgentSdfGridConstraints(
			agent_axis_offsets, num_steps, num_agents, agent_ambient_offsets, workspace_dim, points,
			active_grids, models);
		rows.insert(rows.end(), std::make_move_iterator(grid_row_list.begin()),
			    std::make_move_iterator(grid_row_list.end()));

		if (m == 0 && grad_smooth.norm() < grad_tol) break;

		// H: smooth block is `2 * H_smooth_total` (the factor of 2 converts
		// BuildAxisRhs/AssembleSmoothHessian/AccumulateSO3QuatBlock's shared
		// normal-equations convention `H_n s = g_n` to proxqp's
		// `0.5 z'Hz + g'z`), block-diagonal per agent plus any SO3Quat
		// coupling; slack block is a pure Tikhonov floor (slack appears only
		// LINEARLY in the true cost, `penalty_weight * s`). proxqp reads the
		// upper triangle. The n_smooth^2 scan is cheap relative to the solve
		// (the smooth block is block-diagonal, so most entries are exact
		// zeros and dropped).
		std::vector<Trip> h_trips;
		h_trips.reserve(static_cast<size_t>(n_smooth) * 6 + n);
		for (int j = 0; j < n_smooth; ++j) {
			for (int i = 0; i <= j; ++i) {
				const double v = 2.0 * H_smooth_total(i, j);
				if (v != 0.0) h_trips.emplace_back(i, j, v);
			}
		}
		for (int i = 0; i < n; ++i) h_trips.emplace_back(i, i, 1.0e-10);  // summed with the block diagonal
		SpMat H_sp(n, n);
		H_sp.setFromTriplets(h_trips.begin(), h_trips.end());
		H_sp.makeCompressed();

		// g: smooth gradient on dx_smooth, `penalty_weight` on every slack.
		Eigen::VectorXd g_full(n);
		g_full.head(n_smooth) = grad_smooth;
		if (m > 0) g_full.tail(m).setConstant(penalty_weight);

		// C / l / u: the `m` slack-relaxed penalty rows `a^T dx + s >= -c(x)`
		// (one slack column each), then the trust-region box on dx_smooth,
		// then s >= 0. Every row here is structurally identical across outer
		// iterations -- only coefficients / bounds move.
		std::vector<Trip> c_trips;
		c_trips.reserve(rows.size() * (2 * static_cast<size_t>(workspace_dim) + 1) + n_smooth + m);
		Eigen::VectorXd cl(n_in), cu(n_in);
		for (int r = 0; r < m; ++r) {
			for (const auto& [idx, coeff] : rows[r].coeffs) c_trips.emplace_back(r, idx, coeff);
			c_trips.emplace_back(r, n_smooth + r, 1.0);
			cl(r) = -rows[r].value;
			cu(r) = kProxInf;
		}
		for (int i = 0; i < n_smooth; ++i) {
			const int row = m + i;
			c_trips.emplace_back(row, i, 1.0);
			cl(row) = -trust_radius;
			cu(row) = trust_radius;
		}
		for (int r = 0; r < m; ++r) {
			const int row = m + n_smooth + r;
			c_trips.emplace_back(row, n_smooth + r, 1.0);
			cl(row) = 0.0;
			cu(row) = kProxInf;
		}
		SpMat C_sp(n_in, n);
		C_sp.setFromTriplets(c_trips.begin(), c_trips.end());
		C_sp.makeCompressed();

		// Values-only update() (keeps the symbolic factorization + the
		// equilibration from init) while the H/C sparsity pattern is
		// unchanged, warm-started from the previous solve; full init()
		// otherwise (first solve, a rare SO3Quat Hessian entry crossing
		// zero, or a changed pruned-obstacle/pair shape on a new solve()
		// call). Every subproblem is feasible by construction (dz=0,
		// s=max(0,-c(x))), so a non-SOLVED status is a numerical hiccup, not
		// structural infeasibility -- shrink the trust region and retry
		// cold, same as the qpOASES path did.
		std::vector<long long> h_key = PatternKey(H_sp);
		std::vector<long long> c_key = PatternKey(C_sp);
		const bool reuse = st.initialized && h_key == st.h_pattern && c_key == st.c_pattern;
		if (reuse) {
			st.qp.settings.initial_guess =
				psp::InitialGuessStatus::WARM_START_WITH_PREVIOUS_RESULT;
			st.qp.update(H_sp, g_full, proxsuite::nullopt, proxsuite::nullopt, C_sp, cl, cu,
				     /*update_preconditioner=*/false);
		} else {
			st.qp.settings.initial_guess = psp::InitialGuessStatus::NO_INITIAL_GUESS;
			st.qp.init(H_sp, g_full, proxsuite::nullopt, proxsuite::nullopt, C_sp, cl, cu);
			st.h_pattern = std::move(h_key);
			st.c_pattern = std::move(c_key);
		}
		st.initialized = true;
		st.qp.solve();

		if (st.qp.results.info.status != psp::QPSolverOutput::PROXQP_SOLVED) {
			trust_radius *= 0.25;
			st.initialized = false;
			continue;
		}

		const Eigen::VectorXd z = st.qp.results.x;
		const Eigen::VectorXd dx_smooth = z.head(n_smooth);
		const double predicted_slack_sum = m > 0 ? z.tail(m).sum() : 0.0;

		if (dx_smooth.norm() < 1e-9) break;

		Eigen::MatrixXd candidate_points, candidate_vels;
		ApplyStep(agent_shapes, agent_axis_offsets, agent_ambient_offsets, num_steps, points, vels, dx_smooth,
			  &candidate_points, &candidate_vels);

		const double f_new = TotalSmoothCost(agent_shapes, agent_axis_offsets, agent_ambient_offsets, num_steps,
						      tau, smooth_cost_weights, x0, v0, candidate_points, candidate_vels,
						      ref_points, ref_velocities);
		const double violation_new =
			EvaluateObstacleViolation(num_steps, num_agents, agent_ambient_offsets, workspace_dim,
						   candidate_points, per_agent_obstacles, models) +
			EvaluateAgentPairViolation(num_steps, agent_ambient_offsets, workspace_dim, candidate_points,
						    agent_radii, active_pairs, models) +
			EvaluateAgentSdfGridViolation(num_steps, num_agents, agent_ambient_offsets, workspace_dim,
						       candidate_points, active_grids, models);
		const double phi_new = f_new + penalty_weight * violation_new;

		// Uses H_smooth_total (THIS iteration's smooth Hessian, including
		// any SO3Quat contribution), not the raw constant
		// smooth_hessian_normal -- the merit function's predicted-reduction
		// model must match what the QP was actually built from, or rho
		// becomes meaningless for exactly the SO3Quat rows/cols (matches
		// this file's own top comment's point 1 about the merit function
		// needing to track the true QP model).
		const Eigen::MatrixXd H_smooth_d = (2.0 * H_smooth_total);
		const double predicted_smooth_reduction =
			-(grad_smooth.dot(dx_smooth) + 0.5 * dx_smooth.dot(H_smooth_d * dx_smooth));
		// The QP's own slack values ARE exactly max(0, -(value + a.dx)) at
		// the QP optimum (penalty_weight > 0 drives every slack to its
		// tight lower bound given the row's RHS), so this is the model's
		// own prediction of the post-step violation -- an honest "predicted
		// reduction" in the SAME sense GraphTimingMPC's does for the
		// smooth-only case, just extended to the penalty term.
		const double predicted_violation_reduction = violation_current - predicted_slack_sum;
		const double predicted_reduction =
			predicted_smooth_reduction + penalty_weight * predicted_violation_reduction;
		const double actual_reduction = phi_current - phi_new;
		const double rho = (predicted_reduction > 1e-14) ? actual_reduction / predicted_reduction : 1.0;

		const double step_inf_norm = dx_smooth.lpNorm<Eigen::Infinity>();
		if (rho < 0.25) {
			trust_radius = std::max(0.25 * trust_radius, min_trust_radius);
		} else if (rho > 0.75 && step_inf_norm > 0.9 * trust_radius) {
			trust_radius = std::min(2.0 * trust_radius, max_trust_radius);
		}

		if (rho > 1e-8) {
			points = candidate_points;
			vels = candidate_vels;
			f_current = f_new;
			violation_current = violation_new;
			phi_current = phi_new;
		}
		// else: reject -- points/vels/f_current/violation_current/
		// phi_current stay at the previous (still fully valid) iterate;
		// only trust_radius moved.
	}

	ApplySafetyProjection(num_steps, num_agents, agent_ambient_offsets, workspace_dim, obstacles, models,
			       &points);
	ApplyAgentPairSafetyProjection(num_steps, num_agents, agent_ambient_offsets, workspace_dim, agent_radii,
					models, &points);

	return SqpResult{points, vels, iter, trust_radius};
}

}  // namespace

bool GraphShortPathMPC::solve(const Eigen::VectorXd& x0,
			     const Eigen::VectorXd& v0,
			     const Eigen::VectorXi& /*var_assignments*/,
			     const std::vector<int>& /*remaining_vertices*/,
			     const std::vector<CubicConfigurationSpline>& references) {
	_timer.Start();
	try {
		const int H = static_cast<int>(_num_steps);
		const int num_agents = static_cast<int>(_num_agents);
		const int workspace_dim = _graph->workspace_dim;

		// Total ambient/tangent width, summed across each agent's OWN width
		// -- see the constructor's own comment for why this isn't `num_agents
		// * ` a shared per-agent width any more.
		int total_ambient = 0;
		for (const auto& shape : _agent_shapes) total_ambient += shape.ambient_dim();
		Eigen::MatrixXd ref_points(H, total_ambient);
		Eigen::MatrixXd ref_velocities(H, static_cast<int>(_axes.size()));
		for (int ag = 0; ag < num_agents; ++ag) {
			const auto& [q_ag, qdot_ag] = references.at(ag).eval_multiple(_times);
			ref_points.block(0, _agent_ambient_offsets[ag], H, _agent_shapes[ag].ambient_dim()) = q_ag;
			ref_velocities.block(0, _agent_axis_offsets[ag], H, _agent_shapes[ag].tangent_dim()) = qdot_ag;
		}

		// Warm start: previous cycle's converged trajectory when its shape
		// matches, falling back to the reference itself on the first cycle
		// or after a shape change.
		Eigen::MatrixXd points = ref_points;
		Eigen::MatrixXd vels = ref_velocities;
		if (_has_solved && _points.rows() == H && _points.cols() == ref_points.cols()) {
			points = _points;
		}
		if (_has_solved && _vels.rows() == H && _vels.cols() == ref_velocities.cols()) {
			vels = _vels;
		}

		// Distance-based row pruning (design decision 6: computed ONCE per
		// solve() call, from the REFERENCE trajectory, held fixed for the
		// whole call -- see PruneObstaclesByDistance/PruneAgentPairsByDistance's
		// own comments). Purely a speed lever, not a feasibility one for a
		// TRIVIAL agent (ApplySafetyProjection/ApplyAgentPairSafetyProjection
		// still check every registered obstacle/pair for those). Each agent's
		// swept collision-sphere geometry along the reference is evaluated
		// once here and drives all three prunings.
		const std::vector<AgentReferenceSpheres> ref_spheres = BuildAgentReferenceSpheres(
			ref_points, num_agents, _agent_ambient_offsets, workspace_dim, _agent_collision_models);
		const std::vector<std::vector<ActiveObstacle>> per_agent_obstacles =
			PruneObstaclesByDistance(H, num_agents, workspace_dim, ref_spheres, *_obstacles,
						  _constraint_prune_margin);
		const std::vector<ActivePair> active_pairs =
			PruneAgentPairsByDistance(H, num_agents, ref_spheres, _agent_collision_models,
						   _agent_radii, _constraint_prune_margin);
		const std::vector<ActiveGrid> active_grids =
			PruneAgentSdfGridsByDistance(H, num_agents, ref_spheres, *_obstacles,
						      _constraint_prune_margin);

		SqpResult result = RunTrustRegionSqp(
			_agent_shapes, _axes, _agent_axis_offsets, _agent_ambient_offsets, _agent_collision_models,
			_smooth_hessian_normal,
			_smooth_cost_weights, H, num_agents, workspace_dim, _time_per_step,
			x0, v0, ref_points, ref_velocities, *_obstacles, per_agent_obstacles, active_pairs,
			active_grids, _agent_radii, _penalty_weight, _max_iterations,
			_initial_trust_radius, _max_trust_radius, _min_trust_radius, _grad_tol,
			points, vels, &_qp_state);

		_points = result.points;
		_vels = result.vels;
		_last_iterations = result.iterations;
		_last_trust_radius = result.trust_radius;
		_has_solved = true;
		_last_solve_time = _timer.Tick();
		return true;
	} catch (const std::exception& e) {
		std::cout << "Caught exception in SQP short path solver: " << e.what() << std::endl;
		return false;
	}
}
