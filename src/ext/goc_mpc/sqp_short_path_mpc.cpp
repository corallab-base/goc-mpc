#include "sqp_short_path_mpc.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

#include <qpOASES.hpp>

#include "obstacle_projection.hpp"

using namespace sqp_short_path;

namespace {
using RealMat = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RealVec = Eigen::Matrix<qpOASES::real_t, Eigen::Dynamic, 1>;
constexpr double kInf = 1.0e19;
}  // namespace

struct SqpShortPathMPC::QpState {
	qpOASES::SQProblem qp;
	int n = 0, m_eff = 0;
	bool initialized = false;

	QpState(int n_, int m_eff_) : qp(n_, m_eff_), n(n_), m_eff(m_eff_) {
		qp.setPrintLevel(qpOASES::PL_NONE);
	}
};

SqpShortPathMPC::SqpShortPathMPC(const GraphOfConstraints& graph,
				  unsigned int num_steps,
				  unsigned int num_agents,
				  unsigned int dim,
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
				  double grad_tol)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step),
	  _obstacles(&obstacles),
	  _agent_radii(agent_radii.size() == 0 ? Eigen::VectorXd::Zero(num_agents) : std::move(agent_radii)),
	  _smooth_cost_weights{tracking_weight, velocity_tracking_weight, acceleration_weight},
	  _penalty_weight(penalty_weight),
	  _max_iterations(max_iterations),
	  _initial_trust_radius(initial_trust_radius),
	  _max_trust_radius(max_trust_radius),
	  _min_trust_radius(min_trust_radius),
	  _grad_tol(grad_tol) {

	if (_agent_radii.size() != static_cast<int>(num_agents)) {
		throw std::runtime_error(
			"SqpShortPathMPC: agent_radii must be empty (defaults every agent to "
			"radius 0) or have exactly num_agents entries.");
	}

	_agent_shapes = BuildAgentShapes(graph, static_cast<int>(num_agents));
	for (unsigned int ag = 0; ag < num_agents; ++ag) {
		// `dim` is what every caller (x0/v0/points/vels) slices agent
		// columns with -- if it doesn't match this agent's own ambient
		// width, every later `.segment(ag*dim, dim)` in this file reads
		// the wrong columns (a silent corruption, not a crash: this
		// project's RelWithDebInfo build compiles out eigen_assert, see
		// feedback_eigen_row_col_no_assert in project memory), so this is
		// checked loudly here instead.
		if (_agent_shapes[ag].ambient_dim() != static_cast<int>(dim) ||
		    _agent_shapes[ag].tangent_dim() != static_cast<int>(dim)) {
			throw std::runtime_error(
				"SqpShortPathMPC: agent " + std::to_string(ag) + "'s ambient/"
				"tangent width doesn't match `dim` -- R/Torus blocks only "
				"(ambient_dim==tangent_dim always holds for those), so this "
				"indicates a real spec/dim mismatch, not just a v1 scope gap.");
		}
	}
	_axes = BuildAxisList(_agent_shapes);
	_agent_axis_offsets = BuildAgentAxisOffsets(_agent_shapes);
	_smooth_hessian_normal =
		AssembleSmoothHessian(_axes, static_cast<int>(num_steps), time_per_step, _smooth_cost_weights);

	// Xi.row(i) is offset by one tau from x0/v0's own time -- see
	// GraphShortPathMPC's constructor for the full rationale (Xi.row(0) is
	// treated as the state ONE tau after x0/v0 by the acceleration term
	// throughout this file, so the tracking reference must be sampled at
	// the same offset).
	_times = Eigen::VectorXd(_num_steps);
	for (unsigned int i = 0; i < _num_steps; ++i) {
		_times(i) = (i + 1) * _time_per_step;
	}
	_points = Eigen::MatrixXd::Zero(_num_steps, _num_agents * _dim);
	_vels = Eigen::MatrixXd::Zero(_num_steps, _num_agents * _dim);
}

SqpShortPathMPC::~SqpShortPathMPC() = default;
SqpShortPathMPC::SqpShortPathMPC(SqpShortPathMPC&&) noexcept = default;
SqpShortPathMPC& SqpShortPathMPC::operator=(SqpShortPathMPC&&) noexcept = default;

namespace {

// Retracts every agent's every step by the smooth-cost step vector
// `dx_smooth` (length axes.size()*2*num_steps, same convention as
// sqp_short_path::IdxP/IdxV) -- positions via each agent's own
// CubicConfigurationSpline::Retract (R: plain add, Torus: wrap_pi(x+d)),
// velocities via plain addition (they live in a flat tangent space
// already, no manifold to respect).
void ApplyStep(const std::vector<CubicConfigurationSpline>& agent_shapes,
		const std::vector<int>& agent_axis_offsets, int num_steps, int dim,
		const Eigen::MatrixXd& points, const Eigen::MatrixXd& vels,
		const Eigen::VectorXd& dx_smooth,
		Eigen::MatrixXd* new_points, Eigen::MatrixXd* new_vels) {
	*new_points = points;
	*new_vels = vels;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		const int off = agent_axis_offsets[ag];
		for (int i = 0; i < num_steps; ++i) {
			Eigen::VectorXd dp(dim), dv(dim);
			for (int k = 0; k < dim; ++k) {
				dp(k) = dx_smooth(IdxP(off + k, i, num_steps));
				dv(k) = dx_smooth(IdxV(off + k, i, num_steps));
			}
			const Eigen::VectorXd p_old = points.row(i).segment(ag * dim, dim).transpose();
			const Eigen::VectorXd p_new = agent_shapes[ag].Retract<double>(p_old, dp);
			new_points->row(i).segment(ag * dim, dim) = p_new.transpose();
			new_vels->row(i).segment(ag * dim, dim) += dv.transpose();
		}
	}
}

double TotalSmoothCost(const std::vector<CubicConfigurationSpline>& agent_shapes, int num_steps, double tau,
			const SmoothCostWeights& weights,
			int dim, const Eigen::VectorXd& x0, const Eigen::VectorXd& v0,
			const Eigen::MatrixXd& points, const Eigen::MatrixXd& vels,
			const Eigen::MatrixXd& ref_points, const Eigen::MatrixXd& ref_velocities) {
	double f = 0.0;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		f += EvaluateSmoothCost(
			agent_shapes[ag], num_steps, tau, weights,
			x0.segment(ag * dim, dim), v0.segment(ag * dim, dim),
			points.block(0, ag * dim, num_steps, dim), vels.block(0, ag * dim, num_steps, dim),
			ref_points.block(0, ag * dim, num_steps, dim),
			ref_velocities.block(0, ag * dim, num_steps, dim));
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

// Final hard-feasibility safety net, reusing AdmmShortPathMPC's own
// closed-form projection (obstacle_projection.hpp) -- guarantees the
// RETURNED trajectory clears every registered obstacle regardless of
// whether the SQP loop above fully converged within `max_iterations`
// (design decision 7 in the project plan). This is the actual feasibility
// guarantee; the SQP loop's job is only to make this pass's corrections
// small, exactly mirroring AdmmShortPathMPC's own division of labor
// between its ADMM loop and this same final pass. A single sweep isn't
// enough once obstacles are close enough to interact (confirmed
// empirically by AdmmShortPathMPC's own stress testing -- see that file's
// comment), hence several repeated sweeps (Dykstra-style alternating
// projection) followed by EscapeAlongAxis for anything that's STILL
// violated (the two-obstacle oscillation case above -- confirmed to
// actually occur via a real stress-test trial, not just a theoretical
// worry). Fast-path-only, like everything below -- see
// ApplyAgentPairSafetyProjection's own comment for why, and for the
// intended future toggle once that stops being universally true.
void ApplySafetyProjection(int num_steps, int num_agents, int dim, int workspace_dim,
			    const ObstacleSet& obstacles, Eigen::MatrixXd* points) {
	if (obstacles.obstacles().empty() && !obstacles.has_point_cloud()) {
		return;
	}
	constexpr int kSafetyPassRounds = 10;
	for (int ag = 0; ag < num_agents; ++ag) {
		const Eigen::MatrixXd agent_workspace_traj =
			points->block(0, ag * dim, num_steps, workspace_dim);
		const std::vector<Candidate> candidates =
			gather_candidates(obstacles, agent_workspace_traj, workspace_dim, /*query_margin=*/1.0);
		if (candidates.empty()) continue;
		for (int round = 0; round < kSafetyPassRounds; ++round) {
			for (int i = 0; i < num_steps; ++i) {
				Eigen::VectorXd p = points->row(i).segment(ag * dim, workspace_dim).transpose();
				for (const Candidate& c : candidates) {
					p = project_out(p, c);
				}
				points->row(i).segment(ag * dim, workspace_dim) = p.transpose();
			}
		}
		for (int i = 0; i < num_steps; ++i) {
			Eigen::VectorXd p = points->row(i).segment(ag * dim, workspace_dim).transpose();
			bool any_violated = false;
			for (const Candidate& c : candidates) any_violated |= IsInsideCandidate(p, c);
			if (any_violated) {
				points->row(i).segment(ag * dim, workspace_dim) =
					EscapeAlongAxis(p, candidates, workspace_dim).transpose();
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
void ApplyAgentPairSafetyProjection(int num_steps, int num_agents, int dim, int workspace_dim,
				     const Eigen::VectorXd& agent_radii, Eigen::MatrixXd* points) {
	if (num_agents < 2) {
		return;
	}
	constexpr int kSafetyPassRounds = 10;
	for (int round = 0; round < kSafetyPassRounds; ++round) {
		for (int i = 0; i < num_steps; ++i) {
			for (int ag_a = 0; ag_a < num_agents; ++ag_a) {
				for (int ag_b = ag_a + 1; ag_b < num_agents; ++ag_b) {
					Eigen::VectorXd p_a = points->row(i).segment(ag_a * dim, workspace_dim).transpose();
					Eigen::VectorXd p_b = points->row(i).segment(ag_b * dim, workspace_dim).transpose();
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
					points->row(i).segment(ag_a * dim, workspace_dim) = p_a.transpose();
					points->row(i).segment(ag_b * dim, workspace_dim) = p_b.transpose();
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
		const Eigen::MatrixXd& smooth_hessian_normal,
		const SmoothCostWeights& smooth_cost_weights,
		int num_steps, int num_agents, int dim, int workspace_dim, double tau,
		const Eigen::VectorXd& x0, const Eigen::VectorXd& v0,
		const Eigen::MatrixXd& ref_points, const Eigen::MatrixXd& ref_velocities,
		const ObstacleSet& obstacles, const Eigen::VectorXd& agent_radii,
		double penalty_weight, int max_iterations,
		double initial_trust_radius, double max_trust_radius, double min_trust_radius, double grad_tol,
		Eigen::MatrixXd points, Eigen::MatrixXd vels,
		std::unique_ptr<SqpShortPathMPC::QpState>* qp_state) {

	const int per_axis = 2 * num_steps;
	const int n_smooth = static_cast<int>(axes.size()) * per_axis;
	// Fixed for the whole solve() call (design decision 6): obstacle count
	// (and now agent-pair count) times step count never changes mid-call,
	// only each row's coefficients/value do (re-linearized every outer
	// iteration) -- so qpOASES::SQProblem can be reused (init once,
	// hotstart thereafter) across every outer iteration, and across MPC
	// cycles at a stable size.
	const int num_pairs = num_agents * (num_agents - 1) / 2;
	const int m = num_steps * (num_agents * static_cast<int>(obstacles.obstacles().size()) + num_pairs);
	const int m_eff = std::max(m, 1);
	const int n = n_smooth + m_eff;

	if (!*qp_state || (*qp_state)->n != n || (*qp_state)->m_eff != m_eff) {
		*qp_state = std::make_unique<SqpShortPathMPC::QpState>(n, m_eff);
	}
	qpOASES::SQProblem& qp = (*qp_state)->qp;

	// H_qp is CONSTANT across every outer iteration (design decision 3):
	// the smooth block is `2 * smooth_hessian_normal` (the factor of 2
	// converts BuildAxisRhs/AssembleSmoothHessian's normal-equations
	// convention, `H_n s = g_n`, to qpOASES' `0.5 z'Hz + g'z` convention --
	// see this file's own top comment), the slack block is a pure Tikhonov
	// floor (slack only ever appears LINEARLY in the true cost,
	// `penalty_weight * s`, so it has no real quadratic term of its own).
	RealMat H_qp = RealMat::Zero(n, n);
	H_qp.topLeftCorner(n_smooth, n_smooth) = (2.0 * smooth_hessian_normal).cast<qpOASES::real_t>();
	for (int i = 0; i < n; ++i) H_qp(i, i) += qpOASES::real_t(1e-10);

	double f_current = TotalSmoothCost(agent_shapes, num_steps, tau, smooth_cost_weights, dim, x0, v0,
					    points, vels, ref_points, ref_velocities);
	double violation_current =
		EvaluateObstacleViolation(num_steps, num_agents, dim, workspace_dim, points, obstacles) +
		EvaluateAgentPairViolation(num_steps, num_agents, dim, workspace_dim, points, agent_radii);
	double phi_current = f_current + penalty_weight * violation_current;

	double trust_radius = initial_trust_radius;
	int iter = 0;

	for (; iter < max_iterations; ++iter) {
		if (iter > 0 && trust_radius <= min_trust_radius) break;

		// g_smooth: concatenation of every axis's BuildAxisRhs, evaluated
		// at the CURRENT (points, vels) -- rebuilt every outer iteration
		// (unlike H_qp, this genuinely depends on the current iterate).
		Eigen::VectorXd g_smooth = Eigen::VectorXd::Zero(n_smooth);
		for (int ag = 0; ag < num_agents; ++ag) {
			const int off = agent_axis_offsets[ag];
			const Eigen::VectorXd x0_agent = x0.segment(ag * dim, dim);
			const Eigen::VectorXd v0_agent = v0.segment(ag * dim, dim);
			const Eigen::MatrixXd points_agent = points.block(0, ag * dim, num_steps, dim);
			const Eigen::MatrixXd vels_agent = vels.block(0, ag * dim, num_steps, dim);
			const Eigen::MatrixXd ref_points_agent = ref_points.block(0, ag * dim, num_steps, dim);
			const Eigen::MatrixXd ref_velocities_agent =
				ref_velocities.block(0, ag * dim, num_steps, dim);
			for (int k = 0; k < dim; ++k) {
				const Eigen::VectorXd g_axis = BuildAxisRhs(
					AxisLayout{ag, k}, agent_shapes[ag], num_steps, tau, smooth_cost_weights,
					x0_agent, v0_agent, points_agent, vels_agent,
					ref_points_agent, ref_velocities_agent);
				g_smooth.segment((off + k) * per_axis, per_axis) = g_axis;
			}
		}
		const Eigen::VectorXd grad_smooth = -2.0 * g_smooth;

		std::vector<ConstraintRow> rows = LinearizeObstacleConstraints(
			agent_axis_offsets, num_steps, num_agents, dim, workspace_dim, points, obstacles);
		std::vector<ConstraintRow> pair_rows = LinearizeAgentPairConstraints(
			agent_axis_offsets, num_steps, num_agents, dim, workspace_dim, points, agent_radii);
		rows.insert(rows.end(), std::make_move_iterator(pair_rows.begin()),
			    std::make_move_iterator(pair_rows.end()));

		if (m == 0 && grad_smooth.norm() < grad_tol) break;

		RealVec g_qp = RealVec::Zero(n);
		g_qp.head(n_smooth) = grad_smooth.cast<qpOASES::real_t>();
		for (int r = 0; r < m; ++r) g_qp(n_smooth + r) = qpOASES::real_t(penalty_weight);

		RealMat A_qp = RealMat::Zero(m_eff, n);
		RealVec lbA = RealVec::Constant(m_eff, qpOASES::real_t(-kInf));
		RealVec ubA = RealVec::Constant(m_eff, qpOASES::real_t(kInf));
		for (int r = 0; r < m; ++r) {
			for (const auto& [idx, coeff] : rows[r].coeffs) {
				A_qp(r, idx) += static_cast<qpOASES::real_t>(coeff);
			}
			A_qp(r, n_smooth + r) = qpOASES::real_t(1.0);
			lbA(r) = static_cast<qpOASES::real_t>(-rows[r].value);
		}

		RealVec dx_lb = RealVec::Constant(n, qpOASES::real_t(-kInf));
		RealVec dx_ub = RealVec::Constant(n, qpOASES::real_t(kInf));
		for (int i = 0; i < n_smooth; ++i) {
			dx_lb(i) = qpOASES::real_t(-trust_radius);
			dx_ub(i) = qpOASES::real_t(trust_radius);
		}
		for (int r = 0; r < m; ++r) dx_lb(n_smooth + r) = qpOASES::real_t(0.0);

		qpOASES::int_t nWSR = 200;
		qpOASES::returnValue status;
		if (!(*qp_state)->initialized) {
			status = qp.init(H_qp.data(), g_qp.data(), A_qp.data(),
					  dx_lb.data(), dx_ub.data(), lbA.data(), ubA.data(), nWSR);
			(*qp_state)->initialized = (status == qpOASES::SUCCESSFUL_RETURN);
		} else {
			status = qp.hotstart(H_qp.data(), g_qp.data(), A_qp.data(),
					      dx_lb.data(), dx_ub.data(), lbA.data(), ubA.data(), nWSR);
		}

		if (status != qpOASES::SUCCESSFUL_RETURN) {
			// Unlike GraphTimingMPC's comment for this same branch: here
			// the QP subproblem being unsolvable really can only be a
			// numerical/active-set-homotopy hiccup, not a structural
			// impossibility -- it's LITERALLY always feasible by
			// construction (dz=0, s=max(0,-value) satisfies every row --
			// see this class's own header comment) regardless of how
			// pathological the obstacle geometry is.
			trust_radius *= 0.25;
			(*qp_state)->initialized = false;
			continue;
		}

		RealVec z_q(n);
		qp.getPrimalSolution(z_q.data());
		const Eigen::VectorXd z = z_q.cast<double>();
		const Eigen::VectorXd dx_smooth = z.head(n_smooth);
		const double predicted_slack_sum = m > 0 ? z.tail(m).sum() : 0.0;

		if (dx_smooth.norm() < 1e-9) break;

		Eigen::MatrixXd candidate_points, candidate_vels;
		ApplyStep(agent_shapes, agent_axis_offsets, num_steps, dim, points, vels, dx_smooth,
			  &candidate_points, &candidate_vels);

		const double f_new = TotalSmoothCost(agent_shapes, num_steps, tau, smooth_cost_weights, dim, x0, v0,
						      candidate_points, candidate_vels, ref_points, ref_velocities);
		const double violation_new =
			EvaluateObstacleViolation(num_steps, num_agents, dim, workspace_dim, candidate_points, obstacles) +
			EvaluateAgentPairViolation(num_steps, num_agents, dim, workspace_dim, candidate_points, agent_radii);
		const double phi_new = f_new + penalty_weight * violation_new;

		const Eigen::MatrixXd H_smooth_d = (2.0 * smooth_hessian_normal);
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

	ApplySafetyProjection(num_steps, num_agents, dim, workspace_dim, obstacles, &points);
	ApplyAgentPairSafetyProjection(num_steps, num_agents, dim, workspace_dim, agent_radii, &points);

	return SqpResult{points, vels, iter, trust_radius};
}

}  // namespace

bool SqpShortPathMPC::solve(const Eigen::VectorXd& x0,
			     const Eigen::VectorXd& v0,
			     const Eigen::VectorXi& /*var_assignments*/,
			     const std::vector<int>& /*remaining_vertices*/,
			     const std::vector<CubicConfigurationSpline>& references) {
	_timer.Start();
	try {
		const int H = static_cast<int>(_num_steps);
		const int num_agents = static_cast<int>(_num_agents);
		const int dim = static_cast<int>(_dim);
		const int workspace_dim = _graph->workspace_dim;

		Eigen::MatrixXd ref_points(H, num_agents * dim);
		Eigen::MatrixXd ref_velocities(H, num_agents * dim);
		for (int ag = 0; ag < num_agents; ++ag) {
			const auto& [q_ag, qdot_ag] = references.at(ag).eval_multiple(_times);
			ref_points.block(0, ag * dim, H, dim) = q_ag;
			ref_velocities.block(0, ag * dim, H, dim) = qdot_ag;
		}

		// Warm start: previous cycle's converged trajectory when its shape
		// matches (same policy as GraphShortPathMPC/AdmmShortPathMPC),
		// falling back to the reference itself on the first cycle or after
		// a shape change.
		Eigen::MatrixXd points = ref_points;
		Eigen::MatrixXd vels = ref_velocities;
		if (_has_solved && _points.rows() == H && _points.cols() == ref_points.cols()) {
			points = _points;
		}
		if (_has_solved && _vels.rows() == H && _vels.cols() == ref_velocities.cols()) {
			vels = _vels;
		}

		SqpResult result = RunTrustRegionSqp(
			_agent_shapes, _axes, _agent_axis_offsets, _smooth_hessian_normal, _smooth_cost_weights,
			H, num_agents, dim, workspace_dim, _time_per_step,
			x0, v0, ref_points, ref_velocities, *_obstacles, _agent_radii,
			_penalty_weight, _max_iterations,
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
