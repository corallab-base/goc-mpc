#include "graph_short_path_mpc.hpp"

#include <limits>

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using drake::solvers::MathematicalProgramResult;

using namespace pybind11::literals;
namespace py = pybind11;

ShortPathProblem build_short_path_problem(
	const GraphOfConstraints* graph,
	const ObstacleSet* obstacles,
	double obstacle_repulsion_weight,
	bool use_hard_constraints,
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
	const Eigen::MatrixXd& initial_guess_points,
	const Eigen::MatrixXd& initial_guess_velocities,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const Eigen::VectorXi& var_assignments,
	const std::vector<int> remaining_vertices,
	double tau) {

	using namespace drake::solvers;

	const int num_steps = ref_points.rows();
	const int ambient_dim = ref_points.cols();
	const int tangent_dim = ref_velocities.cols();

	// Create program
	ShortPathProblem problem;

	MatrixXDecisionVariable Xi = problem.prog->NewContinuousVariables(num_steps, ambient_dim, "xi");
	problem.Xi = Xi;

	MatrixXDecisionVariable V = problem.prog->NewContinuousVariables(num_steps, tangent_dim, "v");
	problem.V = V;

	// Set initial guess. `initial_guess_points`/`initial_guess_velocities`
	// are DELIBERATELY separate arguments from `ref_points`/`ref_velocities`
	// (used below for the tracking costs) -- see GraphShortPathMPC::solve's
	// own comment: they're the previous cycle's converged solution when one
	// exists (warm start) and/or, in the obstacle-avoidance path, carry a
	// small symmetry-breaking nudge away from any obstacle center a
	// reference point happens to coincide with (a zero-gradient degenerate
	// point for the constraint below) -- while the tracking costs
	// themselves must still pull toward the real, un-nudged reference.
	problem.prog->SetInitialGuess(Xi, initial_guess_points);
	problem.prog->SetInitialGuess(V, initial_guess_velocities);

	/*
	 * OBJECTIVE FUNCTION
	 */

	// 1. Tracking error objective
	for (int i = 0; i < num_steps; ++i) {
		VectorX<Expression> diff = Xi.row(i) - ref_points.row(i);
		Expression dist = diff.squaredNorm();
		problem.prog->AddQuadraticCost(dist);
	}

	// 1b. Velocity tracking error objective -- `ref_velocities` was
	// previously only used for `SetInitialGuess(V, ...)` above, which seeds
	// the solver's starting point but applies no cost, so nothing in the
	// objective anchored V to it: the QP was free to drift V arbitrarily
	// far from what `references` (the timing MPC's own spline) actually
	// prescribed, constrained only indirectly via the acceleration
	// objective's coupling to Xi. Diagnosed by comparing `spline.eval(0.1)`
	// against the state PyRoboGym actually executed under `position_velocity`
	// mode each cycle (po_goc_mpc.experiments.basic_fmm_experiment): position
	// matched almost exactly (it has this same tracking cost below), but
	// velocity was found to inflate ~1.7-1.8x per cycle and compound, since
	// nothing here penalized it drifting from ref_velocities.
	for (int i = 0; i < num_steps; ++i) {
		VectorX<Expression> vdiff = V.row(i) - ref_velocities.row(i);
		Expression vdist = vdiff.squaredNorm();
		problem.prog->AddQuadraticCost(vdist);
	}

	double tau2 = tau * tau;
	double tau3 = tau * tau2;

	// 2. Scaled acceleration objective
	for (int i = 0; i < num_steps; ++i) {
		if (i == 0) {
			// only take elements for agent positions
			const Eigen::VectorXd xKm1 = x0.segment(0, ambient_dim);
			const Eigen::VectorX<Variable> xK = Xi.row(i);
			const Eigen::VectorXd vKm1 = v0.segment(0, tangent_dim);
			const Eigen::VectorX<Variable> vK = V.row(i);

			const Eigen::VectorX<Expression> a6_tau = 6.0 / tau2 * (-2.0 * (xK - xKm1) + tau * (vK + vKm1));
			const Eigen::VectorX<Expression> b2 = 2.0 / tau2 * (3.0 * (xK - xKm1) - tau * (vK + 2.0 * vKm1));
			const Expression acc_norm = (a6_tau + b2).squaredNorm();
			problem.prog->AddQuadraticCost(acc_norm);
		} else {
			const Eigen::VectorX<Variable> xKm1 = Xi.row(i-1);
			const Eigen::VectorX<Variable> xK = Xi.row(i);
			const Eigen::VectorX<Variable> vKm1 = V.row(i-1);
			const Eigen::VectorX<Variable> vK = V.row(i);

			const Eigen::VectorX<Expression> a6_tau = 6.0 / tau2 * (-2.0 * (xK - xKm1) + tau * (vK + vKm1));
			const Eigen::VectorX<Expression> b2 = 2.0 / tau2 * (3.0 * (xK - xKm1) - tau * (vK + 2.0 * vKm1));
			const Expression acc_norm = (a6_tau + b2).squaredNorm();
			problem.prog->AddQuadraticCost(acc_norm);
		}
	}

	// 3. Inter-agent collision cost -- still disabled (non-convex squared-
	// distance lower bound, can't live in the plain QP path this class
	// used exclusively before stage 2). Stage 3 (ORCA-style reciprocal
	// avoidance) will replace this, following the same pattern stage 2 uses
	// below for environment obstacles: a real inequality constraint, routed
	// through NloptSolver (NOT IpoptSolver -- see item 4's comment below for
	// why) instead of the QP-only dispatcher. Left disabled rather than
	// activated here since it's out of this stage's scope.
	const int num_agents = graph->num_agents;
	const int dim = graph->dim;
	const int workspace_dim_scaffold = graph->workspace_dim;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag_i = 0; ag_i < num_agents; ++ag_i) {
			const Eigen::VectorX<Variable> p_WE_i = Xi.row(i).segment(ag_i * dim, workspace_dim_scaffold);
			for (int ag_j = ag_i + 1; ag_j < num_agents; ++ag_j) {
				const Eigen::VectorX<Variable> p_WE_j = Xi.row(i).segment(ag_j * dim, workspace_dim_scaffold);

				const Expression d_ij = (p_WE_j - p_WE_i).squaredNorm();

				// problem.prog->AddQuadraticConstraint(d_ij,
				// 				     0.0144,
				// 				     10.0);
			}
		}
	}

	// 4. Environment-obstacle avoidance (stage 2) -- spheres and axis-aligned
	// boxes, see ObstacleSet's doc comment (obstacle_set.hpp). For every
	// (step, agent, obstacle): optionally (use_hard_constraints) a hard
	// inequality constraint keeps the agent outside the shape + margin,
	// applied at EVERY step in the horizon, not just step 0 -- otherwise the
	// solver could route straight through the obstacle at an interior step
	// while leaving the endpoints clear -- plus, always, a soft Lorentzian
	// repulsion cost (graceful, visible deflection well before the hard
	// boundary -- reuses the Lorentzian SHAPE validated in an earlier,
	// since-superseded hand-rolled-gradient-descent prototype).
	//
	// use_hard_constraints selects what gets added (both cases route through
	// NloptSolver/LD_SLSQP -- see GraphShortPathMPC::solve):
	//   true (default): the hard constraint is non-convex (same shape of
	//     problem as the disabled inter-agent scaffold above) -- a real
	//     safety guarantee when it converges, but SLSQP is a local SQP
	//     method and its QP subproblem can report the whole NLP infeasible
	//     once several such constraints are simultaneously active
	//     (confirmed empirically: 35-53% failure rate with 2 agents x 2 box
	//     obstacles, in both isolated tests and a real multi-robot
	//     pick-and-place scenario -- not fixable by tuning).
	//   false: no hard constraint at all, only the smooth repulsion cost --
	//     SLSQP's QP subproblem then has nothing to report infeasible about
	//     (there's no explicit inequality to violate), so it reduces to
	//     ordinary local quasi-Newton minimization of a smooth function --
	//     structurally can't fail the same way, at the cost of no hard
	//     safety guarantee (the agent CAN end up inside an obstacle if the
	//     tracking/acceleration cost pulls hard enough against a bounded
	//     penalty -- see the penalty's own doc comment below for why it's
	//     built to grow unboundedly rather than saturate).
	// NOT IpoptSolver for the true case: a first attempt routed the hard
	// constraint through IpoptSolver and was EMPIRICALLY ABANDONED -- even
	// this class's own pre-existing cross-step acceleration-smoothing cost
	// above (item 2), with NO obstacle constraint involved at all, failed to
	// converge under Drake's IpoptSolver in this environment (confirmed via
	// a from-scratch test harness: IPOPT reported zero nonzeros in the
	// Lagrangian Hessian throughout and stalled regardless of max_iter,
	// while the identical cost solves instantly through the normal QP
	// solver).
	//
	// Uses workspace_dim (not a hardcoded 3, unlike the disabled
	// inter-agent block above) to slice each agent's position out of Xi --
	// see stage 1's plan notes on why hardcoding 3 silently breaks a
	// workspace_dim=2 planar agent.
	const int workspace_dim = graph->workspace_dim;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			const auto p = Xi.row(i).segment(ag * dim, workspace_dim);
			for (const Obstacle& obstacle : obstacles->obstacles()) {
				const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
				// p is a ROW (Xi.row(i).segment(...)); center is a COLUMN
				// (Eigen::VectorXd) -- .transpose() p first. Getting this
				// wrong doesn't fail loudly: in a RelWithDebInfo build with
				// Eigen assertions compiled out, a row-minus-column shape
				// mismatch silently produces a garbage/truncated
				// expression instead of throwing (confirmed: without this
				// transpose, the resulting constraint printed as just
				// `pow(xi(0,0), 2)` -- the y/z terms had vanished).
				const Eigen::VectorX<Expression> diff = p.transpose() - center;

				if (obstacle.kind == ObstacleKind::kSphere) {
					const double R = obstacle.params(workspace_dim) + obstacle.margin;
					const double R2 = R * R;
					const Expression d2 = diff.squaredNorm();

					if (use_hard_constraints) {
						problem.prog->AddConstraint(d2, R2, std::numeric_limits<double>::infinity());
					}
					problem.prog->AddCost(obstacle_repulsion_weight * R2 / (d2 + R2));
				} else if (obstacle.kind == ObstacleKind::kBox) {
					// Signed distance to an axis-aligned box (negative
					// inside, per-axis excess distance outside), the
					// standard box-SDF construction: q = |diff| - half_extents;
					// sdf = ||max(q, 0)|| + min(max(q), 0). AddConstraint on
					// the exact sdf (not a squared surrogate, unlike the
					// sphere case above) since the squared-distance-to-surface
					// formula alone is 0 both ON the surface AND anywhere
					// INSIDE the box, and can't tell those apart -- sdf can.
					const Eigen::VectorXd half_extents = obstacle.params.segment(workspace_dim, workspace_dim);
					const double margin = obstacle.margin;

					Expression outside_sq = 0.0;
					Expression max_q;
					for (int k = 0; k < workspace_dim; ++k) {
						const Expression q_k = drake::symbolic::abs(diff(k)) - half_extents(k);
						const Expression clamped = drake::symbolic::max(q_k, 0.0);
						outside_sq += clamped * clamped;
						max_q = (k == 0) ? q_k : drake::symbolic::max(max_q, q_k);
					}
					// sqrt(outside_sq) alone has an INFINITE gradient at
					// outside_sq == 0 -- which is the box's entire interior,
					// not just its boundary (d/dx[sqrt(u)] = 1/(2*sqrt(u)) ->
					// infinity as u -> 0). Any point inside the box during
					// optimization hits that singularity, corrupting autodiff
					// and throwing (confirmed empirically: NLopt/LD_MMA raised
					// SolverSpecificError as soon as the soft cost below was
					// changed to depend on `sdf` instead of `outside_sq`
					// alone). A tiny smoothing epsilon inside the sqrt fixes
					// the gradient everywhere at a negligible (~1e-4) bias to
					// sdf's value.
					const double kSqrtEps2 = 1.0e-8;
					const Expression sdf =
						drake::symbolic::sqrt(outside_sq + kSqrtEps2) + drake::symbolic::min(max_q, 0.0);

					if (use_hard_constraints) {
						problem.prog->AddConstraint(sdf, margin, std::numeric_limits<double>::infinity());
					}
					// Soft repulsion cost: a squared-hinge PENALTY on
					// `margin - sdf` (same shape already used for this
					// project's other soft inequality constraints), not a
					// barrier and not the bounded Lorentzian/sigmoid bump
					// tried earlier. Two failure modes it avoids:
					//   - A bump built from `outside_sq` alone is flat (zero
					//     gradient) throughout the box's entire interior --
					//     `outside_sq` only measures excess distance outside
					//     each face, so it's identically 0 anywhere inside.
					//     Confirmed empirically: with use_hard_constraints=
					//     false (no hard boundary to stop tunneling), a path
					//     pulled inside the box by the tracking cost stayed
					//     there regardless of weight, up to 40x the default.
					//   - A true barrier (e.g. 1/(sdf-margin)) is only valid
					//     in the strictly-feasible region and requires a
					//     solver that guarantees the iterate never crosses
					//     it (that guarantee is exactly what dropping the
					//     hard constraint gives up) -- cross to the
					//     infeasible side and it flips sign, rewarding
					//     further penetration instead of resisting it.
					// The hinge penalty is 0 (no gradient) whenever
					// sdf >= margin -- doesn't fight the tracking cost when
					// there's nothing to avoid -- and grows quadratically
					// (gradient growing linearly) in penetration depth once
					// sdf < margin, including arbitrarily deep inside, with
					// no saturation ceiling: unlike a bounded bump, a deep
					// enough violation eventually outweighs any finite
					// spring stiffness from the acceleration cost (item 2
					// above, whose effective stiffness scales as ~1/tau^4).
					// Continuous first derivative at sdf == margin (the
					// hinge's kink itself is smooth: d/dx[max(x,0)^2] -> 0 as
					// x -> 0 from either side), so no new numerical kink at
					// the boundary.
					const Expression violation = drake::symbolic::max(margin - sdf, 0.0);
					problem.prog->AddCost(obstacle_repulsion_weight * violation * violation);
				}
			}
		}
	}
	if (!obstacles->obstacles().empty()) {
		// Xi/V otherwise have NO other bounds anywhere in this problem --
		// harmless extra insurance for the NLP solver's numerics; doesn't
		// change the feasible region for any realistic trajectory.
		const double kGenericBound = 1.0e3;
		problem.prog->AddBoundingBoxConstraint(
			Eigen::MatrixXd::Constant(num_steps, ambient_dim, -kGenericBound),
			Eigen::MatrixXd::Constant(num_steps, ambient_dim, kGenericBound),
			Xi);
		problem.prog->AddBoundingBoxConstraint(
			Eigen::MatrixXd::Constant(num_steps, tangent_dim, -kGenericBound),
			Eigen::MatrixXd::Constant(num_steps, tangent_dim, kGenericBound),
			V);
	}


	// TODO: Add path constraint
	// for (const auto& [edge_phi_id, edge_op] : graph->get_next_edge_ops(remaining_vertices)) {
	// 	edge_op.short_path_builder(*(problem.prog), edge_phi_id, var_assignments, Xi);
	// }

	return std::move(problem);
}


/*
 * Short Path MPC
 */

GraphShortPathMPC::GraphShortPathMPC(const GraphOfConstraints& graph,
				     unsigned int num_steps,
				     unsigned int num_agents,
				     unsigned int dim,
				     double time_per_step,
				     const ObstacleSet& obstacles,
				     double obstacle_repulsion_weight,
				     bool use_hard_constraints)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step),
	  _obstacles(&obstacles),
	  _obstacle_repulsion_weight(obstacle_repulsion_weight),
	  _use_hard_constraints(use_hard_constraints) {

        /* short path times */
	// Xi.row(i) is offset by one tau from x0/v0's own time -- the
	// acceleration cost's i==0 branch already treats Xi.row(0) as the
	// state ONE tau after x0 (x0/v0 stand in as "xKm1"/"vKm1" for it).
	// _times must agree, so ref_points/ref_velocities (used by the
	// tracking cost) are sampled at the SAME offset times -- otherwise
	// the tracking cost pulls Xi.row(0) toward the reference at t=0
	// (which fill_cubic_splines sets to literally BE x0) while the
	// acceleration cost simultaneously treats it as t=tau, fighting each
	// other at every step 0 of every cycle.
	_times = Eigen::VectorXd(_num_steps);
	for (int i = 0; i < _num_steps; ++i) {
		_times(i) = (i + 1) * _time_per_step;
	}

	/* short path points */
	_points = Eigen::MatrixXd(_num_steps, _dim);
	for (int i = 0; i < _num_steps; ++i) {
		for (int j = 0; j < _dim; ++j) {
			_points(i, j) = 0.0;
		}
	}

	/* short path vels */
	_vels = Eigen::MatrixXd(_num_steps, _dim);
	for (int i = 0; i < _num_steps; ++i) {
		for (int j = 0; j < _dim; ++j) {
			_vels(i, j) = 0.0;
		}
	}
}

bool GraphShortPathMPC::solve(const Eigen::VectorXd& x0,
			      const Eigen::VectorXd& v0,
			      const Eigen::VectorXi& var_assignments,
			      const std::vector<int>& remaining_vertices,
			      const std::vector<CubicConfigurationSpline>& references) {

	_timer.Start();

	int a_dim = references.at(0).ambient_dim();
	int t_dim = references.at(0).tangent_dim();

	Eigen::MatrixXd ref_points(_num_steps, _num_agents * a_dim);
	Eigen::MatrixXd ref_velocities(_num_steps, _num_agents * t_dim);

	for (int ag = 0; ag < _num_agents; ++ag) {
		const auto& [q_ag, qdot_ag] = references[ag].eval_multiple(_times);
		ref_points.block(0, ag * a_dim, _num_steps, a_dim) = q_ag;
		ref_velocities.block(0, ag * t_dim, _num_steps, t_dim) = qdot_ag;
	}
	// Initial guess for the solver -- distinct from `ref_points`/
	// `ref_velocities` (used for the tracking costs, see
	// build_short_path_problem). Fast path (empty obstacles): identical to
	// ref_points/ref_velocities, exactly as before stage 2 -- byte-identical
	// default behavior. Obstacle path: warm-start from the PREVIOUS cycle's
	// converged trajectory (`_points`/`_vels`) when its shape matches
	// (steady-state cycles then need very little solver work, since
	// consecutive ~50ms cycles -- or, in an interactive demo, consecutive
	// drag events -- rarely change much), falling back to ref_points/
	// ref_velocities on the first cycle or after a shape change. Then
	// symmetry-break (positions only): if any per-agent-per-step guess point is CLOSE to a
	// registered sphere's center (not just exactly coincident -- the
	// constraint's gradient there, 2*(p-center), is small in proportion to
	// how close p is to center, and empirically even a near-degenerate
	// (not just exactly zero) starting gradient reliably turned a fast,
	// correct solve into an immediate solver error: confirmed via a real
	// end-to-end GraphOfConstraintsMPC.step() reference spline that placed
	// a point just ~0.001 away from an R=0.2 sphere's center -- far outside
	// a naive exact-equality check, but still degenerate enough to break
	// NLopt/SLSQP) -- nudge it off-axis whenever it's within a proportional
	// fraction of the sphere's own radius, mirroring the small perturbation
	// technique already used for exactly this failure mode in this
	// project's own history.
	Eigen::MatrixXd initial_guess_points = ref_points;
	Eigen::MatrixXd initial_guess_velocities = ref_velocities;
	if (!_obstacles->obstacles().empty()) {
		if (_has_solved && _points.rows() == static_cast<int>(_num_steps) && _points.cols() == ref_points.cols()) {
			initial_guess_points = _points;
		}
		if (_has_solved && _vels.rows() == static_cast<int>(_num_steps) && _vels.cols() == ref_velocities.cols()) {
			initial_guess_velocities = _vels;
		}
		const int workspace_dim = _graph->workspace_dim;
		for (int i = 0; i < static_cast<int>(_num_steps); ++i) {
			for (int ag = 0; ag < static_cast<int>(_num_agents); ++ag) {
				auto p = initial_guess_points.row(i).segment(ag * _dim, workspace_dim);
				for (const Obstacle& obstacle : _obstacles->obstacles()) {
					const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
					if (obstacle.kind == ObstacleKind::kSphere) {
						const double R = obstacle.params(workspace_dim) + obstacle.margin;
						const double kNudgeThreshold = std::max(0.05 * R, 1.0e-3);
						if ((p.transpose() - center).norm() < kNudgeThreshold) {
							p(0) += kNudgeThreshold;
						}
					} else if (obstacle.kind == ObstacleKind::kBox) {
						// Same failure mode as the sphere case (near-zero
						// constraint gradient breaks NLopt/SLSQP), but the
						// degenerate region is the box's WHOLE interior, not
						// a single point -- several consecutive reference
						// steps can land inside at once (confirmed: a 0.1s
						// step spacing against a 0.2-wide box put 3 of 10
						// reference points inside it simultaneously). A tiny
						// epsilon nudge isn't enough to clear that for all
						// of them; push all the way past the box's x extent
						// + margin instead.
						const Eigen::VectorXd half_extents = obstacle.params.segment(workspace_dim, workspace_dim);
						const double margin = obstacle.margin;
						const double kNudgeThreshold = std::max(0.05 * half_extents.minCoeff(), 1.0e-3);
						const Eigen::VectorXd q = (p.transpose() - center).cwiseAbs() - half_extents;
						const double outside_dist = q.cwiseMax(0.0).norm();
						if (outside_dist < kNudgeThreshold) {
							p(0) = center(0) + half_extents(0) + margin + kNudgeThreshold;
						}
					}
				}
			}
		}
	}

	std::unique_ptr<ShortPathProblem> problem;
	try {
		problem = std::make_unique<ShortPathProblem>(
			build_short_path_problem(_graph,
						 _obstacles,
						 _obstacle_repulsion_weight,
						 _use_hard_constraints,
						 ref_points,
						 ref_velocities,
						 initial_guess_points,
						 initial_guess_velocities,
						 x0, v0,
						 var_assignments,
						 remaining_vertices,
						 _time_per_step));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in short path problem construction" << std::endl;
		return false;
	}



	// Solve. Fast path (unchanged from before stage 2): no obstacles
	// registered means no non-convex constraint was added, so the original
	// QP-only dispatcher is byte-identical to today's behavior. Otherwise
	// route through NloptSolver/LD_SLSQP regardless of _use_hard_constraints
	// -- SLSQP's documented failure mode (see build_short_path_problem's own
	// comment) comes specifically from the HARD inequality constraints
	// (multiple simultaneously-active non-convex AddConstraint calls can
	// make an SQP subproblem locally infeasible), not from being SLSQP.
	// When use_hard_constraints is false, build_short_path_problem adds no
	// AddConstraint at all -- only the smooth penalty cost -- so there's
	// nothing for SLSQP to report infeasible about; it's just local
	// quasi-Newton minimization of a smooth function, exactly what SLSQP is
	// built for. (NOT IpoptSolver: see build_short_path_problem's own
	// comment for that dead end.)
	MathematicalProgramResult result;
	try {
		if (_obstacles->obstacles().empty()) {
			result = drake::solvers::Solve(*problem->prog);
		} else {
			drake::solvers::NloptSolver solver;
			problem->prog->SetSolverOption(drake::solvers::NloptSolver::id(), "algorithm", "LD_SLSQP");
			result = solver.Solve(*problem->prog);
		}
	} catch (const std::exception& e) {
		std::cout << "Caught exception in short path solver: " << e.what() << std::endl;
		return false;
	}

	if (result.is_success()) {
		_last_solve_time = _timer.Tick();

		_points = result.GetSolution(problem->Xi);
		_vels = result.GetSolution(problem->V);
		_has_solved = true;
		return true;
	} else {
		return false;
	}
}
