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
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
	const Eigen::MatrixXd& initial_guess_points,
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

	// Set initial guess. `initial_guess_points` is DELIBERATELY a separate
	// argument from `ref_points` (used below for the tracking cost) -- in
	// the obstacle-avoidance path (see GraphShortPathMPC::solve), the
	// solver's starting point needs a small symmetry-breaking nudge away
	// from any obstacle center it happens to coincide with (a zero-gradient
	// degenerate point for the constraint below), while the tracking cost
	// itself must still pull toward the real, un-nudged reference.
	problem.prog->SetInitialGuess(Xi, initial_guess_points);
	problem.prog->SetInitialGuess(V, ref_velocities);

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
	for (int i = 0; i < num_steps; ++i) {
		for (int ag_i = 0; ag_i < num_agents; ++ag_i) {
			const Eigen::VectorX<Variable> p_WE_i = Xi.row(i).segment(ag_i * dim, 3);
			for (int ag_j = ag_i + 1; ag_j < num_agents; ++ag_j) {
				const Eigen::VectorX<Variable> p_WE_j = Xi.row(i).segment(ag_j * dim, 3);

				const Expression d_ij = (p_WE_j - p_WE_i).squaredNorm();

				// problem.prog->AddQuadraticConstraint(d_ij,
				// 				     0.0144,
				// 				     10.0);
			}
		}
	}

	// 4. Environment-obstacle avoidance (stage 2) -- spheres only, see
	// ObstacleSet's doc comment (obstacle_set.hpp) for why. For every
	// (step, agent, obstacle): a hard inequality constraint keeps the agent
	// outside the sphere + margin (the actual safety guarantee, applied at
	// EVERY step in the horizon, not just step 0 -- otherwise the solver
	// could route straight through the obstacle at an interior step while
	// leaving the endpoints clear), plus a soft Lorentzian repulsion cost
	// (graceful, visible deflection well before the hard boundary -- reuses
	// the Lorentzian SHAPE validated in an earlier, since-superseded
	// hand-rolled-gradient-descent prototype; the weight below is a fresh
	// tunable, not that prototype's tuned value, since it doesn't transfer
	// to this cost-normalization context).
	//
	// ||p-center||^2 >= R^2 is non-convex (complement of a ball) -- same
	// shape of problem as the disabled inter-agent scaffold above -- so
	// GraphShortPathMPC::solve routes to NloptSolver (algorithm LD_SLSQP)
	// instead of the QP-only dispatcher whenever any obstacle is
	// registered. NOT IpoptSolver: a first attempt routed this through
	// IpoptSolver and was EMPIRICALLY ABANDONED -- even this class's own
	// pre-existing cross-step acceleration-smoothing cost above (item 2),
	// with NO obstacle constraint involved at all, failed to converge
	// under Drake's IpoptSolver in this environment (confirmed via a
	// from-scratch test harness: IPOPT reported zero nonzeros in the
	// Lagrangian Hessian throughout and stalled regardless of max_iter,
	// while the identical cost solves instantly through the normal QP
	// solver). NLopt/LD_SLSQP was verified (same harness) to converge
	// reliably and quickly on this exact cost+constraint combination.
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
				if (obstacle.kind != ObstacleKind::kSphere) continue;

				const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
				const double R = obstacle.params(3) + obstacle.margin;
				const double R2 = R * R;

				// p is a ROW (Xi.row(i).segment(...)); center is a COLUMN
				// (Eigen::VectorXd) -- .transpose() p first. Getting this
				// wrong doesn't fail loudly: in a RelWithDebInfo build with
				// Eigen assertions compiled out, a row-minus-column shape
				// mismatch silently produces a garbage/truncated
				// expression instead of throwing (confirmed: without this
				// transpose, the resulting constraint printed as just
				// `pow(xi(0,0), 2)` -- the y/z terms had vanished).
				const Eigen::VectorX<Expression> diff = p.transpose() - center;
				const Expression d2 = diff.squaredNorm();

				problem.prog->AddConstraint(d2, R2, std::numeric_limits<double>::infinity());
				problem.prog->AddCost(obstacle_repulsion_weight * R2 / (d2 + R2));
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
				     double obstacle_repulsion_weight)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step),
	  _obstacles(&obstacles),
	  _obstacle_repulsion_weight(obstacle_repulsion_weight) {

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
	// Initial guess for the solver -- distinct from `ref_points` (used
	// for the tracking cost, see build_short_path_problem). Fast path
	// (empty obstacles): identical to ref_points, exactly as before stage
	// 2 -- byte-identical default behavior. Obstacle path: warm-start from
	// the PREVIOUS cycle's converged trajectory (`_points`) when its shape
	// matches (steady-state cycles then need very little solver work,
	// since consecutive ~50ms cycles rarely change much), falling back to
	// ref_points on the first cycle or after a shape change. Then
	// symmetry-break: if any per-agent-per-step guess point is CLOSE to a
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
	if (!_obstacles->obstacles().empty()) {
		if (_has_solved && _points.rows() == static_cast<int>(_num_steps) && _points.cols() == ref_points.cols()) {
			initial_guess_points = _points;
		}
		const int workspace_dim = _graph->workspace_dim;
		for (int i = 0; i < static_cast<int>(_num_steps); ++i) {
			for (int ag = 0; ag < static_cast<int>(_num_agents); ++ag) {
				auto p = initial_guess_points.row(i).segment(ag * _dim, workspace_dim);
				for (const Obstacle& obstacle : _obstacles->obstacles()) {
					if (obstacle.kind != ObstacleKind::kSphere) continue;
					const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
					const double R = obstacle.params(3) + obstacle.margin;
					const double kNudgeThreshold = std::max(0.05 * R, 1.0e-3);
					if ((p.transpose() - center).norm() < kNudgeThreshold) {
						p(0) += kNudgeThreshold;
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
						 ref_points,
						 ref_velocities,
						 initial_guess_points,
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
	// QP-only dispatcher is byte-identical to today's behavior. Only route
	// through NloptSolver -- more expensive per call, appropriate for a
	// non-convex problem -- when there's actually a non-convex obstacle
	// constraint to solve. NloptSolver with the LD_SLSQP algorithm (SQP,
	// not IPOPT's interior-point method) was verified empirically to
	// converge reliably and quickly on this class's costs (see
	// build_short_path_problem's own comment for the IPOPT dead end this
	// replaced).
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
