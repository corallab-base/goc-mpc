#include "graph_short_path_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using drake::solvers::MathematicalProgramResult;

using namespace pybind11::literals;
namespace py = pybind11;

// ---------------------------------------------------------------------------
// QP problem builder (no obstacles)
// ---------------------------------------------------------------------------

ShortPathProblem build_short_path_problem(
	const GraphOfConstraints* graph,
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const Eigen::VectorXi& var_assignments,
	const std::vector<int> remaining_vertices,
	double tau,
	const std::vector<Obstacle>& /*obstacles*/) {

	using namespace drake::solvers;

	const int num_steps   = ref_points.rows();
	const int ambient_dim = ref_points.cols();
	const int tangent_dim = ref_velocities.cols();

	ShortPathProblem problem;

	MatrixXDecisionVariable Xi =
		problem.prog->NewContinuousVariables(num_steps, ambient_dim, "xi");
	problem.Xi = Xi;

	MatrixXDecisionVariable V =
		problem.prog->NewContinuousVariables(num_steps, tangent_dim, "v");
	problem.V = V;

	problem.prog->SetInitialGuess(Xi, ref_points);
	problem.prog->SetInitialGuess(V, ref_velocities);

	// 1. Tracking cost
	for (int i = 0; i < num_steps; ++i) {
		VectorX<Expression> diff = Xi.row(i) - ref_points.row(i);
		problem.prog->AddQuadraticCost(diff.squaredNorm());
	}

	const double tau2 = tau * tau;

	// 2. Acceleration cost (cubic Hermite consistency)
	for (int i = 0; i < num_steps; ++i) {
		if (i == 0) {
			const Eigen::VectorXd      xKm1 = x0.segment(0, ambient_dim);
			const Eigen::VectorX<Variable> xK = Xi.row(i);
			const Eigen::VectorXd      vKm1 = v0.segment(0, tangent_dim);
			const Eigen::VectorX<Variable> vK = V.row(i);
			const Eigen::VectorX<Expression> a6 =
				6.0/tau2 * (-2.0*(xK-xKm1) + tau*(vK+vKm1));
			const Eigen::VectorX<Expression> b2 =
				2.0/tau2 * ( 3.0*(xK-xKm1) - tau*(vK+2.0*vKm1));
			problem.prog->AddQuadraticCost((a6+b2).squaredNorm());
		} else {
			const Eigen::VectorX<Variable> xKm1 = Xi.row(i-1);
			const Eigen::VectorX<Variable> xK   = Xi.row(i);
			const Eigen::VectorX<Variable> vKm1 = V.row(i-1);
			const Eigen::VectorX<Variable> vK   = V.row(i);
			const Eigen::VectorX<Expression> a6 =
				6.0/tau2 * (-2.0*(xK-xKm1) + tau*(vK+vKm1));
			const Eigen::VectorX<Expression> b2 =
				2.0/tau2 * ( 3.0*(xK-xKm1) - tau*(vK+2.0*vKm1));
			problem.prog->AddQuadraticCost((a6+b2).squaredNorm());
		}
	}

	// Path constraints (edge operators)
	for (const auto& [edge_phi_id, edge_op] : graph->get_next_edge_ops(remaining_vertices)) {
		edge_op.short_path_builder(*(problem.prog), edge_phi_id, var_assignments, Xi);
	}

	return std::move(problem);
}


// ---------------------------------------------------------------------------
// Gradient-descent solver for the obstacle-aware case.
//
// Cost = smoothness (CHOMP-style velocity Laplacian)
//      + Lorentzian obstacle repulsion
//
// Fixed endpoints: xi_0 pinned to current agent position x0,
//                  xi_{N-1} pinned to the reference end point.
//
// Gradient descent pushes interior waypoints away from obstacles while the
// smoothness term keeps the path smooth.  No IPOPT, no symbolic expressions.
// ---------------------------------------------------------------------------

static void solve_gd(
	const Eigen::MatrixXd& ref_points,
	const Eigen::VectorXd& x0,
	const std::vector<Obstacle>& obstacles,
	const int    num_agents,
	const int    dim,
	const double tau,
	Eigen::MatrixXd& out_points,
	Eigen::MatrixXd& out_vels)
{
	const int N = ref_points.rows();
	const int D = ref_points.cols();

	// -----------------------------------------------------------------------
	// Initialise from the reference spline.
	// Add a tiny x-perturbation to any point within 1 mm of an obstacle
	// centre to break gradient symmetry (∇L = 0 exactly at centre).
	// -----------------------------------------------------------------------
	Eigen::MatrixXd Xi = ref_points;
	Xi.row(0) = x0.head(D).transpose();  // pin start

	for (int k = 0; k < N; ++k) {
		for (const auto& obs : obstacles) {
			const int ag0 = (obs.agent_id >= 0) ? obs.agent_id : 0;
			const int ag1 = (obs.agent_id >= 0) ? obs.agent_id+1 : num_agents;
			for (int ag = ag0; ag < ag1; ++ag) {
				Eigen::Vector3d p(Xi.row(k).segment(ag * dim, 3));
				if ((p - obs.pos).norm() < 1e-3)
					Xi.row(k).segment(ag * dim, 3)(0) += 1e-3;
			}
		}
	}

	// -----------------------------------------------------------------------
	// Hyper-parameters
	//
	// w_smooth must be >> max obstacle weight so the smoothness term can
	// balance the Lorentzian repulsion and prevent the trajectory from
	// drifting arbitrarily far from the reference.  The equilibrium
	// deflection magnitude ≈ (w_obs * sigma) / (4 * w_smooth * tau).
	// -----------------------------------------------------------------------
	const double w_smooth  = 100.0; // CHOMP smoothness weight
	const double lr        = 0.01;  // gradient-descent step size
	const double grad_clip = 0.5;   // per-step gradient magnitude cap
	const int    max_iter  = 100;

	// -----------------------------------------------------------------------
	// Gradient-descent loop — only interior points (1 … N-2) are updated.
	// -----------------------------------------------------------------------
	for (int iter = 0; iter < max_iter; ++iter) {
		for (int k = 1; k < N - 1; ++k) {
			Eigen::VectorXd g = Eigen::VectorXd::Zero(D);

			// -- CHOMP-style smoothness (velocity Laplacian) --
			// L_smooth = Σ ||xi_{k+1} - xi_k||²
			// ∂L/∂xi_k = 2 * (2*xi_k - xi_{k-1} - xi_{k+1})
			const Eigen::VectorXd xi_k    = Xi.row(k).transpose();
			const Eigen::VectorXd xi_prev = Xi.row(k-1).transpose();
			const Eigen::VectorXd xi_next = Xi.row(k+1).transpose();
			g += 2.0 * w_smooth * (2.0*xi_k - xi_prev - xi_next);

			// -- Lorentzian obstacle repulsion --
			// cost(p) = w * σ² / (d² + σ²)
			// ∂cost/∂p = -2w σ² dp / (d² + σ²)²
			for (const auto& obs : obstacles) {
				const int ag0 = (obs.agent_id >= 0) ? obs.agent_id   : 0;
				const int ag1 = (obs.agent_id >= 0) ? obs.agent_id+1 : num_agents;
				for (int ag = ag0; ag < ag1; ++ag) {
					const Eigen::Vector3d p(Xi.row(k).segment(ag * dim, 3));
					const Eigen::Vector3d dp = p - obs.pos;
					Eigen::Vector3d g_obs;

					if (obs.type == Obstacle::SPHERE) {
						const double s2    = std::pow(obs.r_agent + obs.r_obs, 2.0);
						const double d2    = dp.squaredNorm();
						const double den   = d2 + s2;
						g_obs = -2.0 * obs.weight * s2 / (den * den) * dp;
					} else { // BOX (ellipsoidal approximation)
						const Eigen::Vector3d h =
							obs.half_sizes + Eigen::Vector3d::Constant(obs.r_agent);
						const Eigen::Vector3d e  = dp.array() / h.array();
						const double d2e  = e.squaredNorm();
						const double den  = d2e + 1.0;
						// cost = w / den
						// ∂cost/∂p_i = -2w e_i / (h_i · den²)
						//             = -2w dp_i / (h_i² · den²)
						g_obs = -2.0 * obs.weight / (den * den)
						        * (dp.array() / (h.array() * h.array())).matrix();
					}
					g.segment(ag * dim, 3) += g_obs;
				}
			}

			// Gradient clipping
			const double gn = g.norm();
			if (gn > grad_clip) g *= grad_clip / gn;

			Xi.row(k) -= lr * g.transpose();
		}
	}

	// -----------------------------------------------------------------------
	// Outputs
	// -----------------------------------------------------------------------
	out_points = Xi;

	// Velocities from central differences of the optimised positions.
	out_vels.resize(N, D);
	out_vels.row(0)   = (Xi.row(1) - Xi.row(0)) / tau;
	out_vels.row(N-1) = (Xi.row(N-1) - Xi.row(N-2)) / tau;
	for (int k = 1; k < N-1; ++k)
		out_vels.row(k) = (Xi.row(k+1) - Xi.row(k-1)) / (2.0 * tau);
}


/*
 * Short Path MPC
 */

GraphShortPathMPC::GraphShortPathMPC(const GraphOfConstraints& graph,
				     unsigned int num_steps,
				     unsigned int num_agents,
				     unsigned int dim,
				     double time_per_step)
	: _graph(&graph),
	  _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step) {

	_times = Eigen::VectorXd(_num_steps);
	for (int i = 0; i < _num_steps; ++i)
		_times(i) = i * _time_per_step;

	_points = Eigen::MatrixXd::Zero(_num_steps, _dim);
	_vels   = Eigen::MatrixXd::Zero(_num_steps, _dim);
}

bool GraphShortPathMPC::solve(const Eigen::VectorXd& x0,
			      const Eigen::VectorXd& v0,
			      const Eigen::VectorXi& var_assignments,
			      const std::vector<int>& remaining_vertices,
			      const std::vector<CubicConfigurationSpline>& references) {

	_timer.Start();

	const int a_dim = references.at(0).ambient_dim();
	const int t_dim = references.at(0).tangent_dim();

	Eigen::MatrixXd ref_points(_num_steps, _num_agents * a_dim);
	Eigen::MatrixXd ref_velocities(_num_steps, _num_agents * t_dim);

	for (int ag = 0; ag < _num_agents; ++ag) {
		const auto& [q_ag, qdot_ag] = references[ag].eval_multiple(_times);
		ref_points.block(0, ag * a_dim, _num_steps, a_dim)     = q_ag;
		ref_velocities.block(0, ag * t_dim, _num_steps, t_dim) = qdot_ag;
	}

	// -------------------------------------------------------------------
	// With obstacles: gradient descent (CHOMP-style) — no IPOPT needed.
	// Without obstacles: original QP via Drake.
	// -------------------------------------------------------------------
	if (!_obstacles.empty()) {
		solve_gd(ref_points, x0, _obstacles,
		         _num_agents, _dim, _time_per_step,
		         _points, _vels);
		_last_solve_time = _timer.Tick();
		return true;
	}

	std::unique_ptr<ShortPathProblem> problem;
	try {
		problem = std::make_unique<ShortPathProblem>(
			build_short_path_problem(_graph,
						 ref_points, ref_velocities,
						 x0, v0,
						 var_assignments,
						 remaining_vertices,
						 _time_per_step,
						 _obstacles));
	} catch (const std::exception& e) {
		std::cout << "Short path problem construction failed\n";
		return false;
	}

	MathematicalProgramResult result;
	try {
		result = drake::solvers::Solve(*problem->prog);
	} catch (const std::exception& e) {
		std::cout << "Short path solve failed\n";
		return false;
	}

	if (result.is_success()) {
		_last_solve_time = _timer.Tick();
		_points = result.GetSolution(problem->Xi);
		_vels   = result.GetSolution(problem->V);
		return true;
	}
	return false;
}

void GraphShortPathMPC::add_sphere_obstacle(
		int agent_id, Eigen::Vector3d pos,
		double r_agent, double r_obs, double weight) {
	_obstacles.push_back({Obstacle::SPHERE, agent_id, pos, r_agent, r_obs,
	                      Eigen::Vector3d::Zero(), weight});
}

void GraphShortPathMPC::add_box_obstacle(
		int agent_id, Eigen::Vector3d pos,
		Eigen::Vector3d half_sizes, double r_agent, double weight) {
	_obstacles.push_back({Obstacle::BOX, agent_id, pos, r_agent, 0.0,
	                      half_sizes, weight});
}

void GraphShortPathMPC::clear_obstacles() {
	_obstacles.clear();
}
