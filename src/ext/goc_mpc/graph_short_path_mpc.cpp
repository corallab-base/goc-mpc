#include "graph_short_path_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using drake::solvers::MathematicalProgramResult;

using namespace pybind11::literals;
namespace py = pybind11;

ShortPathProblem build_short_path_problem(
	const GraphOfConstraints* graph,
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
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

	// Set initial guess
	problem.prog->SetInitialGuess(Xi, ref_points);
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

	// 3. Collision cost
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


	// TODO: Add path constraint
	for (const auto& [edge_phi_id, edge_op] : graph->get_next_edge_ops(remaining_vertices)) {
		edge_op.short_path_builder(*(problem.prog), edge_phi_id, var_assignments, Xi);
	}

	return std::move(problem);
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

        /* short path times */
	_times = Eigen::VectorXd(_num_steps);
	for (int i = 0; i < _num_steps; ++i) {
		_times(i) = (i) * _time_per_step;
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

	std::unique_ptr<ShortPathProblem> problem;
	try {
		problem = std::make_unique<ShortPathProblem>(
			build_short_path_problem(_graph,
						 ref_points,
						 ref_velocities,
						 x0, v0,
						 var_assignments,
						 remaining_vertices,
						 _time_per_step));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in short path problem construction" << std::endl;
		return false;
	}



	// Solve
	MathematicalProgramResult result;
	try {
		result = drake::solvers::Solve(*problem->prog);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in solver" << std::endl;
		return false;
	}

	if (result.is_success()) {
		_last_solve_time = _timer.Tick();

		_points = result.GetSolution(problem->Xi);
		_vels = result.GetSolution(problem->V);
		return true;
	} else {
		return false;
	}
}
