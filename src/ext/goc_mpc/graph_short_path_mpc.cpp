#include "graph_short_path_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;

using namespace pybind11::literals;
namespace py = pybind11;

ShortPathProblem build_short_path_problem(
	const Eigen::MatrixXd& ref_points,
	const Eigen::MatrixXd& ref_velocities,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double tau) {

	using namespace drake::solvers;

	const int num_steps = ref_points.rows();
	const int dim = ref_points.cols();

	// Create program
	ShortPathProblem problem;

	MatrixXDecisionVariable xi = problem.prog->NewContinuousVariables(num_steps, dim, "xi");
	problem.xi = xi;

	MatrixXDecisionVariable v = problem.prog->NewContinuousVariables(num_steps, dim, "v");
	problem.v = v;

	// Set initial guess
	problem.prog->SetInitialGuess(xi, ref_points);
	problem.prog->SetInitialGuess(v, ref_velocities);

	/*
	 * OBJECTIVE FUNCTION
	 */

	// 1. Tracking error objective
	for (int i = 0; i < num_steps; ++i) {
		VectorX<Expression> diff = xi.row(i) - ref_points.row(i);
		Expression dist = diff.squaredNorm();
		problem.prog->AddQuadraticCost(dist);
	}

	double tau2 = tau * tau;
	double tau3 = tau * tau2;

	// 2. Scaled acceleration objective
	for (int i = 0; i < num_steps; ++i) {
		if (i == 0) {
			const Eigen::VectorXd xKm1 = x0;
			const Eigen::VectorX<Variable> xK = xi.row(i);
			const Eigen::VectorXd vKm1 = v0;
			const Eigen::VectorX<Variable> vK = v.row(i);

			const Eigen::VectorX<Expression> a6_tau = 6.0 / tau2 * (-2.0 * (xK - xKm1) + tau * (vK + vKm1));
			const Eigen::VectorX<Expression> b2 = 2.0 / tau2 * (3.0 * (xK - xKm1) - tau * (vK + 2.0 * vKm1));
			const Expression acc_norm = (a6_tau + b2).squaredNorm();
			problem.prog->AddQuadraticCost(acc_norm);
		} else {
			const Eigen::VectorX<Variable> xKm1 = xi.row(i-1);
			const Eigen::VectorX<Variable> xK = xi.row(i);
			const Eigen::VectorX<Variable> vKm1 = v.row(i-1);
			const Eigen::VectorX<Variable> vK = v.row(i);

			const Eigen::VectorX<Expression> a6_tau = 6.0 / tau2 * (-2.0 * (xK - xKm1) + tau * (vK + vKm1));
			const Eigen::VectorX<Expression> b2 = 2.0 / tau2 * (3.0 * (xK - xKm1) - tau * (vK + 2.0 * vKm1));
			const Expression acc_norm = (a6_tau + b2).squaredNorm();
			problem.prog->AddQuadraticCost(acc_norm);
		}
	}

	// TODO: Add path constraint

	return std::move(problem);
}


/*
 * Short Path MPC
 */

GraphShortPathMPC::GraphShortPathMPC(unsigned int num_steps,
				     unsigned int num_agents,
				     unsigned int dim,
				     double time_per_step)
	: _num_steps(num_steps),
	  _num_agents(num_agents),
	  _dim(dim),
	  _time_per_step(time_per_step) {

        /* short path times */
	_times = Eigen::VectorXd(_num_steps);
	for (int i = 0; i < _num_steps; ++i) {
		_times(i) = (i+1) * _time_per_step;
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

bool GraphShortPathMPC::solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& v0, const std::vector<CubicSpline>& references) {

	Eigen::MatrixXd ref_points(_num_steps, _num_agents * _dim);
	Eigen::MatrixXd ref_velocities(_num_steps, _num_agents * _dim);

	for (int ag = 0; ag < _num_agents; ++ag) {
		ref_points.block(0, ag * _dim, _num_steps, _dim) = references[ag].eval_multiple(_times, 0);
		ref_velocities.block(0, ag * _dim, _num_steps, _dim) = references[ag].eval_multiple(_times, 1);
	}
	
	struct ShortPathProblem problem = build_short_path_problem(ref_points,
								   ref_velocities,
								   x0, v0, _time_per_step);

	// Solve
	auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		_points = result.GetSolution(problem.xi);
		_vels = result.GetSolution(problem.v);
		return true;
	} else {
		return false;
	}
}
