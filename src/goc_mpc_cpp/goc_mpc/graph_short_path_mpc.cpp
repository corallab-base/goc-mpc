#include "graph_short_path_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;

using namespace pybind11::literals;
namespace py = pybind11;

ShortPathProblem build_short_path_problem(
	const py::array_t<double>& ref_points_np,
	const py::array_t<double>& ref_velocities_np,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double tau) {

	using namespace drake::solvers;

	const ssize_t num_steps = ref_points_np.shape(0);
	const ssize_t dim = ref_points_np.shape(1);

	// Create Eigen objects from input arrays
	Eigen::MatrixXd ref_points(num_steps, dim);
	Eigen::MatrixXd ref_velocities(num_steps, dim);

	auto ref_points_u = ref_points_np.unchecked<2>();
	auto ref_velocities_u = ref_velocities_np.unchecked<2>();

	for (size_t i = 0; i < num_steps; ++i) {
		for (size_t j = 0; j < dim; ++j) {
			ref_points(i, j) = ref_points_u(i, j);
			ref_velocities(i, j) = ref_velocities_u(i, j);
		}
	}

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
				     unsigned int dim,
				     double time_per_step)
	: _num_steps(num_steps),
	  _dim(dim),
	  _time_per_step(time_per_step) {

        /* short path times */
	_times = py::array_t<double>(_num_steps);
	auto times_mut = _times.mutable_unchecked<1>();
	for (int i = 0; i < _num_steps; ++i) {
		times_mut(i) = (i+1) * _time_per_step;
	}

	/* short path points */
	_points = py::array_t<unsigned int>({_num_steps, _dim});
	auto points_mut = _points.mutable_unchecked<2>();
	for (int i = 0; i < _num_steps; ++i) {
		for (int j = 0; j < _dim; ++j) {
			points_mut(i, j) = 0.0;
		}
	}

	/* short path vels */
	_vels = py::array_t<unsigned int>({_num_steps, _dim});
	auto vels_mut = _vels.mutable_unchecked<2>();
	for (int i = 0; i < _num_steps; ++i) {
		for (int j = 0; j < _dim; ++j) {
			vels_mut(i, j) = 0.0;
		}
	}
}

int GraphShortPathMPC::solve(const Eigen::VectorXd& x0, const Eigen::VectorXd& v0, const CubicSpline& reference) {

	py::array_t<double> ref_points = reference.eval_multiple(_times, 0);
	py::array_t<double> ref_velocities = reference.eval_multiple(_times, 1);
	
	struct ShortPathProblem problem = build_short_path_problem(ref_points,
								   ref_velocities,
								   x0, v0, _time_per_step);

	// Solve
	auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		std::cout << "Success" << std::endl;

		const Eigen::MatrixXd xi = result.GetSolution(problem.xi);
		const Eigen::MatrixXd v = result.GetSolution(problem.v);

		auto points_mut = _points.mutable_unchecked<2>();
		for (int i = 0; i < _num_steps; ++i) {
			for (int j = 0; j < _dim; ++j) {
				points_mut(i, j) = xi(i, j);
			}
		}

		auto vels_mut = _vels.mutable_unchecked<2>();
		for (int i = 0; i < _num_steps; ++i) {
			for (int j = 0; j < _dim; ++j) {
				vels_mut(i, j) = v(i, j);
			}
		}
	} else {
		std::cerr << "Optimization failed." << std::endl;
	}

	return 0;
}

py::array_t<double> GraphShortPathMPC::get_points() const {
	return _points;
}

py::array_t<double> GraphShortPathMPC::get_vels() const {
	return _vels;
}

py::array_t<double> GraphShortPathMPC::get_times() const {
	return _times;
}
