#include "graph_ordering_mpc_problem.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;

GraphOrderingProblem build_graph_ordering_problem(
	const py::array_t<double>& waypoints,
	const py::array_t<unsigned int>& graph_np,
	const py::array_t<double>& x0_np,
	const py::array_t<double>& v0_np) {

	using namespace drake::solvers;

	const ssize_t K = waypoints.shape(0);
	const ssize_t d = waypoints.shape(1);

	// Create Eigen objects from input arrays
	Eigen::VectorXd x0(d);
	Eigen::VectorXd v0(d);
	Eigen::MatrixXd wps(K, d);
	Eigen::MatrixXd graph(K, K);

	auto x0_u = x0_np.unchecked<1>();
	auto v0_u = v0_np.unchecked<1>();
	auto wps_u = waypoints.unchecked<2>();
	auto graph_u = graph_np.unchecked<2>();

	for (size_t i = 0; i < K; ++i) {
		for (size_t j = 0; j < K; ++j) {
			graph(i, j) = graph_u(i, j);
		}
		for (size_t j = 0; j < d; ++j) {
			wps(i, j) = wps_u(i, j);
		}
	}

	for (size_t i = 0; i < d; ++i) {
		x0(i) = x0_u(i);
		v0(i) = v0_u(i);
	}

	// Create program
	GraphOrderingProblem problem;

	// Create decision variables
	// p(i,j) = 1 iff waypoint i is at position j
	MatrixXDecisionVariable p = problem.prog->NewBinaryVariables(K, K, "p");
	problem.p = p;

	// Doubly-Stochastic
	// Each waypoint exactly once: sum_k p(i,k) = 1
	for (int i = 0; i < K; ++i) {
		VectorX<Variable> row = p.row(i);
		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, row);
	}
	// Each position filled: sum_i p(i,k) = 1
	for (int j = 0; j < K; ++j) {
		VectorX<Variable> col = p.col(j);
		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, col);
	}

	// Precedence: i must appear before j  ==>  sum_k k*P(i,k) + 1 <= sum_k k*P(j,k)
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			if (graph(i, j) == 1) {
				VectorX<Variable> lhs_vars(2*K);
				Eigen::RowVectorXd lhs_coeffs = Eigen::RowVectorXd::Zero(2*K);
				// pack as [P(i, 0 through n-1), P(j, 0 through n-1)]
				for (int k = 0; k < K; ++k) {
					lhs_vars(k)     = p(i, k);
					lhs_vars(K + k) = p(j, k);
					lhs_coeffs(k)       = k;       // +k * P(i,k)
					lhs_coeffs(K + k)   = -k;      // -k * P(j,k)
				}
				/* lb occurs when P(i,0) = 1 and P(j, K-1) = 1. Therefore k*P(i, k) - k*P(j, k) = -(K - 1). */
				problem.prog->AddLinearConstraint(lhs_coeffs, -(K-1), -1, lhs_vars); // enforces pos(i)+1 <= pos(j)
			}
		}
	}

	// Helpers: squared distances from x0 to xi for all i
	// squared distances between xi and xj for all i,j
	Eigen::VectorXd s2(K);
	Eigen::MatrixXd d2(K, K);
	for (int i = 0; i < K; ++i) {
		s2(i) = (wps.row(i).transpose() - x0).squaredNorm();
		for (int j = 0; j < K; ++j) {
			d2(i,j) = (wps.row(i) - wps.row(j)).squaredNorm();
		}
	}

	// Objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_{i,j} d2(i,j)*p(i,k-1)*p(j,k)

	// The second term is quadratic in binaries; to keep MILP, use a *linear* proxy:
	// e.g., sum_{k=1}^{n-1} sum_{j} (min_i D2(i,j)) * p(j,k)  (lower-bound-ish) or just sum over k of degrees.
	// A better linear surrogate: use a fixed "nearest predecessor" cost C(j,k) = min_i D2(i,j).
	Eigen::VectorXd c_min(K);
	for (int j = 0; j < K; ++j) {
		double m = std::numeric_limits<double>::infinity();
		for (int i = 0; i < K; ++i) if (i != j) m = std::min(m, d2(i,j));
		c_min(j) = std::isfinite(m) ? m : 0.0;
	}
	drake::solvers::LinearCost* obj = nullptr;
	// Build linear objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_j c_min(j)*p(j,k)
	Eigen::VectorXd coeffs(K*K);
	coeffs.setZero();
	int idx = 0;
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j, ++idx) {
			double c = (j == 0) ? s2(i) : c_min(i);
			coeffs(idx) = c;
		}
	}
	drake::VectorX<Variable> allP(K*K);
	idx = 0;
	for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) allP[idx++] = p(i, j);
	problem.prog->AddLinearCost(coeffs, 0.0, allP);

	return std::move(problem);
}
