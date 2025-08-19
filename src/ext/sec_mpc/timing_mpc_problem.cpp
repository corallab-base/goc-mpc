#include "timing_mpc_problem.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;

TimingProblem build_timing_problem(
	const Eigen::MatrixXd& wps,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double time_cost,
	double ctrl_cost,
	bool opt_time_deltas,
	bool opt_last_vel,
	double max_vel,
	double max_acc,
	double max_jerk,
	bool acc_cont,
	double time_cost2) {

	using namespace drake::solvers;

	const int K = wps.rows();
	const int d = wps.cols();

	// Create program
	TimingProblem problem;

	// Create decision variables
	VectorXDecisionVariable time_deltas;
	if (opt_time_deltas) {
		time_deltas = problem.prog->NewContinuousVariables(K, "time_deltas");
		for (int k = 0; k < K; ++k) {
			problem.prog->AddBoundingBoxConstraint(0.01, 10.0, time_deltas(k));
		}
	}
	problem.time_deltas = time_deltas;

	const int vN = opt_last_vel ? K : K - 1;
	MatrixXDecisionVariable v = problem.prog->NewContinuousVariables(vN, d, "v");
	problem.v = v;

	// Set initial guess
	problem.prog->SetInitialGuess(v, Eigen::MatrixXd::Constant(v.rows(), v.cols(), 1.0));
	problem.prog->SetInitialGuess(time_deltas, Eigen::VectorXd::Constant(K, 10.0));

	/*
	 * OBJECTIVE FUNCTION
	 */

	// 1. Linear objective: sum(time_deltas) (OT_f in original code)
	if (time_cost > 0 && opt_time_deltas) {
		problem.prog->AddLinearCost(time_cost * Eigen::RowVectorXd::Ones(K), time_deltas);
	}

	// 2. Also, quadratic time_delta objective : sum_i time_deltas_i^2
	if (time_cost2 > 0. && opt_time_deltas) {
		const Eigen::MatrixXd Q = time_cost2 * Eigen::MatrixXd::Identity(K, K);
		const Eigen::VectorXd b = Eigen::VectorXd::Zero(K);
		problem.prog->AddQuadraticCost(Q, b, time_deltas);
	}

	// 3. Control costs
	if (ctrl_cost > 0) {
		const double s12 = std::sqrt(12.0);

		for (int k = 0; k < vN; ++k) {
		        VectorX<Expression> xK(d), xKm1(d), vK(d), vKm1(d);
		        const Expression tau(time_deltas(k));
			if (k == 0) {
				for (int i = 0; i < d; ++i) {
					xKm1(i) = Expression(x0(i));
					vKm1(i) = Expression(v0(i));
					xK(i)   = Expression(wps(0, i));
					vK(i)   = Expression(v(0, i));
				}
			} else {
				for (int i = 0; i < d; ++i) {
					xKm1(i) = Expression(wps(k-1, i));
					vKm1(i) = Expression(v(k-1, i));
					xK(i)   = Expression(wps(k, i));
					vK(i)   = Expression(v(k, i));
				}
			}
			if (k == 0) {
				xKm1 = x0.unaryExpr([](double x){ return Expression(x); });
				vKm1 = v0.unaryExpr([](double x){ return Expression(x); });
				xK = wps.row(0).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(0).unaryExpr([](Variable x){ return Expression(x); });
			} else {
				xKm1 = wps.row(k-1).unaryExpr([](double x){ return Expression(x); });
				vKm1 = v.row(k-1).unaryExpr([](Variable x){ return Expression(x); });
				xK = wps.row(k).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(k).unaryExpr([](Variable x){ return Expression(x); });
			}
			const VectorX<Expression> D = (xK - xKm1) - (0.5 * tau * (vKm1 + vK));
			const VectorX<Expression> V = (vK - vKm1);
			const VectorX<Expression> tilD = s12 * pow(tau, -1.5) * D;
			const VectorX<Expression> tilV = pow(tau, -0.5) * V;
			const Expression c = tilD.squaredNorm() + tilV.squaredNorm();
			problem.prog->AddCost(c);
		}
	}

	if (max_vel > 0) {
		for (int k = 0; k < vN; ++k) {
			VectorX<Expression> xKm1(d), xK(d), vKm1(d), vK(d);
			const Expression tau(time_deltas(k));
			const Expression inv_tau = pow(tau, -1.0);
			if (k == 0) {
				for (int i = 0; i < d; ++i) {
					xKm1(i) = Expression(x0(i));
					vKm1(i) = Expression(v0(i));
					xK(i)   = Expression(wps(0, i));
					vK(i)   = Expression(v(0, i));
				}
			} else {
				for (int i = 0; i < d; ++i) {
					xKm1(i) = Expression(wps(k-1, i));
					vKm1(i) = Expression(v(k-1, i));
					xK(i)   = Expression(wps(k, i));
					vK(i)   = Expression(v(k, i));
				}
			}

			// Your cubic parameterization (unscaled a,b,c as in your code)
			// c = v0
			const VectorX<Expression> c = vKm1;
			// b = 3*(x1 - x0) - tau*(v1 + 2*v0)
			const VectorX<Expression> b = 3.0*(xK - xKm1) - tau*(vK + 2.0*vKm1);
			// a = -2*(x1 - x0) + tau*(v1 + v0)
			const VectorX<Expression> a = -2.0*(xK - xKm1) + tau*(vK + vKm1);

			// Midpoint velocity surrogate: v_mid = c + (1/tau)*(b + 3/4 a)
			const VectorX<Expression> v_mid = c + inv_tau * (b + 0.75 * a);

			// Enforce |v(0)| <= vmax and |v_mid| <= vmax  (elementwise)
			Eigen::VectorXd lb = Eigen::VectorXd::Constant(d, -max_vel);
			Eigen::VectorXd ub = Eigen::VectorXd::Constant(d,  max_vel);
			problem.prog->AddConstraint(c,     lb, ub);   // v(0) = c
			problem.prog->AddConstraint(v_mid, lb, ub);   // v(tau/2)
		}
	}

	if (max_acc > 0) {
		for (int k = 0; k < vN; ++k) {
			VectorX<Expression> xKm1(d), xK(d), vKm1(d), vK(d);
			const Expression tau(time_deltas(k));
			if (k == 0) {
				xKm1 = x0.unaryExpr([](double x){ return Expression(x); });
				vKm1 = v0.unaryExpr([](double x){ return Expression(x); });
				xK = wps.row(0).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(0).unaryExpr([](Variable x){ return Expression(x); });
			} else {
				xKm1 = wps.row(k-1).unaryExpr([](double x){ return Expression(x); });
				vKm1 = v.row(k-1).unaryExpr([](Variable x){ return Expression(x); });
				xK = wps.row(k).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(k).unaryExpr([](Variable x){ return Expression(x); });
			}

			// b2 = 2/tau^2 * ( 3*(x1 - x0) - tau*(v1 + 2*v0) )
			const Expression inv_tau2 = pow(tau, -2.0);
			const VectorX<Expression> b2 =
				2.0 * inv_tau2 * ( 3.0*(xK - xKm1) - tau*(vK + 2.0*vKm1) );

			// a6_tau = 6/tau^2 * ( -2*(x1 - x0) + tau*(v1 + v0) )
			const VectorX<Expression> a6_tau =
				6.0 * inv_tau2 * ( -2.0*(xK - xKm1) + tau*(vK + vKm1) );

			// Endpoint accelerations
			const VectorX<Expression> acc0  = b2;                 // t = 0
			const VectorX<Expression> accT  = b2 + a6_tau;        // t = tau

			// ||acc(0)||_inf <= amax and ||acc(tau)||_inf <= amax (elementwise)
			Eigen::VectorXd lb = Eigen::VectorXd::Constant(d, -max_acc);
			Eigen::VectorXd ub = Eigen::VectorXd::Constant(d,  max_acc);
			problem.prog->AddConstraint(acc0, lb, ub);
			problem.prog->AddConstraint(accT, lb, ub);
		}
	}

	if (max_jerk > 0) {
		for (int k = 0; k < vN; ++k) {
			VectorX<Expression> xK(d), xKm1(d), vK(d), vKm1(d);
		        const Expression tau(time_deltas(k));
			if (k == 0) {
				xKm1 = x0.unaryExpr([](double x){ return Expression(x); });
				vKm1 = v0.unaryExpr([](double x){ return Expression(x); });
				xK = wps.row(0).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(0).unaryExpr([](Variable x){ return Expression(x); });
			} else {
				xKm1 = wps.row(k-1).unaryExpr([](double x){ return Expression(x); });
				vKm1 = v.row(k-1).unaryExpr([](Variable x){ return Expression(x); });
				xK = wps.row(k).unaryExpr([](double x){ return Expression(x); });
				vK = v.row(k).unaryExpr([](Variable x){ return Expression(x); });
			}

			// a6 = 6/tau^3 * (-2*(x1 - x0) + tau*(v1 + v0))
			const Expression coeff = 6.0 * pow(tau, -3.0);
			const VectorX<Expression> a6 = coeff * ( -2.0 * (xK - xKm1) + tau * (vK + vKm1) );

			// Bound a6 with symmetric bounds:
			Eigen::VectorXd lb2 = Eigen::VectorXd::Constant(d, -max_jerk);
			Eigen::VectorXd ub2 = Eigen::VectorXd::Constant(d,  max_jerk);
			problem.prog->AddConstraint(a6, lb2, ub2);
		}
	}

	// if (acc_cont) {

	// }

	return std::move(problem);
}
