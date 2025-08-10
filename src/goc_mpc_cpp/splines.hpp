/*  ------------------------------------------------------------------
    Copyright (c) 2011-2024 Marc Toussaint
    email: toussaint@tu-berlin.de

    This code is distributed under the MIT License.
    Please see <root-path>/LICENSE for details.
    --------------------------------------------------------------  */

#pragma once

#include <optional>
#include <iostream>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <drake/common/autodiff.h>
#include <drake/solvers/binding.h>
#include <drake/solvers/cost.h>

#include <Eigen/Core>

using namespace pybind11::literals;
namespace py = pybind11;


struct CubicPiece {
	py::array_t<double> a, b, c, d;

	/// Set the cubic piece given positions and velocities at endpoints and duration tau
	void set(const py::array_t<double>& x0,
		 const py::array_t<double>& v0,
		 const py::array_t<double>& x1,
		 const py::array_t<double>& v1,
		 double tau);

	/// Evaluate position, velocity, acceleration at time t
	void eval_into(py::object x,
		       py::object xDot,
		       py::object xDDot,
		       double t) const;

	/// Evaluate the `diff`-th derivative at time t: 0 (pos), 1 (vel), 2 (acc)
	py::array_t<double> eval(double t, unsigned int diff = 0) const;

	// /// Print coefficients to output stream
	// void write(std::ostream& os) const {
	//   os << "a: " << a << " b: " << b << " c: " << c << " d: " << d;
	// }
};


struct CubicSpline {
	std::vector<CubicPiece> pieces;
	py::array_t<double> times;

	/// Set the spline from points, velocities, and time intervals
	void set(const py::array_t<double>& pts,
		 const py::array_t<double>& vels,
		 const py::array_t<double>& times);

	/// Append additional segments to the spline
	void append(const py::array_t<double>& pts,
		    const py::array_t<double>& vels,
		    const py::array_t<double>& times);

	/// Return the index of the spline segment that covers time t
	unsigned int get_piece(double t) const;

	/// Evaluate position, velocity, and acceleration at time t
	void eval_into(py::object x,
		       py::object xDot,
		       py::object xDDot,
		       double t) const;

	/// Evaluate the `diff`-th derivative at time t
	py::array_t<double> eval(double t, unsigned int diff = 0) const;

	/// Evaluate multiple time points
	py::array_t<double> eval_multiple(const py::array_t<double>& T, unsigned int diff = 0) const;

	/// Get the start time of the spline
	double begin() const;

	/// Get the end time of the spline
	double end() const;
};


// CubicSpline based costs and constraints

/**
 * Implements the cost:
 *     ((x1 - x0 - v0 * tau)^2) / tau
 * where x0 and x1 are fixed, and v0, tau are decision variables.
 */
class CubicSplineLeapCost : public drake::solvers::Cost {
public:
	DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(CubicSplineLeapCost);

	CubicSplineLeapCost(const Eigen::VectorXd& x0,
			    const Eigen::VectorXd& v0,
			    const Eigen::MatrixXd& wps,
			    double ctrl_cost,
                            size_t K,
			    bool opt_last_vel,
			    bool opt_time_deltas);

	~CubicSplineLeapCost() override;

private:
	// Evaluates cost with plain double values
	void DoEval(const Eigen::Ref<const Eigen::VectorXd>& x,
		    Eigen::VectorXd* y) const override;

	// Evaluates cost with AutoDiffXd
	void DoEval(const Eigen::Ref<const drake::AutoDiffVecXd>& x,
		    drake::AutoDiffVecXd* y) const override;

	// Evaluates cost with symbolic variables
	void DoEval(const Eigen::Ref<const Eigen::Matrix<drake::symbolic::Variable, -1, 1>>& x,
		    Eigen::Matrix<drake::symbolic::Expression, -1, 1>* y) const override;

	// Internal data
	const Eigen::VectorXd x0_;
	const Eigen::VectorXd v0_;
	const Eigen::MatrixXd wps_;
	const double ctrl_cost_;
	const size_t K_, d_; /* n points of dimension d in the spline (after x0,v0) */
	const bool opt_last_vel_, opt_time_deltas_;
};


// py::array_t<double> CubicSplineMaxJer(const py::array_t<double>& x0,
//                                       const py::array_t<double>& v0,
//                                       const py::array_t<double>& x1,
//                                       const py::array_t<double>& v1,
//                                       double tau,
//                                       const py::array_t<double>& tauJ = py::array_t<double>());

// py::array_t<double> CubicSplineMaxAcc(const py::array_t<double>& x0,
//                                       const py::array_t<double>& v0,
//                                       const py::array_t<double>& x1,
//                                       const py::array_t<double>& v1,
//                                       double tau,
//                                       const py::array_t<double>& tauJ = py::array_t<double>());

// py::array_t<double> CubicSplineAcc0(const py::array_t<double>& x0,
//                                     const py::array_t<double>& v0,
//                                     const py::array_t<double>& x1,
//                                     const py::array_t<double>& v1,
//                                     double tau,
//                                     const py::array_t<double>& tauJ = py::array_t<double>());

// py::array_t<double> CubicSplineAcc1(const py::array_t<double>& x0,
//                                     const py::array_t<double>& v0,
//                                     const py::array_t<double>& x1,
//                                     const py::array_t<double>& v1,
//                                     double tau,
//                                     const py::array_t<double>& tauJ = py::array_t<double>());

// void CubicSplinePosVelAcc(py::array_t<double>& pos,
//                           py::array_t<double>& vel,
//                           py::array_t<double>& acc,
//                           double trel,
//                           const py::array_t<double>& x0,
//                           const py::array_t<double>& v0,
//                           const py::array_t<double>& x1,
//                           const py::array_t<double>& v1,
//                           double tau,
//                           const py::array_t<double>& tauJ = py::array_t<double>());
