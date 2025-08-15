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
#include <pybind11/eigen.h>

#include <drake/common/autodiff.h>
#include <drake/solvers/binding.h>
#include <drake/solvers/cost.h>

#include <Eigen/Core>

using namespace pybind11::literals;
namespace py = pybind11;


struct CubicPiece {
	Eigen::VectorXd _a, _b, _c, _d;

	/// Set the cubic piece given positions and velocities at endpoints and duration tau
	void set(const Eigen::VectorXd& x0,
		 const Eigen::VectorXd& v0,
		 const Eigen::VectorXd& x1,
		 const Eigen::VectorXd& v1,
		 double tau);

	/// Evaluate position, velocity, acceleration at time t
	void eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
		       double t) const;

	/// Evaluate the `diff`-th derivative at time t: 0 (pos), 1 (vel), 2 (acc)
	Eigen::VectorXd eval(double t, unsigned int diff = 0) const;
};


struct CubicSpline {
	std::vector<CubicPiece> _pieces;
	Eigen::VectorXd _times;
	size_t _dim = 0;

	/// Set the spline from points, velocities, and time intervals
	void set(const Eigen::MatrixXd& pts,
		 const Eigen::MatrixXd& vels,
		 const Eigen::VectorXd& times);

	/// Append additional segments to the spline
	void append(const Eigen::Ref<const Eigen::MatrixXd>& pts,
		    const Eigen::Ref<const Eigen::MatrixXd>& vels,
		    const Eigen::Ref<const Eigen::VectorXd>& times);

	/// Return the index of the spline segment that covers time t
	unsigned int get_piece(double t) const;

	/// Evaluate position, velocity, and acceleration at time t
	void eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
		       double t) const;

	/// Evaluate the `diff`-th derivative at time t
	Eigen::VectorXd eval(double t, unsigned int diff = 0) const;

	/// Evaluate multiple time points
	Eigen::MatrixXd eval_multiple(const Eigen::Ref<const Eigen::VectorXd>& T,
				      unsigned int diff = 0) const;

	/// Get the start time of the spline
	double begin() const;

	/// Get the end time of the spline
	double end() const;
};
