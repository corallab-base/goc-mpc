#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include <Eigen/Dense>
#include <optional>
#include <vector>
#include <stdexcept>


struct CubicPiece {
	// Coefficients per dimension (size = dim)
	Eigen::VectorXd _a, _b, _c, _d;  // x(t) = d + t c + t^2 b + t^3 a, for t ∈ [0, tau]

	// Set from endpoint states and duration tau (> 0). Copies data.
	void set(const Eigen::Ref<const Eigen::VectorXd>& x0,
		 const Eigen::Ref<const Eigen::VectorXd>& v0,
		 const Eigen::Ref<const Eigen::VectorXd>& x1,
		 const Eigen::Ref<const Eigen::VectorXd>& v1,
		 double tau);

	// Evaluate position/velocity/acceleration; pass std::nullopt to skip.
	void eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
		       double t) const;

	// Evaluate the diff-th derivative, diff = 0,1,2. For diff>2, you may throw or extend.
	Eigen::VectorXd eval(double t, unsigned int diff = 0) const;

	// Dimension accessor
	inline int dim() const { return static_cast<int>(_d.size()); }
};

struct CubicSpline {
	// Invariants:
	//  - _times.size() >= 2 when initialized
	//  - _pieces.size() == _times.size() - 1
	//  - _times is strictly increasing (absolute knot times)
	//  - all pieces have the same dim == _dim
	std::vector<CubicPiece> _pieces;
	Eigen::VectorXd _times;   // absolute times t0 < t1 < ... < tN
	int _dim = -1;

	// ---------- Helpers / invariants ----------
	inline int dimension() const { return _dim; }
	inline int num_pieces() const { return static_cast<int>(_pieces.size()); }
	inline bool initialized() const { return _times.size() >= 2; }

	// Reset to empty
	void clear();

	// Initialize from waypoints/velocities and absolute times (copies data).
	// pts: (N, dim), vels: (N, dim), times: (N,), strictly increasing
	void set(const Eigen::Ref<const Eigen::MatrixXd>& pts,
		 const Eigen::Ref<const Eigen::MatrixXd>& vels,
		 const Eigen::Ref<const Eigen::VectorXd>& times);

	// Append K_new segments defined by pts/vels and durations or absolute times:
	// If 'times' is strictly increasing and times(0) > 0, interpret as durations
	// relative to end; otherwise document and enforce one convention consistently.
	// Recommended: interpret as durations relative to end (strictly increasing).
	void append(const Eigen::Ref<const Eigen::MatrixXd>& pts,
		    const Eigen::Ref<const Eigen::MatrixXd>& vels,
		    const Eigen::Ref<const Eigen::VectorXd>& times);

	// Return index k such that t ∈ [ _times[k], _times[k+1] ), clamped to [0, num_pieces()-1]
	unsigned int get_piece(double t) const;

	// Evaluate at global time t; pass std::nullopt to skip.
	void eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
		       std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
		       double t) const;

	// Convenience single-output evaluation (diff = 0: x, 1: xDot, 2: xDDot).
	Eigen::VectorXd eval(double t, unsigned int diff = 0) const;

	// Evaluate many times; returns (M × dim) matrix (one row per time).
	Eigen::MatrixXd eval_multiple(const Eigen::Ref<const Eigen::VectorXd>& T,
				      unsigned int diff = 0) const;

	// Start/end times (valid only if initialized)
	inline double begin() const {
		if (_times.size() == 0) throw std::runtime_error("CubicSpline is empty");
		return _times(0);
	}
	inline double end() const {
		if (_times.size() == 0) throw std::runtime_error("CubicSpline is empty");
		return _times(_times.size() - 1);
	}

};
