#pragma once

#include <stdexcept>
#include <vector>

#include <drake/common/eigen_types.h>
#include <drake/common/extract_double.h>

#include <Eigen/Dense>

#include "edge_cost_functor.hpp"

// Axis-aligned regular grid, N-dimensional multilinear interpolant.
//
// Templated on scalar type so it compiles for both double and AutoDiffXd:
// corner *selection* always operates on drake::ExtractDoubleOrThrow(query(i))
// (grid indices are inherently non-differentiable at cell boundaries), while
// interpolation *weights* are computed in T-arithmetic so derivatives w.r.t.
// the query point propagate through ordinary +/-/* on AutoDiffXd. This makes
// the interpolant directly usable inside a Drake Cost functor (via
// MathematicalProgram::AddCost) without any Python round-trip for gradients.
//
// Implements EdgeCostFunctor by treating the grid as an arrival-time-style
// field from a fixed, implicit source (the natural output of solving the
// Eikonal equation once with Fast Marching Method): Evaluate(agent, w_a, w_b)
// looks up w_b only. agent/w_a are accepted for interface uniformity with
// other EdgeCostFunctor implementations (e.g. a future per-agent or
// two-point/displacement-based cost) but are unused here.
class RegularGridInterpolant : public EdgeCostFunctor {
public:
	// origin/spacing describe the grid's axis-aligned coordinate system;
	// shape is the number of grid points per axis; values is the field data
	// flattened in row-major order (matches numpy .ravel() default), with
	// values.size() == product(shape).
	RegularGridInterpolant(Eigen::VectorXd origin,
			       Eigen::VectorXd spacing,
			       std::vector<int> shape,
			       Eigen::VectorXd values)
		: origin_(std::move(origin)),
		  spacing_(std::move(spacing)),
		  shape_(std::move(shape)),
		  values_(std::move(values)) {

		if (origin_.size() != spacing_.size() ||
		    static_cast<size_t>(origin_.size()) != shape_.size()) {
			throw std::invalid_argument(
				"RegularGridInterpolant: origin, spacing, and shape must all "
				"have the same dimension");
		}
		for (int s : shape_) {
			if (s < 2) {
				throw std::invalid_argument(
					"RegularGridInterpolant: every axis must have at least 2 "
					"grid points");
			}
		}
		for (int i = 0; i < spacing_.size(); ++i) {
			if (spacing_(i) <= 0.0) {
				throw std::invalid_argument(
					"RegularGridInterpolant: spacing must be strictly positive");
			}
		}
		long expected_size = 1;
		for (int s : shape_) expected_size *= s;
		if (values_.size() != expected_size) {
			throw std::invalid_argument(
				"RegularGridInterpolant: values.size() must equal the product "
				"of shape");
		}
	}

	int dim() const { return static_cast<int>(shape_.size()); }

	template <typename T>
	T Interpolate(const Eigen::Ref<const drake::VectorX<T>>& query) const {
		const int d = dim();
		if (query.size() != d) {
			throw std::invalid_argument(
				"RegularGridInterpolant::Interpolate: query dimension mismatch");
		}

		// Per-axis: continuous fractional grid coordinate, clamped to
		// [0, shape_(i) - 1], base cell index, and fractional offset in [0, 1].
		std::vector<int> base(d);
		std::vector<T> frac(d);
		for (int i = 0; i < d; ++i) {
			const double q = drake::ExtractDoubleOrThrow(query(i));
			double g = (q - origin_(i)) / spacing_(i);
			const double g_max = static_cast<double>(shape_[i] - 1);
			if (g < 0.0) g = 0.0;
			if (g > g_max) g = g_max;

			int b = static_cast<int>(std::floor(g));
			if (b > shape_[i] - 2) b = shape_[i] - 2;
			base[i] = b;

			// Fractional part kept in T so AutoDiffXd derivatives w.r.t.
			// query(i) propagate through the corner blend below. Clamped
			// query points (g pinned to an endpoint) correctly yield zero
			// sensitivity beyond the grid boundary.
			T t_g = (query(i) - T(origin_(i))) / T(spacing_(i));
			T t_frac = t_g - T(static_cast<double>(b));
			if (drake::ExtractDoubleOrThrow(t_frac) < 0.0) t_frac = T(0.0);
			if (drake::ExtractDoubleOrThrow(t_frac) > 1.0) t_frac = T(1.0);
			frac[i] = t_frac;
		}

		// Blend the 2^d corners of the containing cell.
		T result(0.0);
		const int num_corners = 1 << d;
		for (int corner = 0; corner < num_corners; ++corner) {
			T weight(1.0);
			long flat_index = 0;
			long stride = 1;
			for (int i = d - 1; i >= 0; --i) {
				const bool hi = (corner >> i) & 1;
				weight = weight * (hi ? frac[i] : (T(1.0) - frac[i]));
				const int idx_i = base[i] + (hi ? 1 : 0);
				flat_index += static_cast<long>(idx_i) * stride;
				stride *= shape_[i];
			}
			result += weight * T(values_(flat_index));
		}
		return result;
	}

	double Evaluate(int /*agent*/,
			const Eigen::VectorXd& /*w_a*/,
			const Eigen::VectorXd& w_b) const override {
		return Interpolate<double>(w_b);
	}

	drake::AutoDiffXd Evaluate(int /*agent*/,
				   const drake::VectorX<drake::AutoDiffXd>& /*w_a*/,
				   const drake::VectorX<drake::AutoDiffXd>& w_b) const override {
		return Interpolate<drake::AutoDiffXd>(w_b);
	}

private:
	Eigen::VectorXd origin_;
	Eigen::VectorXd spacing_;
	std::vector<int> shape_;
	Eigen::VectorXd values_;
};
