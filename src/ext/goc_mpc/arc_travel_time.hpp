#pragma once

#include <cmath>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <Eigen/Dense>

// One agent's travel time T(x, y) between two configurations, e.g. a learned
// NTField. GCSWaypointMPC's kMinMaxTravelTime/kAvgTravelTime objectives
// linearize it once per solve, at the centres of each arc's vertex sets.
//
// T reads the leading dim() = lower.size() columns of the agent's row; the
// trailing columns don't affect it.
struct ArcTravelTimeModel {
	// T's domain [lower, upper] (linearization points must lie inside it),
	// and an upper bound on T over that domain, which sizes the routing-time
	// big-M.
	Eigen::VectorXd lower, upper;
	double max_travel_time;

	ArcTravelTimeModel(Eigen::VectorXd lower, Eigen::VectorXd upper, double max_travel_time)
		: lower(std::move(lower)), upper(std::move(upper)), max_travel_time(max_travel_time) {
		if (this->lower.size() == 0)
			throw std::runtime_error("ArcTravelTimeModel: lower/upper must be non-empty");
		if (this->lower.size() != this->upper.size())
			throw std::runtime_error("ArcTravelTimeModel: lower and upper differ in size");
		if (!((this->lower.array() <= this->upper.array()).all()))
			throw std::runtime_error("ArcTravelTimeModel: lower must be <= upper");
		if (!(max_travel_time > 0.0) || !std::isfinite(max_travel_time))
			throw std::runtime_error("ArcTravelTimeModel: max_travel_time must be finite and > 0");
	}
	virtual ~ArcTravelTimeModel() = default;

	int dim() const { return lower.size(); }

	// T (n), dT/dX (n x dim), dT/dY (n x dim) at the rows of X, Y (n x dim).
	virtual std::tuple<Eigen::VectorXd, Eigen::MatrixXd, Eigen::MatrixXd> Linearize(
		const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) const = 0;
};
