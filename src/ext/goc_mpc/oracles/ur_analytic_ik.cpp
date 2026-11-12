#include "ur_analytic_ik.hpp"

#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace {

constexpr double kTwoPi = 2.0 * M_PI;
// acos arguments within this of [-1, 1] are rounding, not unreachability.
constexpr double kAcosSlack = 1e-9;
constexpr double kSingularWrist = 1e-8;

Eigen::Isometry3d Dh(double theta, double d, double a, double alpha) {
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	T.rotate(Eigen::AngleAxisd(theta, Eigen::Vector3d::UnitZ()));
	T.translate(Eigen::Vector3d(a, 0.0, d));
	T.rotate(Eigen::AngleAxisd(alpha, Eigen::Vector3d::UnitX()));
	return T;
}

bool Acos(double c, double* out) {
	if (!(std::abs(c) <= 1.0 + kAcosSlack)) return false;
	*out = std::acos(std::clamp(c, -1.0, 1.0));
	return true;
}

constexpr std::array<double, 6> kAlpha = {M_PI / 2, 0.0, 0.0, M_PI / 2, -M_PI / 2, 0.0};

Eigen::Isometry3d ChainFK(const UrDhParameters& dh, const Eigen::VectorXd& q) {
	const std::array<double, 6> d = {dh.d1, 0.0, 0.0, dh.d4, dh.d5, dh.d6};
	const std::array<double, 6> a = {0.0, dh.a2, dh.a3, 0.0, 0.0, 0.0};
	Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
	for (int i = 0; i < 6; ++i) T = T * Dh(q(i), d[i], a[i], kAlpha[i]);
	return T;
}

// Every branch of base_T_flange that the closed form reaches (up to 8),
// unwrapped and unchecked.
std::vector<Eigen::VectorXd> Branches(const UrDhParameters& dh, const Eigen::Isometry3d& T,
				      double theta6_if_singular) {
	const Eigen::Matrix3d R = T.linear();
	const Eigen::Vector3d p06 = T.translation();
	const Eigen::Vector3d p05 = p06 - dh.d6 * R.col(2);

	std::vector<Eigen::VectorXd> out;
	const double r = std::hypot(p05.x(), p05.y());
	double phi;
	if (r < 1e-12 || !Acos(dh.d4 / r, &phi)) return out;
	const double psi = std::atan2(p05.y(), p05.x());
	for (const double theta1 : {psi + phi + M_PI / 2, psi - phi + M_PI / 2}) {
		const double s1 = std::sin(theta1), c1 = std::cos(theta1);
		double theta5_pos;
		if (!Acos((p06.x() * s1 - p06.y() * c1 - dh.d4) / dh.d5, &theta5_pos)) continue;
		for (const double theta5 : {theta5_pos, -theta5_pos}) {
			const double s5 = std::sin(theta5);
			const double theta6 =
				std::abs(s5) < kSingularWrist
					? theta6_if_singular
					: std::atan2((-R(0, 1) * s1 + R(1, 1) * c1) / s5, (R(0, 0) * s1 - R(1, 0) * c1) / s5);
			const Eigen::Isometry3d t14 = Dh(theta1, dh.d1, 0.0, kAlpha[0]).inverse() * T *
						      (Dh(theta5, dh.d5, 0.0, kAlpha[4]) * Dh(theta6, dh.d6, 0.0, kAlpha[5])).inverse();
			const Eigen::Vector3d p14 = t14.translation();
			const double p14xy = std::hypot(p14.x(), p14.y());
			double theta3_pos;
			if (!Acos((p14xy * p14xy - dh.a2 * dh.a2 - dh.a3 * dh.a3) / (2.0 * dh.a2 * dh.a3), &theta3_pos))
				continue;
			for (const double theta3 : {theta3_pos, -theta3_pos}) {
				const double gamma = std::atan2(dh.a3 * std::sin(theta3), dh.a2 + dh.a3 * std::cos(theta3));
				const double theta2 = std::atan2(p14.y(), p14.x()) - gamma;
				const double theta234 = std::atan2(t14.linear()(1, 0), t14.linear()(0, 0));
				Eigen::VectorXd q(6);
				q << theta1, theta2, theta3, theta234 - theta2 - theta3, theta5, theta6;
				out.push_back(q);
			}
		}
	}
	return out;
}

// value's 2*pi-equivalent in [lower, upper] nearest reference, if any.
bool NearestEquivalent(double value, double reference, double lower, double upper, double* out) {
	if (!(lower <= upper)) return false;
	const double k_lo = std::ceil((lower - value) / kTwoPi), k_hi = std::floor((upper - value) / kTwoPi);
	if (k_lo > k_hi) return false;
	const double k = std::clamp(std::round((reference - value) / kTwoPi), k_lo, k_hi);
	*out = value + k * kTwoPi;
	return true;
}

}  // namespace

UrAnalyticIK::UrAnalyticIK(std::map<int, UrArm> arms) : _arms(std::move(arms)) {
	for (const auto& [agent, arm] : _arms)
		if (arm.lower.size() != 6 || arm.upper.size() != 6)
			throw std::invalid_argument(
				fmt::format("UrAnalyticIK: agent {}: joint limits need 6 entries", agent));
}

static const UrArm& ArmOf(const std::map<int, UrArm>& arms, int agent) {
	const auto it = arms.find(agent);
	if (it == arms.end()) throw std::out_of_range(fmt::format("UrAnalyticIK: no arm for agent {}", agent));
	return it->second;
}

Eigen::Isometry3d UrAnalyticIK::ForwardKinematics(int agent, const Eigen::VectorXd& q) const {
	const UrArm& arm = ArmOf(_arms, agent);
	return arm.world_T_base * ChainFK(arm.dh, q) * arm.flange_T_link;
}

std::vector<Eigen::VectorXd> UrAnalyticIK::Solve(int agent, const Eigen::Isometry3d& world_T_link,
						  const Eigen::VectorXd& reference, const Eigen::VectorXd& lower,
						  const Eigen::VectorXd& upper) const {
	const UrArm& arm = ArmOf(_arms, agent);
	if (reference.size() != 6)
		throw std::invalid_argument(
			fmt::format("UrAnalyticIK: agent {}: reference has {} entries, not 6", agent, reference.size()));
	Eigen::VectorXd lo = arm.lower, hi = arm.upper;
	if (lower.size() == 6) lo = lo.cwiseMax(lower);
	if (upper.size() == 6) hi = hi.cwiseMin(upper);
	const Eigen::Isometry3d base_T_flange = arm.world_T_base.inverse() * world_T_link * arm.flange_T_link.inverse();

	std::vector<Eigen::VectorXd> valid;
	for (const Eigen::VectorXd& raw : Branches(arm.dh, base_T_flange, reference(5))) {
		if (!raw.allFinite()) continue;
		const Eigen::Isometry3d fk = ForwardKinematics(agent, raw);
		if ((fk.translation() - world_T_link.translation()).lpNorm<Eigen::Infinity>() > kTolerance ||
		    (fk.linear() - world_T_link.linear()).lpNorm<Eigen::Infinity>() > kTolerance)
			continue;
		Eigen::VectorXd q(6);
		bool in_limits = true;
		for (int i = 0; i < 6 && in_limits; ++i)
			in_limits = NearestEquivalent(raw(i), reference(i), lo(i), hi(i), &q(i));
		if (!in_limits) continue;
		const bool duplicate = std::any_of(valid.begin(), valid.end(), [&](const Eigen::VectorXd& v) {
			return (v - q).lpNorm<Eigen::Infinity>() < 1e-9;
		});
		if (!duplicate) valid.push_back(q);
	}
	std::stable_sort(valid.begin(), valid.end(), [&](const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
		return (x - reference).squaredNorm() < (y - reference).squaredNorm();
	});
	return valid;
}

std::vector<Eigen::VectorXd> UrAnalyticIK::Solve(const IkRequest& request) const {
	const auto arm_it = _arms.find(request.agent);
	if (arm_it == _arms.end() || arm_it->second.link != request.link) return {};
	return Solve(request.agent, request.world_T_link, request.reference, request.lower, request.upper);
}
