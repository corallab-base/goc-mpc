#pragma once

#include "initialization_oracle.hpp"

#include <Eigen/Geometry>

#include <map>
#include <string>
#include <vector>

// Standard DH parameters of a UR arm (joint i: Rz(theta_i) Tz(d_i) Tx(a_i)
// Rx(alpha_i), alpha = (pi/2, 0, 0, pi/2, -pi/2, 0)).
struct UrDhParameters {
	double d1 = 0.0, a2 = 0.0, a3 = 0.0, d4 = 0.0, d5 = 0.0, d6 = 0.0;
};

// One UR arm as the oracle sees it: world_T_base places the DH base frame,
// flange_T_link puts `link` (the graph's FK link) relative to the DH flange,
// and [lower, upper] bounds the joints.
struct UrArm {
	UrDhParameters dh;
	Eigen::Isometry3d world_T_base = Eigen::Isometry3d::Identity();
	Eigen::Isometry3d flange_T_link = Eigen::Isometry3d::Identity();
	std::string link;
	Eigen::VectorXd lower, upper;
};

// The closed-form UR inverse kinematics, as Lazy-DaSH's transition sampler
// draws its pick/place anchors: every branch (shoulder x wrist x elbow, up to
// 8) of the link's pose that this arm's own FK reproduces and that fits the
// joint limits -- the arm's and the request's both -- each joint taken at its
// 2*pi-equivalent nearest the reference, nearest the reference first. A
// request for another link, or for an agent with no arm, gets none.
//
// At a wrist singularity (sin(theta5) ~ 0) only theta4 + theta6 is
// determined; theta6 is then taken from the reference.
class UrAnalyticIK : public IkSolver {
public:
	// Per agent id.
	explicit UrAnalyticIK(std::map<int, UrArm> arms);

	std::vector<Eigen::VectorXd> Solve(const IkRequest& request) const override;
	bool complete() const override { return true; }

	// The valid solutions for `agent`'s link at world_T_link within the
	// arm's limits and [lower, upper] (when given), nearest `reference`
	// first.
	std::vector<Eigen::VectorXd> Solve(int agent, const Eigen::Isometry3d& world_T_link,
					   const Eigen::VectorXd& reference, const Eigen::VectorXd& lower = {},
					   const Eigen::VectorXd& upper = {}) const;

	// world_T_link at q under the oracle's model of `agent`.
	Eigen::Isometry3d ForwardKinematics(int agent, const Eigen::VectorXd& q) const;

	const std::map<int, UrArm>& arms() const { return _arms; }

	// A branch is kept when its FK reproduces the target to within this
	// (metres, and max rotation-matrix entry).
	static constexpr double kTolerance = 1e-6;

private:
	std::map<int, UrArm> _arms;
};
