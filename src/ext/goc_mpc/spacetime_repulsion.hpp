#pragma once

#include <vector>

#include <Eigen/Core>

#include "obstacle_set.hpp"

// Space-time repulsion between agents' collision spheres, and between those
// spheres and the scene's obstacles.
//
// Each sphere's centre, sampled on a shared time grid, traces a polyline in
// space-time, gamma(t) = (c(t), time_scale * t). Two agents repel through the
// tangent-point kernel of Repulsive Curves (Yu, Schumacher and Crane), in the
// edge-midpoint discretization of Repulsion's near-field kernel
// (repulsive-shells/deps/repulsion, TangentPointEnergy_NearFieldKernel): an
// edge pair (I of agent a, J of agent b), with v the vector between their
// midpoints, scores
//
//     l_I * l_J * (sin(theta_I)^alpha + sin(theta_J)^alpha) * radial(|v| - (r_I + r_J)),
//
// theta_X the angle between edge X's unit tangent and v. The kernel's
// |v|^(alpha - beta) is replaced by `radial` of the gap between the two
// spheres, so the barrier sits at contact rather than at coincident centres:
// the windowed power g^-(beta - alpha) (1 - (g / cutoff)^2)^2 for relax < g <
// cutoff, zero from the cutoff on, and below relax the quadratic with matching
// value, slope and curvature, which stays finite through contact and overlap.
// It is normalized to force 1 at reference_gap. Only pairs of different agents
// count: an agent's own spheres never repel each other.
//
// An obstacle scores each edge as l_I * radial(sdf(midpoint) - r_I), summed
// over obstacles. Both totals are divided by time_scale, so for slow motion
// they integrate over time.
//
// The radial factor vanishes past the cutoff, so a pair can only interact when
// their midpoints are within cutoff + r_I + r_J in space-time; that bounds the
// time gap, and each agent pair's search runs over a band of their samples.
// Agents need not share sample times.

namespace spacetime_repulsion {

struct Kernel {
	double alpha = 2.0;
	double beta = 4.5;
	double time_scale = 1.0;     // m/s
	double cutoff = 0.3;         // sphere gap, m
	double relax = 0.05;         // quadratic below this gap, m
	double reference_gap = 0.1;  // radial force is 1 here
};

void CheckKernel(const Kernel& kernel);

// radial(gap) and its derivative.
struct Radial {
	explicit Radial(const Kernel& kernel);
	double value(double gap, double* slope) const;

	double p, cutoff, relax, h0, h1, h2, inv_force;
};

// One agent's sampled spheres: centers(s * num_spheres + k, :) is sphere k at
// sample s, in workspace coordinates, at times(s) (increasing).
struct AgentSamples {
	Eigen::VectorXd times;    // samples
	Eigen::MatrixXd centers;  // (samples * num_spheres) x workspace_dim
	Eigen::VectorXd radii;    // num_spheres
};

struct Clearance {
	// Smallest gap between spheres of different agents, with each sphere
	// moving linearly between its own samples, at every instant either agent
	// has a sample and in between; and smallest sphere-to-obstacle gap at the
	// samples (both +inf when nothing is measured).
	double inter_agent;
	double obstacle;
};

struct Result {
	double inter_agent = 0.0, obstacle = 0.0;  // energies, before weights
	// d(weighted total)/d(centers) and d(weighted total)/d(times), per agent,
	// in the layouts of centers and times (empty unless asked for).
	std::vector<Eigen::MatrixXd> gradient;
	std::vector<Eigen::VectorXd> time_gradient;
	int pairs = 0;  // edge pairs inside the cutoff
};

// `obstacle_weight` scales the obstacle term in the gradient (the energies
// are returned unweighted).
Result Evaluate(const std::vector<AgentSamples>& agents, const ObstacleSet* obstacles, const Kernel& kernel,
		double obstacle_weight, bool gradient);

Clearance MeasureClearance(const std::vector<AgentSamples>& agents, const ObstacleSet* obstacles);

}  // namespace spacetime_repulsion
