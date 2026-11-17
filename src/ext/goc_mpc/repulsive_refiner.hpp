#pragma once

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "agent_collision_model.hpp"
#include "obstacle_set.hpp"
#include "spacetime_repulsion.hpp"

struct GraphOfConstraints;

// Refines every agent's timing-layer spline against the space-time repulsion
// of spacetime_repulsion.hpp: inter-agent repulsion between the agents'
// collision spheres, and obstacle repulsion against an ObstacleSet.
//
// The decision variables are the splines' own knots -- positions, velocities
// and times, minus whatever each agent's freeze masks hold. Each spline is
// sampled at fixed fractions of its knot intervals (so the samples move with
// the knots and a squeezed interval keeps its samples), and parked after its
// last knot until a shared horizon. Every sample goes through the agent's
// collision model (the graph's registered sphere model, or a single sphere at
// q[:workspace_dim]), and the energy's gradient with respect to the sphere
// centres and sample times is pulled back through the sphere Jacobians and the
// cubic Hermite pieces to the knots. An optional speed penalty keeps each
// block's speed under a limit. R and Torus blocks only, with per-block knot bridging
// (`active`), matching CubicConfigurationSpline::eval.
//
// The descent is L-BFGS with an Armijo backtracking line search; each step's
// largest knot change is capped at `max_step`, and knot times are pushed
// forward to keep every gap at least `min_knot_dt`. It stops as soon as the
// measured clearances (swept between agents over every sample interval, at
// the samples against obstacles) reach `target_clearance` -- before the first
// step when they already do -- or when the energy is zero, the line search
// fails, or after `max_iterations`.

namespace repulsive_refinement {

// One agent's spline knots, as CubicConfigurationSpline::set takes them.
struct AgentKnots {
	Eigen::VectorXd times;      // n, strictly increasing
	Eigen::MatrixXd wps;        // n x ambient_dim
	Eigen::MatrixXd vs;         // n x tangent_dim
	Eigen::MatrixXi active;     // n x num_blocks (empty: all active)
	// Which rows may move (empty: none).
	Eigen::Matrix<bool, Eigen::Dynamic, 1> wps_free, vs_free, times_free;
};

struct Options {
	spacetime_repulsion::Kernel kernel;
	double repulsion_weight = 1.0;
	double obstacle_weight = 1.0;
	// Speed penalty: speed_weight * integral over time of
	// max(0, |v_b| - max_speed)^2, summed over every spline block b (|v_b| the
	// norm of the block's velocity), sampled at the knot-interval samples.
	// speed_weight 0 disables it.
	double speed_weight = 0.0;
	double max_speed = 0.0;
	// Initial sample spacing, s: each knot interval gets samples at fixed
	// fractions of it, about this far apart, so they move with the knots.
	double sample_dt = 0.1;
	// After its last knot each agent is parked until the latest initial end
	// time plus this, s; no knot time may move past it.
	double horizon_margin = 2.0;
	int max_iterations = 100;
	double target_clearance = 0.02;
	double max_step = 0.1;
	// Step metric for (positions, velocities, times).
	Eigen::Vector3d metric_scale = Eigen::Vector3d(1.0, 1.0, 0.25);
	double min_knot_dt = 0.01;
	int memory = 8;
	int max_backtracks = 12;
	double armijo = 1e-4;
	// Sphere radius of agents with no registered collision model, per agent
	// (empty: 0).
	Eigen::VectorXd agent_radii;
};

struct Report {
	bool moved = false;
	int iterations = 0;
	int evaluations = 0;
	// NaN when it stops before evaluating the energy (nothing to refine).
	double energy_initial = 0.0, energy_final = 0.0;
	spacetime_repulsion::Clearance clearance_initial{}, clearance_final{};
	int pairs_initial = 0, pairs_final = 0;
	double time = 0.0;  // s
	// Of `time`: sampling the splines through the collision models, the
	// energy (and its gradient w.r.t. the samples), pulling that gradient
	// back to the knots, and measuring clearance.
	double time_sampling = 0.0, time_energy = 0.0, time_pullback = 0.0, time_clearance = 0.0;
	std::string stop_reason;
};

class Refiner {
public:
	// `obstacles` may be null, and is read by pointer on every Refine.
	Refiner(const GraphOfConstraints& graph, const ObstacleSet* obstacles, Options options);
	~Refiner();

	// Refines `knots` (one entry per agent, graph order) in place.
	Report Refine(std::vector<AgentKnots>& knots) const;

	// The weighted energy at `knots`, and its gradient over the free knot
	// entries (per agent: free position rows, then free velocity rows, then
	// free times, each row's entries in order).
	std::pair<double, Eigen::VectorXd> EnergyAndGradient(const std::vector<AgentKnots>& knots) const;

	const Options& options() const { return _options; }

private:
	Report Run(std::vector<AgentKnots>& knots, double* energy_out, Eigen::VectorXd* gradient_out) const;

	const GraphOfConstraints* _graph;
	const ObstacleSet* _obstacles;
	Options _options;
	std::vector<std::unique_ptr<sqp_short_path::AgentCollisionModel>> _models;
	std::vector<Eigen::VectorXd> _radii;
};

}  // namespace repulsive_refinement
