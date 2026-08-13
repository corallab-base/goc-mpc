#pragma once

#include <vector>

#include <Eigen/Dense>

#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "obstacle_set.hpp"
#include "../configuration_spline.hpp"
#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

// ADMM-based short-horizon obstacle-avoidance MPC -- an alternative to
// GraphShortPathMPC's Drake-MathematicalProgram/NLopt approach for the
// use_hard_constraints=false case, built to structurally never fail AND be
// fast even with obstacle geometry (including large point clouds) that
// changes every cycle.
//
// The key structural fact this exploits: the smooth cost (tracking +
// velocity-tracking + acceleration-smoothing, identical to
// GraphShortPathMPC's items 1/1b/2) is separable PER AXIS, not just per
// agent -- every spatial axis has the SAME (2*num_steps x 2*num_steps)
// tridiagonal Hessian (interleaved [p_0,v_0,p_1,v_1,...]), only the linear
// term differs per axis. Obstacle avoidance is moved OUT of that system
// entirely and handled via closed-form projections in an outer ADMM loop:
//   x-update: solve the smooth QP (now with an added quadratic pull toward
//             each active obstacle's current target) -- an UNCONSTRAINED
//             quadratic minimization, i.e. one linear solve, not an
//             iterative NLP solver call.
//   z-update: project the just-solved position, per candidate obstacle,
//             onto that obstacle's "outside" region -- closed form for
//             spheres, boxes, and point-cloud points (treated as
//             zero-radius spheres), always well-defined, never fails.
//   u-update: standard scaled-dual-variable ADMM update.
// Because the SAME set of candidate obstacles applies to every axis of a
// given agent uniformly (see solve()'s own comment), the Hessian is
// factorized ONCE per agent per solve() call and reused across every axis
// AND every outer ADMM iteration -- only the right-hand side changes.
//
// Point-cloud obstacles (ObstacleSet::set_point_cloud) are pruned to a
// small per-agent candidate set via ONE KD-tree radius query per solve()
// call (not per ADMM iteration) -- see ObstacleSet::query_point_cloud_radius.
struct AdmmShortPathMPC {
	const GraphOfConstraints* _graph;
	unsigned int _num_steps, _num_agents, _dim;
	double _time_per_step;
	Eigen::VectorXd _times;

	const ObstacleSet* _obstacles;

	// ADMM penalty weight. Fixed (not adapted) across iterations/cycles --
	// see solve()'s own comment for why (adapting it would invalidate the
	// cached Hessian factorization).
	double _rho;
	// Fixed outer-iteration count -- see solve()'s own comment on why fixed
	// iterations (not residual-based stopping) is the right choice here.
	unsigned int _num_iterations;
	// How far beyond the reference trajectory's own bounding region to
	// widen the point-cloud KD-tree query -- must cover the max plausible
	// deflection, or a point just outside today's naive bounding box could
	// go unnoticed while still being close enough to matter after the
	// solve pulls the path toward it.
	double _point_cloud_query_margin;

	Eigen::MatrixXd _points;
	Eigen::MatrixXd _vels;
	bool _has_solved = false;

	drake::SteadyTimer _timer;
	double _last_solve_time = 0.0;

	// No default value for `obstacles` -- same lifetime discipline as
	// GraphShortPathMPC (see that class's own comment): storing a pointer
	// to a default-constructed temporary would dangle immediately.
	AdmmShortPathMPC(const GraphOfConstraints& graph,
			  unsigned int num_steps,
			  unsigned int num_agents,
			  unsigned int dim,
			  double time_per_step,
			  const ObstacleSet& obstacles,
			  double rho = 5.0,
			  unsigned int num_iterations = 8,
			  double point_cloud_query_margin = 1.0);

	// `var_assignments`/`remaining_vertices` are accepted but UNUSED --
	// present only so this class's solve() signature matches
	// GraphShortPathMPC::solve()'s exactly, letting GraphOfConstraintsMPC's
	// Python-side call site (goc_mpc.py's _solve_for_short_path) treat
	// self.short_path_mpc as either class without special-casing. They're
	// dead in GraphShortPathMPC::solve() too today (see that class's own
	// comment) -- vestigial scaffolding for graph-node-target handling
	// neither class currently reads.
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const Eigen::VectorXi& var_assignments,
		   const std::vector<int>& remaining_vertices,
		   const std::vector<CubicConfigurationSpline>& references);

	const Eigen::MatrixXd& view_points() { return _points; }
	const Eigen::MatrixXd& view_vels() { return _vels; }
	const Eigen::VectorXd& view_times() { return _times; }
	const ObstacleSet& view_obstacles() { return *_obstacles; }
	double get_last_solve_time() { return _last_solve_time; }
};
