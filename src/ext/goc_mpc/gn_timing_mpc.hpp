#pragma once

#include <map>
#include <set>
#include <vector>

#include <Eigen/Dense>

#include <drake/common/timer.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

// Gauss-Newton-SQP timing solver -- an alternative to GraphTimingMPC's
// Drake-MathematicalProgram/IPOPT(L-BFGS-only) approach for the same
// per-cycle timing problem: same decision variables (per-agent time deltas
// `tau` and interior tangent-space velocities `v`), same per-agent min-
// acceleration cost (compute_ctrl_cost's "psi"), same cross-agent
// LESS_THAN/EQUAL timing constraints (add_agent_interaction_constraints) --
// but solved with a hand-written Ipopt::TNLP (see gn_timing_mpc.cpp)
// instead of going through Drake's generic MathematicalProgram/IpoptSolver
// binding. No max_vel/max_acc hard bounds -- deliberately out of scope, not
// merely unimplemented (see this file's own history if that changes).
//
// The reason this exists as a SEPARATE class rather than a flag on
// GraphTimingMPC: Drake's IpoptSolver_NLP (drake/solvers/
// ipopt_solver_internal.cc) never implements eval_h, so IPOPT is always run
// with hessian_approximation=limited-memory (L-BFGS) through Drake, no
// matter what solver options are set -- there is no path to give IPOPT a
// better Hessian for a generic Drake Cost. This class links directly
// against the system libipopt (bypassing Drake's MathematicalProgram
// entirely for this one problem) specifically to supply the Gauss-Newton
// Hessian: psi is already exactly a sum of squared residuals
// (psi = ||sqrt(12/tau^3)*D||^2 + ||sqrt(1/tau)*V||^2, with D/V as in
// CubicConfigurationSpline::compute_ctrl_cost), so H_GN = J^T J (J = the
// residual Jacobian, which is cheap and exact here since x0/x1 are fixed
// waypoints from the upstream waypoint-solve phase -- only tau/v are free,
// and D/V are bilinear in them) is a far better local curvature model than
// L-BFGS's secant approximation, and is guaranteed PSD, unlike this
// problem's true Hessian (non-convex near the tau lower bound). See the
// design discussion this class came out of for the full argument.
//
// Scope of this first version, intentionally narrower than GraphTimingMPC's
// (throws rather than silently diverging from GraphTimingMPC's behavior for
// anything out of scope):
//   - cost: time_cost (linear) + time_cost2 (diagonal quadratic, both exact,
//     no GN approximation needed) + acceleration_cost * psi (GN-Hessian
//     approximated). energy_cost/arclength_cost/stability_cost from
//     GraphTimingMPC are NOT ported -- this class doesn't accept them.
//   - blocks: R, Torus, SO3Quat (same PositionDelta-based residual math as
//     GraphTimingMPC, reused verbatim via CubicConfigurationSpline::
//     PositionDelta/BlockPositionDelta<double>, evaluated ONCE since x's
//     are fixed constants at this phase -- so no autodiff is ever needed
//     through the wrap/quaternion-log branches themselves, only through
//     the tau/v-dependent D/V math built on top of that constant). SO3Mat
//     throws, matching BlockPositionDelta's own existing stub.
//   - no max_vel/max_acc: the only inequality/equality constraints are the
//     cross-agent LESS_THAN/EQUAL timing constraints, which are exactly
//     linear -- so eval_h's dropped lambda-weighted constraint-Hessian term
//     (see below) is exact, not an approximation, for every constraint
//     this class has.
//   - constraint curvature (lambda-weighted Hessian of g) is dropped from
//     eval_h entirely -- exact here (see above), not merely a standard
//     Gauss-Newton-SQP approximation, since every g row is linear.
//   - only the sparse (real-node-only) solve() path -- no solve_dense()
//     counterpart yet.
struct GnTimingMPC {
	const GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// Persistent output buffers -- same shapes/semantics as
	// GraphTimingMPC's own, so fill_cubic_splines/set_progressed_time/
	// get_next_taus/get_next_nodes are verbatim copies of that class's
	// logic (they only ever read these, never touch Drake).
	std::vector<Eigen::MatrixXd> _wps_list;
	std::vector<Eigen::MatrixXd> _vs_list;
	std::vector<Eigen::VectorXd> _time_deltas_list;
	std::vector<std::vector<int>> _agent_nodes_list;
	std::map<int, int> _agent_spline_length_map;

	double _time_cost;
	double _time_cost2;
	double _acceleration_cost;

	drake::SteadyTimer _timer;
	double _last_solve_time = 0.0;

	GnTimingMPC(const GraphOfConstraints& graph,
		    std::vector<CubicConfigurationSpline> splines,
		    double time_cost = 1e0,
		    double time_cost2 = 0e0,
		    double acceleration_cost = 1.0);

	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const std::vector<int>& remaining_vertices,
		   const Eigen::MatrixXd& waypoints,
		   const Eigen::VectorXi& assignments,
		   const Eigen::VectorXd& t_by_node = Eigen::VectorXd());

	int get_agent_spline_length(int agent) const;
	std::vector<int> get_agent_spline_nodes(int agent) const;

	std::set<int> set_progressed_time(double delta, double tau_cutoff);

	void fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				const Eigen::VectorXd& x0,
				const Eigen::VectorXd& v0) const;

	const std::vector<double> get_next_taus() const;
	const std::vector<int> get_next_nodes() const;

	const std::vector<Eigen::MatrixXd> &view_wps_list() const { return _wps_list; }
	const std::vector<Eigen::MatrixXd> &view_vs_list() const { return _vs_list; }
	const std::vector<Eigen::VectorXd> &view_time_deltas_list() const { return _time_deltas_list; }
	const std::vector<std::vector<int>> &view_agent_nodes_list() const { return _agent_nodes_list; }
	const std::map<int, int> &view_agent_spline_length_map() const { return _agent_spline_length_map; }
	const double get_last_solve_time() { return _last_solve_time; }
};
