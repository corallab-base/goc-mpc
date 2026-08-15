#pragma once

// Shared Gauss-Newton timing-problem plumbing -- the flat decision-vector
// layout (per-agent tau/v ranges, per-segment constant PositionDelta data,
// linear interaction rows) and the objective assembly (value/gradient/
// Gauss-Newton-Hessian-triplets) GraphTimingMPC (graph_timing_mpc.cpp,
// solved via a trust-region Gauss-Newton loop with qpOASES QP subproblems)
// builds on. See graph_timing_mpc.hpp's own doc comment for the math
// itself (D/V residuals, why disp needs no autodiff, etc.).

#include <map>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"

namespace gn_timing {

constexpr double kTauMin = 0.01;
constexpr double kTauMax = 100.0;
// A generic "no bound" convention (used as qpOASES' box bound for an
// "unbounded" variable).
constexpr double kInf = 1e19;

// One agent's tau/v decision-variable ranges within the flat global vector.
struct AgentLayout {
	int tau_offset = 0;
	int v_offset = 0;
	int K = 0;  // agent_spline_length: number of segments == number of taus
};

// One cubic-Hermite segment's layout, plus the per-block constant
// PositionDelta ("disp") data compute_ctrl_cost's D/V residuals are built
// from. xJ/xJm1 are fixed constants at this phase (upstream waypoint-solve
// output), so `disp` is computed ONCE in double via
// CubicConfigurationSpline::PositionDelta -- see graph_timing_mpc.hpp's own
// doc comment for why that means no autodiff is ever needed through the
// wrap/quaternion-log branches themselves.
struct SegmentLayout {
	int tau_idx = 0;
	int v0_idx = -1;   // global column of this segment's v0 block start, or -1 if fixed
	int v1_idx = -1;   // ditto for v1
	Eigen::VectorXd v0_const;  // used when v0_idx < 0
	Eigen::VectorXd v1_const;  // used when v1_idx < 0
	Eigen::VectorXd disp;      // tangent_dim, PositionDelta(xJ, xJm1)
	const CubicConfigurationSpline* spline = nullptr;  // for block_offsets_
};

// One cross-agent LESS_THAN/EQUAL row (add_agent_interaction_constraints'
// counterpart) -- taus_i.head(depth+1)/taus_j.head(depth+1) are contiguous
// global-index ranges by construction (see the *ProblemLayout builders
// below), so the row's Jacobian is a fixed +1/-1 pattern -- exactly linear,
// no autodiff.
struct InteractionRow {
	int tau_offset_i = 0, count_i = 0;
	int tau_offset_j = 0, count_j = 0;
	double min_tau = 0.0;
	AgentInteraction::Type type;
};

struct ProblemLayout {
	int n = 0;
	std::vector<AgentLayout> agents;
	std::vector<SegmentLayout> segments;
	std::vector<InteractionRow> interactions;
	Eigen::VectorXd x_init;
};

Eigen::VectorXd CumsumWithZero(const Eigen::VectorXd& x, int n);

// Computes f/grad_f/Hessian-triplet contributions for the WHOLE objective
// (time_cost + time_cost2 + acceleration_cost * psi) at a given point `x`.
// Any of f_out/grad_out/hess_out may be null when the caller doesn't need
// that piece. time_cost/time_cost2 contribute EXACTLY (already linear/
// diagonal-quadratic -- no Gauss-Newton approximation needed).
//
// acceleration_cost's psi is the only approximated part: each block's D/V
// residual pair is differentiated locally (only tau and that block's own
// free v0/v1 are ever seeded -- different blocks within a segment never
// share v variables, and disp is a precomputed constant, itself already
// manifold-correct -- CubicConfigurationSpline::PositionDelta wrap-adjusts
// Torus and quaternion-logs SO3Quat, not a raw ambient subtraction), and
// every residual's contribution to f/grad/H_GN is accumulated into the
// GLOBAL arrays at that residual's own global variable indices. H_GN
// triplets may repeat the same (row, col) pair many times (e.g. every block
// within a segment touches that segment's shared tau; a shared interior v
// is touched by both the segment ending there and the one starting there)
// -- the caller is expected to sum duplicates
// (Eigen::SparseMatrix::setFromTriplets does this, or a caller building a
// dense Hessian can just accumulate directly), not treat this list as
// already-deduplicated.
void AssembleObjective(
	const ProblemLayout& layout,
	double acceleration_cost,
	double time_cost,
	double time_cost2,
	const Eigen::VectorXd& x,
	double* f_out,
	Eigen::VectorXd* grad_out,
	std::vector<Eigen::Triplet<double>>* hess_out);

// Builds the flat decision-vector layout (agent tau/v ranges, per-segment
// constant data, interaction rows, initial guess) for one solve() call.
// Sparse/real-node-only path: ordering/cross-agent interactions are
// resolved via graph.get_agent_paths, and each agent's waypoint positions
// looked up from the full `waypoints` matrix by resolved node id.
ProblemLayout BuildProblemLayout(
	const GraphOfConstraints& graph,
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<int>& remaining_vertices,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXi& assignments,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const Eigen::VectorXd& t_by_node,
	std::vector<Eigen::MatrixXd>* wps_list_out,
	std::vector<std::vector<int>>* agent_nodes_list_out,
	const std::vector<Eigen::MatrixXd>& prev_vs_list,
	const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
	const std::vector<std::vector<int>>& prev_agent_nodes_list,
	const std::map<int, int>& prev_agent_spline_length_map);

// Dense-waypoint counterpart to BuildProblemLayout: the caller has already
// resolved graph ordering + cross-agent interactions itself (e.g. via
// graph.get_agent_paths, possibly reindexed against a denser per-agent
// sequence with traced interior waypoints spliced in -- see
// GraphOfConstraints::reindex_agent_interactions) and expanded each agent's
// node sequence into `agent_dense_wps`/`agent_dense_node_ids` (one row per
// decision-variable segment target; node id -1 for a synthetic/traced
// interior point). Everything else (per-segment layout, warm start) is
// identical to BuildProblemLayout's own -- both share BuildProblemLayoutCore.
ProblemLayout BuildDenseProblemLayout(
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<Eigen::MatrixXd>& agent_dense_wps,
	const std::vector<std::vector<int>>& agent_dense_node_ids,
	const std::vector<AgentInteraction>& agent_interactions,
	const std::map<std::pair<int, int>, double>& edge_to_min_tau_map,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	std::vector<Eigen::MatrixXd>* wps_list_out,
	std::vector<std::vector<int>>* agent_nodes_list_out,
	const std::vector<Eigen::MatrixXd>& prev_vs_list,
	const std::vector<Eigen::VectorXd>& prev_time_deltas_list,
	const std::vector<std::vector<int>>& prev_agent_nodes_list,
	const std::map<int, int>& prev_agent_spline_length_map);

}  // namespace gn_timing
