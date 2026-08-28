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
// tau: K contiguous columns at [tau_offset, tau_offset + K), one per base
// interval. v: this agent's interior-knot velocities, packed PER BLOCK --
// each block contributes one tangent_size-wide entry per interior knot it
// treats as a real knot; a block bridged past a knot gets NO v column there
// (one longer merged BlockSegment spans it instead). v_offset is just the
// start of that packed region; BlockSegment carries the resolved columns.
struct AgentLayout {
	int tau_offset = 0;
	int v_offset = 0;
	int K = 0;  // agent_spline_length: number of base intervals == number of taus
};

// One merged cubic-Hermite piece for ONE block of one agent's spline,
// spanning active knots k_lo -> k_hi (knot 0 == x0, knot k == this agent's
// k-th resolved node; k_lo/k_hi always land on knots the block treats as
// real). For an unbridged block k_hi == k_lo + 1 -- exactly one base
// interval, reducing to the old per-segment residual. For a block bridged
// past interior knot(s), k_hi - k_lo > 1 and this piece's duration is the
// SUM of the spanned base-interval taus (tau_indices), matching the single
// long Hermite piece CubicConfigurationSpline::set() builds for that block
// under the same active-knot mask. disp / v0_const / v1_const are this
// block's tangent slice only (tangent_size wide); v0_idx/v1_idx point at
// this block's own packed v column (component c is at v{0,1}_idx + c).
struct BlockSegment {
	int agent = 0;
	int block_idx = 0;              // index into spline->block_offsets_
	int k_lo = 0, k_hi = 0;         // this agent's knot indices (StoreResult reads k_hi)
	std::vector<int> tau_indices;   // global columns of the spanned base taus
	int v0_idx = -1;   // global column of this block's v at k_lo, or -1 if fixed
	int v1_idx = -1;   // ditto at k_hi
	Eigen::VectorXd v0_const;  // tangent_size, used when v0_idx < 0
	Eigen::VectorXd v1_const;  // tangent_size, used when v1_idx < 0
	Eigen::VectorXd disp;      // tangent_size, BlockPositionDelta(pos[k_hi], pos[k_lo])
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
	std::vector<BlockSegment> block_segments;
	std::vector<InteractionRow> interactions;
	Eigen::VectorXd x_init;
	// Per-agent (K+1) x num_blocks active-knot mask (1 == real knot for that
	// block, 0 == bridge the block past it; rows 0 and K forced to 1). The
	// SAME matrix GraphTimingMPC::fill_cubic_splines feeds to
	// CubicConfigurationSpline::set_block_active_mask -- built here so the
	// timing solve's merged BlockSegments and the output spline's bridged
	// knots can never disagree. Empty matrix for an agent with K == 0.
	std::vector<Eigen::MatrixXi> agent_block_active;
};

Eigen::VectorXd CumsumWithZero(const Eigen::VectorXd& x, int n);

// Computes f/grad_f/Hessian-triplet contributions for the WHOLE objective
// (time_cost + time_cost2 + acceleration_cost * psi) at a given point `x`.
// Any of f_out/grad_out/hess_out may be null when the caller doesn't need
// that piece. time_cost/time_cost2 contribute EXACTLY (already linear/
// diagonal-quadratic -- no Gauss-Newton approximation needed).
//
// acceleration_cost's psi is the only approximated part: iterated over
// layout.block_segments (one merged Hermite piece per block per bridged
// span), each block-segment's D/V residual pair is differentiated locally
// -- only the piece's own spanned tau(s) and its own free v0/v1 are ever
// seeded. A bridged block's piece spans MORE than one base tau, so all of
// them are seeded and the residual's curvature in their sum produces the
// cross-tau Gauss-Newton Hessian terms. disp is a precomputed constant,
// itself manifold-correct (CubicConfigurationSpline::BlockPositionDelta
// wrap-adjusts Torus and quaternion-logs SO3Quat, not a raw ambient
// subtraction). Every residual's contribution to f/grad/H_GN is accumulated
// into the GLOBAL arrays at its own global variable indices. H_GN triplets
// may repeat the same (row, col) pair many times (block-segments sharing a
// base tau; a shared interior v touched by the block-segment ending there
// and the one starting there) -- the caller is expected to sum duplicates
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

// Builds the flat decision-vector layout (agent tau/v ranges, per-
// block-segment constant data, interaction rows, initial guess, per-agent
// active-knot mask) for one solve() call. Sparse/real-node-only path:
// ordering/cross-agent interactions are resolved via graph.get_agent_paths,
// each agent's waypoint positions looked up from the full `waypoints`
// matrix by resolved node id, and which blocks each node actually pins
// (hence which knots the timing solve merges past) from
// graph.constrained_columns(node, var_assignments).
ProblemLayout BuildProblemLayout(
	const GraphOfConstraints& graph,
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<int>& remaining_vertices,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXi& var_assignments,
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
// interior point). Everything else (per-block-segment layout, warm start,
// active-knot mask) is identical to BuildProblemLayout's own -- both share
// BuildProblemLayoutCore. `graph`/`var_assignments` are still needed (only)
// to resolve constrained_columns for the real node ids in the dense
// sequence -- the SAME graph this timing instance was constructed with.
ProblemLayout BuildDenseProblemLayout(
	const GraphOfConstraints& graph,
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<Eigen::MatrixXd>& agent_dense_wps,
	const std::vector<std::vector<int>>& agent_dense_node_ids,
	const std::vector<AgentInteraction>& agent_interactions,
	const Eigen::VectorXi& var_assignments,
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
