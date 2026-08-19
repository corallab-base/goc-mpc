#pragma once

#include <algorithm>
#include <vector>

#include <Eigen/Dense>

#include "obstacle_set.hpp"

// Closed-form "project a workspace point outside an obstacle, by at least
// its margin" primitive, used by GraphShortPathMPC's fast-path safety-
// projection final pass. See project_out's own doc comment for the actual
// sphere/box math.

// One obstacle candidate in workspace-dim coordinates, dimension-agnostic
// (2 or 3), unified across ObstacleSet's spheres/boxes AND pruned
// point-cloud points (each point-cloud point becomes a zero-radius sphere --
// Euclidean distance is symmetric, so "distance from robot to this fixed
// point" needs no different formula than "distance to a sphere's center").
struct Candidate {
	Eigen::VectorXd center;        // workspace_dim
	Eigen::VectorXd half_extents;  // box only, empty otherwise
	double radius = 0.0;           // sphere/point
	double margin = 0.0;
	bool is_box = false;
};

// Closed-form projection of `p` onto "outside `c`, by at least its margin" --
// always well-defined (mod the ignorable exact-center degenerate case).
inline Eigen::VectorXd project_out(const Eigen::VectorXd& p, const Candidate& c) {
	if (c.is_box) {
		// Isotropic margin approximated as a uniform per-axis expansion of
		// the box: exact on the faces, slightly conservative at
		// edges/corners relative to the true (rounded) isotropic-offset
		// surface GraphShortPathMPC's hard-constraint sdf uses -- chosen so
		// the projection stays a closed-form O(workspace_dim) op.
		const Eigen::VectorXd he = c.half_extents.array() + c.margin;
		const Eigen::VectorXd lo = c.center - he;
		const Eigen::VectorXd hi = c.center + he;
		const bool inside = (p.array() >= lo.array()).all() && (p.array() <= hi.array()).all();
		if (!inside) {
			// Already outside the margin-expanded box, i.e. already
			// FEASIBLE -- a no-op. NOT a clamp-to-box-surface: that formula
			// finds the nearest point ON the box (useful for
			// collision/contact queries), which is the wrong direction here
			// and would pull an already-clear point BACK toward the
			// obstacle regardless of how far away it already is.
			return p;
		}
		// Interior: push out along the LEAST-penetrated axis (shortest exit
		// path) -- same argmax-of-per-axis-excess logic as
		// GraphShortPathMPC's box sdf/nudge (graph_short_path_mpc.cpp).
		const Eigen::VectorXd diff = p - c.center;
		const Eigen::VectorXd excess = diff.cwiseAbs() - he;  // all <= 0 here
		int kstar = 0;
		excess.maxCoeff(&kstar);
		Eigen::VectorXd result = p;
		result(kstar) = c.center(kstar) + (diff(kstar) >= 0.0 ? 1.0 : -1.0) * he(kstar);
		return result;
	} else {
		const double R = c.radius + c.margin;
		const Eigen::VectorXd diff = p - c.center;
		const double d = diff.norm();
		if (d >= R) {
			return p;
		}
		if (d < 1.0e-9) {
			// Degenerate (p exactly at the obstacle's center) -- push along
			// an arbitrary fixed axis. Ignorable by design: obstacles are
			// assumed to never sit exactly on the path's centerline.
			Eigen::VectorXd dir = Eigen::VectorXd::Zero(p.size());
			dir(0) = 1.0;
			return c.center + R * dir;
		}
		return c.center + (R / d) * diff;
	}
}

// A trajectory's own bounding sphere in workspace coordinates -- centered
// on its mean position, radius covering every sample point exactly.
// Shared by gather_candidates' point-cloud query below and
// GraphShortPathMPC's distance-based obstacle/inter-agent-pair QP-row
// pruning (sqp_short_path_layout.hpp's PruneObstaclesByDistance/
// PruneAgentPairsByDistance) -- same "one bounding sphere per agent per
// solve() call" shape, factored out so both can never silently diverge.
struct BoundingSphere {
	Eigen::VectorXd center;
	double radius;
};

inline BoundingSphere TrajectoryBoundingSphere(const Eigen::MatrixXd& ref_points_workspace) {
	const Eigen::VectorXd center = ref_points_workspace.colwise().mean().transpose();
	double max_dist = 0.0;
	for (int i = 0; i < ref_points_workspace.rows(); ++i) {
		max_dist = std::max(max_dist, (ref_points_workspace.row(i).transpose() - center).norm());
	}
	return BoundingSphere{center, max_dist};
}

// Every registered sphere/box (unconditionally -- cheap, small counts) plus,
// if a point cloud is registered, every cloud point within ONE
// bounding-sphere query covering the agent's WHOLE reference trajectory (not
// just its current position) + `query_margin`. One KD-tree query per agent
// per solve() call, not per ADMM/SQP iteration -- see
// ObstacleSet::query_point_cloud_radius's own comment.
inline std::vector<Candidate> gather_candidates(const ObstacleSet& obstacles,
						 const Eigen::MatrixXd& ref_points_workspace,
						 int workspace_dim, double query_margin) {
	std::vector<Candidate> candidates;
	for (const Obstacle& obstacle : obstacles.obstacles()) {
		Candidate c;
		c.center = obstacle.params.segment(0, workspace_dim);
		c.margin = obstacle.margin;
		if (obstacle.kind == ObstacleKind::kSphere) {
			c.is_box = false;
			c.radius = obstacle.params(workspace_dim);
		} else {
			c.is_box = true;
			c.half_extents = obstacle.params.segment(workspace_dim, workspace_dim);
		}
		candidates.push_back(std::move(c));
	}

	if (obstacles.has_point_cloud()) {
		const BoundingSphere traj_sphere = TrajectoryBoundingSphere(ref_points_workspace);
		const Eigen::VectorXd& center = traj_sphere.center;
		const double query_radius = traj_sphere.radius + query_margin;
		std::vector<int> indices = obstacles.query_point_cloud_radius(center, query_radius);
		// Defensive cap: per-candidate work (RHS assembly + projection) is
		// O(num_candidates) per step per iteration, so a pathological query
		// (e.g. a dense sensor cloud with a large nearby surface) could
		// otherwise blow the solve-time budget even though a typical pruned
		// cluster is small (tens to low hundreds). nanoflann's radiusSearch
		// returns results SORTED by distance (nearest first) by default, so
		// truncating is free -- and correct, since the nearest points are
		// what matter most for avoidance.
		constexpr size_t kMaxPointCandidates = 300;
		if (indices.size() > kMaxPointCandidates) {
			indices.resize(kMaxPointCandidates);
		}
		const Eigen::MatrixXd& cloud = obstacles.point_cloud();
		const double point_margin = obstacles.point_cloud_margin();
		for (int idx : indices) {
			Candidate c;
			c.center = cloud.row(idx).transpose();
			c.radius = 0.0;
			c.margin = point_margin;
			c.is_box = false;
			candidates.push_back(std::move(c));
		}
	}
	return candidates;
}
