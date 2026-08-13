#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>
#include <nanoflann.hpp>

// Thin wrapper around nanoflann's Eigen-matrix KD-tree adaptor, fixed to a
// COMPILE-TIME dimension (2 or 3) so nanoflann's inner distance loops get
// fully unrolled -- workspace_dim is always exactly 2 or 3 for the life of a
// graph (GraphOfConstraints enforces this at construction), so there's never
// a need for nanoflann's slower runtime-DIM (-1) mode. ObstacleSet picks
// PointCloudKdTree<2> or <3> at set_point_cloud() time, behind this
// dimension-erased base so ObstacleSet's own type stays undemanding on
// dimension, matching every other part of that class.
class PointCloudKdTreeBase {
public:
	virtual ~PointCloudKdTreeBase() = default;

	// Indices into the SAME point matrix this tree was built from (see
	// PointCloudKdTree<Dim>'s constructor) within `radius` of `center`.
	// `center` must be Dim-long. One call is a single O(log N)-ish tree
	// traversal -- meant to be called once per query region (e.g. once per
	// MPC cycle against the whole horizon's bounding region), not repeated
	// per ADMM inner iteration.
	virtual std::vector<int> radius_search(const Eigen::VectorXd& center, double radius) const = 0;
};

template <int Dim>
class PointCloudKdTree : public PointCloudKdTreeBase {
public:
	using Adaptor = nanoflann::KDTreeEigenMatrixAdaptor<Eigen::MatrixXd, Dim>;

	// `points` (N x Dim, one point per row) must outlive this tree -- the
	// underlying nanoflann adaptor stores a REFERENCE to it, not a copy
	// (mirrors ObstacleSet/GraphShortPathMPC's own pointer-storage
	// discipline elsewhere -- see those classes' own comments on why).
	// ObstacleSet satisfies this by owning `_point_cloud` as a stable member
	// and only ever (re)constructing the tree AFTER assigning new data into
	// it, never before.
	explicit PointCloudKdTree(const Eigen::MatrixXd& points)
		: _adaptor(Dim, std::cref(points)) {}

	std::vector<int> radius_search(const Eigen::VectorXd& center, double radius) const override {
		std::vector<nanoflann::ResultItem<Eigen::Index, double>> matches;
		nanoflann::SearchParameters params;
		// nanoflann's default L2 metric takes/returns SQUARED distances
		// throughout -- square the caller's (real) radius here so callers
		// never have to think about this.
		_adaptor.index_->radiusSearch(center.data(), radius * radius, matches, params);
		std::vector<int> indices;
		indices.reserve(matches.size());
		for (const auto& m : matches) {
			indices.push_back(static_cast<int>(m.first));
		}
		return indices;
	}

private:
	Adaptor _adaptor;
};
