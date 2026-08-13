#include "obstacle_set.hpp"

#include <stdexcept>

#include "point_cloud_kdtree.hpp"

ObstacleSet::ObstacleSet() = default;
ObstacleSet::~ObstacleSet() = default;

void ObstacleSet::set_point_cloud(const Eigen::MatrixXd& points, double margin) {
	if (points.rows() > 0 && points.cols() != 2 && points.cols() != 3) {
		throw std::runtime_error("ObstacleSet::set_point_cloud: points must be N x 2 or N x 3.");
	}
	// Assign the data FIRST, then (re)build the tree referencing this same
	// stable member -- see _point_cloud's own doc comment (obstacle_set.hpp)
	// for why the order matters.
	_point_cloud = points;
	_point_cloud_margin = margin;

	if (_point_cloud.rows() == 0) {
		_kdtree.reset();
		return;
	}
	const int d = static_cast<int>(_point_cloud.cols());
	if (d == 2) {
		_kdtree = std::make_unique<PointCloudKdTree<2>>(_point_cloud);
	} else {
		_kdtree = std::make_unique<PointCloudKdTree<3>>(_point_cloud);
	}
}

std::vector<int> ObstacleSet::query_point_cloud_radius(const Eigen::VectorXd& center, double radius) const {
	if (!_kdtree) {
		return {};
	}
	return _kdtree->radius_search(center, radius);
}
