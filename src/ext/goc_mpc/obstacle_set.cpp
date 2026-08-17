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

void ObstacleSet::set_agent_sdf_grid(int agent, const Eigen::VectorXd& origin, const Eigen::VectorXd& resolution,
				      const Eigen::VectorXi& shape, const Eigen::VectorXd& values,
				      const Eigen::VectorXd& gradient, double margin) {
	const int d = static_cast<int>(origin.size());
	if (d != 2 && d != 3) {
		throw std::runtime_error("ObstacleSet::set_agent_sdf_grid: origin must be 2- or 3-dimensional.");
	}
	if (resolution.size() != d || shape.size() != d) {
		throw std::runtime_error(
			"ObstacleSet::set_agent_sdf_grid: origin/resolution/shape must all be the same size.");
	}
	if ((resolution.array() <= 0.0).any()) {
		throw std::runtime_error("ObstacleSet::set_agent_sdf_grid: resolution entries must be positive.");
	}
	if ((shape.array() < 2).any()) {
		throw std::runtime_error(
			"ObstacleSet::set_agent_sdf_grid: shape entries must be >= 2 (need at least one cell "
			"per axis).");
	}
	const long num_vertices = shape.cast<long>().prod();
	if (values.size() != num_vertices) {
		throw std::runtime_error("ObstacleSet::set_agent_sdf_grid: values must have prod(shape) entries.");
	}
	if (gradient.size() != 0 && gradient.size() != num_vertices * d) {
		throw std::runtime_error(
			"ObstacleSet::set_agent_sdf_grid: gradient must be empty (derive from values) or have "
			"prod(shape)*d entries.");
	}
	AgentSdfGrid grid;
	grid.origin = origin;
	grid.resolution = resolution;
	grid.shape = shape;
	grid.values = values;
	grid.gradient = gradient;
	grid.margin = margin;
	_agent_sdf_grids[agent] = std::move(grid);
}
