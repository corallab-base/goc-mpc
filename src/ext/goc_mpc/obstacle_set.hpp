#pragma once

#include <vector>

#include <Eigen/Dense>

// Extensible registry of static/scene obstacle geometry, passed into
// GraphShortPathMPC (see graph_short_path_mpc.hpp). Deliberately a single,
// open-ended class rather than one flat array parameter per primitive kind
// (e.g. `obstacle_spheres`, `obstacle_boxes`, ...) -- adding a new kind
// later means adding one new `add_*` method here and one new ObstacleKind
// value; it never means touching GraphShortPathMPC's constructor signature
// again.
//
// Stage 1 (this file's initial version) only plumbs sphere obstacles
// through -- no cost/constraint anywhere reads `obstacles()` yet. A later
// stage adds the actual collision cost/constraint that consumes this data.
enum class ObstacleKind {
	kSphere,
	// kBox, kCapsule: added in a later stage, as new enum values only --
	// see add_sphere's doc comment below for the extension pattern.
};

// One registered obstacle. `params`'s layout depends on `kind`:
//   kSphere: [cx, cy, cz, radius]
struct Obstacle {
	ObstacleKind kind;
	Eigen::VectorXd params;
	double margin = 0.0;
};

// Holds registered obstacle geometry. GraphShortPathMPC stores a POINTER to
// an ObstacleSet (mirroring its existing `const GraphOfConstraints* _graph`
// pattern), not an owned copy -- so a caller can keep mutating the same
// long-lived ObstacleSet (register more obstacles, or a future
// set_point_cloud update from a driving thread) and have every subsequent
// solve() see the live data, without reconstructing the MPC. The caller is
// responsible for keeping the ObstacleSet alive for at least as long as any
// GraphShortPathMPC holding a pointer to it (see GraphOfConstraintsMPC's
// `self.obstacles = obstacles`, which exists for exactly this reason).
class ObstacleSet {
public:
	void add_sphere(const Eigen::Vector3d& center, double radius, double margin = 0.0) {
		Obstacle obstacle;
		obstacle.kind = ObstacleKind::kSphere;
		obstacle.params = Eigen::Vector4d(center.x(), center.y(), center.z(), radius);
		obstacle.margin = margin;
		_obstacles.push_back(std::move(obstacle));
	}

	// Future extension points, not built in this stage:
	//   void add_box(const Eigen::Vector3d& center,
	//                const Eigen::Vector3d& half_extents, double margin = 0.0);
	//   void add_capsule(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
	//                     double radius, double margin = 0.0);
	//   void set_point_cloud(const Eigen::MatrixXd& points, double margin = 0.0);
	//     -- bulk (N, 3) storage, kept separate from `_obstacles` below
	//     since a cloud has no natural single radius/margin per point and
	//     shouldn't be flattened into one Obstacle record per point
	//     (explicitly not a sphere-union approximation -- distance-to-cloud
	//     gets its own dedicated representation/cost in a later stage).

	const std::vector<Obstacle>& obstacles() const { return _obstacles; }

private:
	std::vector<Obstacle> _obstacles;
	// Eigen::MatrixXd _point_cloud;  -- added when set_point_cloud lands
};
