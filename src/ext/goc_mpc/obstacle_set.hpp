#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include <drake/common/drake_assert.h>

// Forward-declared only -- nanoflann.hpp (and the resulting template
// instantiations) stay confined to obstacle_set.cpp, not pulled into every
// translation unit that includes this header (graph_short_path_mpc.hpp,
// goc_mpc.cpp, ...).
class PointCloudKdTreeBase;

// Extensible registry of static/scene obstacle geometry, passed into
// GraphShortPathMPC (see graph_short_path_mpc.hpp). Deliberately a single,
// open-ended class rather than one flat array parameter per primitive kind
// (e.g. `obstacle_spheres`, `obstacle_boxes`, ...) -- adding a new kind
// later means adding one new `add_*` method here and one new ObstacleKind
// value; it never means touching GraphShortPathMPC's constructor signature
// again.
//
enum class ObstacleKind {
	kSphere,
	kBox,
	// kCapsule: added later, as a new enum value only.
};

// One registered obstacle. `center`/`half_extents` are sized to whatever
// workspace dimension the caller registered them with (2 or 3) -- NOT
// hardcoded to 3, so a workspace_dim=2 graph never needs a throwaway z
// component. `params`'s layout depends on `kind`, both relative to that
// size `d` (== center.size()):
//   kSphere: [c_0, ..., c_(d-1), radius]        -- radius at index d
//   kBox:    [c_0, ..., c_(d-1), h_0, ..., h_(d-1)]  -- half-extents from index d
// The consumer (build_short_path_problem) reads these offsets relative to
// graph->workspace_dim, not a fixed constant -- it's the caller's job to
// register obstacles sized consistently with that graph's workspace_dim.
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
	// Both declared (not implicitly defaulted inline) and defined in
	// obstacle_set.cpp: `_kdtree` is a unique_ptr<PointCloudKdTreeBase>, and
	// that type is only forward-declared here. An inline `= default`
	// constructor's compiler-generated exception-unwind cleanup path (in
	// case a LATER member's constructor were to throw) needs `_kdtree`'s
	// destructor complete at the point the constructor is DEFINED, and an
	// implicitly-generated destructor needs it at every call site that
	// instantiates ~ObstacleSet() (e.g. pybind's machinery in goc_mpc.cpp)
	// -- both only see the forward declaration there. Defining both out of
	// line, in the .cpp where point_cloud_kdtree.hpp IS included, is the
	// standard fix for a Pimpl-style incomplete-type member.
	ObstacleSet();
	~ObstacleSet();
	// No copies (ObstacleSet is always used by pointer/reference -- see
	// this class's own doc comment -- and `_kdtree` isn't copyable anyway).
	ObstacleSet(const ObstacleSet&) = delete;
	ObstacleSet& operator=(const ObstacleSet&) = delete;

	// `center` may be 2- or 3-dimensional (matching the graph's
	// workspace_dim this obstacle is meant for) -- see Obstacle's own doc
	// comment.
	void add_sphere(const Eigen::VectorXd& center, double radius, double margin = 0.0) {
		Obstacle obstacle;
		obstacle.kind = ObstacleKind::kSphere;
		obstacle.params = Eigen::VectorXd(center.size() + 1);
		obstacle.params.head(center.size()) = center;
		obstacle.params(center.size()) = radius;
		obstacle.margin = margin;
		_obstacles.push_back(std::move(obstacle));
	}

	// `center`/`half_extents` must be the same size as each other, 2 or 3
	// (matching the graph's workspace_dim) -- see Obstacle's own doc comment.
	void add_box(const Eigen::VectorXd& center, const Eigen::VectorXd& half_extents, double margin = 0.0) {
		DRAKE_DEMAND(center.size() == half_extents.size());
		Obstacle obstacle;
		obstacle.kind = ObstacleKind::kBox;
		obstacle.params = Eigen::VectorXd(center.size() + half_extents.size());
		obstacle.params << center, half_extents;
		obstacle.margin = margin;
		_obstacles.push_back(std::move(obstacle));
	}

	// Drops every registered obstacle. Needed to MOVE a registered obstacle
	// (e.g. an interactively-dragged sphere): clear() + add_sphere(new
	// position) each update, rather than accumulating a new record per
	// frame -- there's no update_sphere(index, ...) yet, this is the
	// simplest correct way to represent "the sphere moved" with today's
	// API. Does NOT touch the point cloud (see set_point_cloud) -- that's a
	// bulk, separately-managed representation, not one more entry in
	// `_obstacles`.
	void clear() { _obstacles.clear(); }

	// Bulk point-cloud obstacle representation, kept separate from
	// `_obstacles` above (a cloud has no natural single radius/margin per
	// point and shouldn't be flattened into one Obstacle record per point --
	// explicitly not a sphere-union approximation). `points` is N x d, one
	// point per ROW, d == 2 or 3 matching the graph's workspace_dim. Every
	// call REPLACES the previously-registered cloud wholesale (no
	// incremental update API -- a fresh sensor scan each cycle is the
	// expected caller pattern) and rebuilds the KD-tree index used by
	// query_point_cloud_radius immediately: O(N log N), meant to happen
	// once per caller update (e.g. once per MPC cycle as new sensor data
	// arrives), not repeated during a solve's inner iterations.
	void set_point_cloud(const Eigen::MatrixXd& points, double margin = 0.0);

	// Indices (into point_cloud(), see below) of every point within
	// `radius` of `center` (Dim-long, Dim == point_cloud().cols()). Returns
	// empty if no cloud has been registered yet. ONE query against the
	// whole tree -- intended to be called once per solve with a
	// center/radius covering the ENTIRE reference trajectory's horizon
	// (plus margin), not once per horizon step: mirrors why box/sphere
	// obstacles above are checked at every step, not just step 0 -- a
	// nearby point that only matters at step 6 must still be in the
	// candidate set from step 0's perspective.
	std::vector<int> query_point_cloud_radius(const Eigen::VectorXd& center, double radius) const;

	const Eigen::MatrixXd& point_cloud() const { return _point_cloud; }
	double point_cloud_margin() const { return _point_cloud_margin; }
	bool has_point_cloud() const { return _point_cloud.rows() > 0; }

	// Future extension points, not built in this stage:
	//   void add_capsule(const Eigen::Vector3d& a, const Eigen::Vector3d& b,
	//                     double radius, double margin = 0.0);

	const std::vector<Obstacle>& obstacles() const { return _obstacles; }

private:
	std::vector<Obstacle> _obstacles;

	// N x d (d == 2 or 3), one point per row. Owned (not a pointer/view)
	// specifically so `_kdtree`'s nanoflann adaptor -- which stores a
	// REFERENCE to this matrix, not a copy -- always refers to stable
	// memory: set_point_cloud() assigns here FIRST, then rebuilds _kdtree
	// referencing this same member, never the other order.
	Eigen::MatrixXd _point_cloud;
	double _point_cloud_margin = 0.0;
	std::unique_ptr<PointCloudKdTreeBase> _kdtree;
};
