#pragma once

#include <memory>
#include <unordered_map>
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

// Caller-supplied signed-distance FIELD for one agent's own local vicinity
// (Stage 3's point-cloud replacement for SqpShortPathMPC -- see the project
// plan's "backend-agnostic external SDF-grid/TSDF obstacle plan"). A grid
// VERTEX (i_0, ..., i_{d-1}) sits at world position `origin +
// [i_0*resolution(0), ..., i_{d-1}*resolution(d-1)]`; querying an arbitrary
// point in between multilinearly interpolates (bilinear for d=2, trilinear
// for d=3) the surrounding 2^d vertices -- see sqp_short_path_layout.hpp's
// QueryAgentSdfGrid, the only place this struct's buffers are actually read.
// `margin` is subtracted from the raw interpolated value exactly like
// Obstacle::margin is for spheres/boxes (SphereSdf/BoxSdf,
// sqp_short_path_layout.cpp) -- `values` is assumed to already be a genuine
// signed distance to the nearest obstacle SURFACE (negative inside), with
// no clearance baked in; `margin` adds that clearance uniformly.
//
// `gradient` is OPTIONAL. Empty (default) means "derive the gradient by
// differentiating the SAME multilinear interpolant `values` is queried
// through" -- guarantees value/gradient consistency, which this solver's
// trust-region ratio test depends on (an independently-supplied gradient
// that doesn't match the interpolated value's actual local slope is a
// subtle bug magnet there). Non-empty means the caller is instead handing
// over its own (e.g. cheaper, or backend-native) per-vertex gradient field,
// interpolated the same multilinear way as `values` -- the caller's own
// responsibility to keep consistent with `values`.
struct AgentSdfGrid {
	Eigen::VectorXd origin;      // workspace_dim
	Eigen::VectorXd resolution;  // workspace_dim, cell size per axis, > 0
	Eigen::VectorXi shape;       // workspace_dim, >= 2 per axis
	Eigen::VectorXd values;      // flat row-major, length prod(shape)
	Eigen::VectorXd gradient;    // flat row-major workspace_dim-vectors, length prod(shape)*d, or empty
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

	// Registers/replaces agent `agent`'s own local signed-distance grid
	// wholesale (same per-cycle-refresh contract as set_point_cloud: no
	// incremental update, a fresh crop each call is the expected caller
	// pattern) -- keyed by agent index, unlike every other obstacle kind
	// above, which is shared/global and matched to agents only via
	// distance pruning. A grid is registered PER AGENT instead because the
	// robots this library controls may be far enough apart that one shared
	// crop can't cover all of them; each agent gets its own independently
	// caller-built local map. Deliberately backend-agnostic (no
	// skfmm/nvblox/etc. anywhere in this repo's C++) -- the caller builds
	// `values` (and, optionally, `gradient`) however it wants and hands
	// over the raw buffer; see AgentSdfGrid's own doc comment for the
	// buffer layout and the value/gradient consistency tradeoff.
	//
	// `origin`/`resolution`/`shape` must all be the same size, 2 or 3
	// (matching the graph's workspace_dim this grid is meant for -- not
	// validated against workspace_dim here, same "caller's responsibility"
	// stance set_point_cloud already takes on its own `d`). `resolution`
	// entries must be positive; `shape` entries must be >= 2 (need at
	// least one cell per axis to interpolate at all). `values` must be
	// flat, row-major (last axis fastest, i.e. the same C-order a NumPy
	// array of shape `shape` flattens to), length `prod(shape)`.
	// `gradient` is optional (default empty, meaning "derive from
	// `values`" -- see AgentSdfGrid's own comment): if given, must be flat
	// workspace_dim-vectors in the SAME per-vertex order as `values`,
	// length `prod(shape) * d`.
	void set_agent_sdf_grid(int agent, const Eigen::VectorXd& origin, const Eigen::VectorXd& resolution,
				 const Eigen::VectorXi& shape, const Eigen::VectorXd& values,
				 const Eigen::VectorXd& gradient = Eigen::VectorXd(), double margin = 0.0);

	// Drops agent `agent`'s registered grid, if any. No-op if none
	// registered (mirrors clear()'s "drop everything" for the shared
	// obstacle list, scoped to just this one agent's grid instead).
	void clear_agent_sdf_grid(int agent) { _agent_sdf_grids.erase(agent); }

	// nullptr if agent `agent` has no registered grid.
	const AgentSdfGrid* agent_sdf_grid(int agent) const {
		const auto it = _agent_sdf_grids.find(agent);
		return it == _agent_sdf_grids.end() ? nullptr : &it->second;
	}

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

	// Sparse (most agents may have none) -- see set_agent_sdf_grid's own
	// comment for why this is keyed by agent index instead of being one
	// more entry in `_obstacles`.
	std::unordered_map<int, AgentSdfGrid> _agent_sdf_grids;
};
