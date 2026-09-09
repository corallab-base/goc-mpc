#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

// Configuration-dependent workspace-sphere collision geometry for
// GraphShortPathMPC (v2 plan Stage 3). Each agent gets an AgentCollisionModel
// mapping that agent's configuration to the set of workspace spheres its body
// occupies, with each sphere centre's Jacobian w.r.t. that agent's own
// tangent step -- the generalization of the solver's original
// `fk(q) = q[:workspace_dim]`, single-point-per-agent assumption.
//
// The zero-config case (no model registered) is TrivialCollisionModel: one
// sphere, centre = q[:workspace_dim], radius 0, Jacobian = [I_wd | 0]. Its
// `Eval` reproduces the old fast path exactly, so a solver with every agent
// trivial is byte-identical to before this stage. `is_trivial()` additionally
// gates the closed-form safety-projection passes in graph_short_path_mpc.cpp,
// which have no articulated-FK analogue ("solve fk(q) = target" is IK).
namespace sqp_short_path {

// One workspace collision sphere at a given configuration: world-frame
// centre, radius, and the centre's Jacobian w.r.t. the OWNING agent's
// tangent step (workspace_dim x that agent's tangent_dim). The Jacobian's
// column layout matches the agent's own CubicConfigurationSpline tangent
// ordering -- i.e. column j is d(centre)/d(delta_j) for that agent, which
// the constraint-row assembly scatters onto flat decision-vector index
// IdxP(agent_axis_offset + j, step).
struct WorkspaceSphere {
	Eigen::Vector3d center = Eigen::Vector3d::Zero();  // only leading workspace_dim entries used
	double radius = 0.0;
	// workspace_dim rows, agent_tangent_dim cols.
	Eigen::MatrixXd jac;
};

class AgentCollisionModel {
   public:
	virtual ~AgentCollisionModel() = default;

	// `q_ambient` is this agent's own ambient configuration row (the caller
	// slices `ambient_dim()` columns starting at that agent's ambient
	// offset, e.g. points.row(i).segment(off, model.ambient_dim())).
	// Returns one WorkspaceSphere per body sphere, in a fixed order for the
	// model's lifetime. Called on the SQP hot path (every outer iteration x
	// every horizon step) -- must be Python-free and allocation-light
	// (feedback_cpp_native_cost_functors).
	virtual std::vector<WorkspaceSphere> Eval(
		const Eigen::Ref<const Eigen::VectorXd>& q_ambient) const = 0;

	// Ambient configuration width this model's Eval expects (the owning
	// agent's CubicConfigurationSpline::ambient_dim()).
	virtual int ambient_dim() const = 0;

	// True only for the zero-config single-point model. Gates the
	// closed-form ApplySafetyProjection / ApplyAgentPairSafetyProjection
	// passes (see graph_short_path_mpc.cpp) and lets the inter-agent pair
	// rows keep using the solver's scalar `_agent_radii` for a trivial
	// agent instead of a per-sphere radius.
	virtual bool is_trivial() const { return false; }

	// Exact sphere count (fixed for the model's lifetime) -- used for QP
	// row-count reservations and the per-solve pattern signature.
	virtual int num_spheres() const = 0;
};

// Single point sphere at q[:workspace_dim], Jacobian [I_wd | 0], radius 0.
// `tangent_dim` / `ambient_dim` are the owning agent's own widths.
std::unique_ptr<AgentCollisionModel> MakeTrivialCollisionModel(int workspace_dim, int tangent_dim,
							       int ambient_dim);

// ---------------------------------------------------------------------------
// Non-trivial model registration -- PLAIN DATA. Stored on GraphOfConstraints
// (agent_collision_specs) and turned into an actual AgentCollisionModel by
// GraphShortPathMPC's constructor, so Drake's multibody headers stay out of
// graph_of_constraints.hpp and every TU that includes it.
// ---------------------------------------------------------------------------

struct CollisionSphereSpec {
	// Tier B (kArticulated): name of the plant body this sphere rides on.
	std::string body;
	// Tier A (kRigidConstellation): index into graph._robot_specs[agent] of
	// the block whose pose places this sphere. Ignored for Tier B.
	int block = 0;
	// Sphere centre in the body's / block's local frame.
	Eigen::Vector3d offset = Eigen::Vector3d::Zero();
	double radius = 0.0;
};

struct AgentCollisionSpec {
	enum class Kind {
		kArticulated,        // Tier B: Drake MultibodyPlant parsed from model_path
		kRigidConstellation  // Tier A: closed-form on the agent's own blocks (not built yet)
	};
	Kind kind = Kind::kArticulated;

	// Tier B: a URDF / MJCF / SDF file, parsed once into a MultibodyPlant.
	std::string model_path;
	// Tier B: the root link welded to the world, and its welded world pose
	// (translation + wxyz quaternion). For the ur_description UR5e the root
	// link is "base_link".
	std::string base_link;
	Eigen::Vector3d base_translation = Eigen::Vector3d::Zero();
	Eigen::Vector4d base_quaternion_wxyz = Eigen::Vector4d(1.0, 0.0, 0.0, 0.0);

	std::vector<CollisionSphereSpec> spheres;
};

// Tier B: Drake MultibodyPlant forward kinematics. `q_ambient` (the agent's
// own ambient configuration, which must equal plant.num_positions()) sets
// the plant's positions; each sphere's centre is X_WB(q) * offset and its
// Jacobian is the plant's translational-velocity Jacobian of that point
// w.r.t. q̇ -- which is exactly the solver's tangent step for a fixed-base,
// all-revolute arm registered as one Block::R(n_joints). Construction throws
// unless plant.num_positions() == tangent_dim.
std::unique_ptr<AgentCollisionModel> MakeDrakePlantCollisionModel(
	const AgentCollisionSpec& spec, int workspace_dim, int tangent_dim);

}  // namespace sqp_short_path
