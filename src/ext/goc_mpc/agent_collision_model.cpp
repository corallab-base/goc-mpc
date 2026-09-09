#include "agent_collision_model.hpp"

#include <stdexcept>
#include <string>

#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>
#include <drake/systems/framework/context.h>

namespace sqp_short_path {

namespace {

// q[:workspace_dim] with a constant [I_wd | 0] Jacobian -- exactly the
// solver's original fk fast path, expressed as a one-sphere model so every
// constraint path can loop over spheres uniformly (the trivial model just
// yields a single radius-0 sphere whose row is byte-identical to the old
// point-based row).
class TrivialCollisionModel : public AgentCollisionModel {
   public:
	TrivialCollisionModel(int workspace_dim, int tangent_dim, int ambient_dim)
	    : workspace_dim_(workspace_dim), ambient_dim_(ambient_dim) {
		jac_ = Eigen::MatrixXd::Zero(workspace_dim, tangent_dim);
		for (int k = 0; k < workspace_dim; ++k) jac_(k, k) = 1.0;
	}

	std::vector<WorkspaceSphere> Eval(
		const Eigen::Ref<const Eigen::VectorXd>& q_ambient) const override {
		WorkspaceSphere s;
		s.center.head(workspace_dim_) = q_ambient.head(workspace_dim_);
		s.radius = 0.0;
		s.jac = jac_;
		return {std::move(s)};
	}

	int ambient_dim() const override { return ambient_dim_; }
	bool is_trivial() const override { return true; }
	int num_spheres() const override { return 1; }

   private:
	int workspace_dim_;
	int ambient_dim_;
	Eigen::MatrixXd jac_;
};

}  // namespace

std::unique_ptr<AgentCollisionModel> MakeTrivialCollisionModel(int workspace_dim, int tangent_dim,
							       int ambient_dim) {
	return std::make_unique<TrivialCollisionModel>(workspace_dim, tangent_dim, ambient_dim);
}

namespace {

// Tier B (v2 plan Stage 3c): the agent's configuration is a fixed-base,
// all-revolute arm's joint vector; a Drake MultibodyPlant parsed once from a
// URDF/MJCF supplies each body-sphere's world centre and its analytic
// position Jacobian w.r.t. those joints. Kinematics only -- no dynamics, no
// collision queries (the spheres are supplied by the caller, not read from
// the model's own <collision> geometry).
class DrakePlantCollisionModel : public AgentCollisionModel {
   public:
	DrakePlantCollisionModel(const AgentCollisionSpec& spec, int workspace_dim, int tangent_dim)
	    : workspace_dim_(workspace_dim) {
		drake::multibody::Parser parser(&plant_);
		parser.AddModels(spec.model_path);

		if (spec.base_link.empty()) {
			throw std::runtime_error(
				"MakeDrakePlantCollisionModel: spec.base_link is empty -- name the "
				"root link to weld to the world (e.g. \"base_link\").");
		}
		const drake::math::RigidTransformd X_WBase(
			drake::math::RotationMatrixd(Eigen::Quaterniond(
				spec.base_quaternion_wxyz(0), spec.base_quaternion_wxyz(1),
				spec.base_quaternion_wxyz(2), spec.base_quaternion_wxyz(3))),
			spec.base_translation);
		plant_.WeldFrames(plant_.world_frame(),
				  plant_.GetFrameByName(spec.base_link), X_WBase);
		plant_.Finalize();

		if (plant_.num_positions() != tangent_dim) {
			throw std::runtime_error(
				"MakeDrakePlantCollisionModel: model \"" + spec.model_path +
				"\" has " + std::to_string(plant_.num_positions()) +
				" position DOFs after welding \"" + spec.base_link + "\" but the agent's "
				"tangent_dim is " + std::to_string(tangent_dim) +
				" -- Tier B is scoped to fixed-base all-revolute arms registered as one "
				"Block::R(n_joints); a floating base or mismatched joint count is not "
				"supported yet.");
		}
		ambient_dim_ = plant_.num_positions();
		context_ = plant_.CreateDefaultContext();

		spheres_.reserve(spec.spheres.size());
		for (const CollisionSphereSpec& s : spec.spheres) {
			spheres_.push_back(Sphere{plant_.GetBodyByName(s.body).index(),
						  s.offset.cast<double>(), s.radius});
		}
	}

	std::vector<WorkspaceSphere> Eval(
		const Eigen::Ref<const Eigen::VectorXd>& q_ambient) const override {
		plant_.SetPositions(context_.get(), q_ambient);
		const int nq = plant_.num_positions();

		std::vector<WorkspaceSphere> out;
		out.reserve(spheres_.size());
		Eigen::MatrixXd Jq(3, nq);
		for (const Sphere& s : spheres_) {
			const drake::multibody::RigidBody<double>& body = plant_.get_body(s.body);
			const drake::math::RigidTransformd& X_WB =
				plant_.EvalBodyPoseInWorld(*context_, body);

			WorkspaceSphere ws;
			ws.center = X_WB * s.offset;  // p_WoSk_W = X_WB * p_BoSk_B
			ws.radius = s.radius;

			plant_.CalcJacobianTranslationalVelocity(
				*context_, drake::multibody::JacobianWrtVariable::kQDot,
				body.body_frame(), s.offset, plant_.world_frame(), plant_.world_frame(),
				&Jq);
			// Solver tangent columns are the joint velocities (Block::R),
			// which for an all-revolute fixed-base arm are q̇ -- so Jq maps
			// straight in. Only the leading workspace_dim rows are used
			// downstream, but keep all 3 (WorkspaceSphere.jac is
			// workspace_dim x tangent_dim; slice here).
			ws.jac = Jq.topRows(workspace_dim_);
			out.push_back(std::move(ws));
		}
		return out;
	}

	int ambient_dim() const override { return ambient_dim_; }
	bool is_trivial() const override { return false; }
	int num_spheres() const override { return static_cast<int>(spheres_.size()); }

   private:
	struct Sphere {
		drake::multibody::BodyIndex body;
		Eigen::Vector3d offset;
		double radius;
	};

	// time_step 0 -> continuous plant; we only ever do kinematics on it.
	drake::multibody::MultibodyPlant<double> plant_{0.0};
	std::unique_ptr<drake::systems::Context<double>> context_;
	std::vector<Sphere> spheres_;
	int workspace_dim_;
	int ambient_dim_ = 0;
};

}  // namespace

std::unique_ptr<AgentCollisionModel> MakeDrakePlantCollisionModel(
	const AgentCollisionSpec& spec, int workspace_dim, int tangent_dim) {
	return std::make_unique<DrakePlantCollisionModel>(spec, workspace_dim, tangent_dim);
}

}  // namespace sqp_short_path
