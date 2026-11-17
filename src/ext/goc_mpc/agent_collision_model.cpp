#include "agent_collision_model.hpp"

#include <algorithm>
#include <cmath>
#include <random>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>

#include <drake/math/rigid_transform.h>
#include <drake/math/rotation_matrix.h>
#include <drake/multibody/parsing/parser.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>
#include <drake/multibody/tree/prismatic_joint.h>
#include <drake/multibody/tree/revolute_joint.h>
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

// v2 plan Stage 3c: the agent's configuration is a fixed-base, all-revolute
// arm's joint vector; a Drake MultibodyPlant parsed once from a URDF/MJCF
// supplies each body-sphere's world centre and its analytic position
// Jacobian w.r.t. those joints. Kinematics only -- no dynamics, no collision
// queries (the spheres are supplied by the caller, not read from the model's
// own <collision> geometry).
class DrakePlantCollisionModel : public AgentCollisionModel {
   public:
	DrakePlantCollisionModel(const AgentCollisionSpec& spec, int workspace_dim, int tangent_dim)
	    : workspace_dim_(workspace_dim) {
		drake::multibody::Parser parser(&plant_);
		parser.AddModels(spec.model_path);

		if (spec.base_link.empty()) {
			throw std::runtime_error(
				"MakeDrakePlantCollisionModel: spec.base_link is empty -- name the "
				"model's root link (e.g. \"base_link\" for a ur_description URDF, "
				"\"base\" for a menagerie MJCF).");
		}
		// A ur_description URDF leaves its root link un-jointed (weld it
		// here); a menagerie MJCF's parser already welds `base` to the
		// world at the model's own mount pose. Either way the root ends up
		// welded -- possibly NOT where the caller wants it -- so the actual
		// alignment is done by X_correction_ below, applied to every sphere
		// centre/Jacobian in Eval. That makes base_translation /
		// base_quaternion_wxyz mean "where the root link frame sits in the
		// world", independent of any mount baked into the model file.
		try {
			plant_.WeldFrames(plant_.world_frame(),
					  plant_.GetFrameByName(spec.base_link),
					  drake::math::RigidTransformd::Identity());
		} catch (const std::exception&) {
			// Already welded by the (MJCF) parser -- fine, X_correction_
			// absorbs wherever it put the root.
		}
		plant_.Finalize();

		if (plant_.num_positions() != tangent_dim) {
			throw std::runtime_error(
				"MakeDrakePlantCollisionModel: model \"" + spec.model_path +
				"\" has " + std::to_string(plant_.num_positions()) +
				" position DOFs but the agent's tangent_dim is " +
				std::to_string(tangent_dim) +
				" -- Tier B is scoped to fixed-base all-revolute arms registered as one "
				"Block::R(n_joints); a floating base or mismatched joint count is not "
				"supported yet.");
		}
		ambient_dim_ = plant_.num_positions();
		context_ = plant_.CreateDefaultContext();

		// X_correction_ = X_WBase_wanted * X_WBase_asParsed^{-1}: the rigid
		// transform that moves the root link from wherever the parser
		// welded it to where the caller asked (identity rotation + the
		// given translation, i.e. this matches a JAX FK that zeroes the
		// model's own base mount and applies its own base placement).
		const drake::math::RigidTransformd X_WBase_wanted(
			drake::math::RotationMatrixd(Eigen::Quaterniond(
				spec.base_quaternion_wxyz(0), spec.base_quaternion_wxyz(1),
				spec.base_quaternion_wxyz(2), spec.base_quaternion_wxyz(3))),
			spec.base_translation);
		const drake::math::RigidTransformd X_WBase_asParsed =
			plant_.EvalBodyPoseInWorld(*context_, plant_.GetBodyByName(spec.base_link));
		X_correction_ = X_WBase_wanted * X_WBase_asParsed.inverse();

		spheres_.reserve(spec.spheres.size());
		for (const CollisionSphereSpec& s : spec.spheres) {
			spheres_.push_back(Sphere{plant_.GetBodyByName(s.body).index(),
						  s.offset.cast<double>(), s.radius});
		}

		// reach_: farthest body-sphere centre from the base at the zero
		// configuration (the default context) -- the arm's kinematic
		// footprint radius. broadphase_margin_hint() scales this down to an
		// estimate of the sphere travel a single solve can produce.
		const Eigen::Vector3d p_WBase = X_WBase_asParsed.translation();
		for (const Sphere& s : spheres_) {
			const drake::math::RigidTransformd& X_WB =
				plant_.EvalBodyPoseInWorld(*context_, plant_.get_body(s.body));
			reach_ = std::max(reach_, ((X_WB * s.offset) - p_WBase).norm() + s.radius);
		}
		CompileChain();
	}

	std::vector<WorkspaceSphere> Eval(
		const Eigen::Ref<const Eigen::VectorXd>& q_ambient) const override {
		return chain_ok_ ? EvalChain(q_ambient) : EvalPlant(q_ambient);
	}

	std::vector<WorkspaceSphere> EvalPlant(const Eigen::Ref<const Eigen::VectorXd>& q_ambient) const {
		plant_.SetPositions(context_.get(), q_ambient);
		const int nq = plant_.num_positions();
		const Eigen::Matrix3d R_corr = X_correction_.rotation().matrix();

		std::vector<WorkspaceSphere> out;
		out.reserve(spheres_.size());
		Eigen::MatrixXd Jq(3, nq);
		for (const Sphere& s : spheres_) {
			const drake::multibody::RigidBody<double>& body = plant_.get_body(s.body);
			const drake::math::RigidTransformd& X_WB =
				plant_.EvalBodyPoseInWorld(*context_, body);

			WorkspaceSphere ws;
			ws.center = X_correction_ * (X_WB * s.offset);  // p_WoSk_W then aligned
			ws.radius = s.radius;

			plant_.CalcJacobianTranslationalVelocity(
				*context_, drake::multibody::JacobianWrtVariable::kQDot,
				body.body_frame(), s.offset, plant_.world_frame(), plant_.world_frame(),
				&Jq);
			// Solver tangent columns are the joint velocities (Block::R),
			// which for an all-revolute fixed-base arm are q̇ -- so Jq maps
			// straight in, after the constant frame correction (d(R*p)/dq =
			// R * dp/dq). Only the leading workspace_dim rows are used.
			ws.jac = (R_corr * Jq).topRows(workspace_dim_);
			out.push_back(std::move(ws));
		}
		return out;
	}


	// Closed-form forward kinematics over the plant's own joints: every body
	// on a sphere's path to the world hangs off a weld, revolute or
	// prismatic joint, so each pose is the parent's times the joint's fixed
	// frames and its one-coordinate motion, and each sphere centre's
	// Jacobian column is axis x (centre - joint origin) for a revolute
	// joint and the axis for a prismatic one. The same numbers as the plant
	// (checked at construction), without its per-call caching and
	// allocation. Any other joint leaves EvalPlant in charge.
	std::vector<WorkspaceSphere> EvalChain(const Eigen::Ref<const Eigen::VectorXd>& q) const {
		const int nb = static_cast<int>(links_.size());
		for (int i = 0; i < nb; ++i) {
			const Link& l = links_[i];
			const drake::math::RigidTransformd& X_WP = l.parent < 0 ? identity_ : poses_[l.parent];
			const drake::math::RigidTransformd X_WF = X_WP * l.X_PF;
			drake::math::RigidTransformd X_FM;
			if (l.type == JointType::kRevolute) {
				X_FM = drake::math::RigidTransformd(drake::math::RotationMatrixd(Eigen::AngleAxisd(q(l.q_index), l.axis_F)),
								    Eigen::Vector3d::Zero());
			} else if (l.type == JointType::kPrismatic) {
				X_FM = drake::math::RigidTransformd(l.axis_F * q(l.q_index));
			}
			poses_[i] = X_WF * X_FM * l.X_MB;
			axes_W_[i] = X_WF.rotation() * l.axis_F;
			origins_W_[i] = X_WF.translation();
		}
		const int nq = static_cast<int>(q.size());
		const Eigen::Matrix3d R_corr = X_correction_.rotation().matrix();
		std::vector<WorkspaceSphere> out(sphere_links_.size());
		for (size_t k = 0; k < sphere_links_.size(); ++k) {
			const int b = sphere_links_[k];
			const Eigen::Vector3d p_W = poses_[b] * spheres_[k].offset;
			Eigen::Matrix<double, 3, Eigen::Dynamic> J = Eigen::MatrixXd::Zero(3, nq);
			for (int a = b; a >= 0; a = links_[a].parent) {
				const Link& l = links_[a];
				if (l.type == JointType::kRevolute)
					J.col(l.q_index) = axes_W_[a].cross(p_W - origins_W_[a]);
				else if (l.type == JointType::kPrismatic)
					J.col(l.q_index) = axes_W_[a];
			}
			WorkspaceSphere& ws = out[k];
			ws.center = X_correction_ * p_W;
			ws.radius = spheres_[k].radius;
			ws.jac = (R_corr * J).topRows(workspace_dim_);
		}
		return out;
	}

	void CompileChain() {
		// Each body's inboard joint.
		std::map<drake::multibody::BodyIndex, const drake::multibody::Joint<double>*> inboard;
		for (drake::multibody::JointIndex j : plant_.GetJointIndices())
			inboard[plant_.get_joint(j).child_body().index()] = &plant_.get_joint(j);
		std::map<drake::multibody::BodyIndex, int> slot;
		const drake::multibody::BodyIndex world = plant_.world_body().index();
		// Adds `body` and its ancestors (parents first); false on a joint
		// this can't evaluate.
		std::function<int(drake::multibody::BodyIndex)> add = [&](drake::multibody::BodyIndex body) -> int {
			if (body == world) return -1;
			const auto found = slot.find(body);
			if (found != slot.end()) return found->second;
			const auto it = inboard.find(body);
			if (it == inboard.end()) throw std::runtime_error("no inboard joint");
			const drake::multibody::Joint<double>& joint = *it->second;
			Link l;
			l.parent = add(joint.parent_body().index());
			l.X_PF = joint.frame_on_parent().GetFixedPoseInBodyFrame();
			l.X_MB = joint.frame_on_child().GetFixedPoseInBodyFrame().inverse();
			if (const auto* rj = dynamic_cast<const drake::multibody::RevoluteJoint<double>*>(&joint)) {
				l.type = JointType::kRevolute;
				l.axis_F = rj->revolute_axis();
				l.q_index = rj->position_start();
			} else if (const auto* pj = dynamic_cast<const drake::multibody::PrismaticJoint<double>*>(&joint)) {
				l.type = JointType::kPrismatic;
				l.axis_F = pj->translation_axis();
				l.q_index = pj->position_start();
			} else if (joint.num_positions() == 0) {
				l.type = JointType::kWeld;
			} else {
				throw std::runtime_error("unsupported joint");
			}
			links_.push_back(l);
			return slot[body] = static_cast<int>(links_.size()) - 1;
		};
		try {
			for (const Sphere& s : spheres_) sphere_links_.push_back(add(s.body));
		} catch (const std::runtime_error&) {
			links_.clear();
			sphere_links_.clear();
			return;
		}
		poses_.resize(links_.size());
		axes_W_.resize(links_.size());
		origins_W_.resize(links_.size());
		// The plant is the reference: keep the chain only where it agrees.
		std::mt19937 rng(0);
		std::uniform_real_distribution<double> unit(-M_PI, M_PI);
		chain_ok_ = true;
		for (int trial = 0; trial < 4 && chain_ok_; ++trial) {
			Eigen::VectorXd q(ambient_dim_);
			for (int i = 0; i < ambient_dim_; ++i) q(i) = trial == 0 ? 0.0 : unit(rng);
			const auto a = EvalChain(q), b = EvalPlant(q);
			for (size_t k = 0; k < a.size() && chain_ok_; ++k)
				chain_ok_ = (a[k].center - b[k].center).norm() < 1e-9 && (a[k].jac - b[k].jac).cwiseAbs().maxCoeff() < 1e-9;
		}
	}

	int ambient_dim() const override { return ambient_dim_; }
	bool is_trivial() const override { return false; }
	int num_spheres() const override { return static_cast<int>(spheres_.size()); }
	bool uses_closed_form_kinematics() const { return chain_ok_; }

	// ~1/4 of the arm's reach: a fixed-base arm sweeping a distal joint
	// through the few tenths of a radian a single short-horizon solve
	// typically asks for moves its far spheres by roughly that much, and
	// the pruner only needs an upper bound on that travel (too large just
	// keeps extra inactive rows and -- past ~0.3 for a UR5e-scale arm --
	// balloons ProxQP's cold factorization; too small could drop a contact
	// the solve would have smoothed). Callers whose reference is much
	// farther from the solution can still raise constraint_prune_margin.
	double broadphase_margin_hint() const override { return 0.25 * reach_; }

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
	drake::math::RigidTransformd X_correction_;
	int workspace_dim_;
	int ambient_dim_ = 0;
	double reach_ = 0.0;

	enum class JointType { kWeld, kRevolute, kPrismatic };
	struct Link {
		int parent = -1;  // index into links_, -1 for the world
		drake::math::RigidTransformd X_PF, X_MB;
		JointType type = JointType::kWeld;
		Eigen::Vector3d axis_F = Eigen::Vector3d::Zero();
		int q_index = 0;
	};
	std::vector<Link> links_;          // parents before children
	std::vector<int> sphere_links_;    // per sphere, its body's link
	bool chain_ok_ = false;
	const drake::math::RigidTransformd identity_;
	// Scratch for EvalChain, per link.
	mutable std::vector<drake::math::RigidTransformd> poses_;
	mutable std::vector<Eigen::Vector3d> axes_W_, origins_W_;
};

}  // namespace

std::unique_ptr<AgentCollisionModel> MakeDrakePlantCollisionModel(
	const AgentCollisionSpec& spec, int workspace_dim, int tangent_dim) {
	return std::make_unique<DrakePlantCollisionModel>(spec, workspace_dim, tangent_dim);
}

}  // namespace sqp_short_path
