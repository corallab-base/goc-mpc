#include "agent_collision_model.hpp"

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

}  // namespace sqp_short_path
