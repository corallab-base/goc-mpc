#include "sqp_short_path_layout.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>

namespace sqp_short_path {

namespace {
using Block = CubicConfigurationSpline::Block;
}  // namespace

std::vector<CubicConfigurationSpline> BuildAgentShapes(const GraphOfConstraints& graph, int num_agents) {
	std::vector<CubicConfigurationSpline> shapes;
	shapes.reserve(num_agents);
	for (int ag = 0; ag < num_agents; ++ag) {
		for (const Block& b : graph._robot_specs.at(ag)) {
			if (b.type != Block::Type::R && b.type != Block::Type::Torus &&
			    b.type != Block::Type::SO3Quat) {
				throw std::runtime_error(
					"GraphShortPathMPC: Block::SO3Mat is not supported (agent " +
					std::to_string(ag) + ") -- R/Torus/SO3Quat only.");
			}
		}
		shapes.emplace_back(graph._robot_specs.at(ag));
	}
	return shapes;
}

std::vector<AxisLayout> BuildAxisList(const std::vector<CubicConfigurationSpline>& agent_shapes) {
	std::vector<AxisLayout> axes;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		const int tdim = agent_shapes[ag].tangent_dim();
		for (int k = 0; k < tdim; ++k) {
			axes.push_back(AxisLayout{ag, k});
		}
	}
	return axes;
}

std::vector<int> BuildAgentAxisOffsets(const std::vector<CubicConfigurationSpline>& agent_shapes) {
	std::vector<int> offsets(agent_shapes.size());
	int running = 0;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		offsets[ag] = running;
		running += agent_shapes[ag].tangent_dim();
	}
	return offsets;
}

std::vector<int> BuildAgentAmbientOffsets(const std::vector<CubicConfigurationSpline>& agent_shapes) {
	std::vector<int> offsets(agent_shapes.size());
	int running = 0;
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		offsets[ag] = running;
		running += agent_shapes[ag].ambient_dim();
	}
	return offsets;
}

Eigen::MatrixXd BuildAxisHessianBlock(int num_steps, double tau, const SmoothCostWeights& weights) {
	const int n = 2 * num_steps;
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	const double tau2 = tau * tau;

	// Tracking, velocity-tracking, and the coast-corrected acceleration
	// residual's linear-in-(p,v) coefficients,
	// none of which depend on the current iterate (only the RHS/target
	// does, see BuildAxisRhs) -- so this block is reused unchanged across
	// every outer SQP iteration within one solve() call. `weights` scales
	// each TERM's contribution (a plain multiplier on that term's squared
	// residual, same semantics as GraphTimingMPC's `acceleration_cost`),
	// not the coefficient PATTERN itself.
	auto add_h = [&](std::initializer_list<std::pair<int, double>> terms, double weight) {
		for (auto [i, ci] : terms) {
			for (auto [j, cj] : terms) {
				H(i, j) += weight * ci * cj;
			}
		}
	};

	for (int i = 0; i < num_steps; ++i) {
		add_h({{IdxP(0, i, num_steps), 1.0}}, weights.tracking);  // axis-local index 0 below
		add_h({{IdxV(0, i, num_steps), 1.0}}, weights.velocity_tracking);
		if (i == 0) {
			add_h({{IdxP(0, 0, num_steps), -6.0 / tau2}, {IdxV(0, 0, num_steps), 4.0 / tau}},
			      weights.acceleration);
		} else {
			add_h({{IdxP(0, i, num_steps), -6.0 / tau2}, {IdxP(0, i - 1, num_steps), 6.0 / tau2},
			       {IdxV(0, i, num_steps), 4.0 / tau}, {IdxV(0, i - 1, num_steps), 2.0 / tau}},
			      weights.acceleration);
		}
	}
	return H;
}

Eigen::MatrixXd AssembleSmoothHessian(const std::vector<CubicConfigurationSpline>& agent_shapes,
				       const std::vector<int>& agent_axis_offsets,
				       int num_steps, double tau, const SmoothCostWeights& weights) {
	const Eigen::MatrixXd block = BuildAxisHessianBlock(num_steps, tau, weights);
	const int per_axis = 2 * num_steps;
	int n_axes = 0;
	for (const auto& shape : agent_shapes) n_axes += shape.tangent_dim();
	const int n = n_axes * per_axis;
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	// Place BuildAxisHessianBlock's (iteration-constant) block at every
	// R/Torus tangent column; SO3Quat columns are left zero here -- their
	// real (coupled, iterate-dependent) contribution is added separately,
	// every outer iteration, by AccumulateSO3QuatBlock.
	for (int ag = 0; ag < static_cast<int>(agent_shapes.size()); ++ag) {
		const int off = agent_axis_offsets[ag];
		for (const auto& boff : agent_shapes[ag].block_offsets_) {
			if (boff.type == Block::Type::SO3Quat) continue;
			for (int k = 0; k < boff.tangent_size; ++k) {
				const int a = off + boff.tangent_offset + k;
				H.block(a * per_axis, a * per_axis, per_axis, per_axis) = block;
			}
		}
	}
	return H;
}

Eigen::VectorXd BuildAxisRhs(const AxisLayout& axis,
			      const CubicConfigurationSpline& agent_shape,
			      int num_steps, double tau, const SmoothCostWeights& weights,
			      const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			      const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			      const Eigen::MatrixXd& ref_points_agent,
			      const Eigen::MatrixXd& ref_velocities_agent) {
	const int n = 2 * num_steps;
	Eigen::VectorXd g = Eigen::VectorXd::Zero(n);
	const double tau2 = tau * tau;
	const int k = axis.tangent_col;

	auto add_g = [&](std::initializer_list<std::pair<int, double>> terms, double target, double weight) {
		for (auto [i, ci] : terms) {
			g(i) += weight * target * ci;
		}
	};

	for (int i = 0; i < num_steps; ++i) {
		// Tracking: cost is (p_abs(i)-ref(i))^2, substituting p_abs(i) =
		// p_current(i) + dp(i) linearizes to (dp(i) - target)^2 with
		// target = ref(i) - p_current(i), wrap-aware (BlockPositionDelta's
		// R case is a plain subtraction, Torus case wraps to (-pi,pi]) --
		// exactly the manifold-correct "step that would reach ref exactly
		// from here" to first order.
		// .transpose() both sides explicitly (not relying on Eigen's
		// implicit row<->column vector reshape) -- this project's
		// RelWithDebInfo build compiles out eigen_assert, so a genuine
		// shape mismatch elsewhere in a call chain like this can silently
		// corrupt instead of throwing (see feedback_eigen_row_col_no_assert
		// in project memory); being explicit here costs nothing.
		const Eigen::VectorXd track_delta = agent_shape.PositionDelta<double>(
			ref_points_agent.row(i).transpose(), points_agent.row(i).transpose());
		add_g({{IdxP(0, i, num_steps), 1.0}}, track_delta(k), weights.tracking);

		// Velocity tracking: velocities live in a flat tangent space
		// already (no manifold/wraparound), so this is always a plain
		// subtraction, R or Torus alike.
		const double vtrack_target = ref_velocities_agent(i, k) - vels_agent(i, k);
		add_g({{IdxV(0, i, num_steps), 1.0}}, vtrack_target, weights.velocity_tracking);

		if (i == 0) {
			// disp0 = p_current(0) - x0, wrap-aware -- linearized around the
			// current iterate instead of being an absolute-coordinate
			// target against x0 directly -- see this file's own top
			// comment / the project plan's design decision 3 for the
			// derivation.
			const Eigen::VectorXd disp0 =
				agent_shape.PositionDelta<double>(points_agent.row(0).transpose(), x0_agent);
			const double target0 =
				(6.0 / tau2) * disp0(k) - (4.0 / tau) * vels_agent(0, k) - (2.0 / tau) * v0_agent(k);
			add_g({{IdxP(0, 0, num_steps), -6.0 / tau2}, {IdxV(0, 0, num_steps), 4.0 / tau}},
			      target0, weights.acceleration);
		} else {
			const Eigen::VectorXd disp = agent_shape.PositionDelta<double>(
				points_agent.row(i).transpose(), points_agent.row(i - 1).transpose());
			const double target =
				(6.0 / tau2) * disp(k) - (4.0 / tau) * vels_agent(i, k) - (2.0 / tau) * vels_agent(i - 1, k);
			add_g({{IdxP(0, i, num_steps), -6.0 / tau2}, {IdxP(0, i - 1, num_steps), 6.0 / tau2},
			       {IdxV(0, i, num_steps), 4.0 / tau}, {IdxV(0, i - 1, num_steps), 2.0 / tau}},
			      target, weights.acceleration);
		}
	}
	return g;
}

namespace {

// One coupled 3-vector term of a residual `sum(term.A * delta_term) + b`
// (see AddVec3ResidualNormalEquations) -- the SO3Quat generalization of
// BuildAxisRhs's `{index, scalar coefficient}` Entry pattern to `{index
// triple, 3x3 coefficient}`, needed because an SO3Quat block's 3 tangent
// components are coupled (SO(3) has real curvature -- see
// ComputeSO3QuatResidual), not independently separable the way an R/Torus
// axis's single scalar component is.
struct Vec3Term {
	std::array<int, 3> idx;  // flat decision-vector index (IdxP/IdxV) per tangent component
	Eigen::Matrix3d A;       // this term's contribution to the residual is A * delta_term
};

// Normal-equations accumulation for a vector residual `residual(delta) =
// sum_i(terms[i].A * delta_term_i) + b`, cost = weight*||residual||^2 --
// standard weighted least squares: H += weight*A_a^T A_b for every term
// PAIR (including a==b), g -= weight*A_a^T*b for every term. Generalizes
// BuildAxisHessianBlock/BuildAxisRhs's scalar add_h/add_g to matrix-valued
// coefficients; see this class's own doc comment.
void AddVec3ResidualNormalEquations(Eigen::MatrixXd* H, Eigen::VectorXd* g,
				     const std::vector<Vec3Term>& terms,
				     const Eigen::Vector3d& b, double weight) {
	for (const auto& ta : terms) {
		const Eigen::Vector3d gb = weight * (ta.A.transpose() * b);
		for (int c = 0; c < 3; ++c) (*g)(ta.idx[c]) -= gb(c);
		for (const auto& tb : terms) {
			const Eigen::Matrix3d hb = weight * (ta.A.transpose() * tb.A);
			for (int c = 0; c < 3; ++c) {
				for (int cp = 0; cp < 3; ++cp) {
					(*H)(ta.idx[c], tb.idx[cp]) += hb(c, cp);
				}
			}
		}
	}
}

}  // namespace

void AccumulateSO3QuatBlock(const SO3QuatBlock& block, const CubicConfigurationSpline& agent_shape,
			     int num_steps, double tau, const SmoothCostWeights& weights,
			     const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			     const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			     const Eigen::MatrixXd& ref_points_agent,
			     const Eigen::MatrixXd& ref_velocities_agent,
			     Eigen::MatrixXd* H, Eigen::VectorXd* g) {
	const auto& off = block.offset;
	const double tau2 = tau * tau;
	const int t0 = off.tangent_offset;
	auto idx3 = [&](std::function<int(int, int, int)> IdxFn, int i) {
		return std::array<int, 3>{IdxFn(block.axis_offset + 0, i, num_steps),
					   IdxFn(block.axis_offset + 1, i, num_steps),
					   IdxFn(block.axis_offset + 2, i, num_steps)};
	};

	for (int i = 0; i < num_steps; ++i) {
		// --- Tracking: weight * ||Log(q_new(i)^{-1} q_ref(i))||^2, linearized
		// around the CURRENT q(i) via its own right-perturbation Jacobian
		// (xJ=ref fixed, xJm1=q_current(i) is the decision variable).
		const auto res_track = CubicConfigurationSpline::ComputeSO3QuatResidual(
			off, ref_points_agent.row(i).transpose(), points_agent.row(i).transpose());
		AddVec3ResidualNormalEquations(
			H, g, {Vec3Term{idx3(IdxP, i), res_track.d_value_d_Jm1}}, res_track.value,
			weights.tracking);

		// --- Velocity tracking: flat tangent space, no manifold curvature --
		// residual(delta_v) = (v_current+delta_v) - ref = delta_v + (v_current-ref).
		const Eigen::Vector3d vtrack_b =
			vels_agent.row(i).segment(t0, 3).transpose() -
			ref_velocities_agent.row(i).segment(t0, 3).transpose();
		AddVec3ResidualNormalEquations(
			H, g, {Vec3Term{idx3(IdxV, i), Eigen::Matrix3d::Identity()}}, vtrack_b,
			weights.velocity_tracking);

		// --- Acceleration (coast-corrected), coupling step i (and i-1, or
		// x0/v0 at i==0) -- same coefficient pattern as BuildAxisRhs's own
		// acceleration branch, generalized to 3x3 blocks via
		// ComputeSO3QuatResidual's Jacobians in place of the scalar +-1.
		std::vector<Vec3Term> accel_terms;
		Eigen::Vector3d disp0, v_km1;
		if (i == 0) {
			const auto res_acc = CubicConfigurationSpline::ComputeSO3QuatResidual(
				off, points_agent.row(0).transpose(), x0_agent);
			disp0 = res_acc.value;
			v_km1 = v0_agent.segment(t0, 3);
			accel_terms.push_back(Vec3Term{idx3(IdxP, 0), -(6.0 / tau2) * res_acc.d_value_d_J});
		} else {
			const auto res_acc = CubicConfigurationSpline::ComputeSO3QuatResidual(
				off, points_agent.row(i).transpose(), points_agent.row(i - 1).transpose());
			disp0 = res_acc.value;
			v_km1 = vels_agent.row(i - 1).segment(t0, 3).transpose();
			accel_terms.push_back(Vec3Term{idx3(IdxP, i), -(6.0 / tau2) * res_acc.d_value_d_J});
			accel_terms.push_back(
				Vec3Term{idx3(IdxP, i - 1), -(6.0 / tau2) * res_acc.d_value_d_Jm1});
		}
		const Eigen::Vector3d v_k = vels_agent.row(i).segment(t0, 3).transpose();
		const Eigen::Vector3d accel0 = -(6.0 / tau2) * disp0 + (2.0 / tau) * (2.0 * v_k + v_km1);
		accel_terms.push_back(Vec3Term{idx3(IdxV, i), (4.0 / tau) * Eigen::Matrix3d::Identity()});
		if (i > 0) {
			accel_terms.push_back(
				Vec3Term{idx3(IdxV, i - 1), (2.0 / tau) * Eigen::Matrix3d::Identity()});
		}
		AddVec3ResidualNormalEquations(H, g, accel_terms, accel0, weights.acceleration);
	}
}

double EvaluateSmoothCost(const CubicConfigurationSpline& agent_shape, int num_steps, double tau,
			   const SmoothCostWeights& weights,
			   const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			   const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			   const Eigen::MatrixXd& ref_points_agent, const Eigen::MatrixXd& ref_velocities_agent) {
	double f = 0.0;
	const double tau2 = tau * tau;
	for (int i = 0; i < num_steps; ++i) {
		const Eigen::VectorXd track_delta = agent_shape.PositionDelta<double>(
			ref_points_agent.row(i).transpose(), points_agent.row(i).transpose());
		f += weights.tracking * track_delta.squaredNorm();

		const Eigen::VectorXd vtrack_delta =
			(ref_velocities_agent.row(i) - vels_agent.row(i)).transpose();
		f += weights.velocity_tracking * vtrack_delta.squaredNorm();

		Eigen::VectorXd disp;
		Eigen::VectorXd v_km1;
		if (i == 0) {
			disp = agent_shape.PositionDelta<double>(points_agent.row(0).transpose(), x0_agent);
			v_km1 = v0_agent;
		} else {
			disp = agent_shape.PositionDelta<double>(
				points_agent.row(i).transpose(), points_agent.row(i - 1).transpose());
			v_km1 = vels_agent.row(i - 1).transpose();
		}
		const Eigen::VectorXd v_k = vels_agent.row(i).transpose();
		const Eigen::VectorXd accel = -(6.0 / tau2) * disp + (2.0 / tau) * (2.0 * v_k + v_km1);
		f += weights.acceleration * accel.squaredNorm();
	}
	return f;
}

namespace {

// Sphere/box signed-distance value + gradient at a workspace point `p`,
// mirroring graph_short_path_mpc.cpp's sdf constructions (same epsilon
// trick to keep the gradient finite through the sqrt singularity at zero
// separation) but hand-differentiated here (no Drake symbolic autodiff in
// this solver's QP path -- see the project plan's design decision 1).
constexpr double kSqrtEps2 = 1.0e-8;

struct SdfResult {
	double value;
	Eigen::VectorXd grad;  // d(value)/d(p), workspace_dim
};

// Core sphere sdf math, independent of where `center`/`R` come from --
// shared by SphereSdf (obstacle case, `center` fixed) and
// LinearizeAgentPairConstraints (inter-agent case, `center` is the OTHER
// agent's own position, itself a decision variable -- see that function's
// own comment for how this single-point gradient becomes a two-endpoint
// row via the chain rule).
SdfResult SphereSdfCore(const Eigen::VectorXd& p, const Eigen::VectorXd& center, double R) {
	const Eigen::VectorXd diff = p - center;
	const double d = std::sqrt(diff.squaredNorm() + kSqrtEps2);
	return SdfResult{d - R, diff / d};
}

SdfResult SphereSdf(const Eigen::VectorXd& p, const Obstacle& obstacle, int workspace_dim) {
	const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
	const double R = obstacle.params(workspace_dim) + obstacle.margin;
	return SphereSdfCore(p, center, R);
}

SdfResult BoxSdf(const Eigen::VectorXd& p, const Obstacle& obstacle, int workspace_dim) {
	const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
	const Eigen::VectorXd he = obstacle.params.segment(workspace_dim, workspace_dim);
	const Eigen::VectorXd diff = p - center;

	Eigen::VectorXd q(workspace_dim), clamped(workspace_dim);
	double outside_sq = 0.0;
	double m = -std::numeric_limits<double>::infinity();
	for (int k = 0; k < workspace_dim; ++k) {
		q(k) = std::abs(diff(k)) - he(k);
		clamped(k) = std::max(q(k), 0.0);
		outside_sq += clamped(k) * clamped(k);
		m = std::max(m, q(k));
	}
	const double sdf = std::sqrt(outside_sq + kSqrtEps2) + std::min(m, 0.0);

	Eigen::VectorXd grad = Eigen::VectorXd::Zero(workspace_dim);
	if (outside_sq > 1.0e-12) {
		// At least one axis is genuinely outside -- the sqrt(outside_sq)
		// term dominates the gradient (the min(m,0) term is 0 here since
		// m > 0 whenever any q(k) > 0).
		const double denom = std::sqrt(outside_sq + kSqrtEps2);
		for (int k = 0; k < workspace_dim; ++k) {
			if (q(k) > 0.0) {
				grad(k) = (diff(k) >= 0.0 ? 1.0 : -1.0) * clamped(k) / denom;
			}
		}
	} else {
		// Strictly inside (every q(k) <= 0): only min(m,0) contributes,
		// and only through its argmax axis -- same "push out along the
		// least-penetrated axis" direction obstacle_projection.hpp's
		// project_out uses for this same case.
		int kstar = 0;
		q.maxCoeff(&kstar);
		grad(kstar) = diff(kstar) >= 0.0 ? 1.0 : -1.0;
	}
	return SdfResult{sdf - obstacle.margin, grad};
}

}  // namespace

namespace {
// The distance pruner works with the tighter of the caller's
// `constraint_prune_margin` and the agent's own broadphase hint: a bounded
// body (a fixed-base arm) reports a hint >= 0 so its rows don't balloon when
// the caller left `constraint_prune_margin` at the free-particle default,
// while the trivial point model reports < 0 ("no opinion").
double EffectiveMargin(double prune_margin, double hint) {
	return hint >= 0.0 ? std::min(prune_margin, hint) : prune_margin;
}
double EffectivePairMargin(double prune_margin, double hint_a, double hint_b) {
	// max, not sum: each hint is already a generous per-body travel bound
	// (~1/4 of reach), and a pair only survives to matter if the SQP would
	// actually drive it -- which it only ever does in the SEPARATING
	// direction. Summing just reinstates the free-particle blow-up.
	return (hint_a >= 0.0 && hint_b >= 0.0) ? std::min(prune_margin, std::max(hint_a, hint_b))
					       : prune_margin;
}

// obstacle's own extent proxy for the bounding-sphere prune test: sphere
// radius+margin exactly; box half-extents' Euclidean norm+margin (the
// circumscribing sphere -- conservative, never excludes a box that could
// genuinely matter, at worst includes a corner-adjacent one that turns out
// not to).
double ObstacleExtent(const Obstacle& obstacle, int workspace_dim) {
	if (obstacle.kind == ObstacleKind::kSphere) {
		return obstacle.params(workspace_dim) + obstacle.margin;
	}
	return obstacle.params.segment(workspace_dim, workspace_dim).norm() + obstacle.margin;
}

// Half-open range `[lo, hi)` of horizon steps for which `within(i)` is true,
// widened to the first..last such step (contiguous -- see ActiveObstacle's
// own comment for why a bimodal reference only ever over-widens here).
// `lo == hi == 0` if no step qualifies.
template <typename F>
std::pair<int, int> StepRange(int num_steps, F&& within) {
	int lo = num_steps;
	int hi = 0;
	for (int i = 0; i < num_steps; ++i) {
		if (within(i)) {
			lo = std::min(lo, i);
			hi = std::max(hi, i + 1);
		}
	}
	return lo < hi ? std::pair<int, int>{lo, hi} : std::pair<int, int>{0, 0};
}
}  // namespace

std::vector<AgentReferenceSpheres> BuildAgentReferenceSpheres(
	const Eigen::MatrixXd& ref_points, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim,
	const AgentCollisionModels& models) {
	const int num_steps = static_cast<int>(ref_points.rows());
	std::vector<AgentReferenceSpheres> out(num_agents);
	for (int ag = 0; ag < num_agents; ++ag) {
		AgentReferenceSpheres& rs = out[ag];
		const int K = models[ag]->num_spheres();
		rs.centers.resize(num_steps);
		rs.radii = Eigen::VectorXd::Zero(K);

		Eigen::VectorXd sum = Eigen::VectorXd::Zero(workspace_dim);
		long count = 0;
		for (int i = 0; i < num_steps; ++i) {
			const Eigen::VectorXd q_ag = ref_points.row(i)
							     .segment(agent_ambient_offsets[ag],
								      models[ag]->ambient_dim())
							     .transpose();
			const std::vector<WorkspaceSphere> sph = models[ag]->Eval(q_ag);
			Eigen::MatrixXd C(K, workspace_dim);
			for (int k = 0; k < K; ++k) {
				C.row(k) = sph[k].center.head(workspace_dim).transpose();
				if (i == 0) rs.radii(k) = sph[k].radius;
				sum += sph[k].center.head(workspace_dim);
				++count;
			}
			rs.centers[i] = std::move(C);
		}
		rs.max_radius = K > 0 ? rs.radii.maxCoeff() : 0.0;

		const Eigen::VectorXd bcenter =
			count > 0 ? (sum / static_cast<double>(count)).eval()
				  : Eigen::VectorXd::Zero(workspace_dim);
		double bradius = 0.0;
		for (int i = 0; i < num_steps; ++i)
			for (int k = 0; k < rs.centers[i].rows(); ++k)
				bradius = std::max(bradius,
						   (rs.centers[i].row(k).transpose() - bcenter).norm());
		rs.bound = BoundingSphere{bcenter, bradius};
	}
	return out;
}

std::vector<std::vector<ActiveObstacle>> PruneObstaclesByDistance(
	int num_steps, int num_agents, int workspace_dim,
	const std::vector<AgentReferenceSpheres>& ref_spheres, const AgentCollisionModels& models,
	const ObstacleSet& obstacles, double prune_margin) {
	std::vector<std::vector<ActiveObstacle>> per_agent_obstacles(num_agents);
	if (obstacles.obstacles().empty()) {
		return per_agent_obstacles;
	}
	for (int ag = 0; ag < num_agents; ++ag) {
		const AgentReferenceSpheres& rs = ref_spheres[ag];
		const double margin =
			EffectiveMargin(prune_margin, models[ag]->broadphase_margin_hint());
		for (const Obstacle& obstacle : obstacles.obstacles()) {
			const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
			const double extent = ObstacleExtent(obstacle, workspace_dim);
			if ((center - rs.bound.center).norm() >
			    rs.bound.radius + rs.max_radius + extent + margin) {
				continue;  // coarse whole-trajectory filter
			}
			auto near_at = [&](int i, int k) {
				return (rs.centers[i].row(k).transpose() - center).norm() - rs.radii(k) <=
				       extent + margin;
			};
			const int K = static_cast<int>(rs.radii.size());
			const auto [lo, hi] = StepRange(num_steps, [&](int i) {
				for (int k = 0; k < K; ++k)
					if (near_at(i, k)) return true;
				return false;
			});
			if (lo >= hi) continue;
			std::vector<std::vector<int>> spheres(hi - lo);
			for (int i = lo; i < hi; ++i)
				for (int k = 0; k < K; ++k)
					if (near_at(i, k)) spheres[i - lo].push_back(k);
			per_agent_obstacles[ag].push_back(ActiveObstacle{&obstacle, lo, hi, std::move(spheres)});
		}
	}
	return per_agent_obstacles;
}

std::vector<ActivePair> PruneAgentPairsByDistance(
	int num_steps, int num_agents, const std::vector<AgentReferenceSpheres>& ref_spheres,
	const AgentCollisionModels& models, const Eigen::VectorXd& agent_radii, double prune_margin,
	int max_pairs_per_step) {
	std::vector<ActivePair> active_pairs;
	if (num_agents < 2) {
		return active_pairs;
	}
	// A trivial agent contributes its scalar _agent_radii(ag) as its single
	// sphere radius; a non-trivial one contributes its body spheres' own
	// radii -- matching LinearizeAgentPairConstraints' PairRadius.
	auto sphere_radius = [&](int ag, int k) {
		return models[ag]->is_trivial() ? agent_radii(ag) : ref_spheres[ag].radii(k);
	};
	auto pad = [&](int ag) {
		return models[ag]->is_trivial() ? agent_radii(ag) : ref_spheres[ag].max_radius;
	};
	for (int ag_a = 0; ag_a < num_agents; ++ag_a) {
		for (int ag_b = ag_a + 1; ag_b < num_agents; ++ag_b) {
			const AgentReferenceSpheres& ra = ref_spheres[ag_a];
			const AgentReferenceSpheres& rb = ref_spheres[ag_b];
			const double margin = EffectivePairMargin(prune_margin,
								 models[ag_a]->broadphase_margin_hint(),
								 models[ag_b]->broadphase_margin_hint());
			if ((ra.bound.center - rb.bound.center).norm() >
			    ra.bound.radius + rb.bound.radius + pad(ag_a) + pad(ag_b) + margin) {
				continue;  // coarse whole-trajectory filter
			}
			const int Ka = static_cast<int>(ra.radii.size());
			const int Kb = static_cast<int>(rb.radii.size());
			// Reference-trajectory surface separation of sphere pair
			// (ka, kb) at step i -- the per-step filter threshold AND the
			// top-k ranking key.
			auto sep_at = [&](int i, int ka, int kb) {
				return (ra.centers[i].row(ka) - rb.centers[i].row(kb)).norm() -
				       sphere_radius(ag_a, ka) - sphere_radius(ag_b, kb);
			};
			const auto [lo, hi] = StepRange(num_steps, [&](int i) {
				for (int ka = 0; ka < Ka; ++ka)
					for (int kb = 0; kb < Kb; ++kb)
						if (sep_at(i, ka, kb) <= margin) return true;
				return false;
			});
			if (lo >= hi) continue;
			std::vector<std::vector<std::pair<int, int>>> sphere_pairs(hi - lo);
			for (int i = lo; i < hi; ++i) {
				std::vector<std::pair<double, std::pair<int, int>>> near;
				for (int ka = 0; ka < Ka; ++ka)
					for (int kb = 0; kb < Kb; ++kb) {
						const double s = sep_at(i, ka, kb);
						if (s <= margin) near.emplace_back(s, std::make_pair(ka, kb));
					}
				// Keep only the `max_pairs_per_step` closest at this step
				// (0 = keep all). Only a few sphere pairs are ever the
				// binding contact even when a hundred sit within `margin`.
				if (max_pairs_per_step > 0 &&
				    static_cast<int>(near.size()) > max_pairs_per_step) {
					std::nth_element(near.begin(), near.begin() + max_pairs_per_step,
							 near.end(),
							 [](const auto& x, const auto& y) { return x.first < y.first; });
					near.resize(max_pairs_per_step);
				}
				sphere_pairs[i - lo].reserve(near.size());
				for (const auto& scored : near) sphere_pairs[i - lo].push_back(scored.second);
			}
			active_pairs.push_back(ActivePair{ag_a, ag_b, lo, hi, std::move(sphere_pairs)});
		}
	}
	return active_pairs;
}

namespace {

// One agent's ambient configuration row at horizon step `i`, sliced to the
// width its collision model expects.
Eigen::VectorXd AgentConfigRow(const Eigen::MatrixXd& points, int i, int ambient_offset,
			       const AgentCollisionModel& model) {
	return points.row(i).segment(ambient_offset, model.ambient_dim()).transpose();
}

}  // namespace

const std::vector<WorkspaceSphere>& SphereEvalCache::Get(int ag, int i, const Eigen::MatrixXd& points,
							 int ambient_offset, const AgentCollisionModel& model) {
	std::optional<std::vector<WorkspaceSphere>>& slot = cache_[ag][i];
	if (!slot) {
		slot = model.Eval(AgentConfigRow(points, i, ambient_offset, model));
	}
	return *slot;
}

namespace {

// Chain d(value)/d(centre) (workspace_dim) through a body sphere's tangent
// Jacobian (workspace_dim x agent_tangent_dim) and append the nonzero
// coefficients to `row`, at flat index IdxP(axis_off + j, step) with the
// given sign. For the trivial model's constant [I | 0] Jacobian this
// appends exactly (IdxP(axis_off + k, step), sign * grad_c(k)) for k in
// [0, workspace_dim) -- byte-identical to the old fk fast-path row.
void AppendSphereJacobianRow(ConstraintRow* row, const Eigen::VectorXd& grad_c,
			     const Eigen::MatrixXd& jac, int axis_off, int step, int num_steps,
			     double sign) {
	for (int j = 0; j < static_cast<int>(jac.cols()); ++j) {
		// Skip tangent columns the sphere centre structurally doesn't
		// depend on -- every column past workspace_dim for the trivial
		// [I | 0] Jacobian, so the emitted set is exactly the old
		// fk-fast-path one. For the columns it does depend on, emit the
		// coefficient even if it rounds to zero (the old rows did too).
		if ((jac.col(j).array() != 0.0).any()) {
			row->coeffs.emplace_back(IdxP(axis_off + j, step, num_steps),
						 sign * grad_c.dot(jac.col(j)));
		}
	}
}

}  // namespace

double EvaluateObstacleViolation(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
				  int workspace_dim, const Eigen::MatrixXd& points,
				  const std::vector<std::vector<ActiveObstacle>>& per_agent_obstacles,
				  const AgentCollisionModels& models, SphereEvalCache& sphere_cache) {
	double violation = 0.0;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			if (per_agent_obstacles[ag].empty()) continue;
			const std::vector<WorkspaceSphere>& spheres =
				sphere_cache.Get(ag, i, points, agent_ambient_offsets[ag], *models[ag]);
			for (const ActiveObstacle& ao : per_agent_obstacles[ag]) {
				if (i < ao.step_lo || i >= ao.step_hi) continue;
				for (int k : ao.spheres[i - ao.step_lo]) {
					const WorkspaceSphere& sph = spheres[k];
					const Eigen::VectorXd c = sph.center.head(workspace_dim);
					const double value =
						((ao.obstacle->kind == ObstacleKind::kSphere)
							 ? SphereSdf(c, *ao.obstacle, workspace_dim).value
							 : BoxSdf(c, *ao.obstacle, workspace_dim).value) -
						sph.radius;
					violation += std::max(0.0, -value);
				}
			}
		}
	}
	return violation;
}

std::vector<ConstraintRow> LinearizeObstacleConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim, const Eigen::MatrixXd& points,
	const std::vector<std::vector<ActiveObstacle>>& per_agent_obstacles,
	const AgentCollisionModels& models, SphereEvalCache& sphere_cache) {
	std::vector<ConstraintRow> rows;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			if (per_agent_obstacles[ag].empty()) continue;
			const int axis_off = agent_axis_offsets[ag];
			const std::vector<WorkspaceSphere>& spheres =
				sphere_cache.Get(ag, i, points, agent_ambient_offsets[ag], *models[ag]);
			for (const ActiveObstacle& ao : per_agent_obstacles[ag]) {
				if (i < ao.step_lo || i >= ao.step_hi) continue;
				for (int k : ao.spheres[i - ao.step_lo]) {
					const WorkspaceSphere& sph = spheres[k];
					const Eigen::VectorXd c = sph.center.head(workspace_dim);
					const SdfResult sdf = (ao.obstacle->kind == ObstacleKind::kSphere)
						? SphereSdf(c, *ao.obstacle, workspace_dim)
						: BoxSdf(c, *ao.obstacle, workspace_dim);

					ConstraintRow row;
					row.value = sdf.value - sph.radius;
					row.coeffs.reserve(workspace_dim);
					AppendSphereJacobianRow(&row, sdf.grad, sph.jac, axis_off, i, num_steps, 1.0);
					rows.push_back(std::move(row));
				}
			}
		}
	}
	return rows;
}

namespace {

// The pair-separation radius contributed by one agent's sphere: a trivial
// (single-point) agent contributes the solver's scalar _agent_radii(ag) --
// preserving the point-agent inter-agent separation exactly -- while a
// registered model contributes that body sphere's own radius.
double PairRadius(const AgentCollisionModel& model, const WorkspaceSphere& sph, int ag,
		  const Eigen::VectorXd& agent_radii) {
	return model.is_trivial() ? agent_radii(ag) : sph.radius;
}

}  // namespace

double EvaluateAgentPairViolation(int num_steps, const std::vector<int>& agent_ambient_offsets, int workspace_dim,
				   const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
				   const std::vector<ActivePair>& active_pairs, const AgentCollisionModels& models,
				   SphereEvalCache& sphere_cache) {
	double violation = 0.0;
	for (int i = 0; i < num_steps; ++i) {
		for (const ActivePair& ap : active_pairs) {
			if (i < ap.step_lo || i >= ap.step_hi) continue;
			const int ag_a = ap.ag_a, ag_b = ap.ag_b;
			const std::vector<WorkspaceSphere>& sph_a =
				sphere_cache.Get(ag_a, i, points, agent_ambient_offsets[ag_a], *models[ag_a]);
			const std::vector<WorkspaceSphere>& sph_b =
				sphere_cache.Get(ag_b, i, points, agent_ambient_offsets[ag_b], *models[ag_b]);
			for (const auto& [ka, kb] : ap.sphere_pairs[i - ap.step_lo]) {
				const WorkspaceSphere& sa = sph_a[ka];
				const WorkspaceSphere& sb = sph_b[kb];
				const Eigen::VectorXd c_a = sa.center.head(workspace_dim);
				const Eigen::VectorXd c_b = sb.center.head(workspace_dim);
				const double r_a = PairRadius(*models[ag_a], sa, ag_a, agent_radii);
				const double r_b = PairRadius(*models[ag_b], sb, ag_b, agent_radii);
				const double value = SphereSdfCore(c_b, c_a, r_a + r_b).value;
				violation += std::max(0.0, -value);
			}
		}
	}
	return violation;
}

std::vector<ConstraintRow> LinearizeAgentPairConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, const std::vector<int>& agent_ambient_offsets,
	int workspace_dim, const Eigen::MatrixXd& points, const Eigen::VectorXd& agent_radii,
	const std::vector<ActivePair>& active_pairs, const AgentCollisionModels& models,
	SphereEvalCache& sphere_cache) {
	std::vector<ConstraintRow> rows;
	for (int i = 0; i < num_steps; ++i) {
		for (const ActivePair& ap : active_pairs) {
			if (i < ap.step_lo || i >= ap.step_hi) continue;
			const int ag_a = ap.ag_a, ag_b = ap.ag_b;
			const std::vector<WorkspaceSphere>& sph_a =
				sphere_cache.Get(ag_a, i, points, agent_ambient_offsets[ag_a], *models[ag_a]);
			const std::vector<WorkspaceSphere>& sph_b =
				sphere_cache.Get(ag_b, i, points, agent_ambient_offsets[ag_b], *models[ag_b]);
			for (const auto& [ka, kb] : ap.sphere_pairs[i - ap.step_lo]) {
				const WorkspaceSphere& sa = sph_a[ka];
				const WorkspaceSphere& sb = sph_b[kb];
				const Eigen::VectorXd c_a = sa.center.head(workspace_dim);
				const Eigen::VectorXd c_b = sb.center.head(workspace_dim);
				const double r_a = PairRadius(*models[ag_a], sa, ag_a, agent_radii);
				const double r_b = PairRadius(*models[ag_b], sb, ag_b, agent_radii);

				// Treat sphere b's centre as sphere a's "obstacle center"
				// (SphereSdfCore's math is symmetric in shape; only the
				// sign of the chain rule differs per side). value =
				// ||c_b - c_a|| - (r_a + r_b), so d(value)/d(c_a) = -grad,
				// d(value)/d(c_b) = +grad, each chained through that
				// agent's sphere Jacobian.
				const SdfResult sdf = SphereSdfCore(c_b, c_a, r_a + r_b);

				ConstraintRow row;
				row.value = sdf.value;
				row.coeffs.reserve(2 * workspace_dim);
				AppendSphereJacobianRow(&row, sdf.grad, sa.jac, agent_axis_offsets[ag_a], i,
							num_steps, -1.0);
				AppendSphereJacobianRow(&row, sdf.grad, sb.jac, agent_axis_offsets[ag_b], i,
							num_steps, 1.0);
				rows.push_back(std::move(row));
			}
		}
	}
	return rows;
}

namespace {

// Row-major flat index into an AgentSdfGrid's `values`/`gradient` buffers
// (last axis fastest -- see AgentSdfGrid's own doc comment), generalized
// over `idx.size()` axes so the same helper serves both workspace_dim=2
// and workspace_dim=3 without separate bilinear/trilinear special cases.
long GridFlatIndex(const Eigen::VectorXi& shape, const std::vector<int>& idx) {
	long flat = 0;
	for (int k = 0; k < static_cast<int>(idx.size()); ++k) {
		flat = flat * shape(k) + idx[k];
	}
	return flat;
}

}  // namespace

SdfSample QueryAgentSdfGrid(const AgentSdfGrid& grid, const Eigen::VectorXd& p, int workspace_dim) {
	const int d = workspace_dim;
	// Continuous cell coordinates, clamped into the grid's own extent
	// (constant/zero-order-hold extrapolation past the boundary -- see
	// this function's own doc comment) -- `i0[k]` is clamped to
	// `shape(k)-2` so `i0[k]+1` is always a valid vertex index.
	std::vector<int> i0(d);
	Eigen::VectorXd t(d);
	for (int k = 0; k < d; ++k) {
		double u = (p(k) - grid.origin(k)) / grid.resolution(k);
		u = std::clamp(u, 0.0, static_cast<double>(grid.shape(k) - 1));
		i0[k] = std::min(static_cast<int>(std::floor(u)), grid.shape(k) - 2);
		t(k) = u - i0[k];
	}

	const int num_corners = 1 << d;
	std::vector<int> idx(d);
	auto corner_value = [&](int bits) -> double {
		for (int k = 0; k < d; ++k) idx[k] = i0[k] + ((bits >> k) & 1);
		return grid.values(GridFlatIndex(grid.shape, idx));
	};
	auto corner_weight = [&](int bits, int skip_axis) -> double {
		double w = 1.0;
		for (int k = 0; k < d; ++k) {
			if (k == skip_axis) continue;
			w *= ((bits >> k) & 1) ? t(k) : (1.0 - t(k));
		}
		return w;
	};

	double value = 0.0;
	for (int bits = 0; bits < num_corners; ++bits) {
		value += corner_weight(bits, /*skip_axis=*/-1) * corner_value(bits);
	}

	Eigen::VectorXd grad(d);
	if (grid.gradient.size() > 0) {
		// Interpolate the caller-supplied per-vertex gradient field the
		// SAME multilinear way as `value` (see AgentSdfGrid's own
		// comment for why this is the caller's own consistency
		// responsibility, unlike the derive-by-default path below).
		grad.setZero();
		for (int bits = 0; bits < num_corners; ++bits) {
			for (int k = 0; k < d; ++k) idx[k] = i0[k] + ((bits >> k) & 1);
			const long flat = GridFlatIndex(grid.shape, idx);
			grad += corner_weight(bits, /*skip_axis=*/-1) * grid.gradient.segment(flat * d, d);
		}
	} else {
		// Derive by differentiating the SAME multilinear interpolant
		// `value` was just computed from: d(value)/d(u_j), at fixed
		// other axes, is the (d-1)-linear interpolation (over every
		// OTHER axis only) of the forward difference along axis j --
		// then the chain rule (u_j = (p_j - origin_j)/resolution_j)
		// converts to d(value)/d(p_j).
		for (int j = 0; j < d; ++j) {
			double dvalue_duj = 0.0;
			for (int bits = 0; bits < num_corners; ++bits) {
				if ((bits >> j) & 1) continue;  // iterate the bit_j=0 half only
				const double w = corner_weight(bits, /*skip_axis=*/j);
				dvalue_duj += w * (corner_value(bits | (1 << j)) - corner_value(bits));
			}
			grad(j) = dvalue_duj / grid.resolution(j);
		}
	}

	return SdfSample{value - grid.margin, grad};
}

namespace {
// A grid's own AABB (workspace coordinates) for the pruning distance test
// below -- `margin` deliberately NOT included here (unlike ObstacleExtent's
// sphere/box radius+margin): a grid's `margin` shrinks its FEASIBLE region
// inward (see QueryAgentSdfGrid's `value - grid.margin`), it doesn't grow
// the region a query might plausibly matter over, which is what this AABB
// is for.
struct Aabb {
	Eigen::VectorXd lo, hi;
};
Aabb GridAabb(const AgentSdfGrid& grid) {
	const Eigen::VectorXd extent =
		(grid.shape.cast<double>().array() - 1.0).matrix().cwiseProduct(grid.resolution);
	return Aabb{grid.origin, grid.origin + extent};
}
// Conservative sphere-vs-AABB overlap test: distance from `center` to the
// AABB's nearest point, compared against `radius` -- 0 if `center` is
// already inside the box.
bool SphereOverlapsAabb(const Eigen::VectorXd& center, double radius, const Aabb& box) {
	double dist_sq = 0.0;
	for (int k = 0; k < center.size(); ++k) {
		const double c = std::clamp(center(k), box.lo(k), box.hi(k));
		const double diff = center(k) - c;
		dist_sq += diff * diff;
	}
	return dist_sq <= radius * radius;
}
}  // namespace

std::vector<ActiveGrid> PruneAgentSdfGridsByDistance(
	int num_steps, int num_agents, const std::vector<AgentReferenceSpheres>& ref_spheres,
	const AgentCollisionModels& models, const ObstacleSet& obstacles, double prune_margin) {
	std::vector<ActiveGrid> active_grids(num_agents);
	for (int ag = 0; ag < num_agents; ++ag) {
		const AgentSdfGrid* grid = obstacles.agent_sdf_grid(ag);
		if (!grid) continue;
		const AgentReferenceSpheres& rs = ref_spheres[ag];
		const double margin =
			EffectiveMargin(prune_margin, models[ag]->broadphase_margin_hint());
		const Aabb aabb = GridAabb(*grid);
		if (!SphereOverlapsAabb(rs.bound.center, rs.bound.radius + rs.max_radius + margin,
					aabb)) {
			continue;  // coarse whole-trajectory filter
		}
		const int K = static_cast<int>(rs.radii.size());
		auto near_at = [&](int i, int k) {
			return SphereOverlapsAabb(rs.centers[i].row(k).transpose(),
						  rs.radii(k) + margin, aabb);
		};
		const auto [lo, hi] = StepRange(num_steps, [&](int i) {
			for (int k = 0; k < K; ++k)
				if (near_at(i, k)) return true;
			return false;
		});
		if (lo >= hi) continue;
		std::vector<std::vector<int>> spheres(hi - lo);
		for (int i = lo; i < hi; ++i)
			for (int k = 0; k < K; ++k)
				if (near_at(i, k)) spheres[i - lo].push_back(k);
		active_grids[ag] = ActiveGrid{grid, lo, hi, std::move(spheres)};
	}
	return active_grids;
}

double EvaluateAgentSdfGridViolation(int num_steps, int num_agents, const std::vector<int>& agent_ambient_offsets,
				      int workspace_dim, const Eigen::MatrixXd& points,
				      const std::vector<ActiveGrid>& active_grids, const AgentCollisionModels& models,
				      SphereEvalCache& sphere_cache) {
	double violation = 0.0;
	for (int ag = 0; ag < num_agents; ++ag) {
		const ActiveGrid& ag_grid = active_grids[ag];
		if (!ag_grid.grid) continue;
		for (int i = ag_grid.step_lo; i < ag_grid.step_hi; ++i) {
			const std::vector<WorkspaceSphere>& spheres =
				sphere_cache.Get(ag, i, points, agent_ambient_offsets[ag], *models[ag]);
			for (int k : ag_grid.spheres[i - ag_grid.step_lo]) {
				const WorkspaceSphere& sph = spheres[k];
				const Eigen::VectorXd c = sph.center.head(workspace_dim);
				const double value =
					QueryAgentSdfGrid(*ag_grid.grid, c, workspace_dim).value - sph.radius;
				violation += std::max(0.0, -value);
			}
		}
	}
	return violation;
}

std::vector<ConstraintRow> LinearizeAgentSdfGridConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents,
	const std::vector<int>& agent_ambient_offsets, int workspace_dim, const Eigen::MatrixXd& points,
	const std::vector<ActiveGrid>& active_grids, const AgentCollisionModels& models,
	SphereEvalCache& sphere_cache) {
	std::vector<ConstraintRow> rows;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			const ActiveGrid& ag_grid = active_grids[ag];
			if (!ag_grid.grid || i < ag_grid.step_lo || i >= ag_grid.step_hi) continue;
			const int axis_off = agent_axis_offsets[ag];
			const std::vector<WorkspaceSphere>& spheres =
				sphere_cache.Get(ag, i, points, agent_ambient_offsets[ag], *models[ag]);
			for (int k : ag_grid.spheres[i - ag_grid.step_lo]) {
				const WorkspaceSphere& sph = spheres[k];
				const Eigen::VectorXd c = sph.center.head(workspace_dim);
				const SdfSample sdf = QueryAgentSdfGrid(*ag_grid.grid, c, workspace_dim);

				ConstraintRow row;
				row.value = sdf.value - sph.radius;
				row.coeffs.reserve(workspace_dim);
				AppendSphereJacobianRow(&row, sdf.grad, sph.jac, axis_off, i, num_steps, 1.0);
				rows.push_back(std::move(row));
			}
		}
	}
	return rows;
}

}  // namespace sqp_short_path
