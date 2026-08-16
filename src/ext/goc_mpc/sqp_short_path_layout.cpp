#include "sqp_short_path_layout.hpp"

#include <algorithm>
#include <cmath>
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
			if (b.type != Block::Type::R && b.type != Block::Type::Torus) {
				throw std::runtime_error(
					"SqpShortPathMPC: only Block::R/Block::Torus are supported "
					"(agent " + std::to_string(ag) + " has an SO3Quat/SO3Mat "
					"block -- see the project plan's Stage 4)");
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

Eigen::MatrixXd BuildAxisHessianBlock(int num_steps, double tau) {
	const int n = 2 * num_steps;
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	const double tau2 = tau * tau;

	// Same coefficient pattern as AdmmShortPathMPC::build_hessian
	// (admm_short_path_mpc.cpp) -- tracking, velocity-tracking, and the
	// coast-corrected acceleration residual's linear-in-(p,v) coefficients,
	// none of which depend on the current iterate (only the RHS/target
	// does, see BuildAxisRhs) -- so this block is reused unchanged across
	// every outer SQP iteration within one solve() call.
	auto add_h = [&](std::initializer_list<std::pair<int, double>> terms, double weight) {
		for (auto [i, ci] : terms) {
			for (auto [j, cj] : terms) {
				H(i, j) += weight * ci * cj;
			}
		}
	};

	for (int i = 0; i < num_steps; ++i) {
		add_h({{IdxP(0, i, num_steps), 1.0}}, 1.0);  // tracking (axis-local index 0 below)
		add_h({{IdxV(0, i, num_steps), 1.0}}, 1.0);  // velocity tracking
		if (i == 0) {
			add_h({{IdxP(0, 0, num_steps), -6.0 / tau2}, {IdxV(0, 0, num_steps), 4.0 / tau}}, 1.0);
		} else {
			add_h({{IdxP(0, i, num_steps), -6.0 / tau2}, {IdxP(0, i - 1, num_steps), 6.0 / tau2},
			       {IdxV(0, i, num_steps), 4.0 / tau}, {IdxV(0, i - 1, num_steps), 2.0 / tau}},
			      1.0);
		}
	}
	return H;
}

Eigen::MatrixXd AssembleSmoothHessian(const std::vector<AxisLayout>& axes, int num_steps, double tau) {
	const Eigen::MatrixXd block = BuildAxisHessianBlock(num_steps, tau);
	const int per_axis = 2 * num_steps;
	const int n = static_cast<int>(axes.size()) * per_axis;
	Eigen::MatrixXd H = Eigen::MatrixXd::Zero(n, n);
	for (int a = 0; a < static_cast<int>(axes.size()); ++a) {
		H.block(a * per_axis, a * per_axis, per_axis, per_axis) = block;
	}
	return H;
}

Eigen::VectorXd BuildAxisRhs(const AxisLayout& axis,
			      const CubicConfigurationSpline& agent_shape,
			      int num_steps, double tau,
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
		add_g({{IdxP(0, i, num_steps), 1.0}}, track_delta(k), 1.0);

		// Velocity tracking: velocities live in a flat tangent space
		// already (no manifold/wraparound), so this is always a plain
		// subtraction, R or Torus alike.
		const double vtrack_target = ref_velocities_agent(i, k) - vels_agent(i, k);
		add_g({{IdxV(0, i, num_steps), 1.0}}, vtrack_target, 1.0);

		if (i == 0) {
			// disp0 = p_current(0) - x0, wrap-aware -- the coefficient
			// pattern is identical to AdmmShortPathMPC's i==0 branch
			// (build_rhs), only the target changes (linearized around the
			// current iterate instead of being an absolute-coordinate
			// target against x0 directly -- see this file's own top
			// comment / the project plan's design decision 3 for the
			// derivation).
			const Eigen::VectorXd disp0 =
				agent_shape.PositionDelta<double>(points_agent.row(0).transpose(), x0_agent);
			const double target0 =
				(6.0 / tau2) * disp0(k) - (4.0 / tau) * vels_agent(0, k) - (2.0 / tau) * v0_agent(k);
			add_g({{IdxP(0, 0, num_steps), -6.0 / tau2}, {IdxV(0, 0, num_steps), 4.0 / tau}}, target0, 1.0);
		} else {
			const Eigen::VectorXd disp = agent_shape.PositionDelta<double>(
				points_agent.row(i).transpose(), points_agent.row(i - 1).transpose());
			const double target =
				(6.0 / tau2) * disp(k) - (4.0 / tau) * vels_agent(i, k) - (2.0 / tau) * vels_agent(i - 1, k);
			add_g({{IdxP(0, i, num_steps), -6.0 / tau2}, {IdxP(0, i - 1, num_steps), 6.0 / tau2},
			       {IdxV(0, i, num_steps), 4.0 / tau}, {IdxV(0, i - 1, num_steps), 2.0 / tau}},
			      target, 1.0);
		}
	}
	return g;
}

double EvaluateSmoothCost(const CubicConfigurationSpline& agent_shape, int num_steps, double tau,
			   const Eigen::VectorXd& x0_agent, const Eigen::VectorXd& v0_agent,
			   const Eigen::MatrixXd& points_agent, const Eigen::MatrixXd& vels_agent,
			   const Eigen::MatrixXd& ref_points_agent, const Eigen::MatrixXd& ref_velocities_agent) {
	double f = 0.0;
	const double tau2 = tau * tau;
	for (int i = 0; i < num_steps; ++i) {
		const Eigen::VectorXd track_delta = agent_shape.PositionDelta<double>(
			ref_points_agent.row(i).transpose(), points_agent.row(i).transpose());
		f += track_delta.squaredNorm();

		const Eigen::VectorXd vtrack_delta =
			(ref_velocities_agent.row(i) - vels_agent.row(i)).transpose();
		f += vtrack_delta.squaredNorm();

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
		f += accel.squaredNorm();
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

SdfResult SphereSdf(const Eigen::VectorXd& p, const Obstacle& obstacle, int workspace_dim) {
	const Eigen::VectorXd center = obstacle.params.segment(0, workspace_dim);
	const double R = obstacle.params(workspace_dim) + obstacle.margin;
	const Eigen::VectorXd diff = p - center;
	const double d = std::sqrt(diff.squaredNorm() + kSqrtEps2);
	return SdfResult{d - R, diff / d};
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

double EvaluateObstacleViolation(int num_steps, int num_agents, int dim, int workspace_dim,
				  const Eigen::MatrixXd& points, const ObstacleSet& obstacles) {
	if (obstacles.obstacles().empty()) {
		return 0.0;
	}
	double violation = 0.0;
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			const Eigen::VectorXd p = points.row(i).segment(ag * dim, workspace_dim).transpose();
			for (const Obstacle& obstacle : obstacles.obstacles()) {
				const double value = (obstacle.kind == ObstacleKind::kSphere)
					? SphereSdf(p, obstacle, workspace_dim).value
					: BoxSdf(p, obstacle, workspace_dim).value;
				violation += std::max(0.0, -value);
			}
		}
	}
	return violation;
}

std::vector<ConstraintRow> LinearizeObstacleConstraints(
	const std::vector<int>& agent_axis_offsets, int num_steps, int num_agents, int dim,
	int workspace_dim, const Eigen::MatrixXd& points, const ObstacleSet& obstacles) {
	std::vector<ConstraintRow> rows;
	if (obstacles.obstacles().empty()) {
		return rows;
	}
	for (int i = 0; i < num_steps; ++i) {
		for (int ag = 0; ag < num_agents; ++ag) {
			const Eigen::VectorXd p = points.row(i).segment(ag * dim, workspace_dim).transpose();
			for (const Obstacle& obstacle : obstacles.obstacles()) {
				const SdfResult sdf = (obstacle.kind == ObstacleKind::kSphere)
					? SphereSdf(p, obstacle, workspace_dim)
					: BoxSdf(p, obstacle, workspace_dim);

				ConstraintRow row;
				row.value = sdf.value;
				row.coeffs.reserve(workspace_dim);
				for (int k = 0; k < workspace_dim; ++k) {
					// fk fast path: workspace column k of agent ag IS
					// tangent axis (agent_axis_offsets[ag] + k) at this
					// step -- the leading workspace_dim columns of an
					// R/Torus agent are always position (R), so ambient
					// column k and tangent column k coincide.
					const int axis = agent_axis_offsets[ag] + k;
					row.coeffs.emplace_back(IdxP(axis, i, num_steps), sdf.grad(k));
				}
				rows.push_back(std::move(row));
			}
		}
	}
	return rows;
}

}  // namespace sqp_short_path
