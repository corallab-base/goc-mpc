#include "milp_waypoint_mpc.hpp"
#include "symbolic_constraint_compiler.hpp"

#include <algorithm>
#include <map>
#include <tuple>

using namespace pybind11::literals;
namespace py = pybind11;

using Eigen::VectorX;
using drake::math::RotationMatrix;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using drake::solvers::VectorXDecisionVariable;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::MathematicalProgram;
using namespace pybind11::literals;
namespace py = pybind11;


struct HoldSpec {
	// which object indices (or keypoint indices) are held over this edge
	std::vector<int> held_point_ids;
	// robot index (agent) that holds them — STATIC case
	int robot_ag;
	// optional: end-effector frame name for this robot
	std::string ee_frame_name;  // e.g., "tool0"
};

struct AssignableHoldSpec {
	// which object indices (or keypoint indices) are held over this edge
	std::vector<int> held_point_ids;
	// variable id for robot index (agent) that holds them
	int var;
	// optional: end-effector frame name for this robot
	std::string ee_frame_name;  // e.g., "tool0"
};

inline VectorXDecisionVariable GetARowForVar(const SubgraphOfConstraints& subgraph,
					     const MatrixXDecisionVariable& Assignments,
					     int global_var_id) {
	const int row = subgraph.subgraph_variable_id(global_var_id);
	DRAKE_DEMAND(row >= 0 && row < Assignments.rows());
	return Assignments.row(row);
}

inline Eigen::VectorXd seg_width3(const Eigen::VectorXd& lb,
                                  const Eigen::VectorXd& ub,
                                  int start3, int dim) {
	return (ub.segment(start3, dim) - lb.segment(start3, dim)).cwiseAbs();
}

// Bound ||p_WP - p_WE||_2 on one “side” (0, u, or v).
// If EE is free-body: ee_start3 = robot_ag * robot_dim (where the pos lives).
// If articulated: pass ee_span_hint as a per-axis bound on EE motion; otherwise use a conservative constant.
// `dim` is the robot's workspace_dim (2 or 3) -- both segments read are
// workspace_dim-wide, matching PointWorldFromRow/PoseFromRow's own sizing.
inline double bound_norm_obj_minus_ee_side(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_in_X,              // std::nullopt if articulated
	const Eigen::VectorXd& ee_span_hint, /* meters */
	int dim) {

	const Eigen::VectorXd w_obj = seg_width3(lb, ub, obj_start3, dim);
	Eigen::VectorXd w_ee;
	if (ee_start3_in_X) {
		w_ee = seg_width3(lb, ub, *ee_start3_in_X, dim);
	} else {
		w_ee = ee_span_hint.cwiseAbs();  // articulated, conservative workspace span
	}
	const Eigen::VectorXd w_sum = w_obj + w_ee;
	return w_sum.norm();  // sqrt(sum((w_axis+e_axis)^2))
}

// 0 -> v (initial-layer) exact-rigidity M (identical per component)
inline Eigen::VectorXd M_exact_toX0_componentwise(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_in_X,      // free-body -> pos segment start; else nullopt
	const Eigen::VectorXd& ee_span_hint,    // used when articulated
	int dim) {
	// Both “sides” are 0 and v; the bound structure is the same for both,
	// so the worst-case 2-norm can be upper-bounded the same way on each side.
	const double B0 = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_in_X, ee_span_hint, dim);
	const double Bv = B0;  // same bounding box globally
	const double Mscalar = B0 + Bv;
	return Eigen::VectorXd::Constant(dim, Mscalar);
}

// u -> v exact-rigidity M (identical per component)
inline Eigen::VectorXd M_exact_edge_componentwise(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_u_in_X,    // free-body u; else nullopt
	std::optional<int> ee_start3_v_in_X,    // free-body v; else nullopt
	const Eigen::VectorXd& ee_span_hint_u,  // articulated u
	const Eigen::VectorXd& ee_span_hint_v,  // articulated v
	int dim) {
	const double Bu = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_u_in_X, ee_span_hint_u, dim);
	const double Bv = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_v_in_X, ee_span_hint_v, dim);
	const double Mscalar = Bu + Bv;
	return Eigen::VectorXd::Constant(dim, Mscalar);
}

static void AddHoldRigidityStaticToX0(
	MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints* graph,
	const MatrixXDecisionVariable& X,
	int v,                               // vertex id in the ORIGINAL graph
	const HoldSpec& spec,
	int robot_dim,
	int objs_start,
	int non_robot_dim,
	const Eigen::VectorXd& x0,
	bool exact_rigidity) {

	const int sg_v = subgraph.subgraph_id(v);
	const int workspace_dim = graph->workspace_dim;

	Eigen::Matrix<Expression, Eigen::Dynamic, 1> x0_expr = x0.template cast<Expression>();
	Eigen::Matrix<Expression, Eigen::Dynamic, 1> X_v_expr = X.row(sg_v).template cast<Expression>();

	auto [p_WE_0, R_WE_0] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, x0_expr);
	auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_v_expr);

	if (exact_rigidity) {
		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_0 =
				PointWorldFromRow(x0_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);

			const Eigen::VectorX<Expression> rel_0 =
				R_WE_0.transpose() * (p_WP_0 - p_WE_0);
			const Eigen::VectorX<Expression> rel_v =
				R_WE_v.transpose() * (p_WP_v - p_WE_v);


			switch (graph->robot_rigidity_constraint_degree(spec.robot_ag)) {
			case ConstraintDegree::kQuadratic:
				for (int i = 0; i < workspace_dim; ++i)
					prog.AddQuadraticConstraint(rel_0(i) - rel_v(i), -0.001, 0.001)
						.evaluator()->set_description(fmt::format("-1->{} exact rigidity {}", v, obj_id));
				break;
			case ConstraintDegree::kLinear:
				prog.AddLinearConstraint(rel_0 - rel_v,
						Eigen::VectorXd::Constant(workspace_dim, -0.001),
						Eigen::VectorXd::Constant(workspace_dim, 0.001))
					.evaluator()->set_description(fmt::format("-1->{} exact rigidity {}", v, obj_id));
				break;
			case ConstraintDegree::kGeneral:
				prog.AddConstraint(rel_0 - rel_v,
						Eigen::VectorXd::Constant(workspace_dim, -0.001),
						Eigen::VectorXd::Constant(workspace_dim, 0.001))
					.evaluator()->set_description(fmt::format("-1->{} exact rigidity {}", v, obj_id));
				break;
			}



		}
	} else {
		for (size_t i = 0; i < spec.held_point_ids.size(); ++i) {
			for (size_t j = i + 1; j < spec.held_point_ids.size(); ++j) {
				const int id_i = spec.held_point_ids[i];
				const int id_j = spec.held_point_ids[j];

				const Eigen::VectorX<Expression> p_i_0 =
					PointWorldFromRow(x0_expr, objs_start, non_robot_dim, id_i, workspace_dim);
				const Eigen::VectorX<Expression> p_i_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_i, workspace_dim);
				const Eigen::VectorX<Expression> p_j_0 =
					PointWorldFromRow(x0_expr, objs_start, non_robot_dim, id_j, workspace_dim);
				const Eigen::VectorX<Expression> p_j_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_j, workspace_dim);

				const Eigen::VectorX<Expression> d_0 = p_i_0 - p_j_0;
				const Eigen::VectorX<Expression> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_0.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description(fmt::format("-1->{} p2p distance rigidity", v));
			}
		}


		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_0 =
				PointWorldFromRow(x0_expr,   objs_start, non_robot_dim, obj_id, workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);

			const Eigen::VectorX<Expression> d_0 = p_WP_0 - p_WE_0;
			const Eigen::VectorX<Expression> d_v = p_WP_v - p_WE_v;

			Expression e = d_v.squaredNorm() - d_0.squaredNorm();
			prog.AddQuadraticConstraint(e, 0.0, 0.0)
				.evaluator()->set_description("-1->v r2p distance rigidity");
		}
	}
}

// Add 3 equality constraints for each held point: R^T(p_P - p_E) (u) == R^T(p_P - p_E) (v).
// (I.e., Undoing the end effector rotation on the vector between the point and
// the end effector at node u yields the same as undoing the end effector
// rotation on the vector between the point and the end effector at node v.)
static void AddHoldRigidityStatic(
	MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints* graph,
	const MatrixXDecisionVariable& X,
	int u, int v,                         // graph vertex ids in the *original* graph
	const HoldSpec& spec,
	int robot_dim,
	int objs_start,
	int non_robot_dim,
	bool exact_rigidity) {

	// Map original graph ids to subgraph row ids
	const int sg_u = subgraph.subgraph_id(u);
	const int sg_v = subgraph.subgraph_id(v);
	const int workspace_dim = graph->workspace_dim;

        // Cast Variables -> Expressions and materialize as owning row vectors.
	Eigen::VectorX<drake::symbolic::Expression> X_u_expr =
		X.row(sg_u).template cast<drake::symbolic::Expression>();
	Eigen::VectorX<drake::symbolic::Expression> X_v_expr =
		X.row(sg_v).template cast<drake::symbolic::Expression>();

	auto [p_WE_u, R_WE_u] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_u_expr);
	auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_v_expr);

	if (exact_rigidity) {
		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_u =
				PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);

			const Eigen::VectorX<Expression> rel_u =
				R_WE_u.transpose() * (p_WP_u - p_WE_u);
			const Eigen::VectorX<Expression> rel_v =
				R_WE_v.transpose() * (p_WP_v - p_WE_v);


			switch (graph->robot_rigidity_constraint_degree(spec.robot_ag)) {
			case ConstraintDegree::kQuadratic:
				for (int i = 0; i < workspace_dim; ++i)
					prog.AddQuadraticConstraint(rel_u(i) - rel_v(i), -0.001, 0.001)
						.evaluator()->set_description(fmt::format("{}->{} exact rigidity {}", u, v, obj_id));
				break;
			case ConstraintDegree::kLinear:
				prog.AddLinearConstraint(rel_u - rel_v,
						Eigen::VectorXd::Constant(workspace_dim, -0.001),
						Eigen::VectorXd::Constant(workspace_dim, 0.001))
					.evaluator()->set_description(fmt::format("{}->{} exact rigidity {}", u, v, obj_id));
				break;
			case ConstraintDegree::kGeneral:
				prog.AddConstraint(rel_u - rel_v,
						Eigen::VectorXd::Constant(workspace_dim, -0.001),
						Eigen::VectorXd::Constant(workspace_dim, 0.001))
					.evaluator()->set_description(fmt::format("{}->{} exact rigidity {}", u, v, obj_id));
				break;
			}
		}
	} else {
		for (size_t i = 0; i < spec.held_point_ids.size(); ++i) {
			for (size_t j = i + 1; j < spec.held_point_ids.size(); ++j) {
				const int id_i = spec.held_point_ids[i];
				const int id_j = spec.held_point_ids[j];

				const Eigen::VectorX<Expression> p_i_u =
					PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, id_i, workspace_dim);
				const Eigen::VectorX<Expression> p_i_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_i, workspace_dim);
				const Eigen::VectorX<Expression> p_j_u =
					PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, id_j, workspace_dim);
				const Eigen::VectorX<Expression> p_j_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_j, workspace_dim);

				const Eigen::VectorX<Expression> d_u = p_i_u - p_j_u;
				const Eigen::VectorX<Expression> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_u.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description("u->v p2p distance rigidity");
			}
		}

		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_u =
				PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);

			const Eigen::VectorX<Expression> d_u = p_WP_u - p_WE_u;
			const Eigen::VectorX<Expression> d_v = p_WP_v - p_WE_v;

			Expression e = d_v.squaredNorm() - d_u.squaredNorm();
			prog.AddQuadraticConstraint(e, 0.0, 0.0)
				.evaluator()->set_description("u->v r2p distance rigidity");
		}
	}
}

void AddHoldRigidityAssignableToX0(
	MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints* graph,
	const MatrixXDecisionVariable& X,
	int v,
	const AssignableHoldSpec& spec,
	int robot_dim, int objs_start, int non_robot_dim,
	const Eigen::VectorXd& x0,
	const VectorXDecisionVariable& A_row) {

	using Eigen::Matrix;
	using Eigen::Vector3d;

	const int num_agents = graph->num_agents;
	const int sg_v = subgraph.subgraph_id(v);
	const int workspace_dim = graph->workspace_dim;

	// Cast once.
	Eigen::VectorX<Expression> x0_expr = x0.template cast<Expression>();
	Eigen::VectorX<Expression> X_v_expr = X.row(sg_v).template cast<Expression>();

	// Precompute object world points from x0 / X_v once per object.
	struct ObjPts {
		int id;
		Eigen::VectorX<Expression> p_WP_0;
		Eigen::VectorX<Expression> p_WP_v;
	};
	std::vector<ObjPts> objects;
	objects.reserve(spec.held_point_ids.size());
	for (int obj_id : spec.held_point_ids) {
		ObjPts o;
		o.id = obj_id;
		o.p_WP_0 = PointWorldFromRow(x0_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
		o.p_WP_v = PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
		objects.push_back(std::move(o));
	}

	const double neg_inf = -std::numeric_limits<double>::infinity();
	const Eigen::VectorXd ee_span_hint = Eigen::VectorXd::Constant(workspace_dim, 2.0);

	// For each possible agent k, construct the same residual as static-HoldSpec(robot_ag=k),
	// then gate with +/- M * (1 - A_row(k)).
	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_0, R_WE_0] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, x0_expr);
		auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_v_expr);

		// For free-body agent k:
		const int ee_start3 = k * robot_dim;  // assuming first workspace_dim coords are world position

		// Gate each held point’s exact-rigidity residual.
		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_0 = objects[idx].p_WP_0;
			const auto& p_WP_v = objects[idx].p_WP_v;

			// For each held object (obj_id):
			const int obj_start3 = objs_start + objects[idx].id * non_robot_dim;

			const Eigen::VectorXd Mvec = M_exact_toX0_componentwise(
				graph->_global_x_lb, graph->_global_x_ub,
				obj_start3,
				/*ee_start3_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3)
				: std::nullopt,
				/*ee_span_hint=*/ee_span_hint, workspace_dim);

			// Exact rigidity residual components:
			// rel_0 = R_WE_0^T (p_WP_0 - p_WE_0)
			// rel_v = R_WE_v^T (p_WP_v - p_WE_v)
			// residual = rel_0 - rel_v   (should be ~ 0 when this agent is selected)
			const Eigen::VectorX<Expression> rel_0 = R_WE_0.transpose() * (p_WP_0 - p_WE_0);
			const Eigen::VectorX<Expression> rel_v = R_WE_v.transpose() * (p_WP_v - p_WE_v);
			const Eigen::VectorX<Expression> residual = rel_0 - rel_v;

			// Big-M gating:
			// -M*(1 - A_k) <= residual_i <= M*(1 - A_k)
			// Move to LHS to keep constant bounds:
			// residual_i - M*(1 - A_k) <= 0
			// -residual_i - M*(1 - A_k) <= 0
			for (int j = 0; j < workspace_dim; ++j) {
				const Expression slack = Mvec(j) * (1.0 - A_row(k));


				switch (graph->robot_rigidity_constraint_degree(k)) {
				case ConstraintDegree::kQuadratic:
					prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, +)");
					prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, -)");
					break;
				case ConstraintDegree::kLinear:
					prog.AddLinearConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, +)");
					prog.AddLinearConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, -)");
					break;
				case ConstraintDegree::kGeneral:
					prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, +)");
					prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, -)");
					break;
				}
			}
		}
	}
}

void AddHoldRigidityAssignable(
	MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints* graph,
	const MatrixXDecisionVariable& X,
	int u, int v,
	const AssignableHoldSpec& spec,
	int robot_dim, int objs_start, int non_robot_dim,
	const VectorXDecisionVariable& A_row) {

	using drake::symbolic::Expression;
	using Eigen::Matrix;

	// std::cout << "ADDING ASSIGNABLE HOLD SPEC FOR " << u << "->" << v << std::endl;

	// Map original graph ids to subgraph row ids
	const int sg_u = subgraph.subgraph_id(u);
	const int sg_v = subgraph.subgraph_id(v);
	const int workspace_dim = graph->workspace_dim;

	// Cast Variables -> Expressions
	Eigen::VectorX<Expression> X_u_expr = X.row(sg_u).template cast<Expression>();
	Eigen::VectorX<Expression> X_v_expr = X.row(sg_v).template cast<Expression>();

	const int num_agents = graph->num_agents;

	// Precompute object world points for all held ids at u and v.
	struct ObjPtsUV {
		int id;
		Eigen::VectorX<Expression> p_WP_u;
		Eigen::VectorX<Expression> p_WP_v;
	};
	std::vector<ObjPtsUV> objects;
	objects.reserve(spec.held_point_ids.size());
	for (int obj_id : spec.held_point_ids) {
		ObjPtsUV o;
		o.id = obj_id;
		o.p_WP_u = PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
		o.p_WP_v = PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id, workspace_dim);
		objects.push_back(std::move(o));
	}

	const double neg_inf = -std::numeric_limits<double>::infinity();
	const Eigen::VectorXd ee_span_hint = Eigen::VectorXd::Constant(workspace_dim, 2.0);

	// For each possible agent k, construct the same residual as static-HoldSpec(robot_ag=k),
	// then gate with +/- M * (1 - A_row(k)).
	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_u, R_WE_u] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_u_expr);
		auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_v_expr);

		// For free-body agent k:
		const int ee_start3 = k * robot_dim;  // assuming first workspace_dim coords are world position

		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_u = objects[idx].p_WP_u;
			const auto& p_WP_v = objects[idx].p_WP_v;

			// Exact rigidity residual across the edge:
			// rel_u = R_WE_u^T (p_WP_u - p_WE_u)
			// rel_v = R_WE_v^T (p_WP_v - p_WE_v)
			// residual = rel_u - rel_v  ≈ 0 when this agent is selected
			const Eigen::VectorX<Expression> rel_u = R_WE_u.transpose() * (p_WP_u - p_WE_u);
			const Eigen::VectorX<Expression> rel_v = R_WE_v.transpose() * (p_WP_v - p_WE_v);
			const Eigen::VectorX<Expression> residual = rel_u - rel_v;

			// For each held object (obj_id):
			const int obj_start3 = objs_start + objects[idx].id * non_robot_dim;

			// u->v:
			const Eigen::VectorXd Mvec_uv = M_exact_edge_componentwise(
				graph->_global_x_lb, graph->_global_x_ub,
				obj_start3,
				/*ee_start3_u_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				/*ee_start3_v_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				/*ee_span_hint_u=*/ee_span_hint,
				/*ee_span_hint_v=*/ee_span_hint, workspace_dim);

			// Big-M gating per component:
			// residual_j - M*(1 - A_k) <= 0
			// -residual_j - M*(1 - A_k) <= 0

			for (int j = 0; j < workspace_dim; ++j) {
				const Expression slack = Mvec_uv(j) * (1.0 - A_row(k));


				switch (graph->robot_rigidity_constraint_degree(k)) {
				case ConstraintDegree::kQuadratic:
					prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, +)");
					prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, -)");
					break;
				case ConstraintDegree::kLinear:
					prog.AddLinearConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, +)");
					prog.AddLinearConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, -)");
					break;
				case ConstraintDegree::kGeneral:
					prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, +)");
					prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, -)");
					break;
				}
			}
		}
	}
}


// Gated exact-rigidity check between an arbitrary reference config row and a
// candidate config row, for a single already-known holding agent (HoldSpec).
// Used to re-apply the endpoint rigidity invariant at an interior node that
// might be scheduled between a hold's own two endpoints — see Constraint 14
// in BuildGraphWaypointProblem. `gate` is expected to be forced to 1 when the
// candidate node is genuinely between the hold's endpoints (and free to be 0
// otherwise), mirroring GetOrAddBetweennessIndicator's contract.
static void AddHoldRigidityStaticGated(
	MathematicalProgram& prog,
	const GraphOfConstraints* graph,
	const Eigen::VectorX<Expression>& ref_expr,
	const Eigen::VectorX<Expression>& cand_expr,
	const HoldSpec& spec,
	int robot_dim, int objs_start, int non_robot_dim,
	const Variable& gate,
	const std::string& desc) {

	using Eigen::Matrix;
	const double neg_inf = -std::numeric_limits<double>::infinity();
	const int workspace_dim = graph->workspace_dim;

	const int k = spec.robot_ag;
	const int ee_start3 = k * robot_dim;
	const Eigen::VectorXd ee_span_hint = Eigen::VectorXd::Constant(workspace_dim, 2.0);

	auto [p_WE_ref, R_WE_ref]   = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, ref_expr);
	auto [p_WE_cand, R_WE_cand] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, cand_expr);

	for (int obj_id : spec.held_point_ids) {
		const Eigen::VectorX<Expression> p_WP_ref  = PointWorldFromRow(ref_expr,  objs_start, non_robot_dim, obj_id, workspace_dim);
		const Eigen::VectorX<Expression> p_WP_cand = PointWorldFromRow(cand_expr, objs_start, non_robot_dim, obj_id, workspace_dim);

		const Eigen::VectorX<Expression> rel_ref  = R_WE_ref.transpose()  * (p_WP_ref  - p_WE_ref);
		const Eigen::VectorX<Expression> rel_cand = R_WE_cand.transpose() * (p_WP_cand - p_WE_cand);
		const Eigen::VectorX<Expression> residual = rel_ref - rel_cand;

		const int obj_start3 = objs_start + obj_id * non_robot_dim;
		const Eigen::VectorXd Mvec = M_exact_edge_componentwise(
			graph->_global_x_lb, graph->_global_x_ub,
			obj_start3,
			graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
			graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
			ee_span_hint, ee_span_hint, workspace_dim);

		for (int j = 0; j < workspace_dim; ++j) {
			const Expression slack = Mvec(j) * (1.0 - Expression(gate));

			switch (graph->robot_rigidity_constraint_degree(k)) {
			case ConstraintDegree::kQuadratic:
				prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, +)");
				prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, -)");
				break;
			case ConstraintDegree::kLinear:
				prog.AddLinearConstraint(residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, +)");
				prog.AddLinearConstraint(-residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, -)");
				break;
			case ConstraintDegree::kGeneral:
				prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, +)");
				prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
					.evaluator()->set_description(desc + " (gated, -)");
				break;
			}
		}
	}
}

// Same as AddHoldRigidityStaticGated, but for an AssignableHoldSpec: loops
// over every candidate agent k (as AddHoldRigidityAssignable does), and
// additionally gates each residual by `gate` on top of the per-agent
// selection A_row(k) — tight only when both the agent is selected AND the
// candidate node is genuinely between the hold's endpoints.
static void AddHoldRigidityAssignableGated(
	MathematicalProgram& prog,
	const GraphOfConstraints* graph,
	const Eigen::VectorX<Expression>& ref_expr,
	const Eigen::VectorX<Expression>& cand_expr,
	const AssignableHoldSpec& spec,
	int robot_dim, int objs_start, int non_robot_dim,
	const VectorXDecisionVariable& A_row,
	const Variable& gate,
	const std::string& desc) {

	using Eigen::Matrix;
	const double neg_inf = -std::numeric_limits<double>::infinity();
	const int num_agents = graph->num_agents;
	const int workspace_dim = graph->workspace_dim;
	const Eigen::VectorXd ee_span_hint = Eigen::VectorXd::Constant(workspace_dim, 2.0);

	struct ObjPts {
		int id;
		Eigen::VectorX<Expression> p_WP_ref;
		Eigen::VectorX<Expression> p_WP_cand;
	};
	std::vector<ObjPts> objects;
	objects.reserve(spec.held_point_ids.size());
	for (int obj_id : spec.held_point_ids) {
		objects.push_back(ObjPts{
			obj_id,
			PointWorldFromRow(ref_expr,  objs_start, non_robot_dim, obj_id, workspace_dim),
			PointWorldFromRow(cand_expr, objs_start, non_robot_dim, obj_id, workspace_dim)});
	}

	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_ref, R_WE_ref]   = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, ref_expr);
		auto [p_WE_cand, R_WE_cand] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, cand_expr);
		const int ee_start3 = k * robot_dim;

		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_ref  = objects[idx].p_WP_ref;
			const auto& p_WP_cand = objects[idx].p_WP_cand;

			const Eigen::VectorX<Expression> rel_ref  = R_WE_ref.transpose()  * (p_WP_ref  - p_WE_ref);
			const Eigen::VectorX<Expression> rel_cand = R_WE_cand.transpose() * (p_WP_cand - p_WE_cand);
			const Eigen::VectorX<Expression> residual = rel_ref - rel_cand;

			const int obj_start3 = objs_start + objects[idx].id * non_robot_dim;
			const Eigen::VectorXd Mvec = M_exact_edge_componentwise(
				graph->_global_x_lb, graph->_global_x_ub,
				obj_start3,
				graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				ee_span_hint, ee_span_hint, workspace_dim);

			for (int j = 0; j < workspace_dim; ++j) {
				const Expression slack =
					Mvec(j) * ((1.0 - A_row(k)) + (1.0 - Expression(gate)));

				switch (graph->robot_rigidity_constraint_degree(k)) {
				case ConstraintDegree::kQuadratic:
					prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, +)");
					prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, -)");
					break;
				case ConstraintDegree::kLinear:
					prog.AddLinearConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, +)");
					prog.AddLinearConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, -)");
					break;
				case ConstraintDegree::kGeneral:
					prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, +)");
					prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description(desc + " (gated, -)");
					break;
				}
			}
		}
	}
}

// Compute the L1 distance between two symbolic configuration expressions,
// summed over a spline's R blocks only (non-Euclidean blocks — Torus, SO3 —
// are skipped; geodesic distance over those is not yet implemented). One
// slack variable t_i per component (t_i >= |x1_i - x2_i|) is introduced and
// the returned expression is sum(t) over all R blocks.
Expression ComputeL1Distance(
	MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<Expression>& x1,
	const VecX<Expression>& x2,
	const std::string& tag) {

	const double kInf = std::numeric_limits<double>::infinity();
	Expression dist(0.0);

	for (const auto& off : spline.block_offsets_) {
		if (off.type != CubicConfigurationSpline::Block::Type::R) continue;
		const int a0 = off.ambient_offset;
		const int aN = off.ambient_size;

		auto t = prog.NewContinuousVariables(aN, "l1_t_" + tag);
		prog.AddBoundingBoxConstraint(0.0, kInf, t);

		for (int i = 0; i < aN; ++i) {
			const Expression diff = x1(a0 + i) - x2(a0 + i);
			prog.AddLinearConstraint( diff - Expression(t(i)), -kInf, 0.0);
			prog.AddLinearConstraint(-diff - Expression(t(i)), -kInf, 0.0);
			dist += Expression(t(i));
		}
	}

	return dist;
}

// Compute the L2 (Euclidean) distance between two symbolic configuration
// expressions, summed over a spline's R blocks only (same scope as
// ComputeL1Distance). One nonnegative scalar d_blk per R block is introduced,
// constrained via a Lorentz cone (d_blk >= ||x1_blk - x2_blk||_2), and the
// returned expression is sum(d_blk) over all R blocks.
Expression ComputeL2Distance(
	MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<Expression>& x1,
	const VecX<Expression>& x2,
	const std::string& tag) {

	const double kInf = std::numeric_limits<double>::infinity();
	Expression dist(0.0);

	for (const auto& off : spline.block_offsets_) {
		if (off.type != CubicConfigurationSpline::Block::Type::R) continue;
		const int a0 = off.ambient_offset;
		const int aN = off.ambient_size;

		auto d_blk = prog.NewContinuousVariables(1, "l2_d_" + tag)(0);
		prog.AddBoundingBoxConstraint(0.0, kInf, d_blk);

		VecX<Expression> z(aN + 1);
		z(0) = Expression(d_blk);
		for (int i = 0; i < aN; ++i)
			z(i + 1) = x1(a0 + i) - x2(a0 + i);
		prog.AddLorentzConeConstraint(z);

		dist += Expression(d_blk);
	}

	return dist;
}

// Compile b_same: binary MILP variable = 1 iff variables r0 and r1 are
// assigned to the same agent in the current (or previous) solve.
drake::symbolic::Variable CompileVarEq(
	MathematicalProgram& prog,
	int r0, int r1,
	const MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments)
{
	const double kInf = std::numeric_limits<double>::infinity();
	int r0_local = subgraph.subgraph_variable_id(r0);
	int r1_local = subgraph.subgraph_variable_id(r1);
	int r0_prev  = (r0 < prev_var_assignments.size()) ? prev_var_assignments(r0) : -1;
	int r1_prev  = (r1 < prev_var_assignments.size()) ? prev_var_assignments(r1) : -1;

	auto b_same = prog.NewBinaryVariables(1, "b_same")(0);

	if (r0_prev >= 0 && r1_prev >= 0) {
		double val = (r0_prev == r1_prev) ? 1.0 : 0.0;
		prog.AddLinearConstraint(Expression(b_same), val, val);
		return b_same;
	}

	int num_agents = Assignments.cols();
	Expression sum_s;
	for (int j = 0; j < num_agents; ++j) {
		Expression a0 = (r0_prev >= 0) ? Expression(r0_prev == j ? 1.0 : 0.0)
		                               : Expression(Assignments(r0_local, j));
		Expression a1 = (r1_prev >= 0) ? Expression(r1_prev == j ? 1.0 : 0.0)
		                               : Expression(Assignments(r1_local, j));
		auto s_j = prog.NewBinaryVariables(1, "s_j")(0);
		prog.AddLinearConstraint(Expression(s_j) - a0,       -kInf, 0.0);   // s_j ≤ a0
		prog.AddLinearConstraint(Expression(s_j) - a1,       -kInf, 0.0);   // s_j ≤ a1
		prog.AddLinearConstraint(Expression(s_j) - a0 - a1,  -1.0, kInf);   // s_j ≥ a0+a1-1
		sum_s += Expression(s_j);
	}
	prog.AddLinearConstraint(Expression(b_same) - sum_s, 0.0, 0.0);
	return b_same;
}

// Compile a conditional ordering formula to a binary MILP variable.
// Returns a variable b such that b=1 iff the formula holds.
// Recognised atoms: VarEq (r_sym_i == r_sym_j) and BinVarEq (bv_sym == 0/1).
// Compound formulas use Drake FormulaKind::And / Or / Not.
drake::symbolic::Variable CompileConditionToBinary(
	MathematicalProgram& prog,
	const drake::symbolic::Formula& f,
	const MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments,
	const GraphOfConstraints* graph,
	const std::vector<drake::symbolic::Variable>& cond_binary_vars)
{
	using drake::symbolic::FormulaKind;
	using drake::symbolic::get_lhs_expression;
	using drake::symbolic::get_rhs_expression;
	using drake::symbolic::get_operands;
	using drake::symbolic::get_operand;
	using drake::symbolic::is_variable;
	using drake::symbolic::get_variable;
	const double kInf = std::numeric_limits<double>::infinity();

	auto new_binary = [&](const std::string& name) {
		return prog.NewBinaryVariables(1, name)(0);
	};

	switch (f.get_kind()) {

	case FormulaKind::Eq: {
		const Expression& lhs = get_lhs_expression(f);
		const Expression& rhs = get_rhs_expression(f);

		// VarEq: both sides are assignment symbolic vars
		if (is_variable(lhs) && is_variable(rhs)) {
			auto lv = get_variable(lhs);
			auto rv = get_variable(rhs);
			if (graph->_sym_id_to_variable_id.count(lv.get_id()) &&
			    graph->_sym_id_to_variable_id.count(rv.get_id())) {
				int r0 = graph->_sym_id_to_variable_id.at(lv.get_id());
				int r1 = graph->_sym_id_to_variable_id.at(rv.get_id());
				return CompileVarEq(prog, r0, r1, Assignments, subgraph, prev_var_assignments);
			}
		}

		// BinVarEq: one side is a binary cond var, other is a 0/1 constant
		auto try_binary_eq = [&](const Expression& var_expr, const Expression& val_expr)
			-> std::optional<drake::symbolic::Variable>
		{
			if (!is_variable(var_expr)) return {};
			auto sv = get_variable(var_expr);
			if (!graph->_sym_id_to_binary_cond_id.count(sv.get_id())) return {};
			int idx = graph->_sym_id_to_binary_cond_id.at(sv.get_id());
			double val = val_expr.Evaluate();
			const auto& gv = cond_binary_vars[idx];
			auto b = new_binary("b_bveq");
			if (val == 1.0) {
				prog.AddLinearConstraint(Expression(b) - Expression(gv), 0.0, 0.0);
			} else {
				prog.AddLinearConstraint(Expression(b) + Expression(gv), 1.0, 1.0);
			}
			return b;
		};
		if (auto r = try_binary_eq(lhs, rhs)) return *r;
		if (auto r = try_binary_eq(rhs, lhs)) return *r;

		throw std::runtime_error("Unsupported Eq atom in conditional edge formula");
	}

	case FormulaKind::And: {
		auto b = new_binary("b_and");
		Expression sum;
		int n = 0;
		for (const auto& sub : get_operands(f)) {
			auto b_sub = CompileConditionToBinary(prog, sub, Assignments, subgraph,
			                                      prev_var_assignments, graph, cond_binary_vars);
			prog.AddLinearConstraint(Expression(b) - Expression(b_sub), -kInf, 0.0);
			sum += Expression(b_sub);
			++n;
		}
		prog.AddLinearConstraint(Expression(b) - sum, -(n - 1.0), kInf);   // b ≥ sum-(n-1)
		return b;
	}

	case FormulaKind::Or: {
		auto b = new_binary("b_or");
		Expression sum;
		for (const auto& sub : get_operands(f)) {
			auto b_sub = CompileConditionToBinary(prog, sub, Assignments, subgraph,
			                                      prev_var_assignments, graph, cond_binary_vars);
			prog.AddLinearConstraint(Expression(b) - Expression(b_sub), 0.0, kInf);
			sum += Expression(b_sub);
		}
		prog.AddLinearConstraint(Expression(b) - sum, -kInf, 0.0);
		return b;
	}

	case FormulaKind::Not: {
		auto b_sub = CompileConditionToBinary(prog, get_operand(f), Assignments, subgraph,
		                                      prev_var_assignments, graph, cond_binary_vars);
		auto b = new_binary("b_not");
		prog.AddLinearConstraint(Expression(b) + Expression(b_sub), 1.0, 1.0);
		return b;
	}

	default:
		throw std::runtime_error("Unsupported FormulaKind in conditional edge ordering");
	}
}

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: t(t_idx_w) weakly between t(t_idx_u) and t(t_idx_v)  ⟹  b = 1.
// This is one-directional by design: b is free to be 0 when the node isn't
// between (so unrelated edge phis stay unconstrained), but can never be
// forced to 0 when it genuinely is between (so the caller's gated constraint
// can't be silently skipped). Built from two big-M half-plane implications
// ANDed together, mirroring the And-combination in CompileConditionToBinary.
drake::symbolic::Variable GetOrAddBetweennessIndicator(
	MathematicalProgram& prog,
	const VectorXDecisionVariable& t_var,
	int t_idx_u, int t_idx_w, int t_idx_v,
	double M_time,
	std::map<std::tuple<int, int, int>, drake::symbolic::Variable>& cache)
{
	const auto key = std::make_tuple(t_idx_u, t_idx_w, t_idx_v);
	auto it = cache.find(key);
	if (it != cache.end()) return it->second;

	const double kInf = std::numeric_limits<double>::infinity();
	// Must clear the solver's default numerical tolerances (Gurobi's
	// FeasibilityTol/IntFeasTol are ~1e-6/1e-5) with margin, or a
	// same-time tie (t(u) == t(w) == t(v), e.g. nodes with no forced
	// routing) can let b1/b2 slack to 0 well within tolerance, silently
	// defeating the "forced to 1 when genuinely between" guarantee.
	const double eps = 1e-3;

	auto b1 = prog.NewBinaryVariables(1, "b_after_u")(0);   // t(w) >= t(u)  ⟹  b1 = 1
	auto b2 = prog.NewBinaryVariables(1, "b_before_v")(0);  // t(w) <= t(v)  ⟹  b2 = 1

	prog.AddLinearConstraint(
		Expression(t_var(t_idx_u)) - Expression(t_var(t_idx_w)) + M_time * Expression(b1),
		eps, kInf);
	prog.AddLinearConstraint(
		Expression(t_var(t_idx_w)) - Expression(t_var(t_idx_v)) + M_time * Expression(b2),
		eps, kInf);

	auto b = prog.NewBinaryVariables(1, "b_between")(0);
	prog.AddLinearConstraint(Expression(b) - Expression(b1), -kInf, 0.0);
	prog.AddLinearConstraint(Expression(b) - Expression(b2), -kInf, 0.0);
	prog.AddLinearConstraint(Expression(b) - Expression(b1) - Expression(b2), -1.0, kInf);

	cache[key] = b;
	return b;
}

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: [t(t_idx_a1), t(t_idx_b1)] overlaps [t(t_idx_a2), t(t_idx_b2)]  <=>  b = 1
// (intervals overlap iff t(a1) <= t(b2) AND t(a2) <= t(b1)). Fully
// bidirectional: b is forced to 0 when the intervals genuinely do NOT
// overlap (not just free to be 0), via a matched reverse-direction big-M
// constraint per side, in addition to the existing "genuinely overlapping
// ⟹ b=1" constraints. Needed because Constraint 14b (the only caller) uses
// b=1 to RELAX a stationary-object constraint -- a false positive there
// silently lets an unheld/not-yet-held object drift, which is exactly the
// bug this fixes (see Constraint 14b's own comment). Only an unavoidable
// +-eps dead zone remains, right at an exact interval-boundary tie.
//
// Kept as a separate function/cache from GetOrAddBetweennessIndicator
// (which remains one-directional, deliberately not changed here): that one
// gates Constraint 13/14a, where b=1 APPLIES an extra invariant rather than
// relaxing one, so a false positive there is redundant rather than unsound,
// and retightening it isn't needed to fix this bug.
drake::symbolic::Variable GetOrAddIntervalOverlapIndicator(
	MathematicalProgram& prog,
	const VectorXDecisionVariable& t_var,
	int t_idx_a1, int t_idx_b1, int t_idx_a2, int t_idx_b2,
	double M_time,
	std::map<std::tuple<int, int, int, int>, drake::symbolic::Variable>& cache)
{
	const auto key = std::make_tuple(t_idx_a1, t_idx_b1, t_idx_a2, t_idx_b2);
	auto it = cache.find(key);
	if (it != cache.end()) return it->second;

	const double kInf = std::numeric_limits<double>::infinity();
	const double eps = 1e-3;

	auto c1 = prog.NewBinaryVariables(1, "c_a1_le_b2")(0);  // t(a1) <= t(b2)  <=>  c1 = 1
	auto c2 = prog.NewBinaryVariables(1, "c_a2_le_b1")(0);  // t(a2) <= t(b1)  <=>  c2 = 1

	// t_idx_a1 == t_idx_b2 (or t_idx_a2 == t_idx_b1) means the SAME t_var
	// entry is being compared to itself -- e.g. an edge that ends exactly at
	// a hold's own start node (the edge leading up to a pick-up). The two
	// spans are then merely ADJACENT, not overlapping: the edge's own span
	// is strictly before the hold even begins (the grasp happens AT the
	// shared node, not during any part of the edge), so this must resolve
	// to "no overlap" (0), not "trivially true, therefore overlapping" (1)
	// -- pinning it to 1 was backwards, and is what was letting the held
	// object drift free (via the wrongly-activated gated/relaxed branch)
	// right up to and including the hold's own boundary node. It's also the
	// case the bidirectional big-M pair below can't encode directly: at an
	// exact tie the difference is identically zero, not just numerically
	// close to zero, so demanding a +-eps margin on both sides of it
	// simultaneously is unsatisfiable regardless of which value we want.
	if (t_idx_a1 == t_idx_b2) {
		prog.AddLinearEqualityConstraint(Expression(c1), 0.0);
	} else {
		// t(a1) <= t(b2)  ⟹  c1 = 1
		prog.AddLinearConstraint(
			Expression(t_var(t_idx_a1)) - Expression(t_var(t_idx_b2)) + M_time * Expression(c1),
			eps, kInf);
		// c1 = 1  ⟹  t(a1) <= t(b2) (with margin) -- the previously-missing direction.
		prog.AddLinearConstraint(
			Expression(t_var(t_idx_b2)) - Expression(t_var(t_idx_a1)) + M_time * (1.0 - Expression(c1)),
			eps, kInf);
	}

	if (t_idx_a2 == t_idx_b1) {
		prog.AddLinearEqualityConstraint(Expression(c2), 0.0);
	} else {
		// t(a2) <= t(b1)  ⟹  c2 = 1
		prog.AddLinearConstraint(
			Expression(t_var(t_idx_a2)) - Expression(t_var(t_idx_b1)) + M_time * Expression(c2),
			eps, kInf);
		// c2 = 1  ⟹  t(a2) <= t(b1) (with margin) -- the previously-missing direction.
		prog.AddLinearConstraint(
			Expression(t_var(t_idx_b1)) - Expression(t_var(t_idx_a2)) + M_time * (1.0 - Expression(c2)),
			eps, kInf);
	}

	auto b = prog.NewBinaryVariables(1, "b_overlap")(0);
	prog.AddLinearConstraint(Expression(b) - Expression(c1), -kInf, 0.0);
	prog.AddLinearConstraint(Expression(b) - Expression(c2), -kInf, 0.0);
	prog.AddLinearConstraint(Expression(b) - Expression(c1) - Expression(c2), -1.0, kInf);

	cache[key] = b;
	return b;
}

GraphWaypointProblem BuildGraphWaypointProblem(
	GraphOfConstraints* graph,
	std::shared_ptr<std::vector<CubicConfigurationSpline>> splines,
	const std::vector<int>& remaining_vertices,
	Eigen::VectorXd x0,
	Eigen::MatrixXd previous_X,
	Eigen::VectorXi previous_var_assignments,
	bool enforce_rigidity,
	bool relax_binary_vars,
	WaypointObjective objective) {

	const int num_agents = graph->num_agents;
	const int num_objects = graph->num_objects;

	const SubgraphOfConstraints subgraph(graph, remaining_vertices);

	using namespace drake::solvers;

	// Create problem struct
	GraphWaypointProblem problem;

	// Create initial program
	std::unique_ptr<MathematicalProgram> prog_ptr = std::make_unique<MathematicalProgram>();
	MathematicalProgram& prog = *prog_ptr;

	// record the subgraph
	problem.subgraph = std::make_unique<SubgraphOfConstraints>(subgraph);

	const int robot_dim = graph->dim;
	const int objs_start = graph->num_agents * graph->dim;
	const int non_robot_dim = graph->non_robot_dim;

	// Routing graph N = {depot=0} ∪ task_nodes(1..n_V) ∪ {sink=n_V+1}, |N|=n_N.
	const int n_V = subgraph.num_nodes();
	const int n_N = n_V + 2;
	const int depot_idx = 0;
	const int sink_idx  = n_N - 1;

	auto ri       = [&](int sg_v) -> int { return sg_v + 1; };

	const double kInf     = std::numeric_limits<double>::infinity();
	const double x_range  = (graph->_global_x_ub - graph->_global_x_lb).maxCoeff();
	const double M_cost   = robot_dim * x_range;
	const double M_time   = (n_N - 1) * M_cost + 1.0;

	///////////////////////////////////////////////////////////////////////
        //                          Declare Variables                        //
        ///////////////////////////////////////////////////////////////////////

	problem.n_N  = n_N;

	// One row per assignable variable in the subgraph.
	// Rows are indexed by subgraph.subgraph_variable_id(var).
	// Constraint builders in graph_of_constraints.cpp gate their per-agent
	// constraints on these binary variables; constraint 8 below links them
	// to routing via inflow ≥ A(variable_k, agent).
	MatrixXDecisionVariable Assignments;
	if (subgraph.num_variables() > 0)
		Assignments = prog.NewBinaryVariables(
			subgraph.num_variables(), num_agents, "Assignments");
	problem.Assignments = Assignments;
	(void)relax_binary_vars;

	// W: continuous configuration variables (n x m*d+o).
	MatrixXDecisionVariable W = prog.NewContinuousVariables(subgraph.num_nodes(), num_agents * robot_dim + num_objects * non_robot_dim, "W");
	problem.W = W;

	// Z: binary per-agent path indicators
	VectorXDecisionVariable Z_var    = prog.NewBinaryVariables(num_agents * n_N * n_N, "Z");
	problem.Z    = Z_var;
	auto z_idx_fn = [&](int j, int a, int b) -> int { return j * n_N * n_N + a * n_N + b; };

	// t: per-node timings that increase monotonically along paths
	VectorXDecisionVariable t_var    = prog.NewContinuousVariables(n_N, "t");
	problem.t    = t_var;

	// e: auxiliary variables matching edge cost for used edges
	VectorXDecisionVariable e_var    = prog.NewContinuousVariables(num_agents * n_N * n_N, "e");
	problem.e    = e_var;
	auto e_idx_fn = [&](int j, int a, int b) -> int { return j * n_N * n_N + a * n_N + b; };

	// L: auxiliary variable for maximum path time
	VectorXDecisionVariable L_ms_var = prog.NewContinuousVariables(1, "L");
	problem.L_ms = L_ms_var;

	// One binary MILP variable per binary condition symbolic variable in the graph.
	std::vector<drake::symbolic::Variable> cond_binary_vars;
	for (int i = 0; i < static_cast<int>(graph->_binary_cond_sym_vars.size()); ++i) {
		cond_binary_vars.push_back(prog.NewBinaryVariables(1, "bv_" + std::to_string(i))(0));
	}

	///////////////////////////////////////////////////////////////////////
        //                             Constraints                           //
        ///////////////////////////////////////////////////////////////////////

	// Variable Bound Constraints
	for (int i = 0; i < W.rows(); ++i) {
		prog.AddBoundingBoxConstraint(graph->_global_x_lb, graph->_global_x_ub, W.row(i).transpose())
			.evaluator()->set_description("configuration space bounds");
	}

	prog.AddBoundingBoxConstraint(0.0, kInf, t_var);
	prog.AddBoundingBoxConstraint(0.0, kInf, e_var);
	prog.AddBoundingBoxConstraint(0.0, kInf, L_ms_var);

	// t(depot) = 0
	prog.AddLinearEqualityConstraint(
		Eigen::RowVectorXd::Ones(1), 0.0, t_var.segment(depot_idx, 1));

	// Self-loops forbidden
	for (int j = 0; j < num_agents; ++j)
		for (int i = 0; i < n_N; ++i)
			prog.AddLinearEqualityConstraint(
				Eigen::RowVectorXd::Ones(1), 0.0,
				Z_var.segment(z_idx_fn(j, i, i), 1));

	// Constraint 1: makespan ≥ Σ_{a,b} e[j,a,b] for each agent j
	for (int j = 0; j < num_agents; ++j) {
		VectorXDecisionVariable ej_L(n_N * n_N + 1);
		ej_L.head(n_N * n_N) = e_var.segment(j * n_N * n_N, n_N * n_N);
		ej_L(n_N * n_N) = L_ms_var(0);
		Eigen::RowVectorXd c(n_N * n_N + 1);
		c.head(n_N * n_N).setOnes();
		c(n_N * n_N) = -1.0;
		prog.AddLinearConstraint(c, -kInf, 0.0, ej_L);
	}

	// Constraints 2 & 3: cost-linking (big-M) and zero cost on sink arcs
	for (int j = 0; j < num_agents; ++j) {

		// Helper for getting waypoint vector for agent j at node node_ri
		auto get_w = [&](int node_ri) -> Eigen::VectorX<Expression> {
			Eigen::VectorX<Expression> w(robot_dim);
			if (node_ri == depot_idx) {
				for (int k = 0; k < robot_dim; ++k)
					w(k) = Expression(x0(j * robot_dim + k));
			} else {
				// task node: subgraph index (node_ri - 1)
				w = W.row(node_ri - 1).segment(j * robot_dim, robot_dim)
				      .template cast<Expression>();
			}
			return w;
		};

		for (int a = 0; a < n_N; ++a) {
			for (int b = 0; b < n_N; ++b) {
				// Skip self loops
				if (a == b) continue;

				// Constraint 3: arcs touching sink carry no cost
				if (a == sink_idx || b == sink_idx) {
					prog.AddLinearEqualityConstraint(
						Eigen::RowVectorXd::Ones(1), 0.0,
						e_var.segment(e_idx_fn(j, a, b), 1));
					continue;
				}

				// Constraint 2: e[j,a,b] links to the objective's distance metric
				// (L1 or L2, over R-type spline blocks only) when arc is used.
				Eigen::VectorX<Expression> w_a = get_w(a);
				Eigen::VectorX<Expression> w_b = get_w(b);
				const std::string tag = fmt::format("j{}_a{}_b{}", j, a, b);
				Expression dist;
				switch (objective) {
				case WaypointObjective::kMinMaxL1:
				case WaypointObjective::kAvgL1:
					dist = ComputeL1Distance(prog, splines->at(j), w_a, w_b, tag);
					break;
				case WaypointObjective::kMinMaxL2:
				case WaypointObjective::kAvgL2:
					dist = ComputeL2Distance(prog, splines->at(j), w_a, w_b, tag);
					break;
				}
				const Variable e_jab = e_var(e_idx_fn(j, a, b));
				const Variable z_jab = Z_var(z_idx_fn(j, a, b));
				// lower: e ≥ dist - M*(1-Z)  →  e + M - M*Z - dist ≥ 0
				prog.AddLinearConstraint(
					Expression(e_jab) + M_cost
					- M_cost * Expression(z_jab) - dist,
					0.0, kInf);
				// upper: e ≤ M*Z
				prog.AddLinearConstraint(
					Expression(e_jab) - M_cost * Expression(z_jab),
					-kInf, 0.0);
			}
		}
	}

	for (int j = 0; j < num_agents; ++j) {
		// 3: Depart depot exactly once
		{
			Expression depart;
			for (int v2 = 1; v2 < n_N; ++v2)
				depart += Expression(Z_var(z_idx_fn(j, depot_idx, v2)));
			prog.AddLinearEqualityConstraint(depart, 1.0);
		}
		// 4: Arrive at sink exactly once
		{
			Expression arrive;
			for (int a2 = 0; a2 < sink_idx; ++a2)
				arrive += Expression(Z_var(z_idx_fn(j, a2, sink_idx)));
			prog.AddLinearEqualityConstraint(arrive, 1.0);
		}
		// 5: No return to depot
		for (int v2 = 1; v2 < n_N; ++v2)
			prog.AddLinearEqualityConstraint(
				Eigen::RowVectorXd::Ones(1), 0.0,
				Z_var.segment(z_idx_fn(j, v2, depot_idx), 1));
		// 6: No departure from sink to task nodes
		for (int v2 = 1; v2 < sink_idx; ++v2)
			prog.AddLinearEqualityConstraint(
				Eigen::RowVectorXd::Ones(1), 0.0,
				Z_var.segment(z_idx_fn(j, sink_idx, v2), 1));
		// 7: Flow conservation at task nodes
		for (int sg_v2 = 0; sg_v2 < n_V; ++sg_v2) {
			const int rv = ri(sg_v2);
			Expression in_flow, out_flow;
			for (int a2 = 0; a2 < n_N; ++a2) {
				if (a2 == rv) continue;
				in_flow  += Expression(Z_var(z_idx_fn(j, a2, rv)));
				out_flow += Expression(Z_var(z_idx_fn(j, rv, a2)));
			}
			prog.AddLinearEqualityConstraint(in_flow - out_flow, 0.0);
		}
	}

	// Constraint 8: inflow(k,v) = A(variable_k, k) for each agent k.
	// Equality (not just ≥) prevents zero-cost sub-tour phantoms: an agent may
	// visit task node v if and only if it is assigned the variable at v.
	// For static phis: inflow(static_agent→v) = 1 (forced visit).
	for (const auto& [phi_id, op] : subgraph.get_subgraph_ops()) {
		const int sg_v2 = subgraph.subgraph_id(op.node);
		if (sg_v2 == -1) continue;
		const int rv = ri(sg_v2);
		if (graph->phi_to_variable_map.contains(phi_id)) {
			const int var = graph->phi_to_variable_map.at(phi_id);
			const int variable_k = subgraph.subgraph_variable_id(var);
			if (variable_k < 0) continue;
			for (int k = 0; k < num_agents; ++k) {
				Expression in_flow;
				for (int a2 = 0; a2 < n_N; ++a2)
					if (a2 != rv) in_flow += Expression(Z_var(z_idx_fn(k, a2, rv)));
				// inflow[k→v] = A(variable_k, k): assignment ↔ visit (no phantom sub-tours)
				prog.AddLinearEqualityConstraint(
					in_flow - Expression(Assignments(variable_k, k)), 0.0);
			}
		}
		if (graph->_phi_to_static_assignment_map.contains(phi_id)) {
			const int k = graph->_phi_to_static_assignment_map.at(phi_id);
			Expression in_flow;
			for (int a2 = 0; a2 < n_N; ++a2)
				if (a2 != rv) in_flow += Expression(Z_var(z_idx_fn(k, a2, rv)));
			prog.AddLinearEqualityConstraint(in_flow, 1.0);
		}
	}

	// Constraint 8b: assignment commits (see add_variable_commit /
	// GraphOfConstraints::committed_assignments) -- once a plan has pinned a
	// variable to whichever agent it resolved to at some earlier commit
	// node, force every remaining node/hold that still references it through
	// that same agent. Pin the whole row (not just the committed cell) --
	// the row's other cells aren't otherwise guaranteed to be forced to 0 by
	// something else (that normally falls out of a var_id node constraint's
	// own "exactly one agent" emission in CompileSymbolicNodeConstraint, but
	// a committed variable referenced only via a hold at this point in the
	// plan would have no such emission left in the subgraph) -- and the
	// solution-extraction scan in Solve()/SolveCoordinated (`for k: if
	// val>0.5: ... break`) takes the first 1-valued cell it finds, so a
	// stray free cell at a lower agent index would silently report the
	// wrong agent even though the intended one is also pinned to 1.
	for (const auto& [var, committed_agent] : graph->committed_assignments) {
		const int variable_k = subgraph.subgraph_variable_id(var);
		if (variable_k < 0) continue;
		for (int k = 0; k < num_agents; ++k) {
			prog.AddLinearEqualityConstraint(
				Expression(Assignments(variable_k, k)), k == committed_agent ? 1.0 : 0.0);
		}
	}

	// Constraint 9: time propagation — Z[j,a,b]=1 ⟹ t[b] ≥ t[a] + e[j,a,b]
	for (int j = 0; j < num_agents; ++j) {
		for (int a = 0; a <= n_V; ++a) {   // a ∈ {depot=0} ∪ task routing indices
			for (int b = 1; b <= n_V; ++b) {  // b ∈ task nodes only
				if (a == b) continue;
				// t[b] - t[a] - e[j,a,b] + M_time*(1-Z[j,a,b]) ≥ 0
				prog.AddLinearConstraint(
					Expression(t_var(b)) - Expression(t_var(a))
					- Expression(e_var(e_idx_fn(j, a, b)))
					+ M_time - M_time * Expression(Z_var(z_idx_fn(j, a, b))),
					0.0, kInf);
			}
		}
	}

	// Constraint 10: DAG ordering — hard (structure edges) or conditional (_conditional_ordering_map)
	auto add_ordering_constraint = [&](int u_node, int v_node,
	                                   const drake::symbolic::Formula* cond) {
		const int sg_u = subgraph.subgraph_id(u_node);
		const int sg_v = subgraph.subgraph_id(v_node);
		if (sg_u < 0 || sg_v < 0) return;
		if (cond == nullptr) {
			// Unconditional: hard ordering t(u) ≤ t(v)
			prog.AddLinearConstraint(
				Expression(t_var(ri(sg_u))) - Expression(t_var(ri(sg_v))),
				-kInf, 0.0);
		} else {
			// Conditional: t(u) ≤ t(v) only when b_cond = 1
			// Enforced as: t(u) - t(v) - M_time + M_time*b_cond ≤ 0
			auto b_cond = CompileConditionToBinary(
				prog, *cond, Assignments, subgraph,
				previous_var_assignments, graph, cond_binary_vars);
			prog.AddLinearConstraint(
				Expression(t_var(ri(sg_u))) - Expression(t_var(ri(sg_v)))
				- M_time + M_time * Expression(b_cond),
				-kInf, 0.0);
		}
	};

	for (const auto& edge : subgraph.structure.edges()) {
		const int u_node = edge.u;
		const int v_node = edge.e->to;
		add_ordering_constraint(u_node, v_node, nullptr);
	}
	for (const auto& [uv, formula] : graph->_conditional_ordering_map) {
		add_ordering_constraint(uv.first, uv.second, &formula);
	}

	// Constraint 11: Add assignment variable constraints from registry
	for (const auto& [var_phi_id, var_op] : graph->get_var_ops()) {
		var_op.builder(prog, subgraph, var_phi_id, W, Assignments);
	}


	// Constraint 12: Add node constraints/costs from registry
	for (const auto& [phi_id, op] : subgraph.get_subgraph_ops()) {
		op.builder(prog, subgraph, phi_id, W, Assignments);
	}

	// Constraint 12b: node constraints from the unified symbolic API
	// (add_constraint / add_assignable_constraint) — compiled directly from
	// the stored Formula, mirroring what the retired DeferredOp builder
	// closures used to do inline. See symbolic_constraint_compiler.*.
	for (const auto& [phi_id, rec] : subgraph.get_subgraph_symbolic_ops()) {
		CompileSymbolicNodeConstraint(prog, subgraph, *graph, rec, W, Assignments);
	}

	// Constraint 13: Add edge constraints/costs from edge registry

	// Edge phis are registered against two specific graph nodes (u, v), but
	// node order is a routing decision (Z/t above), not fixed identity: some
	// other subgraph node w with no DAG relationship to u or v may end up
	// scheduled (by any agent) with t(u) <= t(w) <= t(v). For edge ops whose
	// constraint is an independent per-endpoint invariant (interior_builder
	// set — see add_robot_holding_cube_constraint), re-apply that invariant,
	// gated by a betweenness indicator, at every such interior node so it
	// isn't silently unconstrained mid-hold.
	std::map<std::tuple<int, int, int>, Variable> betweenness_cache;
	for (const auto& [edge_phi_id, edge_op] : subgraph.get_subgraph_edge_ops()) {
		edge_op.waypoint_builder(prog, subgraph, edge_phi_id, W, Assignments,
					 previous_X.row(edge_op.u_node));

		if (!edge_op.interior_builder) continue;
		const int sg_u13 = subgraph.subgraph_id(edge_op.u_node);
		const int sg_v13 = subgraph.subgraph_id(edge_op.v_node);
		if (sg_u13 < 0 || sg_v13 < 0) continue;
		for (int sg_w13 = 0; sg_w13 < n_V; ++sg_w13) {
			if (sg_w13 == sg_u13 || sg_w13 == sg_v13) continue;
			const Variable& b_between = GetOrAddBetweennessIndicator(
				prog, t_var, ri(sg_u13), ri(sg_w13), ri(sg_v13), M_time, betweenness_cache);
			edge_op.interior_builder(prog, subgraph, edge_phi_id, W, Assignments,
						  sg_w13, b_between);
		}
	}

	// Constraint 13b: edge constraints from the unified symbolic API
	// (add_edge_constraint). A relational formula (rec.along_edge == false)
	// is a single relation coupling both endpoints — no interior_builder
	// equivalent needed. An "along the edge" formula (rec.along_edge ==
	// true) IS a per-endpoint invariant though, so — mirroring Constraint
	// 13's interior_builder loop above for the older DeferredEdgeOp API —
	// re-apply it, betweenness-gated, at every other subgraph node that
	// could end up scheduled between u_node and v_node in the solved route.
	for (const auto& [edge_phi_id, rec] : subgraph.get_subgraph_symbolic_edge_ops()) {
		CompileSymbolicEdgeConstraint(prog, subgraph, *graph, rec, W, Assignments, previous_X);

		if (!rec.along_edge) continue;
		const int sg_u13b = subgraph.subgraph_id(rec.u_node);
		const int sg_v13b = subgraph.subgraph_id(rec.v_node);
		if (sg_u13b < 0 || sg_v13b < 0) continue;
		for (int sg_w13b = 0; sg_w13b < n_V; ++sg_w13b) {
			if (sg_w13b == sg_u13b || sg_w13b == sg_v13b) continue;
			const Variable& b_between = GetOrAddBetweennessIndicator(
				prog, t_var, ri(sg_u13b), ri(sg_w13b), ri(sg_v13b), M_time, betweenness_cache);
			Expression gate(b_between);
			CompileAlongEdgeConstraintAtNode(prog, subgraph, *graph, rec, W, Assignments, sg_w13b, &gate);
		}
	}

	// Constraint 14: Dynamics (quasi-static manipulation / rigidity + stationary-object) constraints
	//
	// Rather than pre-computing which nodes/edges a held object could possibly
	// span via DAG topology (topological layers + concurrent-edge overlap),
	// both invariants below are gated directly on MILP betweenness/overlap
	// indicators over the actual solved node timings t_var. A node that is
	// DAG-ordered between a hold's own endpoints already has its betweenness
	// indicator forced to 1 by Constraint 10's ordering constraints, so
	// structurally-interior nodes on the hold's own path are handled for
	// free; nodes on other agents' paths that might land in between at solve
	// time are handled by the (free-unless-forced) MILP gate — see
	// GetOrAddBetweennessIndicator / GetOrAddIntervalOverlapIndicator above.
	if (enforce_rigidity) {

		// 14a: rigidity — the held point's pose relative to the holding
		// end-effector must stay fixed from pick-up to put-down.
		for (const auto& [hold_id, hold] : subgraph.get_subgraph_hold_ops()) {
			const bool is_static = hold.robot_ag.has_value();

			const int sg_v14 = subgraph.subgraph_id(hold.v_node);
			if (sg_v14 < 0) continue;  // target not yet in this horizon
			const int sg_u14 = subgraph.subgraph_id(hold.u_node);

			HoldSpec static_spec;
			AssignableHoldSpec assignable_spec;
			VectorXDecisionVariable A_row;
			if (is_static) {
				static_spec.held_point_ids = hold.held_point_ids;
				static_spec.robot_ag = hold.robot_ag.value();
				static_spec.ee_frame_name = "ee_link";
			} else {
				assignable_spec.held_point_ids = hold.held_point_ids;
				assignable_spec.var = hold.var_id.value();
				assignable_spec.ee_frame_name = "ee_link";
				A_row = GetARowForVar(subgraph, Assignments, assignable_spec.var);
			}

			Eigen::VectorX<Expression> ref_expr;
			int t_idx_ref;
			if (sg_u14 >= 0) {
				ref_expr = W.row(sg_u14).template cast<Expression>();
				t_idx_ref = ri(sg_u14);

				// Endpoint (unconditional): a real, always-traversed DAG edge.
				if (is_static) {
					AddHoldRigidityStatic(prog, subgraph, graph, W, hold.u_node, hold.v_node,
							       static_spec, robot_dim, objs_start, non_robot_dim,
							       /*exact_rigidity=*/true);
				} else {
					AddHoldRigidityAssignable(prog, subgraph, graph, W, hold.u_node, hold.v_node,
								   assignable_spec, robot_dim, objs_start, non_robot_dim, A_row);
				}
			} else {
				// x0-boundary edge: the hold started before this horizon.
				ref_expr = x0.template cast<Expression>();
				t_idx_ref = depot_idx;

				if (is_static) {
					AddHoldRigidityStaticToX0(prog, subgraph, graph, W, hold.v_node,
								   static_spec, robot_dim, objs_start, non_robot_dim,
								   x0, /*exact_rigidity=*/true);
				} else {
					AddHoldRigidityAssignableToX0(prog, subgraph, graph, W, hold.v_node,
								       assignable_spec, robot_dim, objs_start, non_robot_dim,
								       x0, A_row);

					// Not pinning the holder here anymore: add_assignable_hold
					// auto-registers its own `u` (pick-up) node as a commit
					// trigger for `var` (see commit_trigger_node_to_var), so by
					// the time hold.u_node has left the subgraph (this branch),
					// GraphOfConstraintsMPC has already armed
					// graph->committed_assignments for this variable and
					// Constraint 8b above pins the whole row -- same "don't
					// reassign this hold's holder mid-grasp" outcome, on the
					// same cycle (committed_assignments captures the resolved
					// agent the cycle `u` completes, same as previous_var_
					// assignments would have here), but via one shared
					// mechanism instead of two, and pinning every other cell
					// to 0 too rather than just the held one.
				}
			}

			// Interior: nodes that might be scheduled between the hold's
			// endpoints, structurally (forced) or via another agent's route (free).
			for (int sg_w14 = 0; sg_w14 < n_V; ++sg_w14) {
				if (sg_w14 == sg_u14 || sg_w14 == sg_v14) continue;
				const Variable& gate = GetOrAddBetweennessIndicator(
					prog, t_var, t_idx_ref, ri(sg_w14), ri(sg_v14), M_time, betweenness_cache);
				Eigen::VectorX<Expression> cand_expr = W.row(sg_w14).template cast<Expression>();
				const std::string desc = fmt::format("interior rigidity {}->{}", hold.u_node, hold.v_node);
				if (is_static) {
					AddHoldRigidityStaticGated(prog, graph, ref_expr, cand_expr, static_spec,
								    robot_dim, objs_start, non_robot_dim, gate, desc);
				} else {
					AddHoldRigidityAssignableGated(prog, graph, ref_expr, cand_expr, assignable_spec,
									robot_dim, objs_start, non_robot_dim, A_row, gate, desc);
				}
			}
		}
	}

	// 14b: stationary objects — an object not currently being held must not
	// move between two graph nodes. Unlike 14a, this only ever compares
	// plain position segments (W.row(...).segment(...)) with no rotation
	// applied, so it's linear (or, when gated by an overlap indicator,
	// mixed-integer-linear) regardless of robot kind -- there's no
	// configuration space for which it becomes nonlinear the way 14a's
	// rotated relative-offset check can. So it's always enforced,
	// independent of enforce_rigidity.
	{
		std::map<std::tuple<int, int, int, int>, Variable> overlap_cache;

		auto stationary_edge = [&](const Eigen::VectorX<Expression>& seg_a_full, int sg_b,
					    int t_idx_a, int t_idx_b) {
			for (int obj = 0; obj < num_objects; ++obj) {
				const int start = objs_start + obj * non_robot_dim;
				Eigen::VectorX<Expression> seg_a = seg_a_full.segment(start, non_robot_dim);
				Eigen::VectorX<Expression> seg_b = W.row(sg_b).segment(start, non_robot_dim)
									.template cast<Expression>();

				// Every hold (anywhere in the graph) that might move this object.
				std::vector<int> relevant_holds;
				for (const auto& [other_hold_id, other_hold] : graph->hold_ops) {
					if (std::find(other_hold.held_point_ids.begin(), other_hold.held_point_ids.end(), obj) !=
					    other_hold.held_point_ids.end()) {
						relevant_holds.push_back(other_hold_id);
					}
				}

				Expression gate_sum;
				bool any_gate = false;
				for (int other_hold_id : relevant_holds) {
					const auto& other_hold = graph->hold_ops.at(other_hold_id);
					const int sg_other_u = subgraph.subgraph_id(other_hold.u_node);
					const int sg_other_v = subgraph.subgraph_id(other_hold.v_node);
					if (sg_other_v < 0) continue;  // can't reason about it within this horizon
					const int t_idx_other_u = (sg_other_u >= 0) ? ri(sg_other_u) : depot_idx;
					const int t_idx_other_v = ri(sg_other_v);

					const Variable& overlap = GetOrAddIntervalOverlapIndicator(
						prog, t_var, t_idx_a, t_idx_b, t_idx_other_u, t_idx_other_v,
						M_time, overlap_cache);
					gate_sum += Expression(overlap);
					any_gate = true;
				}

				if (!any_gate) {
					prog.AddLinearEqualityConstraint(seg_a - seg_b, Eigen::VectorXd::Zero(non_robot_dim))
						.evaluator()->set_description(fmt::format("stationary point {}", obj));
					continue;
				}

				const double m_stat = (graph->_global_x_ub.segment(start, non_robot_dim) -
							graph->_global_x_lb.segment(start, non_robot_dim))
							.cwiseAbs().maxCoeff();
				for (int i = 0; i < non_robot_dim; ++i) {
					const Expression slack = m_stat * gate_sum;
					prog.AddLinearConstraint(seg_a(i) - seg_b(i) - slack, -kInf, 0.0)
						.evaluator()->set_description(fmt::format("gated stationary point {}", obj));
					prog.AddLinearConstraint(-seg_a(i) + seg_b(i) - slack, -kInf, 0.0)
						.evaluator()->set_description(fmt::format("gated stationary point {}", obj));
				}
			}
		};

		for (auto edge : subgraph.structure.edges()) {
			const int sg_u14b = subgraph.subgraph_id(edge.u);
			const int sg_v14b = subgraph.subgraph_id(edge.e->to);
			stationary_edge(W.row(sg_u14b).template cast<Expression>(), sg_v14b,
					 ri(sg_u14b), ri(sg_v14b));
		}
		for (int v14b : subgraph.structure.sources()) {
			const int sg_v14b = subgraph.subgraph_id(v14b);
			stationary_edge(x0.template cast<Expression>(), sg_v14b, depot_idx, ri(sg_v14b));
		}
	}


	// TODO: Constraint 15: Manifold Constraints

	//
	// QUATERNION / ROTATION MATRIX CONSTRAINTS
	//

	// for (int ag = 0; ag < num_agents; ++ag) {
	// 	switch (graph->robot_kind(ag)) {
	// 	case RobotKind::kPosQuat:
	// 	{
	// 		for (int node = 0; node < subgraph.num_nodes(); ++node) {
	// 			VectorX<Expression> ag_quat = AsExprRow(X.row(node).segment(ag * robot_dim + 3, 4));
	// 			Expression ag_quat_norm = ag_quat.squaredNorm();
	// 			const double tol = 0.001;
	// 			prog.AddQuadraticConstraint(ag_quat_norm, 1.0-tol, 1.0+tol)
	// 				.evaluator()->set_description("unit quaternion constraint");
	// 		}
	// 		break;
	// 	}
	// 	case RobotKind::kPosRotMat:
	// 	{
	// 		for (int node = 0; node < subgraph.num_nodes(); ++node) {
	// 			VectorXDecisionVariable R_flat = X.row(node).segment(ag * robot_dim + 3, 9);
	// 			Eigen::Matrix<Variable, 3, 3> R;
	// 			for (int i = 0; i < 3; ++i)
	// 				for (int j = 0; j < 3; ++j)
	// 					R(i, j) = R_flat(i * 3 + j);

	// 			Eigen::Matrix<Expression, 3, 3> orthogonality_residual = R.transpose() * R - Eigen::Matrix3d::Identity();
	// 			const double tol = 0.001;
	// 			for (int i = 0; i < 3; ++i)
	// 				for (int j = i; j < 3; ++j)
	// 					prog.AddQuadraticConstraint(orthogonality_residual(i, j), 0.0-tol, 0.0+tol)
	// 						.evaluator()->set_description("rotation matrix orthogonality constraint");

	// 			Eigen::Matrix<Expression, 3, 1> right_handedness_residual = R.col(0).cross(R.col(1)) - R.col(2);
	// 			for (int i = 0; i < 3; ++i)
	// 				prog.AddQuadraticConstraint(right_handedness_residual(i), 0.0-tol, 0.0+tol)
	// 					.evaluator()->set_description("rotation matrix right-handedness constraint");
	// 		}
	// 		break;
	// 	}
	// 	default:
	// 		break;
	// 	}
	// }

	///////////////////////////////////////////////////////////////////////
        //                              Objective                            //
        ///////////////////////////////////////////////////////////////////////

	// Objective: MinMax variants minimize the makespan (Constraint 1 forces
	// L_ms_var to bound every agent's total route cost); Avg variants minimize
	// the average total route cost across agents directly from e_var.
	switch (objective) {
	case WaypointObjective::kMinMaxL1:
	case WaypointObjective::kMinMaxL2:
		prog.AddLinearCost(Eigen::VectorXd::Ones(1), 0.0, L_ms_var);
		break;
	case WaypointObjective::kAvgL1:
	case WaypointObjective::kAvgL2: {
		if (num_agents == 0) throw std::runtime_error("kAvg* objectives require num_agents > 0");
		Expression total_cost;
		for (int i = 0; i < e_var.size(); ++i) total_cost += Expression(e_var(i));
		prog.AddLinearCost(total_cost * (1.0 / num_agents));
		break;
	}
	}

	problem.prog = std::move(prog_ptr);

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

MILPWaypointMPC::MILPWaypointMPC(GraphOfConstraints& graph,
				   std::vector<CubicConfigurationSpline> splines,
				   WaypointSolver solver,
				   bool enforce_rigidity,
				   WaypointObjective objective)
	: GraphWaypointMPC(graph, std::move(splines), objective),
	  _solver(solver),
	  _enforce_rigidity(enforce_rigidity) {
	// Routing outputs are sized on first successful solve.
	_t_routing = Eigen::VectorXd();
	_Z_routing = Eigen::VectorXd();
}

using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;

void PrintSolverReport(GraphWaypointProblem* problem,
		       const MathematicalProgramResult& result,
		       double tol = 1e-6) {
	using std::cout;
	using std::endl;

	cout << "=== Drake Solve() Report ===\n";
	cout << "Solver:           " << result.get_solver_id().name() << "\n";
	// to_string(SolutionResult) is supported; if your Drake is older, cast to int.
	cout << "SolutionResult:   " << to_string(result.get_solution_result()) << "\n";
	cout << "Success?          " << (result.is_success() ? "yes" : "no") << "\n";

	// Print optimal cost if available.
	if (std::isfinite(result.get_optimal_cost())) {
		cout << "Optimal cost:     " << result.get_optimal_cost() << "\n";
	}

	// Try to obtain a decision vector to evaluate constraints.
	Eigen::VectorXd x;
	bool have_x = false;
	try {
		x = result.GetSolution(problem->prog->decision_variables());
		have_x = true;
	} catch (const std::exception&) {
		// Fall back to initial guess (may be zero) if solver did not return x.
		if (problem->prog->decision_variables().size() > 0) {
			x = problem->prog->initial_guess();
			have_x = true;
			cout << "(No returned solution vector; using initial guess for diagnostics.)\n";
		}
	}

	// Constraint violation scan.
	if (have_x) {
		try {
			auto print_binding_violation = [&](const auto& bindings, const char* kind) {
				for (const auto& b : bindings) {
					const auto& c = *b.evaluator();
					const auto& vars = b.variables();
					Eigen::VectorXd x_b = result.GetSolution(vars);

					double tol = 0.05;
					if (!c.CheckSatisfied(x_b, tol)) {
						cout << "Violation [" << kind << "]: "
						     << " (constraint: " << c.get_description() << ")\n";
					} else {
						cout << "Satisfied [" << kind << "]: "
						     << " (constraint: " << c.get_description() << ")\n";
					}
				}
			};

			cout << "--- Constraint violations (inf-norm > " << tol << ") ---\n";
			print_binding_violation(problem->prog->bounding_box_constraints(),   "BoundingBox");
			print_binding_violation(problem->prog->linear_equality_constraints(),"LinEq");
			print_binding_violation(problem->prog->linear_constraints(),         "LinIneq");
			print_binding_violation(problem->prog->lorentz_cone_constraints(),   "Lorentz");
			print_binding_violation(problem->prog->rotated_lorentz_cone_constraints(),"RotLorentz");
			print_binding_violation(problem->prog->quadratic_constraints(),      "Quadratic");
			print_binding_violation(problem->prog->exponential_cone_constraints(),"ExpCone");
			print_binding_violation(problem->prog->generic_constraints(),        "Generic");
		} catch (const std::exception& e) {
			cout << "Exception in printing binding violations: " << e.what() << endl;
		}
	} else {
		cout << "(No decision vector available to evaluate constraints.)\n";
	}

	// Try to print some solver-specific details (best-effort).
	try {
		const auto& sid = result.get_solver_id().name();
		if (sid == "Ipopt") {
			struct IpoptDetails {
				int status; int iterations; double objective; double dual_inf;
				double constr_viol; double comp_viol; double primal_step; double dual_step;
			};

			for (const std::string& s : result.GetInfeasibleConstraintNames(*problem->prog, tol)) {
				cout << "infeasible constraint: " << s << std::endl;
			}

			// If your Drake provides IpoptSolver::Details, use that exact type.
			// Example (adjust to your Drake version):
			const auto& d = result.get_solver_details<drake::solvers::IpoptSolver>();
			cout << "Ipopt status: " << d.ConvertStatusToString() << endl;
		} else if (sid == "NLopt") {
			const auto& d = result.get_solver_details<drake::solvers::NloptSolver>();
			cout << "TODO: Print nlopt information" << endl;
		} else if (sid == "OSQP") {
			// Example (adjust to your Drake version):
			// const auto& d = result.get_solver_details<OsqpSolver>();
			// cout << "OSQP status: " << d.status_val << ", iters: " << d.iter
			//      << ", obj: " << d.obj_val << "\n";
		} else if (sid == "Gurobi") {
			// const auto& d = result.get_solver_details<GurobiSolver>();
			// cout << "Gurobi status: " << d.optimization_status << ", iters: "
			//      << d.iteration_count << ", mip_gap: " << d.mip_relative_gap << "\n";
		}
	} catch (const std::exception& e) {
		cout << "(Could not print solver-specific details: " << e.what() << ")\n";
	}

	cout << "=== End Report ===\n";
}

bool MILPWaypointMPC::Solve(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	_timer.Start();

	// Warm start configurations
	if (_first_cycle) {
		for (int v : remaining_vertices) {
			_waypoints.row(v) = x0.transpose();
		}
		_first_cycle = false;
	}

	// Live-rebind: a node's row in `_waypoints`, once it's no longer in
	// remaining_vertices, is otherwise whatever SolveWithGurobi/
	// SolveWithMosek's write-back loop last planned for it while still
	// active -- the nominal MILP-solved target, not the real state the
	// graph actually reached. That's stale for a relational edge
	// constraint anchored at that node (e.g. a rigid-transport edge):
	// CompileSymbolicEdgeConstraint substitutes a passed endpoint's row as
	// a constant read from `_waypoints` (passed through as `previous_X` --
	// see symbolic_constraint_compiler.cpp), so it would otherwise
	// propagate the *planned* transform forward even once the real one has
	// diverged (e.g. a grasp whose real standoff doesn't match the planned
	// one). So the first Solve() call that sees a node freshly missing
	// from remaining_vertices (vs. the previous call) overwrites that
	// node's row with the real x0 given on THIS call, once, before
	// anything reads it as previous_X -- not every cycle after, since the
	// contract is a frozen constant once a node has passed, not a
	// continuously-tracked one. Mirrors EvolutionaryWaypointSolver's
	// equivalent fix (evolutionary_waypoint_solver/mpc.py's solve()).
	const std::set<int> remaining_set(remaining_vertices.begin(), remaining_vertices.end());
	if (_has_prev_remaining_set) {
		for (int v : _prev_remaining_set) {
			if (remaining_set.contains(v)) continue;
			_waypoints.row(v) = x0.transpose();
		}
	}
	_prev_remaining_set = remaining_set;
	_has_prev_remaining_set = true;

	switch (_solver) {
	case WaypointSolver::kGurobi:
		return SolveWithGurobi(remaining_vertices, x0);
	case WaypointSolver::kMosek:
		return SolveWithMosek(remaining_vertices, x0);
	}
	return false;
}

bool MILPWaypointMPC::SolveWithMosek(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	GraphWaypointProblem problem = BuildGraphWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		_enforce_rigidity, _enforce_rigidity, _objective);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);

	if (result.is_success()) {

		_last_solve_time = _timer.Tick();

		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				const int variable_id = _graph->phi_to_variable_map.at(i);
				const int variable_k = problem.subgraph->subgraph_variable_id(variable_id);
				if (variable_k >= 0 && problem.Assignments.rows() > 0) {
					for (int j = 0; j < num_agents; ++j) {
						const double val = result.GetSolution(problem.Assignments(variable_k, j));
						if (val > 0.5) {
							_assignments(i) = j;
							_var_assignments(variable_id) = j;
							break;
						}
					}
				} else {
					_assignments(i) = -1;
					_var_assignments(variable_id) = -1;
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd W_flat = result.GetSolution(problem.W);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			_waypoints.row(v) = W_flat.row(i);
		}
		if (problem.n_N > 0) {
			_t_routing = result.GetSolution(problem.t);
			_Z_routing = result.GetSolution(problem.Z);
			_t_by_node_id.setZero();
			for (int v : remaining_vertices) {
				const int sg_v = problem.subgraph->subgraph_id(v);
				_t_by_node_id(v) = _t_routing(sg_v + 1);
			}
		}
		return true;
	}

	return false;
}

bool MILPWaypointMPC::SolveWithGurobi(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	GraphWaypointProblem problem = BuildGraphWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		_enforce_rigidity, _enforce_rigidity, _objective);

	// Solve
	drake::solvers::GurobiSolver solver;
	auto result = solver.Solve(*problem.prog);

	if (result.is_success()) {

		_last_solve_time = _timer.Tick();

		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				const int variable_id = _graph->phi_to_variable_map.at(i);
				const int variable_k = problem.subgraph->subgraph_variable_id(variable_id);
				if (variable_k >= 0 && problem.Assignments.rows() > 0) {
					for (int j = 0; j < num_agents; ++j) {
						const double val = result.GetSolution(problem.Assignments(variable_k, j));
						if (val > 0.5) {
							_assignments(i) = j;
							_var_assignments(variable_id) = j;
							break;
						}
					}
				} else {
					_assignments(i) = -1;
					_var_assignments(variable_id) = -1;
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd W_flat = result.GetSolution(problem.W);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			_waypoints.row(v) = W_flat.row(i);
		}
		if (problem.n_N > 0) {
			_t_routing = result.GetSolution(problem.t);
			_Z_routing = result.GetSolution(problem.Z);
			_t_by_node_id.setZero();
			for (int v : remaining_vertices) {
				const int sg_v = problem.subgraph->subgraph_id(v);
				_t_by_node_id(v) = _t_routing(sg_v + 1);
			}
		}
		return true;
	}

	std::cerr << "No feasible solution across all assignments.\n";

	if (result.is_success()) {
	} else {
		PrintSolverReport(&problem, result, 1e-6);
		return false;
	}

	return false;
}
