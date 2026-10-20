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
// If EE is free-body: ee_start3 = graph->agent_col_offset(robot_ag) (where the pos lives).
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
				PointWorldFromRow(x0_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);

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
					PointWorldFromRow(x0_expr, graph->object_col_offset(id_i), graph->object_ambient_dim(id_i), workspace_dim);
				const Eigen::VectorX<Expression> p_i_v =
					PointWorldFromRow(X_v_expr, graph->object_col_offset(id_i), graph->object_ambient_dim(id_i), workspace_dim);
				const Eigen::VectorX<Expression> p_j_0 =
					PointWorldFromRow(x0_expr, graph->object_col_offset(id_j), graph->object_ambient_dim(id_j), workspace_dim);
				const Eigen::VectorX<Expression> p_j_v =
					PointWorldFromRow(X_v_expr, graph->object_col_offset(id_j), graph->object_ambient_dim(id_j), workspace_dim);

				const Eigen::VectorX<Expression> d_0 = p_i_0 - p_j_0;
				const Eigen::VectorX<Expression> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_0.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description(fmt::format("-1->{} p2p distance rigidity", v));
			}
		}


		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_0 =
				PointWorldFromRow(x0_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);

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
				PointWorldFromRow(X_u_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);

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
					PointWorldFromRow(X_u_expr, graph->object_col_offset(id_i), graph->object_ambient_dim(id_i), workspace_dim);
				const Eigen::VectorX<Expression> p_i_v =
					PointWorldFromRow(X_v_expr, graph->object_col_offset(id_i), graph->object_ambient_dim(id_i), workspace_dim);
				const Eigen::VectorX<Expression> p_j_u =
					PointWorldFromRow(X_u_expr, graph->object_col_offset(id_j), graph->object_ambient_dim(id_j), workspace_dim);
				const Eigen::VectorX<Expression> p_j_v =
					PointWorldFromRow(X_v_expr, graph->object_col_offset(id_j), graph->object_ambient_dim(id_j), workspace_dim);

				const Eigen::VectorX<Expression> d_u = p_i_u - p_j_u;
				const Eigen::VectorX<Expression> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_u.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description("u->v p2p distance rigidity");
			}
		}

		for (int obj_id : spec.held_point_ids) {
			const Eigen::VectorX<Expression> p_WP_u =
				PointWorldFromRow(X_u_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
			const Eigen::VectorX<Expression> p_WP_v =
				PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);

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
		o.p_WP_0 = PointWorldFromRow(x0_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
		o.p_WP_v = PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
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
		const int ee_start3 = graph->agent_col_offset(k);  // assuming first workspace_dim coords are world position

		// Gate each held point’s exact-rigidity residual.
		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_0 = objects[idx].p_WP_0;
			const auto& p_WP_v = objects[idx].p_WP_v;

			// For each held object (obj_id):
			const int obj_start3 = graph->object_col_offset(objects[idx].id);

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
		o.p_WP_u = PointWorldFromRow(X_u_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
		o.p_WP_v = PointWorldFromRow(X_v_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
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
		const int ee_start3 = graph->agent_col_offset(k);  // assuming first workspace_dim coords are world position

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
			const int obj_start3 = graph->object_col_offset(objects[idx].id);

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
	const Variable& gate,
	const std::string& desc) {

	using Eigen::Matrix;
	const double neg_inf = -std::numeric_limits<double>::infinity();
	const int workspace_dim = graph->workspace_dim;

	const int k = spec.robot_ag;
	const int ee_start3 = graph->agent_col_offset(k);
	const Eigen::VectorXd ee_span_hint = Eigen::VectorXd::Constant(workspace_dim, 2.0);

	auto [p_WE_ref, R_WE_ref]   = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, ref_expr);
	auto [p_WE_cand, R_WE_cand] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, cand_expr);

	for (int obj_id : spec.held_point_ids) {
		const Eigen::VectorX<Expression> p_WP_ref  = PointWorldFromRow(ref_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);
		const Eigen::VectorX<Expression> p_WP_cand = PointWorldFromRow(cand_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim);

		const Eigen::VectorX<Expression> rel_ref  = R_WE_ref.transpose()  * (p_WP_ref  - p_WE_ref);
		const Eigen::VectorX<Expression> rel_cand = R_WE_cand.transpose() * (p_WP_cand - p_WE_cand);
		const Eigen::VectorX<Expression> residual = rel_ref - rel_cand;

		const int obj_start3 = graph->object_col_offset(obj_id);
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
			PointWorldFromRow(ref_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim),
			PointWorldFromRow(cand_expr, graph->object_col_offset(obj_id), graph->object_ambient_dim(obj_id), workspace_dim)});
	}

	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_ref, R_WE_ref]   = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, ref_expr);
		auto [p_WE_cand, R_WE_cand] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, cand_expr);
		const int ee_start3 = graph->agent_col_offset(k);

		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_ref  = objects[idx].p_WP_ref;
			const auto& p_WP_cand = objects[idx].p_WP_cand;

			const Eigen::VectorX<Expression> rel_ref  = R_WE_ref.transpose()  * (p_WP_ref  - p_WE_ref);
			const Eigen::VectorX<Expression> rel_cand = R_WE_cand.transpose() * (p_WP_cand - p_WE_cand);
			const Eigen::VectorX<Expression> residual = rel_ref - rel_cand;

			const int obj_start3 = graph->object_col_offset(objects[idx].id);
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

	const SubgraphOfConstraints subgraph(graph, remaining_vertices);

	using namespace drake::solvers;

	// Create problem struct
	GraphWaypointProblem problem;

	// Create initial program
	std::unique_ptr<MathematicalProgram> prog_ptr = std::make_unique<MathematicalProgram>();
	MathematicalProgram& prog = *prog_ptr;

	// record the subgraph
	problem.subgraph = std::make_unique<SubgraphOfConstraints>(subgraph);

	// Routing graph N = {depot=0} ∪ task_nodes(1..n_V) ∪ {sink=n_V+1}, |N|=n_N.
	const int n_V = subgraph.num_nodes();
	const int n_N = n_V + 2;
	const int depot_idx = 0;
	const int sink_idx  = n_N - 1;

	auto ri       = [&](int sg_v) -> int { return sg_v + 1; };

	const double kInf     = std::numeric_limits<double>::infinity();
	const double x_range  = (graph->_global_x_ub - graph->_global_x_lb).maxCoeff();
	// Big-M for the routing/cost-linking constraints below (2 & 3, and M_time
	// off of it) -- must upper-bound the arc cost regardless of which agent
	// (now possibly a different width from any other) traverses it, hence
	// the WIDEST agent's own width, not a uniform `robot_dim`.
	int max_agent_width = 0;
	for (int j = 0; j < num_agents; ++j) max_agent_width = std::max(max_agent_width, graph->robot_ambient_dim(j));
	const double M_cost   = max_agent_width * x_range;
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

	// W: continuous configuration variables (n x graph->total_dim -- agents
	// then objects, each its own width, not a uniform per-agent/per-object
	// stride -- see GraphOfConstraints::agent_col_offset/object_col_offset).
	MatrixXDecisionVariable W = prog.NewContinuousVariables(subgraph.num_nodes(), graph->total_dim, "W");
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

	const std::vector<Variable> cond_binary_vars = AddConditionBinaryVars(prog, *graph);

	// Visits and arc pruning (see RoutingPresolve).
	const RoutingPresolve presolve = BuildRoutingPresolve(*graph, subgraph);
	const auto& roles_at = presolve.roles_at;
	const auto& required_at = presolve.required_at;
	auto arc_allowed = [&](int j, int a, int b) { return presolve.arc_allowed(j, a, b); };

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

	// Constraint 1: makespan = t(sink), the time the last agent finishes,
	// waiting included (Constraint 9 bounds t(sink) by every agent's last
	// task node time).
	prog.AddLinearEqualityConstraint(
		Expression(L_ms_var(0)) - Expression(t_var(sink_idx)), 0.0);

	// Constraints 2 & 3: cost-linking (big-M) and zero cost on sink arcs
	for (int j = 0; j < num_agents; ++j) {

		// Helper for getting waypoint vector for agent j at node node_ri --
		// agent j's own width/offset, not a uniform robot_dim/j*robot_dim
		// stride.
		const int j_offset = graph->agent_col_offset(j);
		const int j_width = graph->robot_ambient_dim(j);
		auto get_w = [&](int node_ri) -> Eigen::VectorX<Expression> {
			Eigen::VectorX<Expression> w(j_width);
			if (node_ri == depot_idx) {
				for (int k = 0; k < j_width; ++k)
					w(k) = Expression(x0(j_offset + k));
			} else {
				// task node: subgraph index (node_ri - 1)
				w = W.row(node_ri - 1).segment(j_offset, j_width)
				      .template cast<Expression>();
			}
			return w;
		};

		for (int a = 0; a < n_N; ++a) {
			for (int b = 0; b < n_N; ++b) {
				// Skip self loops
				if (a == b) continue;

				// Pruned arcs (see arc_allowed above): unused, no cost
				if (!arc_allowed(j, a, b)) {
					prog.AddBoundingBoxConstraint(0.0, 0.0, Z_var(z_idx_fn(j, a, b)));
					prog.AddLinearEqualityConstraint(
						Eigen::RowVectorXd::Ones(1), 0.0,
						e_var.segment(e_idx_fn(j, a, b), 1));
					continue;
				}

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

	// Constraint 8: visits. Agent k visits task node v (inflow 1) iff it
	// holds one of the roles referenced at v, or a constraint at v pins
	// agent_q(k) directly (a required visit); it holds at most one role per
	// node, and every role goes to exactly one agent:
	//   required:  inflow(k,v) = 1,  Σ_{r at v} A(r,k) ≤ 1
	//   otherwise: inflow(k,v) = Σ_{r at v} A(r,k) ≤ 1  (0 at a node with no roles)
	// (roles_at / required_at: computed above, before the arc pruning.)
	{
		AddRoleExactlyOneAgent(prog, subgraph, Assignments, num_agents);

		for (int sg_v2 = 0; sg_v2 < n_V; ++sg_v2) {
			const int rv = ri(sg_v2);
			for (int k = 0; k < num_agents; ++k) {
				Expression in_flow, alpha;
				int num_roles = 0;
				for (int a2 = 0; a2 < n_N; ++a2)
					if (a2 != rv) in_flow += Expression(Z_var(z_idx_fn(k, a2, rv)));
				for (int var : roles_at[sg_v2]) {
					const int variable_k = subgraph.subgraph_variable_id(var);
					if (variable_k < 0) continue;
					alpha += Expression(Assignments(variable_k, k));
					++num_roles;
				}
				if (num_roles > 1) prog.AddLinearConstraint(alpha, -kInf, 1.0);
				if (required_at[sg_v2].contains(k))
					prog.AddLinearEqualityConstraint(in_flow, 1.0);
				else
					prog.AddLinearEqualityConstraint(in_flow - alpha, 0.0);
			}
		}
	}

	// Constraint 8b: assignment commits.
	AddCommittedAssignments(prog, *graph, subgraph, Assignments);

	// Constraint 9: time propagation — Z[j,a,b]=1 ⟹ t[b] ≥ t[a] + e[j,a,b]
	// (into the sink e = 0 by Constraint 3, so t(sink) ≥ each agent's finish time)
	for (int j = 0; j < num_agents; ++j) {
		for (int a = 0; a <= n_V; ++a) {   // a ∈ {depot=0} ∪ task routing indices
			for (int b = 1; b <= sink_idx; ++b) {  // b ∈ task nodes ∪ {sink}
				if (a == b || !arc_allowed(j, a, b)) continue;
				// t[b] - t[a] - e[j,a,b] + M_time*(1-Z[j,a,b]) ≥ 0
				prog.AddLinearConstraint(
					Expression(t_var(b)) - Expression(t_var(a))
					- Expression(e_var(e_idx_fn(j, a, b)))
					+ M_time - M_time * Expression(Z_var(z_idx_fn(j, a, b))),
					0.0, kInf);
			}
		}
	}

	// Constraint 10: DAG ordering -- hard (structure edges) or conditional (_conditional_ordering_map)
	AddOrderingConstraints(prog, *graph, subgraph, t_var, M_time, Assignments,
			       previous_var_assignments, cond_binary_vars);

	// Constraint 11: assignment-variable constraints (add_variable_constraint /
	// add_variable_ineq_constraint).
	AddAssignmentVariableConstraints(prog, *graph, subgraph, Assignments);

	// Constraint 12: node constraints from the unified symbolic API
	// (add_constraint / add_assignable_constraint), compiled from the stored
	// Formula. See symbolic_constraint_compiler.*.
	for (const auto& [phi_id, rec] : subgraph.get_subgraph_symbolic_ops()) {
		CompileSymbolicNodeConstraint(prog, subgraph, *graph, rec, W, Assignments);
	}

	// Constraint 13: edge constraints from the unified symbolic API
	// (add_edge_constraint). A relational formula (rec.along_edge == false)
	// is a single relation coupling both endpoints. An "along the edge"
	// formula (rec.along_edge == true) is a per-endpoint invariant, and node
	// order is a routing decision (Z/t above), not fixed identity: another
	// subgraph node w with no DAG relationship to u or v may end up scheduled
	// with t(u) <= t(w) <= t(v). So re-apply it, betweenness-gated, at every
	// other subgraph node so it isn't silently unconstrained mid-edge.
	std::map<std::tuple<int, int, int>, Variable> betweenness_cache;
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
			// A rigid carry (HoldDeclaration.rigid) constrains the held
			// point's offset in the end-effector's OWN frame, which is
			// nonlinear in the holder's configuration -- 14a below is the
			// linear translation-only form. Raise rather than silently
			// enforce the weaker relation.
			if (hold.rigid) {
				throw std::runtime_error(
					"MILPWaypointMPC: hold " + std::to_string(hold_id) +
					" was declared rigid=True (offset fixed in the end-effector's own "
					"frame); that relation is nonlinear -- use the evolutionary "
					"waypoint solver, or declare the hold translation-only.");
			}
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
							       static_spec,
							       /*exact_rigidity=*/true);
				} else {
					AddHoldRigidityAssignable(prog, subgraph, graph, W, hold.u_node, hold.v_node,
								   assignable_spec, A_row);
				}
			} else {
				// x0-boundary edge: the hold started before this horizon.
				ref_expr = x0.template cast<Expression>();
				t_idx_ref = depot_idx;

				if (is_static) {
					AddHoldRigidityStaticToX0(prog, subgraph, graph, W, hold.v_node,
								   static_spec,
								   x0, /*exact_rigidity=*/true);
				} else {
					AddHoldRigidityAssignableToX0(prog, subgraph, graph, W, hold.v_node,
								       assignable_spec,
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
					AddHoldRigidityStaticGated(prog, graph, ref_expr, cand_expr, static_spec, gate, desc);
				} else {
					AddHoldRigidityAssignableGated(prog, graph, ref_expr, cand_expr, assignable_spec, A_row, gate, desc);
				}
			}
		}
	}

	// 14b: stationary objects (see AddStationaryObjectConstraints). Always
	// enforced, independent of enforce_rigidity: it only compares plain
	// object position columns, so it's linear regardless of robot kind.
	AddStationaryObjectConstraints(prog, *graph, subgraph, W.template cast<Expression>(),
				       x0, t_var, M_time);


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

	// Objective: MinMax variants minimize the makespan (Constraint 1 sets
	// L_ms_var = t(sink), the time the last agent finishes, waiting
	// included); Avg variants minimize the average total route cost across
	// agents directly from e_var.
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

bool MILPWaypointMPC::Solve(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	BeginSolve(remaining_vertices, x0);

	switch (_solver) {
	case WaypointSolver::kGurobi:
		return SolveWith(drake::solvers::GurobiSolver(), remaining_vertices, x0);
	case WaypointSolver::kMosek:
		return SolveWith(drake::solvers::MosekSolver(), remaining_vertices, x0);
	}
	return false;
}

bool MILPWaypointMPC::SolveWith(
	const drake::solvers::SolverInterface& solver,
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	GraphWaypointProblem problem = BuildGraphWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		_enforce_rigidity, _enforce_rigidity, _objective);

	drake::solvers::MathematicalProgramResult result;
	solver.Solve(*problem.prog, std::nullopt, std::nullopt, &result);

	if (!result.is_success()) {
		std::cerr << "No feasible solution across all assignments.\n";
		PrintSolverReport(*problem.prog, result, 1e-6);
		return false;
	}

	_last_solve_time = _timer.Tick();

	StoreAssignments(*problem.subgraph, problem.Assignments, result);

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
