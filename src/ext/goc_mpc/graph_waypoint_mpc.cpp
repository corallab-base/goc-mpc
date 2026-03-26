#include "graph_waypoint_mpc.hpp"

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

inline Eigen::Vector3d seg_width3(const Eigen::VectorXd& lb,
                                  const Eigen::VectorXd& ub,
                                  int start3) {
	return (ub.segment<3>(start3) - lb.segment<3>(start3)).cwiseAbs();
}

// Bound ||p_WP - p_WE||_2 on one “side” (0, u, or v).
// If EE is free-body: ee_start3 = robot_ag * robot_dim (where the 3 pos live).
// If articulated: pass ee_span_hint as a per-axis bound on EE motion; otherwise use a conservative constant.
inline double bound_norm_obj_minus_ee_side(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_in_X,              // std::nullopt if articulated
	const Eigen::Vector3d& ee_span_hint /* meters */) {

	const Eigen::Vector3d w_obj = seg_width3(lb, ub, obj_start3);
	Eigen::Vector3d w_ee;
	if (ee_start3_in_X) {
		w_ee = seg_width3(lb, ub, *ee_start3_in_X);
	} else {
		w_ee = ee_span_hint.cwiseAbs();  // articulated, conservative workspace span
	}
	const Eigen::Vector3d w_sum = w_obj + w_ee;
	return w_sum.norm();  // sqrt((wx+ex)^2 + (wy+ey)^2 + (wz+ez)^2)
}

// 0 -> v (initial-layer) exact-rigidity M (identical per component)
inline Eigen::Vector3d M_exact_toX0_componentwise(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_in_X,      // free-body -> pos segment start; else nullopt
	const Eigen::Vector3d& ee_span_hint) {  // used when articulated
	// Both “sides” are 0 and v; the bound structure is the same for both,
	// so the worst-case 2-norm can be upper-bounded the same way on each side.
	const double B0 = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_in_X, ee_span_hint);
	const double Bv = B0;  // same bounding box globally
	const double Mscalar = B0 + Bv;
	return Eigen::Vector3d::Constant(Mscalar);
}

// u -> v exact-rigidity M (identical per component)
inline Eigen::Vector3d M_exact_edge_componentwise(
	const Eigen::VectorXd& lb, const Eigen::VectorXd& ub,
	int obj_start3,
	std::optional<int> ee_start3_u_in_X,    // free-body u; else nullopt
	std::optional<int> ee_start3_v_in_X,    // free-body v; else nullopt
	const Eigen::Vector3d& ee_span_hint_u,  // articulated u
	const Eigen::Vector3d& ee_span_hint_v) {// articulated v
	const double Bu = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_u_in_X, ee_span_hint_u);
	const double Bv = bound_norm_obj_minus_ee_side(lb, ub, obj_start3, ee_start3_v_in_X, ee_span_hint_v);
	const double Mscalar = Bu + Bv;
	return Eigen::Vector3d::Constant(Mscalar);
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

	Eigen::Matrix<Expression, Eigen::Dynamic, 1> x0_expr = x0.template cast<Expression>();
	Eigen::Matrix<Expression, Eigen::Dynamic, 1> X_v_expr = X.row(sg_v).template cast<Expression>();

	auto [p_WE_0, R_WE_0] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, x0_expr);
	auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_v_expr);

	if (exact_rigidity) {
		for (int obj_id : spec.held_point_ids) {
			const Eigen::Matrix<Expression,3,1> p_WP_0 =
				PointWorldFromRow(x0_expr, objs_start, non_robot_dim, obj_id);
			const Eigen::Matrix<Expression,3,1> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);

			const Eigen::Matrix<Expression,3,1> rel_0 =
				R_WE_0.transpose() * (p_WP_0 - p_WE_0);
			const Eigen::Matrix<Expression,3,1> rel_v =
				R_WE_v.transpose() * (p_WP_v - p_WE_v);


			if (graph->robot_is_pos_rot_mat(spec.robot_ag)) {
				// If the rotation matrix is directly
				// represented in the decision variables, R_WE_0/v^T * p_WE/P_0/v is quadratic
				for (int i = 0; i < 3; ++i) {
					prog.AddQuadraticConstraint(rel_0(i) - rel_v(i), -0.001, 0.001)
						.evaluator()->set_description(fmt::format("-1->{} exact rigidity {}", v, obj_id));
				}
			} else {
				prog.AddConstraint(rel_0 - rel_v,
						   Eigen::Vector3d::Constant(3, -0.001),
						   Eigen::Vector3d::Constant(3, 0.001))
					.evaluator()->set_description(fmt::format("-1->{} exact rigidity {}", v, obj_id));
			}



		}
	} else {
		for (size_t i = 0; i < spec.held_point_ids.size(); ++i) {
			for (size_t j = i + 1; j < spec.held_point_ids.size(); ++j) {
				const int id_i = spec.held_point_ids[i];
				const int id_j = spec.held_point_ids[j];

				const Eigen::Matrix<Expression,3,1> p_i_0 =
					PointWorldFromRow(x0_expr, objs_start, non_robot_dim, id_i);
				const Eigen::Matrix<Expression,3,1> p_i_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_i);
				const Eigen::Matrix<Expression,3,1> p_j_0 =
					PointWorldFromRow(x0_expr, objs_start, non_robot_dim, id_j);
				const Eigen::Matrix<Expression,3,1> p_j_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_j);

				const Eigen::Matrix<Expression,3,1> d_0 = p_i_0 - p_j_0;
				const Eigen::Matrix<Expression,3,1> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_0.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description(fmt::format("-1->{} p2p distance rigidity", v));
			}
		}


		for (int obj_id : spec.held_point_ids) {
			const Eigen::Matrix<Expression,3,1> p_WP_0 =
				PointWorldFromRow(x0_expr,   objs_start, non_robot_dim, obj_id);
			const Eigen::Matrix<Expression,3,1> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);

			const Eigen::Matrix<Expression,3,1> d_0 = p_WP_0 - p_WE_0;
			const Eigen::Matrix<Expression,3,1> d_v = p_WP_v - p_WE_v;

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

        // Cast Variables -> Expressions and materialize as owning row vectors.
	Eigen::VectorX<drake::symbolic::Expression> X_u_expr =
		X.row(sg_u).template cast<drake::symbolic::Expression>();
	Eigen::VectorX<drake::symbolic::Expression> X_v_expr =
		X.row(sg_v).template cast<drake::symbolic::Expression>();

	auto [p_WE_u, R_WE_u] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_u_expr);
	auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, spec.robot_ag, spec.ee_frame_name, X_v_expr);

	if (exact_rigidity) {
		for (int obj_id : spec.held_point_ids) {
			const Eigen::Matrix<Expression,3,1> p_WP_u =
				PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id);
			const Eigen::Matrix<Expression,3,1> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);

			const Eigen::Matrix<Expression,3,1> rel_u =
				R_WE_u.transpose() * (p_WP_u - p_WE_u);
			const Eigen::Matrix<Expression,3,1> rel_v =
				R_WE_v.transpose() * (p_WP_v - p_WE_v);


			if (graph->robot_is_pos_rot_mat(spec.robot_ag)) {
				// If the rotation matrix is directly
				// represented in the decision variables, R_WE_0/v^T * p_WE/P_0/v is quadratic
				for (int i = 0; i < 3; ++i) {
					prog.AddQuadraticConstraint(rel_u(i) - rel_v(i), -0.001, 0.001)
						.evaluator()->set_description(fmt::format("{}->{} exact rigidity {}", u, v, obj_id));
				}
			} else {
				prog.AddConstraint(rel_u - rel_v,
						   Eigen::Vector3d::Constant(3, -0.001),
						   Eigen::Vector3d::Constant(3, 0.001))
					.evaluator()->set_description(fmt::format("{}->{} exact rigidity {}", u, v, obj_id));
			}
		}
	} else {
		for (size_t i = 0; i < spec.held_point_ids.size(); ++i) {
			for (size_t j = i + 1; j < spec.held_point_ids.size(); ++j) {
				const int id_i = spec.held_point_ids[i];
				const int id_j = spec.held_point_ids[j];

				const Eigen::Matrix<Expression,3,1> p_i_u =
					PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, id_i);
				const Eigen::Matrix<Expression,3,1> p_i_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_i);
				const Eigen::Matrix<Expression,3,1> p_j_u =
					PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, id_j);
				const Eigen::Matrix<Expression,3,1> p_j_v =
					PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, id_j);

				const Eigen::Matrix<Expression,3,1> d_u = p_i_u - p_j_u;
				const Eigen::Matrix<Expression,3,1> d_v = p_i_v - p_j_v;
				Expression e = d_v.squaredNorm() - d_u.squaredNorm();
				prog.AddQuadraticConstraint(e, 0.0, 0.0)
					.evaluator()->set_description("u->v p2p distance rigidity");
			}
		}

		for (int obj_id : spec.held_point_ids) {
			const Eigen::Matrix<Expression,3,1> p_WP_u =
				PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id);
			const Eigen::Matrix<Expression,3,1> p_WP_v =
				PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);

			const Eigen::Matrix<Expression,3,1> d_u = p_WP_u - p_WE_u;
			const Eigen::Matrix<Expression,3,1> d_v = p_WP_v - p_WE_v;

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

	// Cast once.
	Eigen::VectorX<Expression> x0_expr = x0.template cast<Expression>();
	Eigen::VectorX<Expression> X_v_expr = X.row(sg_v).template cast<Expression>();

	// Precompute object world points from x0 / X_v once per object.
	struct ObjPts {
		int id;
		Matrix<Expression,3,1> p_WP_0;
		Matrix<Expression,3,1> p_WP_v;
	};
	std::vector<ObjPts> objects;
	objects.reserve(spec.held_point_ids.size());
	for (int obj_id : spec.held_point_ids) {
		ObjPts o;
		o.id = obj_id;
		o.p_WP_0 = PointWorldFromRow(x0_expr, objs_start, non_robot_dim, obj_id);
		o.p_WP_v = PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);
		objects.push_back(std::move(o));
	}

	const double neg_inf = -std::numeric_limits<double>::infinity();
	Eigen::Vector3d ee_span_hint(2.0, 2.0, 2.0);

	// For each possible agent k, construct the same residual as static-HoldSpec(robot_ag=k),
	// then gate with +/- M * (1 - A_row(k)).
	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_0, R_WE_0] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, x0_expr);
		auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_v_expr);

		// For free-body agent k:
		const int ee_start3 = k * robot_dim;  // assuming first 3 are world position

		// Gate each held point’s exact-rigidity residual.
		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_0 = objects[idx].p_WP_0;
			const auto& p_WP_v = objects[idx].p_WP_v;

			// For each held object (obj_id):
			const int obj_start3 = objs_start + objects[idx].id * non_robot_dim;

			const Eigen::Vector3d Mvec = M_exact_toX0_componentwise(
				graph->_global_x_lb, graph->_global_x_ub,
				obj_start3,
				/*ee_start3_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3)
				: std::nullopt,
				/*ee_span_hint=*/ee_span_hint);

			// Exact rigidity residual components:
			// rel_0 = R_WE_0^T (p_WP_0 - p_WE_0)
			// rel_v = R_WE_v^T (p_WP_v - p_WE_v)
			// residual = rel_0 - rel_v   (should be ~ 0 when this agent is selected)
			const Matrix<Expression,3,1> rel_0 = R_WE_0.transpose() * (p_WP_0 - p_WE_0);
			const Matrix<Expression,3,1> rel_v = R_WE_v.transpose() * (p_WP_v - p_WE_v);
			const Matrix<Expression,3,1> residual = rel_0 - rel_v;

			// Big-M gating:
			// -M*(1 - A_k) <= residual_i <= M*(1 - A_k)
			// Move to LHS to keep constant bounds:
			// residual_i - M*(1 - A_k) <= 0
			// -residual_i - M*(1 - A_k) <= 0
			for (int j = 0; j < 3; ++j) {
				const Expression slack = Mvec(j) * (1.0 - A_row(k));


				if (graph->robot_is_pos_rot_mat(k)) {
					// If the rotation matrix is directly
					// represented in the decision variables, R_WE_0/v^T * p_WE/P_0/v is quadratic

					// Upper bound residual: residual - slack <= 0
					prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, +)");

					// Lower bound residual: -residual - slack <= 0
					prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, -)");

				} else {
					// Upper bound residual: residual - slack <= 0
					prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, +)");

					// Lower bound residual: -residual - slack <= 0
					prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("0->v exact rigidity (assignable, -)");
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

	// Cast Variables -> Expressions
	Eigen::VectorX<Expression> X_u_expr = X.row(sg_u).template cast<Expression>();
	Eigen::VectorX<Expression> X_v_expr = X.row(sg_v).template cast<Expression>();

	const int num_agents = graph->num_agents;

	// Precompute object world points for all held ids at u and v.
	struct ObjPtsUV {
		int id;
		Matrix<Expression,3,1> p_WP_u;
		Matrix<Expression,3,1> p_WP_v;
	};
	std::vector<ObjPtsUV> objects;
	objects.reserve(spec.held_point_ids.size());
	for (int obj_id : spec.held_point_ids) {
		ObjPtsUV o;
		o.id = obj_id;
		o.p_WP_u = PointWorldFromRow(X_u_expr, objs_start, non_robot_dim, obj_id);
		o.p_WP_v = PointWorldFromRow(X_v_expr, objs_start, non_robot_dim, obj_id);
		objects.push_back(std::move(o));
	}

	const double neg_inf = -std::numeric_limits<double>::infinity();
	Eigen::Vector3d ee_span_hint(2.0, 2.0, 2.0);

	// For each possible agent k, construct the same residual as static-HoldSpec(robot_ag=k),
	// then gate with +/- M * (1 - A_row(k)).
	for (int k = 0; k < num_agents; ++k) {
		auto [p_WE_u, R_WE_u] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_u_expr);
		auto [p_WE_v, R_WE_v] = PoseFromRow<Expression>(graph, k, spec.ee_frame_name, X_v_expr);

		// For free-body agent k:
		const int ee_start3 = k * robot_dim;  // assuming first 3 are world position

		for (size_t idx = 0; idx < spec.held_point_ids.size(); ++idx) {
			const auto& p_WP_u = objects[idx].p_WP_u;
			const auto& p_WP_v = objects[idx].p_WP_v;

			// Exact rigidity residual across the edge:
			// rel_u = R_WE_u^T (p_WP_u - p_WE_u)
			// rel_v = R_WE_v^T (p_WP_v - p_WE_v)
			// residual = rel_u - rel_v  ≈ 0 when this agent is selected
			const Matrix<Expression,3,1> rel_u = R_WE_u.transpose() * (p_WP_u - p_WE_u);
			const Matrix<Expression,3,1> rel_v = R_WE_v.transpose() * (p_WP_v - p_WE_v);
			const Matrix<Expression,3,1> residual = rel_u - rel_v;

			// For each held object (obj_id):
			const int obj_start3 = objs_start + objects[idx].id * non_robot_dim;

			// u->v:
			const Eigen::Vector3d Mvec_uv = M_exact_edge_componentwise(
				graph->_global_x_lb, graph->_global_x_ub,
				obj_start3,
				/*ee_start3_u_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				/*ee_start3_v_in_X=*/graph->robot_is_free_body(k) ? std::make_optional(ee_start3) : std::nullopt,
				/*ee_span_hint_u=*/ee_span_hint,
				/*ee_span_hint_v=*/ee_span_hint);

			// Big-M gating per component:
			// residual_j - M*(1 - A_k) <= 0
			// -residual_j - M*(1 - A_k) <= 0
			
			for (int j = 0; j < 3; ++j) {
				const Expression slack = Mvec_uv(j) * (1.0 - A_row(k));


				if (graph->robot_is_pos_rot_mat(k)) {
					// If the rotation matrix is directly
					// represented in the decision variables, R_WE_0/v^T * p_WE/P_0/v is quadratic

					// Upper bound residual: residual - slack <= 0
					prog.AddQuadraticConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, +)");

					// Lower bound residual: -residual - slack <= 0
					prog.AddQuadraticConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, -)");


				} else {
					// Upper bound residual: residual - slack <= 0
					prog.AddConstraint(residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, +)");

					// Lower bound residual: -residual - slack <= 0
					prog.AddConstraint(-residual(j) - slack, neg_inf, 0.0)
						.evaluator()->set_description("u->v exact rigidity (assignable, -)");
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
	bool relax_binary_vars) {

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

	MatrixXDecisionVariable Assignments;
	if (relax_binary_vars) {
		// Define the assignment variables as continuous so that they
		// can be pinned manually.
		Assignments = prog.NewContinuousVariables(subgraph.num_variables(),
							  num_agents, "Assignments");
		problem.Assignments = Assignments;

		Eigen::MatrixXd lb = Eigen::MatrixXd::Zero(Assignments.rows(), Assignments.cols());
		Eigen::MatrixXd ub = Eigen::MatrixXd::Ones(Assignments.rows(), Assignments.cols());
		problem.A_bounds = std::make_unique<drake::solvers::Binding<drake::solvers::BoundingBoxConstraint>>(prog.AddBoundingBoxConstraint(lb, ub, problem.Assignments));
		problem.A_bounds->evaluator()->set_description("continuous agent assignment bounds");

		for (int i = 0; i < subgraph.num_variables(); ++i) {
			prog.AddLinearEqualityConstraint(
				Eigen::RowVectorXd::Ones(num_agents), 1.0, Assignments.row(i))
				.evaluator()->set_description("exclusive assignment");
		}
	} else {
		if (subgraph.num_variables() > 0) {
			Assignments = prog.NewBinaryVariables(subgraph.num_variables(),
							      num_agents, "Assignments");
			problem.Assignments = Assignments;
		}

		for (int i = 0; i < subgraph.num_variables(); ++i) {
			prog.AddLinearEqualityConstraint(
				Eigen::RowVectorXd::Ones(num_agents),
				1.0, Assignments.row(i))
				.evaluator()->set_description("exclusive assignment");;
		}
	}

	const int robot_dim = graph->dim;
	const int objs_start = graph->num_agents * graph->dim;
	const int non_robot_dim = graph->non_robot_dim;

	// x: continuous configuration variables (n x m*d+o).
	MatrixXDecisionVariable X = prog.NewContinuousVariables(subgraph.num_nodes(), num_agents * robot_dim + num_objects * non_robot_dim, "X");
	problem.X = X;

	for (int i = 0; i < X.rows(); ++i) {
		prog.AddBoundingBoxConstraint(graph->_global_x_lb, graph->_global_x_ub, X.row(i).transpose())
			.evaluator()->set_description("configuration space bounds");
	}

	//
	// QUATERNION / ROTATION MATRIX CONSTRAINTS
	//

	for (int ag = 0; ag < num_agents; ++ag) {
		if (graph->robot_is_pos_quat(ag)) {
			for (int node = 0; node < subgraph.num_nodes(); ++node) {
				VectorX<Expression> ag_quat = AsExprRow(X.row(node).segment(ag * robot_dim + 3, 4));
				Expression ag_quat_norm = ag_quat.squaredNorm();
				const double tol = 0.001;
				prog.AddQuadraticConstraint(ag_quat_norm, 1.0-tol, 1.0+tol)
					.evaluator()->set_description("unit quaternion constraint");
			}
		} else if (graph->robot_is_pos_rot_mat(ag)) {
			for (int node = 0; node < subgraph.num_nodes(); ++node) {

				VectorXDecisionVariable R_flat = X.row(node).segment(ag * robot_dim + 3, 9);
				Eigen::Matrix<Variable, 3, 3> R;
				for (int i = 0; i < 3; ++i) {
					for (int j = 0; j < 3; ++j) {
						R(i, j) = R_flat(i * 3 + j);
					}
				}

			        // orthogonality constraint: R^T R = I
				Eigen::Matrix<Expression, 3, 3> orthogonality_residual = R.transpose() * R - Eigen::Matrix3d::Identity();

				const double tol = 0.001;
				for (int i = 0; i < 3; ++i) {
					for (int j = i; j < 3; ++j) {
						prog.AddQuadraticConstraint(orthogonality_residual(i, j), 0.0-tol, 0.0+tol)
							.evaluator()->set_description("rotation matrix orthogonality constraint");
					}
				}

				// right-handedness: col1 x col2 = col3
				Eigen::Matrix<Expression, 3, 1> right_handedness_residual = R.col(0).cross(R.col(1)) - R.col(2);
				for (int i = 0; i < 3; ++i) {
					prog.AddQuadraticConstraint(right_handedness_residual(i), 0.0-tol, 0.0+tol)
						.evaluator()->set_description("rotation matrix right-handedness constraint");
				}
			}
		}
	}

	///////////////////////////////////////////////////////////////////////
        //           OBJECTIVE FUNCTION AND RIGIDITY CONSTRAINT              //
        ///////////////////////////////////////////////////////////////////////

	//
        // ACT 1: INITIAL EDGES FROM CURRENT STATE TO NEXT NODES
        //

	//
        // PART 1: IDENTIFY EDGES SPANNING OBJECT MANIPULATION
        //

	const auto& next_edge_ops = graph->get_next_edge_ops(remaining_vertices);
	std::set<int> possibly_manipulated_cubes_for_initial_layer;
	std::vector<HoldSpec> hold_specs_for_initial_layer;
	std::vector<AssignableHoldSpec> assignable_hold_specs_for_initial_layer;
	for (const auto& [edge_phi_id, op] : next_edge_ops) {
		{
			// If you have a static assignment for this edge op, use it.
			auto it = graph->_edge_phi_to_static_assignment_map.find(edge_phi_id);
			if (it != graph->_edge_phi_to_static_assignment_map.end()) {
				const int ag = it->second;
				HoldSpec spec;
				spec.robot_ag = ag;
				spec.ee_frame_name = "ee_link";
				spec.held_point_ids.assign(op.cubes.begin(), op.cubes.end());

				hold_specs_for_initial_layer.push_back(spec);
			}
		}
		{
			// If you have a variable assignment for this edge, use it
			auto it = graph->edge_phi_to_variable_map.find(edge_phi_id);
			if (it != graph->edge_phi_to_variable_map.end()) {
				const int var = it->second;
				AssignableHoldSpec spec;
				spec.var = var;
				spec.ee_frame_name = "ee_link";
				spec.held_point_ids.assign(op.cubes.begin(), op.cubes.end());

				assignable_hold_specs_for_initial_layer.push_back(spec);
			}
		}
		possibly_manipulated_cubes_for_initial_layer.insert(
			op.cubes.begin(), op.cubes.end());
	}

	//
        // PART 2: ADD LAYER 0 COST FOR EDGES TO NEXT NODES IN SUBGRAPH AND RIGIDITY CONSTRAINTS
        // IF CURRENTLY MANIPULATING ANYTHING
        //

	for (auto v : subgraph.structure.sources()) {
		int sg_v = subgraph.subgraph_id(v);

		// First, costs to minimize across transitions from x0 to the source
		// nodes in the subgraph.
		for (int ag = 0; ag < num_agents; ++ag) {
			VectorX<Expression> x0_exps(robot_dim);
			for (int k = 0; k < robot_dim; ++k) {
				x0_exps(k) = Expression(x0(ag * robot_dim + k));
			}
			Expression dist = splines->at(ag).squared_distance(x0_exps, X.row(sg_v).segment(ag * robot_dim, robot_dim));
			prog.AddCost(dist);
		}

		// Second, for the source nodes, add CONSTRAINTS saything that a block
		// does not move from x0 unless it is moved.
		for (int obj = 0; obj < num_objects; ++obj) {
			const int start = objs_start + obj * non_robot_dim;

			// Grab the decision-variable segment (as Expressions)
			// Note: X.row(sg_v) is RowVector<Expression>, so transpose to column.
			Eigen::VectorX<drake::symbolic::Expression> X_seg =
				X.row(sg_v).segment(start, non_robot_dim).transpose();

			// Grab the constant target segment
			Eigen::VectorXd x0_seg = x0.segment(start, non_robot_dim);

			if (!possibly_manipulated_cubes_for_initial_layer.contains(obj)) {
				// Enforce X_seg == x0_seg (all objects not effected by an edge constraint)
				prog.AddLinearEqualityConstraint(X_seg, x0_seg)
					.evaluator()->set_description(fmt::format("0->{} stationary point {}", v, obj));
			}
		}

		// Third, optionally enforce rigidity between points and
		// end-effectors to enable solving for robot poses based on
		// keypoint requirements
		if (enforce_rigidity) {
			for (const AssignableHoldSpec& hold_spec : assignable_hold_specs_for_initial_layer) {
				auto A_row = GetARowForVar(subgraph, Assignments, hold_spec.var);
				AddHoldRigidityAssignableToX0(
					prog, subgraph, graph, X,
					v, hold_spec,
					/*robot_dim=*/robot_dim,
					/*objs_start=*/objs_start,
					/*non_robot_dim=*/non_robot_dim,
					x0, A_row);
			}

			for (const HoldSpec& hold_spec : hold_specs_for_initial_layer) {
				AddHoldRigidityStaticToX0(
					prog, subgraph, graph, X,
					v, hold_spec,
					/*robot_dim=*/robot_dim,
					/*objs_start=*/objs_start,
					/*non_robot_dim=*/non_robot_dim,
					x0,
					/*exact_rigidity=*/true);
			}
		}

		// Fourth, ensure reassignments aren't done while holding a block??
		// This is a heuristic that maybe will be replaced.
		for (const AssignableHoldSpec& hold_spec : assignable_hold_specs_for_initial_layer) {
			int prev_assignment = previous_var_assignments(hold_spec.var);
			auto A_row = GetARowForVar(subgraph, Assignments, hold_spec.var);
			if (prev_assignment != -1) {
				prog.AddLinearEqualityConstraint(A_row(prev_assignment) == 1)
					.evaluator()->set_description(fmt::format("{} non-changing constraint", hold_spec.var));
			}
		}
	}

	//
        // ACT 2: EDGES BEYOND NEXT NODES
        //

	//
        // PART 3: IDENTIFY CONCURRENT EDGES FOR EACH EDGE IN GRAPH, AND
        // POSSIBLY MANIPULATED CUBES ACROSS EACH EDGE USING CONCURRENT EDGES
        //

	auto result = subgraph.structure.compute_concurrent_edges_overlap();

	std::map<std::pair<int, int>, std::set<int>> possibly_manipulated_cubes_during_each_edge;

	// Loop over all edges in the subgraph
	for (std::size_t i = 0; i < result.edges.size(); ++i) {
		const auto& e = result.edges[i];
		int a = e.u;
		int b = e.v;
		const std::pair<int, int> e_pair = std::make_pair(a, b);

		// Also consider the edge itself
		if (graph->edge_to_phis_map.contains(e_pair)) {
			for (int edge_phi_id : graph->edge_to_phis_map.at(e_pair)) {
				DeferredEdgeOp& op = graph->edge_ops.at(edge_phi_id);

				possibly_manipulated_cubes_during_each_edge[e_pair].insert(
					op.cubes.begin(), op.cubes.end());
			}
		}

		// Iterate over concurrent edges
		for (int j : result.concurrent[i]) {
			const auto& ce = result.edges[j];
			int u = ce.u;
			int v = ce.v;
			const std::pair<int, int> ce_pair = std::make_pair(u, v);

			if (graph->edge_to_phis_map.contains(ce_pair)) {
				for (int edge_phi_id : graph->edge_to_phis_map.at(ce_pair)) {
					DeferredEdgeOp& op = graph->edge_ops.at(edge_phi_id);

					possibly_manipulated_cubes_during_each_edge[e_pair].insert(
						op.cubes.begin(), op.cubes.end());
				}
			}
		}
	}

	std::map<int, std::vector<HoldSpec>> hold_specs_during_each_layer;
	std::map<int, std::vector<AssignableHoldSpec>> assignable_hold_specs_during_each_layer;
	const auto& layers = subgraph.structure.topological_layer_cut_snapshot(
		[&graph,
		 &hold_specs_during_each_layer,
		 &assignable_hold_specs_during_each_layer]
		(int level_k, int u, int v) {
			// this callback is called for all u, v where u is in
			// the layers less than or equal to k (the current) to
			// any layer greater than k, before moving on to
			// processing layer k+1.  Therefore, it can be used to
			// accumulate all the possibly manipulated cubes before
			// nodes in layer k+1.
			if (graph->edge_to_phis_map.contains(std::make_pair(u, v))) {

				for (int edge_phi_id : graph->edge_to_phis_map.at(std::make_pair(u, v))) {
					DeferredEdgeOp& op = graph->edge_ops.at(edge_phi_id);

					// If you have a static assignment for this edge op, use it.
					{
						auto it = graph->_edge_phi_to_static_assignment_map.find(edge_phi_id);
						if (it != graph->_edge_phi_to_static_assignment_map.end()) {
							const int ag = it->second;
							HoldSpec spec;
							spec.robot_ag = ag;
							spec.ee_frame_name = "ee_link";
							spec.held_point_ids.assign(op.cubes.begin(), op.cubes.end());

							hold_specs_during_each_layer[level_k].push_back(spec);
						}
					}
					{
						// If you have a variable assignment for this edge, use it
						auto it = graph->edge_phi_to_variable_map.find(edge_phi_id);
						if (it != graph->edge_phi_to_variable_map.end()) {
							const int var = it->second;
							AssignableHoldSpec spec;
							spec.var = var;
							spec.ee_frame_name = "ee_link";
							spec.held_point_ids.assign(op.cubes.begin(), op.cubes.end());

							assignable_hold_specs_during_each_layer[level_k].push_back(spec);
						}
					}
				}
			}
		});

	// Add inter-waypoint costs for all edges
	// You control the cost body via add_edge_cost
	// Iterate edges of the subgraph
	for (auto edge : subgraph.structure.edges()) {
		int u = edge.u;
		int sg_u = subgraph.subgraph_id(u);
		int v = edge.e->to;
		int sg_v = subgraph.subgraph_id(v);
		std::pair<int, int> e_pair = std::make_pair(u, v);
		for (int ag = 0; ag < num_agents; ++ag) {
			Expression dist = splines->at(ag).squared_distance(X.row(sg_u).segment(ag * robot_dim, robot_dim),
									   X.row(sg_v).segment(ag * robot_dim, robot_dim));
			prog.AddCost(dist);
		}

		for (int obj = 0; obj < num_objects; ++obj) {
			const int start = objs_start + obj * non_robot_dim;

			// Grab the segment from node u
			Eigen::VectorX<drake::symbolic::Expression> X_seg_u =
				X.row(sg_u).segment(start, non_robot_dim).transpose();

			// Grab the segment from node v (as Expressions)
			// Note: X.row(sg_v) is RowVector<Expression>, so transpose to column.
			Eigen::VectorX<drake::symbolic::Expression> X_seg_v =
				X.row(sg_v).segment(start, non_robot_dim).transpose();

			// Only enforce stationary if there is no phi at this layer
			// that interferes with this cube.
			if (possibly_manipulated_cubes_during_each_edge.contains(e_pair)) {

				if (!possibly_manipulated_cubes_during_each_edge.at(e_pair).contains(obj)) {
					prog.AddLinearEqualityConstraint(
						X_seg_u - X_seg_v, Eigen::VectorXd::Zero(non_robot_dim))
						.evaluator()->set_description(fmt::format("{}->{} stationary point {}", u, v, obj));
				}
			} else {
				// std::cout << "no information about possibly manipulated cubes for " << e_pair.first << "->" << e_pair.second << "? shouldn't happen" << std::endl;
			// 	prog.AddLinearEqualityConstraint(
			// 		X_seg_u - X_seg_v, Eigen::VectorXd::Zero(non_robot_dim))
			// 		.evaluator()->set_description(fmt::format("{}->{} stationary point {}", u, v, obj));;
			}

			const int layer = layers.node_to_level.at(u);
			if (enforce_rigidity) {
				if (assignable_hold_specs_during_each_layer.contains(layer)) {
					for (const AssignableHoldSpec& hold_spec : assignable_hold_specs_during_each_layer.at(layer)) {
						auto A_row = GetARowForVar(subgraph, Assignments, hold_spec.var);
						AddHoldRigidityAssignable(
							prog, subgraph, graph, X,
							u, v, hold_spec,
							/*robot_dim=*/robot_dim,
							/*objs_start=*/objs_start,
							/*non_robot_dim=*/non_robot_dim,
							A_row);
					}
				}

				if (hold_specs_during_each_layer.contains(layer)) {
					for (const HoldSpec& hold_spec : hold_specs_during_each_layer.at(layer)) {
						AddHoldRigidityStatic(
							prog, subgraph, graph, X,
							u, v, hold_spec,
							/*robot_dim=*/robot_dim,
							/*objs_start=*/objs_start,
							/*non_robot_dim=*/non_robot_dim,
							/*exact_rigidity=*/true);
					}
				}
			}
		}
	}

	// Add constraints/costs from registry
	for (const auto& [phi_id, op] : subgraph.get_subgraph_ops()) {
		op.builder(prog, subgraph, phi_id, X, Assignments);
	}

	// Add constraints/costs from edge registry
	for (const auto& [edge_phi_id, edge_op] : subgraph.get_subgraph_edge_ops()) {
		edge_op.waypoint_builder(prog, subgraph, edge_phi_id, X, Assignments,
					 previous_X.row(edge_op.u_node));
	}

	// Add variable assignment constraints from registry
	for (const auto& [var_phi_id, var_op] : graph->get_var_ops()) {
		var_op.builder(prog, subgraph, var_phi_id, X, Assignments);
	}

	problem.prog = std::move(prog_ptr);

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(GraphOfConstraints& graph,
				   std::vector<CubicConfigurationSpline> splines)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))) {
	// Allocate persistent output buffers.
	_waypoints = Eigen::MatrixXd::Zero(_graph->structure.num_nodes(), _graph->total_dim);
	_assignments = Eigen::VectorXi::Constant(_graph->num_phis, -1);
	_var_assignments = Eigen::VectorXi::Constant(_graph->num_variables, -1);
	_first_cycle = true;
}


using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::IpoptSolverDetails;

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

inline void PinAssignmentScenario(
	const drake::solvers::Binding<drake::solvers::BoundingBoxConstraint>& box,
	const MatrixXDecisionVariable& A,
	const std::vector<int>& agent_of_row) {
	const int Z = A.rows(), M = A.cols();
	Eigen::VectorXd lb(Z*M), ub(Z*M);
	for (int i = 0; i < Z; ++i) {
		for (int k = 0; k < M; ++k) {
			const bool on = (k == agent_of_row[i]);
			lb[i*M + k] = on ? 1.0 : 0.0;
			ub[i*M + k] = on ? 1.0 : 0.0;
		}
	}
	auto* bb = const_cast<drake::solvers::BoundingBoxConstraint*>(box.evaluator().get());
	bb->set_bounds(lb, ub);
}

inline void UnpinAssignments(
	const drake::solvers::Binding<drake::solvers::BoundingBoxConstraint>& box,
	int Z, int M) {
	Eigen::VectorXd lb = Eigen::VectorXd::Zero(Z*M);
	Eigen::VectorXd ub = Eigen::VectorXd::Ones(Z*M);
	auto* bb = const_cast<drake::solvers::BoundingBoxConstraint*>(box.evaluator().get());
	bb->set_bounds(lb, ub);
}

struct EnumSolveResult {
	bool success{false};
	double best_cost{std::numeric_limits<double>::infinity()};
	std::vector<int> best_agent_of_row;
	Eigen::MatrixXd best_X;            // optional: store best continuous solution
	Eigen::MatrixXd best_Assignments;  // will be one-hot for the winner
};

inline bool NextCombo(std::vector<int>& idx,
                      const std::vector<std::vector<int>>& choices) {
	// idx[j] is an index into choices[j]
	for (int i = (int)idx.size()-1; i >= 0; --i) {
		if (++idx[i] < (int)choices[i].size()) return true;
		idx[i] = 0;
	}
	return false; // wrapped around -> done
}

EnumSolveResult EnumerateAllAssignmentsAndSolve(
	GraphWaypointProblem* problem,                        // built once
	const std::vector<std::vector<int>>& choices_per_row, // size Z
	const Eigen::MatrixXd* warmstart_X = nullptr) {

	using drake::solvers::IpoptSolver;
	using drake::solvers::SnoptSolver;
	using drake::solvers::NloptSolver;
	IpoptSolver solver;
	if (!solver.available()) throw std::runtime_error("IPOPT not available.");

	const int Z = problem->Assignments.rows();
	const int M = problem->Assignments.cols();

	EnumSolveResult out;
	out.best_agent_of_row.resize(Z, -1);

	// mixed-radix indices
	std::vector<int> idx(Z, 0);

	// optional continuous warm start
	if (warmstart_X) problem->prog->SetInitialGuess(problem->X, *warmstart_X);

	// iterate all combinations
	bool first = true;
	do {
		// materialize this scenario's agent_of_row
		std::vector<int> agent_of_row(Z);
		for (int i = 0; i < Z; ++i) agent_of_row[i] = choices_per_row[i][idx[i]];

		// pin this scenario in-place
		PinAssignmentScenario(*(problem->A_bounds), problem->Assignments, agent_of_row);

		// give Assignments a matching initial guess (helps a lot)
		Eigen::MatrixXd A0 = Eigen::MatrixXd::Zero(Z, M);
		for (int i = 0; i < Z; ++i) A0(i, agent_of_row[i]) = 1.0;
		problem->prog->SetInitialGuess(problem->Assignments, A0);

		// solve
		drake::solvers::MathematicalProgramResult res;
		solver.Solve(*(problem->prog), {}, {}, &res);

		if (res.is_success()) {
			const double cost = res.get_optimal_cost();
			if (cost < out.best_cost) {
				out.success = true;
				out.best_cost = cost;
				out.best_agent_of_row = agent_of_row;
				out.best_X = res.GetSolution(problem->X);
				out.best_Assignments = res.GetSolution(problem->Assignments);
			}
			// chain warm-start from the latest good X
			problem->prog->SetInitialGuess(problem->X, res.GetSolution(problem->X));
		} else {
			std::cout << "NO SUCCESS FOR ASSIGNMENTS:\n" << A0 << std::endl;
			// PrintSolverReport(problem, res, 1e-6);
		}

		first = false;
	} while (NextCombo(idx, choices_per_row));

	// relax bounds back to [0,1] if you’ll reuse the program later
	UnpinAssignments(*(problem->A_bounds), Z, M);

	return out;
}

bool GraphWaypointMPC::solve(
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

	const bool enforce_rigidity = false;

	if (enforce_rigidity) {
		// return SolveWithBranchAndBoundPlusROPTLIB(remaining_vertices, x0);
		return SolveWithEnumerationAndIPOPT(remaining_vertices, x0);
		// return SolveWithGurobi(remaining_vertices, x0, true);
		// return SolveWithMosek(remaining_vertices, x0, true);
	} else {
		return SolveWithGurobi(remaining_vertices, x0, false);
		// return SolveWithMosek(remaining_vertices, x0, false);
	}
}

bool GraphWaypointMPC::SolveWithMosek(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0,
	bool enforce_rigidity_and_relax_binary_vars) {

	GraphWaypointProblem problem = BuildGraphWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		enforce_rigidity_and_relax_binary_vars,
		enforce_rigidity_and_relax_binary_vars);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);

	if (result.is_success()) {

		_last_solve_time = _timer.Tick();

		const int num_remaining_nodes = remaining_vertices.size();
		const int num_subgraph_assignables = problem.Assignments.rows();
		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;
		const int dim = _graph->dim;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				int variable_id = _graph->phi_to_variable_map.at(i);
				int subgraph_variable_id = problem.subgraph->subgraph_variable_id(variable_id);
				if (subgraph_variable_id != -1) {
					for (int j = 0; j < num_agents; ++j) {
						const double val = result.GetSolution(problem.Assignments(subgraph_variable_id, j));
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

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			_waypoints.row(v) = X_flat.row(i);
		}
		return true;
	}

	return false;
}

bool GraphWaypointMPC::SolveWithGurobi(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0,
	bool enforce_rigidity_and_relax_binary_vars) {

	GraphWaypointProblem problem = BuildGraphWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		enforce_rigidity_and_relax_binary_vars,
		enforce_rigidity_and_relax_binary_vars);

	// Solve
	drake::solvers::GurobiSolver solver;

	drake::solvers::SolverOptions options;
	options.SetOption(solver.id(), "NonConvex", 2);

	auto result = solver.Solve(*problem.prog, std::nullopt, options);

	if (result.is_success()) {

		_last_solve_time = _timer.Tick();

		const int num_remaining_nodes = remaining_vertices.size();
		const int num_subgraph_assignables = problem.Assignments.rows();
		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;
		const int dim = _graph->dim;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				int variable_id = _graph->phi_to_variable_map.at(i);
				int subgraph_variable_id = problem.subgraph->subgraph_variable_id(variable_id);
				if (subgraph_variable_id != -1) {
					for (int j = 0; j < num_agents; ++j) {
						const double val = result.GetSolution(problem.Assignments(subgraph_variable_id, j));
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

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		for (int v : remaining_vertices) {
			const int i = problem.subgraph->subgraph_id(v);
			// Eigen::RowVectorXd row(num_agents * dim + );
			// for (int j = 0; j < num_agents; ++j) {
			// 	row.segment(j * dim, dim) = X_flat.row(i).segment(j * dim * num_agents + j);
			// }
			_waypoints.row(v) = X_flat.row(i);
		}
		return true;
	}

	std::cerr << "No feasible solution across all assignments.\n";

	// if (result.is_success()) {
	// } else {
	// 	// PrintSolverReport(*(problem.prog), result, 1e-6);
	// 	return false;
	// }

	return false;
}

bool GraphWaypointMPC::SolveWithEnumerationAndIPOPT(
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	std::unique_ptr<GraphWaypointProblem> problem;
	try {
		problem = std::make_unique<GraphWaypointProblem>(
			BuildGraphWaypointProblem(
				_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
				/*enforce_rigidity=*/true,
				/*relax_binary_vars=*/true));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in waypoint problem construction: " << e.what() << std::endl;
		return false;
	}

	// Enumerate per-row choices (all agents, or pruned)
	std::vector<std::vector<int>> choices_per_row(problem->Assignments.rows());
	for (auto& v : choices_per_row) {
		v.resize(_graph->num_agents);
		std::iota(v.begin(), v.end(), 0);  // {0,1,...,M-1}
	}

	// Warm start assignments
	// for (int i = 0; i < num_phis; ++i) {
	// 	if (_graph->phi_to_variable_map.contains(i)) {
	// 		int variable_id = _graph->phi_to_variable_map.at(i);
	// 		int subgraph_variable_id = problem->subgraph->subgraph_variable_id(variable_id);
	// 		if (subgraph_variable_id != -1) {
	// 			int j = best.best_agent_of_row.at(subgraph_variable_id);
	// 			_assignments(i) = j;
	// 			_var_assignments(variable_id) = j;
	// 		} else {
	// 			_assignments(i) = -1;
	// 			_var_assignments(variable_id) = -1;
	// 		}
	// 	} else {
	// 		_assignments(i) = -1;
	// 	}
	// }

	for (int v : remaining_vertices) {
		const int i = problem->subgraph->subgraph_id(v);
		problem->prog->SetInitialGuess(problem->X.row(i), _waypoints.row(v));
	}

	// Solve

	EnumSolveResult best;
	try {
		best = EnumerateAllAssignmentsAndSolve(problem.get(), choices_per_row);
	} catch (const std::exception& e) {
		std::cout << "Caught exception in waypoint solver" << std::endl;
		return false;
	}

	if (best.success) {

		_last_solve_time = _timer.Tick();

		const int num_remaining_nodes = remaining_vertices.size();
		const int num_subgraph_assignables = problem->Assignments.rows();
		const int num_phis = _graph->num_phis;
		const int num_agents = _graph->num_agents;
		const int dim = _graph->dim;

		for (int i = 0; i < num_phis; ++i) {
			if (_graph->phi_to_variable_map.contains(i)) {
				int variable_id = _graph->phi_to_variable_map.at(i);
				int subgraph_variable_id = problem->subgraph->subgraph_variable_id(variable_id);
				if (subgraph_variable_id != -1) {
					int j = best.best_agent_of_row.at(subgraph_variable_id);
					_assignments(i) = j;
					_var_assignments(variable_id) = j;
				} else {
					// Don't reset assignments if they've
					// passed. Remember them. Only change
					// them if they come back into the
					// problem and are subsequently changed.

					// _assignments(i) = -1;
					// _var_assignments(variable_id) = -1;
				}
			} else {
				_assignments(i) = -1;
			}
		}

		Eigen::MatrixXd X_flat = best.best_X;
		for (int v : remaining_vertices) {
			const int i = problem->subgraph->subgraph_id(v);
			// Eigen::RowVectorXd row(num_agents * dim + );
			// for (int j = 0; j < num_agents; ++j) {
			// 	row.segment(j * dim, dim) = X_flat.row(i).segment(j * dim * num_agents + j);
			// }
			_waypoints.row(v) = X_flat.row(i);
		}
		return true;
	}

	std::cerr << "No feasible solution across all assignments.\n";

	// if (result.is_success()) {
	// } else {
	// 	PrintSolverReport(*(problem->prog), result, 1e-6);
	// 	return false;
	// }

	return false;
}
