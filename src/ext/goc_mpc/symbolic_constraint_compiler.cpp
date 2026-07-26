#include "symbolic_constraint_compiler.hpp"

#include <limits>

using drake::symbolic::Expression;

namespace {

double ComputeViolation(const drake::symbolic::Formula& f,
                        const drake::symbolic::Environment& env) {
	using drake::symbolic::FormulaKind;
	switch (f.get_kind()) {
	case FormulaKind::Eq:
		return std::abs(get_lhs_expression(f).Evaluate(env) -
		                get_rhs_expression(f).Evaluate(env));
	case FormulaKind::Leq:
		return std::max(0.0, get_lhs_expression(f).Evaluate(env) -
		                     get_rhs_expression(f).Evaluate(env));
	case FormulaKind::Geq:
		return std::max(0.0, get_rhs_expression(f).Evaluate(env) -
		                     get_lhs_expression(f).Evaluate(env));
	case FormulaKind::Lt:
		return std::max(0.0, get_lhs_expression(f).Evaluate(env) -
		                     get_rhs_expression(f).Evaluate(env));
	case FormulaKind::Gt:
		return std::max(0.0, get_rhs_expression(f).Evaluate(env) -
		                     get_lhs_expression(f).Evaluate(env));
	case FormulaKind::And: {
		double v = 0.0;
		for (const auto& sub : get_operands(f))
			v = std::max(v, ComputeViolation(sub, env));
		return v;
	}
	default:
		return f.Evaluate(env) ? 0.0 : 1.0;
	}
}

// Signed counterpart to ComputeViolation, for callers that compare against a
// hard tol=0.00 (GraphOfConstraintsMPC._backtrack's edge-phi check) rather
// than a small positive fuzz tolerance (phi_tolerance's node-completion
// check). ComputeViolation clips inequality residuals to >= 0, so a
// comfortably-satisfied Leq/Geq atom reads identically (0.0) to one sitting
// exactly on the boundary -- fine against tol > 0, but indistinguishable
// from "violated" against tol == 0. This version leaves the sign in place so
// "satisfied with margin" is strictly negative and passes a hard `< 0.0`
// check. Only meaningful for inequality-only formulas (see
// get_next_edge_ops's IsInequalityFormula gate) -- the Eq/default cases
// below exist only as a safe fallback, not as an intended calling
// convention, since an exact-equality residual can never read as negative.
double ComputeSignedViolation(const drake::symbolic::Formula& f,
                               const drake::symbolic::Environment& env) {
	using drake::symbolic::FormulaKind;
	switch (f.get_kind()) {
	case FormulaKind::Eq:
		return std::abs(get_lhs_expression(f).Evaluate(env) -
		                get_rhs_expression(f).Evaluate(env));
	case FormulaKind::Leq:
	case FormulaKind::Lt:
		return get_lhs_expression(f).Evaluate(env) -
		       get_rhs_expression(f).Evaluate(env);
	case FormulaKind::Geq:
	case FormulaKind::Gt:
		return get_rhs_expression(f).Evaluate(env) -
		       get_lhs_expression(f).Evaluate(env);
	case FormulaKind::And: {
		double v = -std::numeric_limits<double>::infinity();
		for (const auto& sub : get_operands(f))
			v = std::max(v, ComputeSignedViolation(sub, env));
		return v;
	}
	default:
		return f.Evaluate(env) ? -1.0 : 1.0;
	}
}

// Conservative big-M: maximum absolute value any component of x could take.
double ConservativeStateBigM(const GraphOfConstraints& graph) {
	return graph._global_x_ub.array().abs().maxCoeff() +
	       (-graph._global_x_lb).array().abs().maxCoeff() + 1.0;
}

// Recursively adds big-M gated linear equalities/inequalities for formula ff,
// enforced only when every expression in `gates` evaluates to 1 (each gate
// contributes its own M*(1-gate) slack term, so this stays linear -- no
// bilinear "AND of two binaries" term is needed). An empty `gates` list makes
// every atom unconditional (slack=0). Shared by CompileSymbolicNodeConstraint's
// assignable-var branch and CompileAlongEdgeConstraintAtNode below.
void AddBigMGatedFormula(drake::solvers::MathematicalProgram& prog,
                         const drake::symbolic::Formula& ff,
                         const std::vector<Expression>& gates,
                         double M) {
	using drake::symbolic::FormulaKind;
	using drake::symbolic::is_conjunction;
	using drake::symbolic::is_relational;
	using drake::symbolic::get_operands;
	using drake::symbolic::get_lhs_expression;
	using drake::symbolic::get_rhs_expression;
	const double neg_inf = -std::numeric_limits<double>::infinity();

	if (is_conjunction(ff)) {
		for (const auto& sub_f : get_operands(ff)) AddBigMGatedFormula(prog, sub_f, gates, M);
		return;
	}
	if (!is_relational(ff)) return;

	Expression slack = Expression::Zero();
	for (const auto& g : gates) slack += M * (1.0 - g);

	Expression e = get_lhs_expression(ff) - get_rhs_expression(ff);
	if (ff.get_kind() == FormulaKind::Eq) {
		prog.AddLinearConstraint(e - slack, neg_inf, 0.0);
		prog.AddLinearConstraint(-e - slack, neg_inf, 0.0);
	} else if (ff.get_kind() == FormulaKind::Geq) {
		prog.AddLinearConstraint(-e - slack, neg_inf, 0.0);
	} else if (ff.get_kind() == FormulaKind::Leq) {
		prog.AddLinearConstraint(e - slack, neg_inf, 0.0);
	}
}

}  // namespace

void CompileSymbolicNodeConstraint(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicNodeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments) {

	const int n_agents = graph.num_agents;
	const int d = graph.dim;
	const int n_objects = graph.num_objects;
	const int obj_dim = graph.non_robot_dim;

	if (!rec.multi_var_ids.empty()) {
		// Multi-variable disjunction: enumerate all n_agents^k combos and emit Or.
		const int sg_node = subgraph.subgraph_id(rec.node);
		const int k = static_cast<int>(rec.multi_var_ids.size());
		std::vector<int> combo(k, 0);

		drake::symbolic::Formula f_or;
		bool first = true;

		auto advance = [&]() -> bool {
			for (int p = k - 1; p >= 0; --p) {
				if (++combo[p] < n_agents) return true;
				combo[p] = 0;
			}
			return false;
		};

		do {
			drake::symbolic::Substitution sub;
			for (int p = 0; p < k; ++p) {
				const auto& pvec = graph._var_agent_q_vars.at(rec.multi_var_ids[p]);
				for (int j = 0; j < d; ++j)
					sub[pvec[j]] = Expression(W(sg_node, combo[p] * d + j));
			}
			for (int ag = 0; ag < n_agents; ++ag)
				for (int j = 0; j < d; ++j)
					sub[graph._agent_q_vars[ag][j]] = Expression(W(sg_node, ag * d + j));
			for (int o = 0; o < n_objects; ++o)
				for (int j = 0; j < obj_dim; ++j)
					sub[graph._object_q_vars[o][j]] = Expression(
						W(sg_node, n_agents * d + o * obj_dim + j));
			drake::symbolic::Formula f_sub = rec.formula.Substitute(sub);
			if (first) { f_or = f_sub; first = false; }
			else        f_or = f_or || f_sub;
		} while (advance());

		prog.AddConstraint(f_or);
		return;
	}

	if (rec.var_id.has_value()) {
		const int var = rec.var_id.value();
		const int sg_node = subgraph.subgraph_id(rec.node);
		const int variable_k = subgraph.subgraph_variable_id(var);
		if (variable_k < 0) return;

		const auto& var_agent_vars = graph._var_agent_q_vars.at(var);
		const double M_sym = ConservativeStateBigM(graph);

		for (int i = 0; i < n_agents; ++i) {
			drake::symbolic::Substitution sub;
			for (int j = 0; j < d; ++j)
				sub[var_agent_vars[j]] = Expression(W(sg_node, i * d + j));
			for (int kk = 0; kk < n_agents; ++kk)
				for (int j = 0; j < d; ++j)
					sub[graph._agent_q_vars[kk][j]] = Expression(W(sg_node, kk * d + j));
			for (int o = 0; o < n_objects; ++o)
				for (int j = 0; j < obj_dim; ++j)
					sub[graph._object_q_vars[o][j]] = Expression(
						W(sg_node, n_agents * d + o * obj_dim + j));
			drake::symbolic::Formula f_sub = rec.formula.Substitute(sub);
			Expression s = Assignments(variable_k, i);
			AddBigMGatedFormula(prog, f_sub, {s}, M_sym);
		}

		// Exactly one agent is assigned to this variable.
		{
			Expression sum_A = Expression::Zero();
			for (int ia = 0; ia < n_agents; ++ia) sum_A += Assignments(variable_k, ia);
			prog.AddLinearEqualityConstraint(sum_A, 1.0);
		}
		return;
	}

	// Plain node constraint (no assignable var placeholders referenced).
	const int sg_node = subgraph.subgraph_id(rec.node);
	drake::symbolic::Substitution sub;
	for (int i = 0; i < n_agents; ++i)
		for (int j = 0; j < d; ++j)
			sub[graph._agent_q_vars[i][j]] = Expression(W(sg_node, i * d + j));
	for (int o = 0; o < n_objects; ++o)
		for (int j = 0; j < obj_dim; ++j)
			sub[graph._object_q_vars[o][j]] = Expression(W(sg_node, n_agents * d + o * obj_dim + j));
	prog.AddConstraint(rec.formula.Substitute(sub));
}

void CompileAlongEdgeConstraintAtNode(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	int sg_node,
	const Expression* extra_gate) {

	const int n_agents = graph.num_agents;
	const int d = graph.dim;
	const int n_objects = graph.num_objects;
	const int obj_dim = graph.non_robot_dim;

	if (rec.var_id.has_value()) {
		const int var = rec.var_id.value();
		const int variable_k = subgraph.subgraph_variable_id(var);
		if (variable_k < 0) return;

		const auto& var_agent_vars = graph._var_agent_q_vars.at(var);
		const double M_sym = ConservativeStateBigM(graph);

		for (int i = 0; i < n_agents; ++i) {
			drake::symbolic::Substitution sub;
			for (int j = 0; j < d; ++j)
				sub[var_agent_vars[j]] = Expression(W(sg_node, i * d + j));
			for (int kk = 0; kk < n_agents; ++kk)
				for (int j = 0; j < d; ++j)
					sub[graph._agent_q_vars[kk][j]] = Expression(W(sg_node, kk * d + j));
			for (int o = 0; o < n_objects; ++o)
				for (int j = 0; j < obj_dim; ++j)
					sub[graph._object_q_vars[o][j]] = Expression(
						W(sg_node, n_agents * d + o * obj_dim + j));
			drake::symbolic::Formula f_sub = rec.formula.Substitute(sub);
			std::vector<Expression> gates = {Assignments(variable_k, i)};
			if (extra_gate) gates.push_back(*extra_gate);
			AddBigMGatedFormula(prog, f_sub, gates, M_sym);
		}
		return;
	}

	// Literal-agent / object-only formula.
	drake::symbolic::Substitution sub;
	for (int i = 0; i < n_agents; ++i)
		for (int j = 0; j < d; ++j)
			sub[graph._agent_q_vars[i][j]] = Expression(W(sg_node, i * d + j));
	for (int o = 0; o < n_objects; ++o)
		for (int j = 0; j < obj_dim; ++j)
			sub[graph._object_q_vars[o][j]] = Expression(W(sg_node, n_agents * d + o * obj_dim + j));
	drake::symbolic::Formula f_sub = rec.formula.Substitute(sub);

	if (!extra_gate) {
		prog.AddConstraint(f_sub);
	} else {
		AddBigMGatedFormula(prog, f_sub, {*extra_gate}, ConservativeStateBigM(graph));
	}
}

void CompileSymbolicEdgeConstraint(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const drake::solvers::MatrixXDecisionVariable& W,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const Eigen::MatrixXd& previous_X) {

	if (rec.along_edge) {
		// Endpoints only -- re-application at interior subgraph nodes the
		// edge might span (betweenness-gated) is driven by the MILP build
		// loop (BuildGraphWaypointProblem), which calls
		// CompileAlongEdgeConstraintAtNode directly with a betweenness gate.
		const int sg_u_node = subgraph.subgraph_id(rec.u_node);
		const int sg_v_node = subgraph.subgraph_id(rec.v_node);
		if (sg_u_node >= 0) CompileAlongEdgeConstraintAtNode(prog, subgraph, graph, rec, W, Assignments, sg_u_node, nullptr);
		if (sg_v_node >= 0) CompileAlongEdgeConstraintAtNode(prog, subgraph, graph, rec, W, Assignments, sg_v_node, nullptr);
		return;
	}

	const int n_agents = graph.num_agents;
	const int d = graph.dim;
	const int n_objects = graph.num_objects;
	const int obj_dim = graph.non_robot_dim;

	const int sg_u_node = subgraph.subgraph_id(rec.u_node);
	const int sg_v_node = subgraph.subgraph_id(rec.v_node);

	// A relational formula's u/v side is normally a live decision-variable
	// row (W(sg_node, ...)) -- but this edge is compiled as soon as EITHER
	// endpoint remains (see SubgraphOfConstraints' constructor), so the
	// OTHER endpoint may have already passed out of remaining_vertices, in
	// which case subgraph_id returns -1 and there's no W row for it.
	// Substitute previous_X's row instead (a numeric constant) -- see this
	// function's header comment for why that's the real, live-rebound
	// state rather than a stale plan.
	auto row_expr = [&](int sg_node, int node, int col) -> Expression {
		return sg_node >= 0 ? Expression(W(sg_node, col)) : Expression(previous_X(node, col));
	};

	drake::symbolic::Substitution sub;
	for (int i = 0; i < n_agents; ++i) {
		for (int j = 0; j < d; ++j) {
			sub[graph._agent_q_vars_u[i][j]] = row_expr(sg_u_node, rec.u_node, i * d + j);
			sub[graph._agent_q_vars_v[i][j]] = row_expr(sg_v_node, rec.v_node, i * d + j);
		}
	}
	for (int o = 0; o < n_objects; ++o) {
		for (int j = 0; j < obj_dim; ++j) {
			sub[graph._object_q_vars_u[o][j]] = row_expr(sg_u_node, rec.u_node, n_agents * d + o * obj_dim + j);
			sub[graph._object_q_vars_v[o][j]] = row_expr(sg_v_node, rec.v_node, n_agents * d + o * obj_dim + j);
		}
	}
	prog.AddConstraint(rec.formula.Substitute(sub));
}

double EvaluateSymbolicNodeConstraint(
	const GraphOfConstraints& graph,
	const SymbolicNodeConstraint& rec,
	const Eigen::VectorXd& x,
	int assigned_agent) {

	if (!rec.multi_var_ids.empty()) {
		// Matches the pre-refactor eval_fn_multi stub (always 0.0).
		return 0.0;
	}

	const int n_agents = graph.num_agents;
	const int d = graph.dim;
	const int n_objects = graph.num_objects;
	const int obj_dim = graph.non_robot_dim;

	drake::symbolic::Environment env;
	if (rec.var_id.has_value()) {
		const auto& var_agent_vars = graph._var_agent_q_vars.at(rec.var_id.value());
		for (int j = 0; j < d; ++j)
			env.insert(var_agent_vars[j], x[assigned_agent * d + j]);
	}
	for (int i = 0; i < n_agents; ++i)
		for (int j = 0; j < d; ++j)
			env.insert(graph._agent_q_vars[i][j], x[i * d + j]);
	for (int o = 0; o < n_objects; ++o)
		for (int j = 0; j < obj_dim; ++j)
			env.insert(graph._object_q_vars[o][j], x[n_agents * d + o * obj_dim + j]);
	return ComputeViolation(rec.formula, env);
}

double EvaluateSymbolicEdgeConstraint(
	const GraphOfConstraints& graph,
	const SymbolicEdgeConstraint& rec,
	const Eigen::VectorXd& x,
	const Eigen::VectorXi& var_assignments) {

	const int n_agents = graph.num_agents;
	const int d = graph.dim;
	const int n_objects = graph.num_objects;
	const int obj_dim = graph.non_robot_dim;

	drake::symbolic::Environment env;

	if (rec.along_edge) {
		// "Along the edge" formulas only ever reference the plain
		// agent_q/object_q/var_agent_q placeholder set (see
		// add_edge_constraint) -- a single joint configuration snapshot
		// substitutes them all, same as a node constraint's evaluator.
		if (rec.var_id.has_value()) {
			const auto& var_agent_vars = graph._var_agent_q_vars.at(rec.var_id.value());
			const int assigned_agent = var_assignments(rec.var_id.value());
			for (int j = 0; j < d; ++j)
				env.insert(var_agent_vars[j], x[assigned_agent * d + j]);
		}
		for (int i = 0; i < n_agents; ++i)
			for (int j = 0; j < d; ++j)
				env.insert(graph._agent_q_vars[i][j], x[i * d + j]);
		for (int o = 0; o < n_objects; ++o)
			for (int j = 0; j < obj_dim; ++j)
				env.insert(graph._object_q_vars[o][j], x[n_agents * d + o * obj_dim + j]);
		return ComputeSignedViolation(rec.formula, env);
	}

	// Relational formula coupling agent_q_u/object_q_u (u side) and
	// agent_q_v/object_q_v (v side). Runtime backtracking only ever has one
	// observed state (the current x), not the two separate solved waypoints
	// add_edge_constraint's formula was written against for u_node/v_node --
	// so both sides substitute from that same x. That's moot in practice for
	// a purely relational formula (e.g. the rigid-transport equality), which
	// collapses to a tautology here, since get_next_edge_ops only ever
	// routes inequality-only formulas through evaluate_edge_phi's tol=0.00
	// backtracking path (see IsInequalityFormula there); an equality edge
	// phi's invariant isn't observable from a single snapshot anyway.
	for (int i = 0; i < n_agents; ++i) {
		for (int j = 0; j < d; ++j) {
			env.insert(graph._agent_q_vars_u[i][j], x[i * d + j]);
			env.insert(graph._agent_q_vars_v[i][j], x[i * d + j]);
		}
	}
	for (int o = 0; o < n_objects; ++o) {
		for (int j = 0; j < obj_dim; ++j) {
			env.insert(graph._object_q_vars_u[o][j], x[n_agents * d + o * obj_dim + j]);
			env.insert(graph._object_q_vars_v[o][j], x[n_agents * d + o * obj_dim + j]);
		}
	}
	return ComputeSignedViolation(rec.formula, env);
}
