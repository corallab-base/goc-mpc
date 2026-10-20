#include "waypoint_common.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <optional>

#include <fmt/format.h>

#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/nlopt_solver.h>

using drake::symbolic::Expression;
using drake::symbolic::Variable;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::VectorXDecisionVariable;

bool RoutingPresolve::precedes(int a, int b) const {
	if (a == depot || b == sink()) return a != b;
	if (a == sink() || b == depot) return false;
	return before[a - 1][b - 1];
}

bool RoutingPresolve::arc_allowed(int j, int a, int b) const {
	if (a != depot && a != sink() && b != depot && b != sink() && precedes(b, a))
		return false;
	for (int sg_c = 0; sg_c < n_V; ++sg_c) {
		const int c = ri(sg_c);
		if (c != a && c != b && required_at[sg_c].contains(j) && precedes(a, c) && precedes(c, b))
			return false;
	}
	return true;
}

RoutingPresolve BuildRoutingPresolve(const GraphOfConstraints& graph,
				     const SubgraphOfConstraints& subgraph) {
	RoutingPresolve p;
	const int n_V = p.n_V = subgraph.num_nodes();
	auto& roles_at = p.roles_at;
	auto& required_at = p.required_at;
	roles_at.resize(n_V);
	required_at.resize(n_V);

	auto add_refs = [&](int node, const drake::symbolic::Variables& free_vars,
			    const PlaceholderVarFamily<int>& var_family,
			    const PlaceholderVarFamily<int>& agent_family) {
		const int sg = subgraph.subgraph_id(node);
		if (sg < 0) return;
		for (int var : var_family.KeysReferencedBy(free_vars)) roles_at[sg].insert(var);
		for (int ag : agent_family.KeysReferencedBy(free_vars)) required_at[sg].insert(ag);
	};

	for (const auto& [phi_id, rec] : subgraph.get_subgraph_symbolic_ops()) {
		if (graph._phi_routing_exempt.count(phi_id)) continue;
		const auto free_vars = rec.formula.GetFreeVariables();
		add_refs(rec.node, free_vars, graph._var_agent_q, graph._agent_q);
		const int sg = subgraph.subgraph_id(rec.node);
		if (sg < 0) continue;
		for (const auto& key : graph._var_agent_link_pos.KeysReferencedBy(free_vars)) roles_at[sg].insert(key.first);
		for (const auto& key : graph._var_agent_link_rot.KeysReferencedBy(free_vars)) roles_at[sg].insert(key.first);
		for (const auto& key : graph._agent_link_pos.KeysReferencedBy(free_vars)) required_at[sg].insert(key.first);
		for (const auto& key : graph._agent_link_rot.KeysReferencedBy(free_vars)) required_at[sg].insert(key.first);
		if (graph._phi_to_static_assignment_map.contains(phi_id))
			required_at[sg].insert(graph._phi_to_static_assignment_map.at(phi_id));
	}
	for (const auto& [edge_phi_id, rec] : subgraph.get_subgraph_symbolic_edge_ops()) {
		const auto free_vars = rec.formula.GetFreeVariables();
		for (int node : {rec.u_node, rec.v_node})
			add_refs(node, free_vars, graph._var_agent_q, graph._agent_q);
		add_refs(rec.u_node, free_vars, graph._var_agent_q_u, graph._agent_q_u);
		add_refs(rec.v_node, free_vars, graph._var_agent_q_v, graph._agent_q_v);
	}
	for (const auto& [hold_id, hold] : subgraph.get_subgraph_hold_ops()) {
		for (int node : {hold.u_node, hold.v_node}) {
			const int sg = subgraph.subgraph_id(node);
			if (sg < 0) continue;
			if (hold.robot_ag.has_value()) required_at[sg].insert(*hold.robot_ag);
			else roles_at[sg].insert(*hold.var_id);
		}
	}

	auto& before = p.before;
	before.assign(n_V, std::vector<bool>(n_V, false));
	for (const auto& edge : subgraph.structure.edges()) {
		const int sg_u = subgraph.subgraph_id(edge.u);
		const int sg_v = subgraph.subgraph_id(edge.e->to);
		if (sg_u >= 0 && sg_v >= 0 && sg_u != sg_v) before[sg_u][sg_v] = true;
	}
	for (int k = 0; k < n_V; ++k)
		for (int u = 0; u < n_V; ++u)
			if (before[u][k])
				for (int v = 0; v < n_V; ++v)
					if (before[k][v]) before[u][v] = true;
	return p;
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

std::vector<Variable> AddConditionBinaryVars(MathematicalProgram& prog,
					     const GraphOfConstraints& graph) {
	std::vector<Variable> cond_binary_vars;
	for (int i = 0; i < static_cast<int>(graph._binary_cond_sym_vars.size()); ++i) {
		cond_binary_vars.push_back(prog.NewBinaryVariables(1, "bv_" + std::to_string(i))(0));
	}
	return cond_binary_vars;
}

void AddRoleExactlyOneAgent(MathematicalProgram& prog,
			    const SubgraphOfConstraints& subgraph,
			    const MatrixXDecisionVariable& Assignments,
			    int num_agents) {
	for (int var_k = 0; var_k < subgraph.num_variables(); ++var_k) {
		Expression sum_A;
		for (int k = 0; k < num_agents; ++k) sum_A += Expression(Assignments(var_k, k));
		prog.AddLinearEqualityConstraint(sum_A, 1.0);
	}
}

// Once a plan has pinned a variable to whichever agent it resolved to at
// some earlier commit node, force every remaining node/hold that still
// references it through that same agent. Pin the whole row (not just the
// committed cell) -- the row's other cells aren't otherwise guaranteed to be
// forced to 0 by something else (that normally falls out of a var_id node
// constraint's own "exactly one agent" emission in
// CompileSymbolicNodeConstraint, but a committed variable referenced only via
// a hold at this point in the plan would have no such emission left in the
// subgraph) -- and the solution-extraction scan (`for k: if val>0.5: ...
// break`) takes the first 1-valued cell it finds, so a stray free cell at a
// lower agent index would silently report the wrong agent even though the
// intended one is also pinned to 1.
void AddCommittedAssignments(MathematicalProgram& prog,
			     const GraphOfConstraints& graph,
			     const SubgraphOfConstraints& subgraph,
			     const MatrixXDecisionVariable& Assignments) {
	const int num_agents = graph.num_agents;
	for (const auto& [var, committed_agent] : graph.committed_assignments) {
		const int variable_k = subgraph.subgraph_variable_id(var);
		if (variable_k < 0) continue;
		for (int k = 0; k < num_agents; ++k) {
			prog.AddLinearEqualityConstraint(
				Expression(Assignments(variable_k, k)), k == committed_agent ? 1.0 : 0.0);
		}
	}
}

void AddOrderingConstraints(MathematicalProgram& prog,
			    const GraphOfConstraints& graph,
			    const SubgraphOfConstraints& subgraph,
			    const VectorXDecisionVariable& t_var,
			    double M_time,
			    const MatrixXDecisionVariable& Assignments,
			    const Eigen::VectorXi& previous_var_assignments,
			    const std::vector<Variable>& cond_binary_vars) {
	const double kInf = std::numeric_limits<double>::infinity();
	const auto ri = RoutingPresolve::ri;
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
				previous_var_assignments, &graph, cond_binary_vars);
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
	for (const auto& [uv, formula] : graph._conditional_ordering_map) {
		add_ordering_constraint(uv.first, uv.second, &formula);
	}
}

void AddAssignmentVariableConstraints(MathematicalProgram& prog,
				      const GraphOfConstraints& graph,
				      const SubgraphOfConstraints& subgraph,
				      const MatrixXDecisionVariable& Assignments) {
	const int num_agents = graph.num_agents;
	for (const auto& [var, allowed] : graph.var_candidates) {
		const int variable_k = subgraph.subgraph_variable_id(var);
		if (variable_k < 0) continue;
		for (int k = 0; k < num_agents; ++k)
			if (!allowed.contains(k))
				prog.AddLinearEqualityConstraint(Expression(Assignments(variable_k, k)), 0.0);
	}
	for (const auto& [var1, var2] : graph.var_ineq_pairs) {
		const int variable1_k = subgraph.subgraph_variable_id(var1);
		const int variable2_k = subgraph.subgraph_variable_id(var2);
		if (variable1_k < 0 || variable2_k < 0) continue;
		for (int k = 0; k < num_agents; ++k)
			prog.AddLinearEqualityConstraint(
				Expression(Assignments(variable1_k, k)) + Expression(Assignments(variable2_k, k)), 1.0);
	}
}

// Unlike the hold-rigidity rows, this only ever compares plain object
// position segments with no rotation applied, so it's linear (or, when gated
// by an overlap indicator, mixed-integer-linear) regardless of robot kind.
// Each structure edge (and each source node, against x0) keeps every object
// still, gated off by an interval-overlap indicator against every hold that
// might move it.
void AddStationaryObjectConstraints(MathematicalProgram& prog,
				    const GraphOfConstraints& graph,
				    const SubgraphOfConstraints& subgraph,
				    const drake::MatrixX<Expression>& state,
				    const Eigen::VectorXd& x0,
				    const VectorXDecisionVariable& t_var,
				    double M_time) {
	const double kInf = std::numeric_limits<double>::infinity();
	const int num_objects = graph.num_objects;
	const auto ri = RoutingPresolve::ri;
	const int depot_idx = RoutingPresolve::depot;
	std::map<std::tuple<int, int, int, int>, Variable> overlap_cache;

	auto stationary_edge = [&](const Eigen::VectorX<Expression>& seg_a_full, int sg_b,
				    int t_idx_a, int t_idx_b) {
		for (int obj = 0; obj < num_objects; ++obj) {
			const int start = graph.object_col_offset(obj);
			const int width = graph.object_ambient_dim(obj);
			Eigen::VectorX<Expression> seg_a = seg_a_full.segment(start, width);
			Eigen::VectorX<Expression> seg_b = state.row(sg_b).segment(start, width).transpose();

			// Every hold (anywhere in the graph) that might move this object.
			std::vector<int> relevant_holds;
			for (const auto& [other_hold_id, other_hold] : graph.hold_ops) {
				if (std::find(other_hold.held_point_ids.begin(), other_hold.held_point_ids.end(), obj) !=
				    other_hold.held_point_ids.end()) {
					relevant_holds.push_back(other_hold_id);
				}
			}

			Expression gate_sum;
			bool any_gate = false;
			for (int other_hold_id : relevant_holds) {
				const auto& other_hold = graph.hold_ops.at(other_hold_id);
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
				prog.AddLinearEqualityConstraint(seg_a - seg_b, Eigen::VectorXd::Zero(width))
					.evaluator()->set_description(fmt::format("stationary point {}", obj));
				continue;
			}

			const double m_stat = (graph._global_x_ub.segment(start, width) -
						graph._global_x_lb.segment(start, width))
						.cwiseAbs().maxCoeff();
			for (int i = 0; i < width; ++i) {
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
		stationary_edge(state.row(sg_u14b).transpose(), sg_v14b,
				 ri(sg_u14b), ri(sg_v14b));
	}
	for (int v14b : subgraph.structure.sources()) {
		const int sg_v14b = subgraph.subgraph_id(v14b);
		stationary_edge(x0.template cast<Expression>(), sg_v14b, depot_idx, ri(sg_v14b));
	}
}

void PrintSolverReport(const MathematicalProgram& prog,
		       const MathematicalProgramResult& result,
		       double tol) {
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
		x = result.GetSolution(prog.decision_variables());
		have_x = true;
	} catch (const std::exception&) {
		// Fall back to initial guess (may be zero) if solver did not return x.
		if (prog.decision_variables().size() > 0) {
			x = prog.initial_guess();
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
			print_binding_violation(prog.bounding_box_constraints(),   "BoundingBox");
			print_binding_violation(prog.linear_equality_constraints(),"LinEq");
			print_binding_violation(prog.linear_constraints(),         "LinIneq");
			print_binding_violation(prog.lorentz_cone_constraints(),   "Lorentz");
			print_binding_violation(prog.rotated_lorentz_cone_constraints(),"RotLorentz");
			print_binding_violation(prog.quadratic_constraints(),      "Quadratic");
			print_binding_violation(prog.exponential_cone_constraints(),"ExpCone");
			print_binding_violation(prog.generic_constraints(),        "Generic");
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

			for (const std::string& s : result.GetInfeasibleConstraintNames(prog, tol)) {
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
