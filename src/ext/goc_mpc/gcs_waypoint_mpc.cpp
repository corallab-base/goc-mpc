#include "gcs_waypoint_mpc.hpp"
#include "symbolic_constraint_compiler.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fmt/format.h>

#include <drake/common/symbolic/decompose.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/mosek_solver.h>

using drake::symbolic::Expression;
using drake::symbolic::Formula;
using drake::symbolic::FormulaKind;
using drake::symbolic::Substitution;
using drake::symbolic::Variable;
using drake::symbolic::Variables;
using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::VectorXDecisionVariable;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Affine rows A x + b <= 0 (in) and A x + b == 0 (eq) over some vector x.
struct AffineRows {
	Eigen::MatrixXd A_in, A_eq;
	Eigen::VectorXd b_in, b_eq;

	explicit AffineRows(int dim) : A_in(0, dim), A_eq(0, dim), b_in(0), b_eq(0) {}

	void Append(const AffineRows& o) {
		auto stack = [](Eigen::MatrixXd& A, Eigen::VectorXd& b,
				 const Eigen::MatrixXd& A2, const Eigen::VectorXd& b2) {
			Eigen::MatrixXd A_new(A.rows() + A2.rows(), A.cols());
			A_new << A, A2;
			Eigen::VectorXd b_new(b.size() + b2.size());
			b_new << b, b2;
			A = std::move(A_new);
			b = std::move(b_new);
		};
		stack(A_in, b_in, o.A_in, o.b_in);
		stack(A_eq, b_eq, o.A_eq, o.b_eq);
	}
};

// The atoms of a conjunction of linear relations, as expressions e with
// e <= 0 (in) or e == 0 (eq).
void CollectAtoms(const Formula& f, std::vector<Expression>* in, std::vector<Expression>* eq,
		  const std::string& where) {
	using drake::symbolic::get_lhs_expression;
	using drake::symbolic::get_rhs_expression;
	switch (f.get_kind()) {
	case FormulaKind::True:
		return;
	case FormulaKind::And:
		for (const auto& sub : drake::symbolic::get_operands(f)) CollectAtoms(sub, in, eq, where);
		return;
	case FormulaKind::Eq:
		eq->push_back(get_lhs_expression(f) - get_rhs_expression(f));
		return;
	case FormulaKind::Leq:
	case FormulaKind::Lt:
		in->push_back(get_lhs_expression(f) - get_rhs_expression(f));
		return;
	case FormulaKind::Geq:
	case FormulaKind::Gt:
		in->push_back(get_rhs_expression(f) - get_lhs_expression(f));
		return;
	default:
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC: {}: only conjunctions of linear (in)equalities are supported, got {}",
			where, f.to_string()));
	}
}

// Rows of f, a conjunction of affine relations over `vars` only.
AffineRows RowsOver(const Formula& f, const drake::VectorX<Variable>& vars, const std::string& where) {
	const Variables allowed(vars);
	for (const auto& v : f.GetFreeVariables()) {
		if (!allowed.include(v))
			throw std::runtime_error(fmt::format(
				"GCSWaypointMPC: {}: unsupported placeholder {} in {}", where, v.get_name(), f.to_string()));
	}
	std::vector<Expression> in, eq;
	CollectAtoms(f, &in, &eq, where);
	AffineRows rows(vars.size());
	auto decompose = [&](const std::vector<Expression>& es, Eigen::MatrixXd* A, Eigen::VectorXd* b) {
		if (es.empty()) return;
		drake::VectorX<Expression> e(es.size());
		for (size_t i = 0; i < es.size(); ++i) e(i) = es[i];
		A->resize(es.size(), vars.size());
		b->resize(es.size());
		try {
			drake::symbolic::DecomposeAffineExpressions(e, vars, A, b);
		} catch (const std::exception& ex) {
			throw std::runtime_error(fmt::format(
				"GCSWaypointMPC: {}: only linear constraints are supported ({})", where, ex.what()));
		}
	};
	decompose(in, &rows.A_in, &rows.b_in);
	decompose(eq, &rows.A_eq, &rows.b_eq);
	return rows;
}

// (x, lam) in the perspective of {x : rows}: A x + b lam <= 0 and
// A x + b lam == 0, each relaxed to |.| <= slack when slack is given.
void AddPerspectiveRows(MathematicalProgram& prog, const AffineRows& rows,
			const drake::VectorX<Expression>& x, const Expression& lam,
			const Expression* slack = nullptr) {
	if (rows.A_in.rows() > 0) {
		drake::VectorX<Expression> e = rows.A_in.cast<Expression>() * x + rows.b_in.cast<Expression>() * lam;
		if (slack) e -= drake::VectorX<Expression>::Constant(e.size(), *slack);
		prog.AddLinearConstraint(e, Eigen::VectorXd::Constant(e.size(), -kInf), Eigen::VectorXd::Zero(e.size()));
	}
	if (rows.A_eq.rows() > 0) {
		const drake::VectorX<Expression> e = rows.A_eq.cast<Expression>() * x + rows.b_eq.cast<Expression>() * lam;
		if (!slack) {
			prog.AddLinearEqualityConstraint(e, Eigen::VectorXd::Zero(e.size()));
		} else {
			const drake::VectorX<Expression> s = drake::VectorX<Expression>::Constant(e.size(), *slack);
			prog.AddLinearConstraint(e - s, Eigen::VectorXd::Constant(e.size(), -kInf), Eigen::VectorXd::Zero(e.size()));
			prog.AddLinearConstraint(-e - s, Eigen::VectorXd::Constant(e.size(), -kInf), Eigen::VectorXd::Zero(e.size()));
		}
	}
}

// The atoms of f (already substituted down to decision variables), enforced
// only when every gate is 1: each gate contributes M * (1 - gate) of slack.
void AddGatedAtoms(MathematicalProgram& prog, const Formula& f, const std::vector<Expression>& gates,
		   double M, const std::string& where) {
	std::vector<Expression> in, eq;
	CollectAtoms(f, &in, &eq, where);
	Expression slack = Expression::Zero();
	for (const auto& g : gates) slack += M * (1.0 - g);
	try {
		for (const auto& e : in) prog.AddLinearConstraint(e - slack, -kInf, 0.0);
		for (const auto& e : eq) {
			if (gates.empty()) {
				prog.AddLinearEqualityConstraint(e, 0.0);
			} else {
				prog.AddLinearConstraint(e - slack, -kInf, 0.0);
				prog.AddLinearConstraint(-e - slack, -kInf, 0.0);
			}
		}
	} catch (const std::exception& ex) {
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC: {}: only linear constraints are supported ({})", where, ex.what()));
	}
}

}  // namespace

GCSWaypointProblem BuildGCSWaypointProblem(
	GraphOfConstraints* graph,
	std::shared_ptr<std::vector<CubicConfigurationSpline>> splines,
	const std::vector<int>& remaining_vertices,
	const Eigen::VectorXd& x0,
	const Eigen::MatrixXd& previous_X,
	const Eigen::VectorXi& previous_var_assignments,
	bool enforce_rigidity,
	WaypointObjective objective,
	SubtourElimination subtour) {

	const int num_agents = graph->num_agents;
	const int num_objects = graph->num_objects;

	const SubgraphOfConstraints subgraph(graph, remaining_vertices);

	GCSWaypointProblem problem;
	std::unique_ptr<MathematicalProgram> prog_ptr = std::make_unique<MathematicalProgram>();
	MathematicalProgram& prog = *prog_ptr;
	problem.subgraph = std::make_unique<SubgraphOfConstraints>(subgraph);

	// Routing graph N = {depot=0} ∪ task_nodes(1..n_V) ∪ {sink=n_V+1}.
	const int n_V = subgraph.num_nodes();
	const int n_N = n_V + 2;
	const int depot_idx = RoutingPresolve::depot;
	const int sink_idx = n_N - 1;
	const auto ri = RoutingPresolve::ri;

	// Same bounds as BuildGraphWaypointProblem: M_cost upper-bounds any arc's
	// cost, M_time any routing time -- here the horizon of the time copies.
	const double x_range = (graph->_global_x_ub - graph->_global_x_lb).maxCoeff();
	int max_agent_width = 0;
	for (int j = 0; j < num_agents; ++j) max_agent_width = std::max(max_agent_width, graph->robot_ambient_dim(j));
	const double M_cost = max_agent_width * x_range;
	const double M_time = (n_N - 1) * M_cost + 1.0;
	const double M_sym = ConservativeStateBigM(*graph);

	///////////////////////////////////////////////////////////////////////
	//                          Declare Variables                        //
	///////////////////////////////////////////////////////////////////////

	MatrixXDecisionVariable Assignments;
	if (subgraph.num_variables() > 0)
		Assignments = prog.NewBinaryVariables(subgraph.num_variables(), num_agents, "Assignments");
	problem.Assignments = Assignments;

	VectorXDecisionVariable t_var = prog.NewContinuousVariables(n_N, "t");
	problem.t = t_var;
	VectorXDecisionVariable L_ms_var = prog.NewContinuousVariables(1, "L");
	problem.L = L_ms_var;

	const std::vector<Variable> cond_binary_vars = AddConditionBinaryVars(prog, *graph);
	const RoutingPresolve presolve = BuildRoutingPresolve(*graph, subgraph);

	// Global variable id of each subgraph variable row.
	std::vector<int> row_var(subgraph.num_variables(), -1);
	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) row_var[row] = var;

	auto is_candidate = [&](int var, int j) {
		const auto it = graph->var_candidates.find(var);
		return it == graph->var_candidates.end() || it->second.contains(j);
	};

	// Mode graph vertices: per agent, one per (node, role it may hold there),
	// plus a NO_ROLE vertex on each required visit.
	problem.graphs.resize(num_agents);
	std::vector<std::vector<std::vector<int>>> vertices_at(
		num_agents, std::vector<std::vector<int>>(n_V));
	for (int j = 0; j < num_agents; ++j) {
		auto& g = problem.graphs[j];
		for (int sg = 0; sg < n_V; ++sg) {
			for (int var : presolve.roles_at[sg]) {
				const int row = subgraph.subgraph_variable_id(var);
				if (row < 0 || !is_candidate(var, j)) continue;
				vertices_at[j][sg].push_back(g.vertices.size());
				g.vertices.push_back({sg, row});
			}
			if (presolve.required_at[sg].contains(j)) {
				vertices_at[j][sg].push_back(g.vertices.size());
				g.vertices.push_back({sg, GCSModeGraph::kNoRole});
			}
		}
	}

	// Vertex weights: A(k, j) for a role vertex, 1 - (roles j holds at the
	// node) for a NO_ROLE vertex; visit(j, sg) = 1 iff j visits sg.
	auto alpha = [&](int j, int sg) {
		Expression a = Expression::Zero();
		for (int v : vertices_at[j][sg]) {
			const int mode = problem.graphs[j].vertices[v].mode;
			if (mode != GCSModeGraph::kNoRole) a += Expression(Assignments(mode, j));
		}
		return a;
	};
	auto weight = [&](int j, int v) -> Expression {
		const auto& vx = problem.graphs[j].vertices[v];
		return vx.mode != GCSModeGraph::kNoRole ? Expression(Assignments(vx.mode, j)) : 1.0 - alpha(j, vx.sg);
	};
	auto visit = [&](int j, int sg) {
		Expression y = Expression::Zero();
		for (int v : vertices_at[j][sg]) y += weight(j, v);
		return y;
	};

	// Vertex position copies, each in the perspective of its vertex set:
	// the global box, intersected below with every constraint on that
	// agent or role alone at that node.
	problem.P.resize(num_agents);
	std::vector<std::vector<AffineRows>> vertex_rows(num_agents);
	for (int j = 0; j < num_agents; ++j) {
		const int d = graph->robot_ambient_dim(j);
		const Eigen::VectorXd lb = graph->_global_x_lb.segment(graph->agent_col_offset(j), d);
		const Eigen::VectorXd ub = graph->_global_x_ub.segment(graph->agent_col_offset(j), d);
		AffineRows box(d);
		box.A_in.resize(2 * d, d);
		box.A_in << Eigen::MatrixXd::Identity(d, d), -Eigen::MatrixXd::Identity(d, d);
		box.b_in.resize(2 * d);
		box.b_in << -ub, lb;
		for (const auto& vx : problem.graphs[j].vertices) {
			problem.P[j].push_back(prog.NewContinuousVariables(d, fmt::format("P_{}_{}_{}", j, vx.sg, vx.mode)));
			vertex_rows[j].push_back(box);
		}
	}

	// Agent j's position at sg (0 when j doesn't visit it), and role row's
	// output: its holder's position there.
	auto pos_agent = [&](int j, int sg) {
		drake::VectorX<Expression> p = drake::VectorX<Expression>::Zero(graph->robot_ambient_dim(j));
		for (int v : vertices_at[j][sg]) p += problem.P[j][v].cast<Expression>();
		return p;
	};
	auto role_out = [&](int row, int sg) {
		const int width = graph->_var_agent_q.Vars(row_var[row]).size();
		drake::VectorX<Expression> r = drake::VectorX<Expression>::Zero(width);
		for (int j = 0; j < num_agents; ++j) {
			for (int v : vertices_at[j][sg]) {
				if (problem.graphs[j].vertices[v].mode != row) continue;
				if (problem.P[j][v].size() != width)
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: role {} has candidates of differing widths", row_var[row]));
				r += problem.P[j][v].cast<Expression>();
			}
		}
		return r;
	};

	// Object positions: plain variables at every node.
	problem.O.resize(num_objects);
	for (int o = 0; o < num_objects; ++o) {
		const int w = graph->object_ambient_dim(o);
		problem.O[o] = prog.NewContinuousVariables(n_V, w, fmt::format("O_{}", o));
		const Eigen::VectorXd lb = graph->_global_x_lb.segment(graph->object_col_offset(o), w);
		const Eigen::VectorXd ub = graph->_global_x_ub.segment(graph->object_col_offset(o), w);
		for (int sg = 0; sg < n_V; ++sg)
			prog.AddBoundingBoxConstraint(lb, ub, problem.O[o].row(sg).transpose());
	}
	auto obj_pos = [&](int o, int sg) -> drake::VectorX<Expression> {
		return problem.O[o].row(sg).transpose().cast<Expression>();
	};

	auto add_params = [&](Substitution* sub) {
		graph->_param.SubstituteRange(sub, graph->num_params(), [&](int p, int) {
			return Expression(graph->view_param_values()(p)); });
	};

	///////////////////////////////////////////////////////////////////////
	//                             Constraints                           //
	///////////////////////////////////////////////////////////////////////

	// Compiles f -- over the plain agent_q/var_agent_q/object_q placeholders
	// -- at subgraph node sg. A constraint on a single agent or a single
	// role is intersected into the sets of that agent's (role's) vertices
	// at sg, so it holds in perspective on every vertex and arc copy. Any
	// other constraint is added as rows on the positions it couples, which
	// is exact: an agent's position at a node it must visit, and a role's
	// output, are linear in the vertex copies. `gate` (an interior node of
	// an along-edge constraint) makes the rows conditional instead -- on
	// the gate, and on the agent in each slot actually being at sg.
	auto compile_at_node = [&](const Formula& f, int sg, bool routes, const Expression* gate,
				   const std::string& where) {
		const Variables fv = f.GetFreeVariables();
		const std::vector<int> agents = graph->_agent_q.KeysReferencedBy(fv);
		const std::vector<int> vars = graph->_var_agent_q.KeysReferencedBy(fv);
		const std::vector<int> objects = graph->_object_q.KeysReferencedBy(fv);
		for (int var : vars) {
			if (!routes)
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: {}: routes=False constraints on a role are not supported "
					"(its holder need not visit the node)", where));
			// As MILP's CompileSymbolicNodeConstraint: a variable with no
			// row in this horizon contributes nothing.
			if (subgraph.subgraph_variable_id(var) < 0) return;
		}
		Substitution sub_p;
		add_params(&sub_p);
		const Formula f_p = f.Substitute(sub_p);

		if (agents.size() == 1 && vars.empty() && objects.empty()) {
			const int j = agents[0];
			const AffineRows rows = RowsOver(f_p, graph->_agent_q.Vars(j), where);
			for (int v : vertices_at[j][sg]) {
				if (gate == nullptr) {
					vertex_rows[j][v].Append(rows);
				} else {
					const Expression slack = M_sym * (1.0 - *gate);
					AddPerspectiveRows(prog, rows, problem.P[j][v].cast<Expression>(), weight(j, v), &slack);
				}
			}
			return;
		}
		if (vars.size() == 1 && agents.empty() && objects.empty()) {
			const int var = vars[0];
			const int row = subgraph.subgraph_variable_id(var);
			const AffineRows rows = RowsOver(f_p, graph->_var_agent_q.Vars(var), where);
			for (int j = 0; j < num_agents; ++j) {
				if (!is_candidate(var, j)) continue;
				for (int v : vertices_at[j][sg]) {
					if (gate == nullptr) {
						if (problem.graphs[j].vertices[v].mode == row) vertex_rows[j][v].Append(rows);
					} else {
						// The holder's position at sg, in whatever mode it's there.
						const Expression slack = M_sym * ((1.0 - *gate) + (1.0 - Expression(Assignments(row, j))));
						AddPerspectiveRows(prog, rows, problem.P[j][v].cast<Expression>(), weight(j, v), &slack);
					}
				}
			}
			return;
		}

		std::vector<Expression> gates;
		if (gate) gates.push_back(*gate);
		Substitution sub = sub_p;
		for (int j : agents) {
			const auto& ph = graph->_agent_q.Vars(j);
			const drake::VectorX<Expression> p = pos_agent(j, sg);
			for (int c = 0; c < ph.size(); ++c) sub[ph[c]] = p(c);
			if (!presolve.required_at[sg].contains(j)) gates.push_back(visit(j, sg));
		}
		for (int o : objects) {
			const auto& ph = graph->_object_q.Vars(o);
			const drake::VectorX<Expression> p = obj_pos(o, sg);
			for (int c = 0; c < ph.size(); ++c) sub[ph[c]] = p(c);
		}
		if (gate != nullptr && !vars.empty()) {
			if (vars.size() > 1)
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: {}: along-edge constraints on more than one role are not supported", where));
			const int var = vars[0];
			const int row = subgraph.subgraph_variable_id(var);
			const auto& ph = graph->_var_agent_q.Vars(var);
			for (int j = 0; j < num_agents; ++j) {
				if (!is_candidate(var, j)) continue;
				Substitution sub_j = sub;
				const drake::VectorX<Expression> p = pos_agent(j, sg);
				for (int c = 0; c < ph.size(); ++c) sub_j[ph[c]] = p(c);
				std::vector<Expression> gates_j = gates;
				gates_j.push_back(Expression(Assignments(row, j)));
				gates_j.push_back(visit(j, sg));
				RequireFullySubstituted(f, sub_j);
				AddGatedAtoms(prog, f.Substitute(sub_j), gates_j, M_sym, where);
			}
			return;
		}
		for (int var : vars) {
			const auto& ph = graph->_var_agent_q.Vars(var);
			const drake::VectorX<Expression> r = role_out(subgraph.subgraph_variable_id(var), sg);
			for (int c = 0; c < ph.size(); ++c) sub[ph[c]] = r(c);
		}
		RequireFullySubstituted(f, sub);
		AddGatedAtoms(prog, f.Substitute(sub), gates, M_sym, where);
	};

	// Node constraints from the unified symbolic API.
	for (const auto& [phi_id, rec] : subgraph.get_subgraph_symbolic_ops()) {
		compile_at_node(rec.formula, subgraph.subgraph_id(rec.node),
				!graph->_phi_routing_exempt.count(phi_id), nullptr,
				fmt::format("node constraint {} at node {}", phi_id, rec.node));
	}

	// Edge constraints. A relational formula couples the u_/v_ positions
	// (a passed endpoint reads previous_X, as in the MILP). An "along the
	// edge" formula holds at both endpoints, and -- betweenness-gated -- at
	// every other node that may end up scheduled between them.
	std::map<std::tuple<int, int, int>, Variable> betweenness_cache;
	for (const auto& [edge_phi_id, rec] : subgraph.get_subgraph_symbolic_edge_ops()) {
		const std::string where = fmt::format("edge constraint {} on ({}, {})", edge_phi_id, rec.u_node, rec.v_node);
		const int sg_u = subgraph.subgraph_id(rec.u_node);
		const int sg_v = subgraph.subgraph_id(rec.v_node);
		if (rec.along_edge) {
			if (sg_u >= 0) compile_at_node(rec.formula, sg_u, true, nullptr, where);
			if (sg_v >= 0) compile_at_node(rec.formula, sg_v, true, nullptr, where);
			if (sg_u < 0 || sg_v < 0) continue;
			for (int sg_w = 0; sg_w < n_V; ++sg_w) {
				if (sg_w == sg_u || sg_w == sg_v) continue;
				const Variable& b_between = GetOrAddBetweennessIndicator(
					prog, t_var, ri(sg_u), ri(sg_w), ri(sg_v), M_time, betweenness_cache);
				const Expression gate(b_between);
				compile_at_node(rec.formula, sg_w, true, &gate, where);
			}
			continue;
		}

		const Variables fv = rec.formula.GetFreeVariables();
		Substitution sub;
		add_params(&sub);
		auto bind_agents = [&](const PlaceholderVarFamily<int>& family, int sg_node, int node) {
			for (int j : family.KeysReferencedBy(fv)) {
				const auto& ph = family.Vars(j);
				const drake::VectorX<Expression> p = sg_node >= 0
					? pos_agent(j, sg_node)
					: drake::VectorX<Expression>(previous_X.row(node).segment(
						graph->agent_col_offset(j), ph.size()).transpose().cast<Expression>());
				for (int c = 0; c < ph.size(); ++c) sub[ph[c]] = p(c);
			}
		};
		auto bind_objects = [&](const PlaceholderVarFamily<int>& family, int sg_node, int node) {
			for (int o : family.KeysReferencedBy(fv)) {
				const auto& ph = family.Vars(o);
				const drake::VectorX<Expression> p = sg_node >= 0
					? obj_pos(o, sg_node)
					: drake::VectorX<Expression>(previous_X.row(node).segment(
						graph->object_col_offset(o), ph.size()).transpose().cast<Expression>());
				for (int c = 0; c < ph.size(); ++c) sub[ph[c]] = p(c);
			}
		};
		bind_agents(graph->_agent_q_u, sg_u, rec.u_node);
		bind_agents(graph->_agent_q_v, sg_v, rec.v_node);
		bind_objects(graph->_object_q_u, sg_u, rec.u_node);
		bind_objects(graph->_object_q_v, sg_v, rec.v_node);
		RequireFullySubstituted(rec.formula, sub);
		AddGatedAtoms(prog, rec.formula.Substitute(sub), {}, M_sym, where);
	}

	// Hold rigidity (MILP's Constraint 14a): the held object's offset from
	// the holder's end effector is the same at the hold's reference (u, or
	// x0 when the hold began before this horizon) and at v -- exact at the
	// endpoints, which the holder visits, and betweenness-gated at every
	// other node, conditional on the holder actually being there.
	if (enforce_rigidity) {
		const int wd = graph->workspace_dim;
		// The leading min(width, workspace_dim) position columns, zero-padded
		// to workspace_dim (as PoseFromRow/PointWorldFromRow read them).
		auto padded = [&](const drake::VectorX<Expression>& q, int width) {
			drake::VectorX<Expression> p = drake::VectorX<Expression>::Zero(wd);
			const int n = std::min(width, wd);
			p.head(n) = q.head(n);
			return p;
		};
		const double M_rigid = 2.0 * M_sym;
		for (const auto& [hold_id, hold] : subgraph.get_subgraph_hold_ops()) {
			const std::string where = fmt::format("hold {} on ({}, {})", hold_id, hold.u_node, hold.v_node);
			if (hold.rigid) {
				throw std::runtime_error(
					"GCSWaypointMPC: hold " + std::to_string(hold_id) +
					" was declared rigid=True (offset fixed in the end-effector's own "
					"frame); that relation is nonlinear -- use the evolutionary "
					"waypoint solver, or declare the hold translation-only.");
			}
			const int sg_v = subgraph.subgraph_id(hold.v_node);
			if (sg_v < 0) continue;  // target not yet in this horizon
			const int sg_u = subgraph.subgraph_id(hold.u_node);
			const int t_idx_ref = sg_u >= 0 ? ri(sg_u) : depot_idx;

			std::vector<int> holders;
			int row = -1;
			if (hold.robot_ag.has_value()) {
				holders = {*hold.robot_ag};
			} else {
				row = subgraph.subgraph_variable_id(*hold.var_id);
				DRAKE_DEMAND(row >= 0);
				for (int j = 0; j < num_agents; ++j)
					if (is_candidate(*hold.var_id, j)) holders.push_back(j);
			}
			for (int j : holders) {
				if (graph->robot_rigidity_constraint_degree(j) != ConstraintDegree::kLinear)
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: {}: only holders whose rigidity relation is linear "
						"(point masses) are supported", where));
			}

			// Holder's end-effector position at an endpoint (exact), and at
			// the reference.
			auto ee_at_endpoint = [&](int sg) {
				drake::VectorX<Expression> e = drake::VectorX<Expression>::Zero(wd);
				for (int j : holders) {
					for (int v : vertices_at[j][sg]) {
						if (row >= 0 && problem.graphs[j].vertices[v].mode != row) continue;
						e += padded(problem.P[j][v].cast<Expression>(), EEPositionWidth(graph, j));
					}
				}
				return e;
			};
			drake::VectorX<Expression> ee_ref = drake::VectorX<Expression>::Zero(wd);
			if (sg_u >= 0) {
				ee_ref = ee_at_endpoint(sg_u);
			} else {
				for (int j : holders) {
					const drake::VectorX<Expression> x0_j = x0.segment(
						graph->agent_col_offset(j), graph->robot_ambient_dim(j)).cast<Expression>();
					const Expression held = row >= 0 ? Expression(Assignments(row, j)) : Expression(1.0);
					ee_ref += held * padded(x0_j, EEPositionWidth(graph, j));
				}
			}
			const drake::VectorX<Expression> ee_v = ee_at_endpoint(sg_v);

			for (int o : hold.held_point_ids) {
				const int w_o = graph->object_ambient_dim(o);
				const drake::VectorX<Expression> obj_ref = sg_u >= 0
					? padded(obj_pos(o, sg_u), w_o)
					: padded(x0.segment(graph->object_col_offset(o), w_o).cast<Expression>(), w_o);
				const drake::VectorX<Expression> rel_ref = obj_ref - ee_ref;
				const drake::VectorX<Expression> residual = rel_ref - (padded(obj_pos(o, sg_v), w_o) - ee_v);
				// Same tolerances as the MILP: a static hold's endpoint rows
				// are a +-0.001 band, an assignable hold's are exact.
				if (row < 0) {
					prog.AddLinearConstraint(residual, Eigen::VectorXd::Constant(wd, -0.001),
								 Eigen::VectorXd::Constant(wd, 0.001));
				} else {
					prog.AddLinearEqualityConstraint(residual, Eigen::VectorXd::Zero(wd));
				}

				for (int sg_w = 0; sg_w < n_V; ++sg_w) {
					if (sg_w == sg_u || sg_w == sg_v) continue;
					const Variable& b_between = GetOrAddBetweennessIndicator(
						prog, t_var, t_idx_ref, ri(sg_w), ri(sg_v), M_time, betweenness_cache);
					for (int j : holders) {
						const drake::VectorX<Expression> rel_w =
							padded(obj_pos(o, sg_w), w_o) - padded(pos_agent(j, sg_w), EEPositionWidth(graph, j));
						Expression slack = M_rigid * ((1.0 - Expression(b_between)) + (1.0 - visit(j, sg_w)));
						if (row >= 0) slack += M_rigid * (1.0 - Expression(Assignments(row, j)));
						const drake::VectorX<Expression> res_w = rel_ref - rel_w;
						for (int c = 0; c < wd; ++c) {
							prog.AddLinearConstraint(res_w(c) - slack, -kInf, 0.0);
							prog.AddLinearConstraint(-res_w(c) - slack, -kInf, 0.0);
						}
					}
				}
			}
		}
	}

	// Every vertex copy in the perspective of its (now complete) set.
	for (int j = 0; j < num_agents; ++j)
		for (size_t v = 0; v < problem.graphs[j].vertices.size(); ++v)
			AddPerspectiveRows(prog, vertex_rows[j][v], problem.P[j][v].cast<Expression>(), weight(j, v));

	// An agent holds at most one role per node.
	for (int j = 0; j < num_agents; ++j) {
		for (int sg = 0; sg < n_V; ++sg) {
			int num_roles = 0;
			for (int v : vertices_at[j][sg])
				if (problem.graphs[j].vertices[v].mode != GCSModeGraph::kNoRole) ++num_roles;
			if (num_roles > 1) prog.AddLinearConstraint(alpha(j, sg), -kInf, 1.0);
		}
	}

	prog.AddBoundingBoxConstraint(0.0, M_time, t_var);
	prog.AddLinearEqualityConstraint(Expression(t_var(depot_idx)), 0.0);
	// Makespan = t(sink), when the last route ends (every route below ends at t(sink)).
	prog.AddLinearEqualityConstraint(Expression(L_ms_var(0)) - Expression(t_var(sink_idx)), 0.0);

	// Routes on the mode graphs, with arc position and time copies.
	problem.Z.resize(num_agents);
	Expression total_cost = Expression::Zero();
	for (int j = 0; j < num_agents; ++j) {
		auto& g = problem.graphs[j];
		const int d = graph->robot_ambient_dim(j);
		const int n_vert = g.vertices.size();
		const Eigen::VectorXd x0_j = x0.segment(graph->agent_col_offset(j), d);
		auto vertex_ri = [&](int v) { return ri(g.vertices[v].sg); };

		for (int v = 0; v < n_vert; ++v)
			if (presolve.arc_allowed(j, depot_idx, vertex_ri(v))) g.arcs.push_back({GCSModeGraph::kDepot, v});
		if (presolve.arc_allowed(j, depot_idx, sink_idx)) g.arcs.push_back({GCSModeGraph::kDepot, GCSModeGraph::kSink});
		for (int a = 0; a < n_vert; ++a)
			for (int b = 0; b < n_vert; ++b)
				if (g.vertices[a].sg != g.vertices[b].sg && presolve.arc_allowed(j, vertex_ri(a), vertex_ri(b)))
					g.arcs.push_back({a, b});
		for (int v = 0; v < n_vert; ++v)
			if (presolve.arc_allowed(j, vertex_ri(v), sink_idx)) g.arcs.push_back({v, GCSModeGraph::kSink});

		const int L = g.arcs.size();
		const VectorXDecisionVariable Z = prog.NewBinaryVariables(L, fmt::format("Z_{}", j));
		problem.Z[j] = Z;
		const VectorXDecisionVariable sigma = prog.NewContinuousVariables(L, fmt::format("sigma_{}", j));
		const VectorXDecisionVariable rho = prog.NewContinuousVariables(L, fmt::format("rho_{}", j));
		prog.AddBoundingBoxConstraint(0.0, kInf, sigma);
		prog.AddBoundingBoxConstraint(0.0, kInf, rho);

		// Tail/head position copies: X = x0 Z out of the depot; no head copy
		// into the sink.
		std::vector<drake::VectorX<Expression>> X(L), Y(L);
		std::vector<std::vector<int>> out_arcs(n_vert), in_arcs(n_vert);
		Expression depart, arrive;
		for (int a = 0; a < L; ++a) {
			const auto [tail, head] = g.arcs[a];
			if (tail == GCSModeGraph::kDepot) {
				X[a] = x0_j.cast<Expression>() * Expression(Z(a));
				depart += Expression(Z(a));
				prog.AddLinearEqualityConstraint(Expression(sigma(a)), 0.0);
			} else {
				X[a] = prog.NewContinuousVariables(d, fmt::format("X_{}_{}", j, a)).cast<Expression>();
				out_arcs[tail].push_back(a);
				AddPerspectiveRows(prog, vertex_rows[j][tail], X[a], Expression(Z(a)));
			}
			if (head == GCSModeGraph::kSink) {
				arrive += Expression(Z(a));
				prog.AddLinearConstraint(Expression(rho(a)) - Expression(sigma(a)), 0.0, kInf);
			} else {
				Y[a] = prog.NewContinuousVariables(d, fmt::format("Y_{}_{}", j, a)).cast<Expression>();
				in_arcs[head].push_back(a);
				AddPerspectiveRows(prog, vertex_rows[j][head], Y[a], Expression(Z(a)));

				const std::string tag = fmt::format("j{}_a{}", j, a);
				Expression dist;
				switch (objective) {
				case WaypointObjective::kMinMaxL1:
				case WaypointObjective::kAvgL1:
					dist = ComputeL1Distance(prog, splines->at(j), Y[a], X[a], tag);
					break;
				case WaypointObjective::kMinMaxL2:
				case WaypointObjective::kAvgL2:
					dist = ComputeL2Distance(prog, splines->at(j), Y[a], X[a], tag);
					break;
				}
				prog.AddLinearConstraint(Expression(rho(a)) - Expression(sigma(a)) - dist, 0.0, kInf);
				total_cost += dist;
			}
			prog.AddLinearConstraint(Expression(sigma(a)) - M_time * Expression(Z(a)), -kInf, 0.0);
			prog.AddLinearConstraint(Expression(rho(a)) - M_time * Expression(Z(a)), -kInf, 0.0);
		}
		prog.AddLinearEqualityConstraint(depart, 1.0);
		prog.AddLinearEqualityConstraint(arrive, 1.0);

		// Every route ends at t(sink).
		Expression finish;
		for (int a = 0; a < L; ++a)
			if (g.arcs[a].second == GCSModeGraph::kSink) finish += Expression(rho(a));
		prog.AddLinearEqualityConstraint(finish - Expression(t_var(sink_idx)), 0.0);

		// Degree, copy coupling, and vertex time copies theta = weight * t(v).
		for (int v = 0; v < n_vert; ++v) {
			const Expression w = weight(j, v);
			const drake::VectorX<Expression> P_v = problem.P[j][v].cast<Expression>();
			Expression in_flow, out_flow, in_rho, out_sigma;
			drake::VectorX<Expression> out_X = drake::VectorX<Expression>::Zero(d);
			drake::VectorX<Expression> in_Y = drake::VectorX<Expression>::Zero(d);
			for (int a : out_arcs[v]) { out_flow += Expression(Z(a)); out_X += X[a]; out_sigma += Expression(sigma(a)); }
			for (int a : in_arcs[v])  { in_flow += Expression(Z(a));  in_Y += Y[a];  in_rho += Expression(rho(a)); }
			prog.AddLinearEqualityConstraint(in_flow - w, 0.0);
			prog.AddLinearEqualityConstraint(out_flow - w, 0.0);
			prog.AddLinearEqualityConstraint(out_X - P_v, Eigen::VectorXd::Zero(d));
			prog.AddLinearEqualityConstraint(in_Y - P_v, Eigen::VectorXd::Zero(d));

			const Variable theta = prog.NewContinuousVariables(1, fmt::format("theta_{}_{}", j, v))(0);
			const Expression t_v(t_var(vertex_ri(v)));
			prog.AddLinearConstraint(Expression(theta), 0.0, kInf);
			prog.AddLinearConstraint(Expression(theta) - M_time * w, -kInf, 0.0);
			prog.AddLinearConstraint(t_v - Expression(theta), 0.0, kInf);
			prog.AddLinearConstraint(t_v - Expression(theta) - M_time * (1.0 - w), -kInf, 0.0);
			prog.AddLinearEqualityConstraint(out_sigma - Expression(theta), 0.0);
			prog.AddLinearEqualityConstraint(in_rho - Expression(theta), 0.0);
		}

		if (subtour == SubtourElimination::kMCF) {
			std::vector<int> vertex_node(n_vert);
			for (int v = 0; v < n_vert; ++v) vertex_node[v] = g.vertices[v].sg;
			std::vector<Expression> visits(n_V);
			for (int c = 0; c < n_V; ++c) visits[c] = visit(j, c);
			AddMultiCommodityFlow(prog, g.arcs, Z, vertex_node, visits, std::to_string(j));
		}
	}

	// Shared with the MILP (waypoint_common.*).
	AddRoleExactlyOneAgent(prog, subgraph, Assignments, num_agents);
	AddCommittedAssignments(prog, *graph, subgraph, Assignments);
	AddOrderingConstraints(prog, *graph, subgraph, t_var, M_time, Assignments,
			       previous_var_assignments, cond_binary_vars);
	AddAssignmentVariableConstraints(prog, *graph, subgraph, Assignments);
	{
		drake::MatrixX<Expression> obj_state = drake::MatrixX<Expression>::Zero(n_V, graph->total_dim);
		for (int o = 0; o < num_objects; ++o)
			for (int sg = 0; sg < n_V; ++sg)
				obj_state.row(sg).segment(graph->object_col_offset(o), graph->object_ambient_dim(o)) =
					obj_pos(o, sg).transpose();
		AddStationaryObjectConstraints(prog, *graph, subgraph, obj_state, x0, t_var, M_time);
	}

	///////////////////////////////////////////////////////////////////////
	//                              Objective                            //
	///////////////////////////////////////////////////////////////////////

	switch (objective) {
	case WaypointObjective::kMinMaxL1:
	case WaypointObjective::kMinMaxL2:
		prog.AddLinearCost(Eigen::VectorXd::Ones(1), 0.0, L_ms_var);
		break;
	case WaypointObjective::kAvgL1:
	case WaypointObjective::kAvgL2:
		if (num_agents == 0) throw std::runtime_error("kAvg* objectives require num_agents > 0");
		prog.AddLinearCost(total_cost * (1.0 / num_agents));
		break;
	}

	problem.prog = std::move(prog_ptr);
	return problem;
}

/*
 * Waypoint MPC
 */

GCSWaypointMPC::GCSWaypointMPC(GraphOfConstraints& graph,
			       std::vector<CubicConfigurationSpline> splines,
			       WaypointSolver solver,
			       bool enforce_rigidity,
			       WaypointObjective objective,
			       SubtourElimination subtour)
	: GraphWaypointMPC(graph, std::move(splines), objective),
	  _solver(solver),
	  _enforce_rigidity(enforce_rigidity),
	  _subtour(subtour) {}

bool GCSWaypointMPC::Solve(const std::vector<int>& remaining_vertices,
			   const Eigen::VectorXd& x0) {
	BeginSolve(remaining_vertices, x0);

	switch (_solver) {
	case WaypointSolver::kGurobi:
		return SolveWith(drake::solvers::GurobiSolver(), remaining_vertices, x0);
	case WaypointSolver::kMosek:
		return SolveWith(drake::solvers::MosekSolver(), remaining_vertices, x0);
	case WaypointSolver::kIPOPT:
		break;
	}
	throw std::runtime_error("GCSWaypointMPC: solver must be kGurobi or kMosek");
}

bool GCSWaypointMPC::SolveWith(const drake::solvers::SolverInterface& solver,
			       const std::vector<int>& remaining_vertices,
			       const Eigen::VectorXd& x0) {

	GCSWaypointProblem problem = BuildGCSWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		_enforce_rigidity, _objective, _subtour);

	MathematicalProgramResult result;
	solver.Solve(*problem.prog, std::nullopt, std::nullopt, &result);

	if (!result.is_success()) {
		std::cerr << "GCSWaypointMPC: no feasible solution.\n";
		PrintSolverReport(*problem.prog, result, 1e-6);
		return false;
	}

	_last_solve_time = _timer.Tick();

	const SubgraphOfConstraints& subgraph = *problem.subgraph;
	StoreAssignments(subgraph, problem.Assignments, result);

	const int n_V = subgraph.num_nodes();
	std::vector<int> node_of_sg(n_V, -1);
	for (int v : remaining_vertices) node_of_sg[subgraph.subgraph_id(v)] = v;

	_t_routing = result.GetSolution(problem.t);
	_t_by_node_id.setZero();
	for (int v : remaining_vertices) _t_by_node_id(v) = _t_routing(RoutingPresolve::ri(subgraph.subgraph_id(v)));

	for (int o = 0; o < _graph->num_objects; ++o) {
		const Eigen::MatrixXd O = result.GetSolution(problem.O[o]);
		for (int v : remaining_vertices)
			_waypoints.row(v).segment(_graph->object_col_offset(o), O.cols()) = O.row(subgraph.subgraph_id(v));
	}

	// Each agent's route; its visited rows from the vertex copies, the rest
	// interpolated per block (linear spline: geodesic per block) in time
	// between its neighbouring visits, holding after the last.
	_routes.assign(_graph->num_agents, {});
	for (int j = 0; j < _graph->num_agents; ++j) {
		const GCSModeGraph& g = problem.graphs[j];
		const int off = _graph->agent_col_offset(j);
		const int d = _graph->robot_ambient_dim(j);
		const Eigen::VectorXd Z = result.GetSolution(problem.Z[j]);

		std::vector<double> times = {0.0};
		std::vector<Eigen::VectorXd> pts = {x0.segment(off, d)};
		std::set<int> visited;
		int at = GCSModeGraph::kDepot;
		for (size_t step = 0; step <= g.vertices.size(); ++step) {
			int next = GCSModeGraph::kSink;
			for (size_t a = 0; a < g.arcs.size(); ++a)
				if (g.arcs[a].first == at && Z(a) > 0.5) { next = g.arcs[a].second; break; }
			if (next == GCSModeGraph::kSink) break;
			const int sg = g.vertices[next].sg;
			const int node = node_of_sg[sg];
			const Eigen::VectorXd q = result.GetSolution(problem.P[j][next]);
			_routes[j].push_back(node);
			visited.insert(node);
			_waypoints.row(node).segment(off, d) = q;
			const double t_node = _t_routing(RoutingPresolve::ri(sg));
			if (t_node > times.back() + 1e-9) {
				times.push_back(t_node);
				pts.push_back(q);
			}
			at = next;
		}

		if (pts.size() == 1) {
			for (int v : remaining_vertices)
				if (!visited.contains(v)) _waypoints.row(v).segment(off, d) = pts[0];
			continue;
		}
		CubicConfigurationSpline interp(_splines->at(j).spec_);
		interp.set_linear(true);
		Eigen::MatrixXd P(pts.size(), d);
		for (size_t k = 0; k < pts.size(); ++k) P.row(k) = pts[k];
		interp.set(P, Eigen::MatrixXd::Zero(pts.size(), interp.tan_dim_),
			   Eigen::Map<const Eigen::VectorXd>(times.data(), times.size()));
		for (int v : remaining_vertices)
			if (!visited.contains(v)) _waypoints.row(v).segment(off, d) = interp.eval(_t_by_node_id(v)).q_ambient;
	}
	return true;
}
