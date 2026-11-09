#include "gcs_waypoint_mpc.hpp"
#include "symbolic_constraint_compiler.hpp"
#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <Eigen/LU>

#include <drake/common/nice_type_name.h>
#include <drake/common/symbolic/decompose.h>
#include <drake/solvers/aggregate_costs_constraints.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/solve.h>

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

// An enclosure [lo, hi] of e over boxes on its variables (interval
// arithmetic; sin/cos as [-1, 1]). Throws on an unsupported operation or a
// variable with no box.
struct Interval {
	double lo, hi;
};

Interval Mul(Interval a, Interval b) {
	const double p[] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
	return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}

Interval Bound(const Expression& e, const std::unordered_map<Variable::Id, Interval>& box) {
	using namespace drake::symbolic;
	if (is_constant(e)) {
		const double c = get_constant_value(e);
		return {c, c};
	}
	if (is_variable(e)) {
		const auto it = box.find(get_variable(e).get_id());
		if (it == box.end())
			throw std::runtime_error("GCSWaypointMPC: no box for " + get_variable(e).get_name());
		return it->second;
	}
	if (is_addition(e)) {
		Interval r{get_constant_in_addition(e), get_constant_in_addition(e)};
		for (const auto& [term, coef] : get_expr_to_coeff_map_in_addition(e)) {
			const Interval t = Mul(Bound(term, box), {coef, coef});
			r = {r.lo + t.lo, r.hi + t.hi};
		}
		return r;
	}
	if (is_multiplication(e)) {
		Interval r{get_constant_in_multiplication(e), get_constant_in_multiplication(e)};
		for (const auto& [base, exponent] : get_base_to_exponent_map_in_multiplication(e))
			r = Mul(r, Bound(pow(base, exponent), box));
		return r;
	}
	if (is_pow(e)) {
		const Interval b = Bound(get_first_argument(e), box);
		const Expression& ex = get_second_argument(e);
		if (!is_constant(ex)) throw std::runtime_error("GCSWaypointMPC: non-constant power in " + e.to_string());
		const double p = get_constant_value(ex);
		if (p == std::round(p) && p >= 0) {
			const double a = std::pow(b.lo, p), c = std::pow(b.hi, p);
			if (static_cast<long>(p) % 2 == 1) return {a, c};
			const double hi = std::max(a, c);
			return {(b.lo <= 0.0 && b.hi >= 0.0) ? 0.0 : std::min(a, c), hi};
		}
		if (b.lo < 0.0) throw std::runtime_error("GCSWaypointMPC: fractional power of a possibly negative base");
		const double a = std::pow(b.lo, p), c = std::pow(b.hi, p);
		return {std::min(a, c), std::max(a, c)};
	}
	if (is_division(e)) {
		const Interval d = Bound(get_second_argument(e), box);
		if (d.lo <= 0.0 && d.hi >= 0.0) throw std::runtime_error("GCSWaypointMPC: division by a possibly zero " + e.to_string());
		return Mul(Bound(get_first_argument(e), box), {1.0 / d.hi, 1.0 / d.lo});
	}
	if (is_sin(e) || is_cos(e)) return {-1.0, 1.0};
	if (is_sqrt(e)) {
		const Interval a = Bound(get_argument(e), box);
		return {std::sqrt(std::max(a.lo, 0.0)), std::sqrt(std::max(a.hi, 0.0))};
	}
	if (is_abs(e)) {
		const Interval a = Bound(get_argument(e), box);
		const double hi = std::max(std::abs(a.lo), std::abs(a.hi));
		return {(a.lo <= 0.0 && a.hi >= 0.0) ? 0.0 : std::min(std::abs(a.lo), std::abs(a.hi)), hi};
	}
	if (is_exp(e)) {
		const Interval a = Bound(get_argument(e), box);
		return {std::exp(a.lo), std::exp(a.hi)};
	}
	throw std::runtime_error("GCSWaypointMPC: no interval bound for " + e.to_string());
}

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

// lb <= e <= ub, e over the placeholders `over`.
struct NonlinearAtom {
	Expression e;
	double lb, ub;
	drake::VectorX<Variable> over;
};

// Rows of f, a conjunction of relations over `vars` only. Non-affine atoms
// go to `nonlinear` when given; without it they throw.
AffineRows RowsOver(const Formula& f, const drake::VectorX<Variable>& vars, const std::string& where,
		    std::vector<NonlinearAtom>* nonlinear = nullptr) {
	const Variables allowed(vars);
	for (const auto& v : f.GetFreeVariables()) {
		if (!allowed.include(v))
			throw std::runtime_error(fmt::format(
				"GCSWaypointMPC: {}: unsupported placeholder {} in {}", where, v.get_name(), f.to_string()));
	}
	std::vector<Expression> in, eq;
	CollectAtoms(f, &in, &eq, where);
	if (nonlinear) {
		auto split = [&](std::vector<Expression>& es, double lb) {
			std::vector<Expression> affine;
			for (const auto& e : es) {
				if (drake::symbolic::IsAffine(drake::Vector1<Expression>(e), allowed))
					affine.push_back(e);
				else
					nonlinear->push_back({e, lb, 0.0, vars});
			}
			es = std::move(affine);
		};
		split(in, -kInf);
		split(eq, 0.0);
	}
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
				"GCSWaypointMPC: {}: only linear constraints are supported here; nonlinear ones only "
				"on a single agent's or role's position in a node constraint ({})", where, ex.what()));
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
			"GCSWaypointMPC: {}: only linear constraints are supported here; nonlinear ones only "
			"on a single agent's or role's position in a node constraint ({})", where, ex.what()));
	}
}

// A set {x : rows}'s Chebyshev centre (within the affine hull of its
// equalities) and the bounding box of its leading `box_cols` columns.
struct SetGeometry {
	Eigen::VectorXd centre, lo, hi;
	bool empty = false;
};

// One LP block for `rows`: the centre/radius problem and, per box column, a
// min and a max copy. Blocks share no variables, so their costs add.
struct SetGeometryBlock {
	VectorXDecisionVariable centre;
	std::vector<VectorXDecisionVariable> lo, hi;
};

SetGeometryBlock AddSetGeometryBlock(MathematicalProgram& prog, const AffineRows& rows, int box_cols, double r_max) {
	const int d = rows.A_in.cols();
	auto add_membership = [&](const VectorXDecisionVariable& x) {
		if (rows.A_eq.rows() > 0) prog.AddLinearEqualityConstraint(rows.A_eq, -rows.b_eq, x);
		if (rows.A_in.rows() > 0)
			prog.AddLinearConstraint(rows.A_in, Eigen::VectorXd::Constant(rows.A_in.rows(), -kInf), -rows.b_in, x);
	};

	SetGeometryBlock block;
	block.centre = prog.NewContinuousVariables(d, "centre");
	const drake::symbolic::Variable r = prog.NewContinuousVariables(1, "radius")(0);
	prog.AddBoundingBoxConstraint(0.0, r_max, r);
	// Distances within the affine hull: a_i's component along its null space.
	Eigen::MatrixXd N = Eigen::MatrixXd::Identity(d, d);
	if (rows.A_eq.rows() > 0) {
		const Eigen::FullPivLU<Eigen::MatrixXd> lu(rows.A_eq);
		N = lu.rank() == d ? Eigen::MatrixXd(d, 0) : Eigen::MatrixXd(lu.kernel());
	}
	if (rows.A_eq.rows() > 0) prog.AddLinearEqualityConstraint(rows.A_eq, -rows.b_eq, block.centre);
	if (rows.A_in.rows() > 0) {
		Eigen::MatrixXd A(rows.A_in.rows(), d + 1);
		A << rows.A_in, (rows.A_in * N).rowwise().norm();
		VectorXDecisionVariable cr(d + 1);
		cr << block.centre, r;
		prog.AddLinearConstraint(A, Eigen::VectorXd::Constant(A.rows(), -kInf), -rows.b_in, cr);
	}
	prog.AddLinearCost(-Expression(r));

	for (int c = 0; c < box_cols; ++c) {
		block.lo.push_back(prog.NewContinuousVariables(d, "lo"));
		block.hi.push_back(prog.NewContinuousVariables(d, "hi"));
		add_membership(block.lo.back());
		add_membership(block.hi.back());
		prog.AddLinearCost(Expression(block.lo.back()(c)) - Expression(block.hi.back()(c)));
	}
	return block;
}

// Every set's geometry from one LP; if that is infeasible, each set alone,
// marking the ones with no point as empty. `sets` pairs each set with its
// number of box columns.
std::vector<SetGeometry> AnalyzeSets(const std::vector<std::pair<const AffineRows*, int>>& sets, double r_max) {
	auto read = [&](const MathematicalProgramResult& result, const SetGeometryBlock& block) {
		const int box_cols = block.lo.size();
		SetGeometry g;
		g.centre = result.GetSolution(block.centre);
		g.lo.resize(box_cols);
		g.hi.resize(box_cols);
		for (int c = 0; c < box_cols; ++c) {
			g.lo(c) = result.GetSolution(block.lo[c](c));
			g.hi(c) = result.GetSolution(block.hi[c](c));
		}
		return g;
	};

	std::vector<SetGeometry> out(sets.size());
	{
		MathematicalProgram prog;
		std::vector<SetGeometryBlock> blocks;
		for (const auto& [rows, box_cols] : sets) blocks.push_back(AddSetGeometryBlock(prog, *rows, box_cols, r_max));
		const MathematicalProgramResult result = drake::solvers::Solve(prog);
		if (result.is_success()) {
			for (size_t i = 0; i < sets.size(); ++i) out[i] = read(result, blocks[i]);
			return out;
		}
	}
	for (size_t i = 0; i < sets.size(); ++i) {
		MathematicalProgram prog;
		const SetGeometryBlock block = AddSetGeometryBlock(prog, *sets[i].first, sets[i].second, r_max);
		const MathematicalProgramResult result = drake::solvers::Solve(prog);
		if (result.is_success()) {
			out[i] = read(result, block);
		} else {
			out[i].empty = true;
		}
	}
	return out;
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
	SubtourElimination subtour,
	double length_weight,
	const std::vector<std::shared_ptr<ArcTravelTimeModel>>& travel_time_models,
	double max_tangent_halfwidth) {

	const int num_agents = graph->num_agents;
	const bool travel_time = objective == WaypointObjective::kMinMaxTravelTime ||
				 objective == WaypointObjective::kAvgTravelTime;
	if (travel_time && static_cast<int>(travel_time_models.size()) != num_agents)
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC: travel-time objectives need one ArcTravelTimeModel per agent ({}), got {}",
			num_agents, travel_time_models.size()));
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
	// Under travel-time objectives, the models' own bounds (each arc's
	// tangent is checked against its model's below).
	const double x_range = (graph->_global_x_ub - graph->_global_x_lb).maxCoeff();
	int max_agent_width = 0;
	for (int j = 0; j < num_agents; ++j) max_agent_width = std::max(max_agent_width, graph->robot_ambient_dim(j));
	double M_cost = max_agent_width * x_range;
	if (travel_time) {
		M_cost = 0.0;
		for (const auto& model : travel_time_models) M_cost = std::max(M_cost, model->max_travel_time);
	}
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
	std::vector<std::vector<std::vector<NonlinearAtom>>> vertex_nonlinear(num_agents);
	// Boxes on the object variables and their per-vertex copies, for
	// bounding atoms over them.
	std::unordered_map<Variable::Id, Interval> copy_box;
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
			vertex_nonlinear[j].emplace_back();
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
		for (int sg = 0; sg < n_V; ++sg) {
			prog.AddBoundingBoxConstraint(lb, ub, problem.O[o].row(sg).transpose());
			for (int c = 0; c < w; ++c) copy_box[problem.O[o](sg, c).get_id()] = {lb(c), ub(c)};
		}
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
	//
	// FK placeholders (agent_link_pos/rot) become their agent's symbolic
	// FK over agent_q first; a role's (var_agent_link_pos/rot), each
	// candidate holder's FK over var_agent_q.
	auto compile_at_node = [&](const Formula& f_raw, int sg, bool routes, const Expression* gate,
				   const std::string& where) {
		const Formula f = f_raw.Substitute(graph->SymbolicFKSubstitution(f_raw.GetFreeVariables()));
		const Variables fv = f.GetFreeVariables();
		const std::vector<int> agents = graph->_agent_q.KeysReferencedBy(fv);
		const std::vector<int> objects = graph->_object_q.KeysReferencedBy(fv);
		std::set<int> var_set;
		for (int var : graph->_var_agent_q.KeysReferencedBy(fv)) var_set.insert(var);
		for (const auto& key : graph->_var_agent_link_pos.KeysReferencedBy(fv)) var_set.insert(key.first);
		for (const auto& key : graph->_var_agent_link_rot.KeysReferencedBy(fv)) var_set.insert(key.first);
		const std::vector<int> vars(var_set.begin(), var_set.end());
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
			std::vector<NonlinearAtom> nonlinear;
			const AffineRows rows = RowsOver(f_p, graph->_agent_q.Vars(j), where, gate ? nullptr : &nonlinear);
			for (int v : vertices_at[j][sg]) {
				if (gate == nullptr) {
					vertex_rows[j][v].Append(rows);
					vertex_nonlinear[j][v].insert(vertex_nonlinear[j][v].end(), nonlinear.begin(), nonlinear.end());
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
			for (int j = 0; j < num_agents; ++j) {
				if (!is_candidate(var, j)) continue;
				const Formula f_j = f_p.Substitute(graph->SymbolicFKSubstitution(f_p.GetFreeVariables(), var, j));
				std::vector<NonlinearAtom> nonlinear;
				const AffineRows rows = RowsOver(f_j, graph->_var_agent_q.Vars(var), where, gate ? nullptr : &nonlinear);
				for (int v : vertices_at[j][sg]) {
					if (gate == nullptr) {
						if (problem.graphs[j].vertices[v].mode != row) continue;
						vertex_rows[j][v].Append(rows);
						vertex_nonlinear[j][v].insert(vertex_nonlinear[j][v].end(), nonlinear.begin(), nonlinear.end());
					} else {
						// The holder's position at sg, in whatever mode it's there.
						const Expression slack = M_sym * ((1.0 - *gate) + (1.0 - Expression(Assignments(row, j))));
						AddPerspectiveRows(prog, rows, problem.P[j][v].cast<Expression>(), weight(j, v), &slack);
					}
				}
			}
			return;
		}
		// One agent and objects, with nonlinear atoms (e.g. fk(q) == object_q
		// + offset): the objects become this node's object variables, the
		// nonlinear atoms join the agent's one vertex's unscaled copy and the
		// affine ones are plain rows -- exact only because the agent surely
		// visits sg, in that one vertex.
		if (agents.size() == 1 && vars.empty() && !objects.empty() && gate == nullptr) {
			const int j = agents[0];
			Substitution sub_o;
			for (int o : objects) {
				const auto& ph = graph->_object_q.Vars(o);
				for (int c = 0; c < ph.size(); ++c) sub_o[ph[c]] = problem.O[o](sg, c);
			}
			auto seed_objects = [&] {
				for (int o : objects)
					for (int c = 0; c < graph->object_ambient_dim(o); ++c)
						problem.nonlinear_seed.emplace_back(problem.O[o](sg, c), x0(graph->object_col_offset(o) + c));
			};
			std::vector<Expression> in, eq;
			CollectAtoms(f_p.Substitute(sub_o), &in, &eq, where);
			auto affine = [](const Expression& e) { return drake::symbolic::IsAffine(drake::Vector1<Expression>(e)); };
			const bool any_nonlinear = std::any_of(in.begin(), in.end(), [&](const auto& e) { return !affine(e); }) ||
						   std::any_of(eq.begin(), eq.end(), [&](const auto& e) { return !affine(e); });
			if (any_nonlinear) {
				if (!presolve.required_at[sg].contains(j) || vertices_at[j][sg].size() != 1)
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: {}: a nonlinear constraint coupling agent {} with objects needs "
						"the agent to surely visit the node, in one mode", where, j));
				const int v = vertices_at[j][sg][0];
				const auto& ph = graph->_agent_q.Vars(j);
				Substitution sub_q;
				const drake::VectorX<Expression> p = pos_agent(j, sg);
				for (int c = 0; c < ph.size(); ++c) sub_q[ph[c]] = p(c);
				auto add = [&](const Expression& e, double lb) {
					if (affine(e))
						prog.AddLinearConstraint(drake::Vector1<Expression>(e.Substitute(sub_q)),
									 drake::Vector1d(lb), drake::Vector1d(0.0));
					else
						vertex_nonlinear[j][v].push_back({e, lb, 0.0, ph});
				};
				for (const auto& e : in) add(e, -kInf);
				for (const auto& e : eq) add(e, 0.0);
				seed_objects();
				return;
			}
		}
		// One role and objects, with nonlinear atoms: per candidate holder,
		// every atom joins its role vertex's unscaled copy, over that
		// vertex's own copy of the objects (equal to O whenever the vertex
		// is used, big-M on the global box), so no atom involves binaries.
		if (vars.size() == 1 && agents.empty() && !objects.empty() && gate == nullptr) {
			const int var = vars[0];
			const int row = subgraph.subgraph_variable_id(var);
			auto atoms_for = [&](int j, std::vector<Expression>* in, std::vector<Expression>* eq) {
				const Formula f_j = f_p.Substitute(graph->SymbolicFKSubstitution(f_p.GetFreeVariables(), var, j));
				CollectAtoms(f_j, in, eq, where);
			};
			auto affine = [](const Expression& e) { return drake::symbolic::IsAffine(drake::Vector1<Expression>(e)); };
			bool any_nonlinear = false;
			for (int j = 0; j < num_agents && !any_nonlinear; ++j) {
				if (!is_candidate(var, j)) continue;
				std::vector<Expression> in, eq;
				atoms_for(j, &in, &eq);
				any_nonlinear = std::any_of(in.begin(), in.end(), [&](const auto& e) { return !affine(e); }) ||
						std::any_of(eq.begin(), eq.end(), [&](const auto& e) { return !affine(e); });
			}
			if (any_nonlinear) {
				const auto& ph = graph->_var_agent_q.Vars(var);
				for (int j = 0; j < num_agents; ++j) {
					if (!is_candidate(var, j)) continue;
					std::vector<Expression> in, eq;
					atoms_for(j, &in, &eq);
					for (int v : vertices_at[j][sg]) {
						if (problem.graphs[j].vertices[v].mode != row) continue;
						const Expression unused = 1.0 - weight(j, v);
						Substitution sub_o;
						for (int o : objects) {
							const int w = graph->object_ambient_dim(o);
							const Eigen::VectorXd lb = graph->_global_x_lb.segment(graph->object_col_offset(o), w);
							const Eigen::VectorXd ub = graph->_global_x_ub.segment(graph->object_col_offset(o), w);
							const Eigen::VectorXd M = 2.0 * lb.cwiseAbs().cwiseMax(ub.cwiseAbs());
							if (!M.allFinite())
								throw std::runtime_error(fmt::format(
									"GCSWaypointMPC: {}: object {} has an unbounded state box", where, o));
							const VectorXDecisionVariable Oc = prog.NewContinuousVariables(
								w, fmt::format("O_{}_{}_{}_{}", o, j, sg, row));
							prog.AddBoundingBoxConstraint(lb, ub, Oc);
							const auto& ph_o = graph->_object_q.Vars(o);
							for (int c = 0; c < w; ++c) {
								copy_box[Oc(c).get_id()] = {lb(c), ub(c)};
								sub_o[ph_o[c]] = Oc(c);
								const Expression diff = problem.O[o](sg, c) - Oc(c);
								prog.AddLinearConstraint(diff - M(c) * unused, -kInf, 0.0);
								prog.AddLinearConstraint(-diff - M(c) * unused, -kInf, 0.0);
								const double x0_c = x0(graph->object_col_offset(o) + c);
								problem.nonlinear_seed.emplace_back(Oc(c), x0_c);
								problem.nonlinear_seed.emplace_back(problem.O[o](sg, c), x0_c);
							}
						}
						for (const auto& e : in) vertex_nonlinear[j][v].push_back({e.Substitute(sub_o), -kInf, 0.0, ph});
						for (const auto& e : eq) vertex_nonlinear[j][v].push_back({e.Substitute(sub_o), 0.0, 0.0, ph});
					}
				}
				return;
			}
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
				// Translation-only: the rows below read only the holder's
				// leading position columns, linear for any free body.
				if (!graph->robot_is_free_body(j))
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: {}: only free-body holders are supported "
						"(articulated holders need FK)", where));
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

	// A vertex with nonlinear atoms gets an unscaled copy Q of its position:
	// Q lies in the vertex's affine set unconditionally, satisfies its
	// nonlinear atoms whenever the vertex is used (each scaled by the
	// vertex's weight), and equals P then (big-M on the global box).
	for (int j = 0; j < num_agents; ++j) {
		const int d = graph->robot_ambient_dim(j);
		const Eigen::VectorXd lb = graph->_global_x_lb.segment(graph->agent_col_offset(j), d);
		const Eigen::VectorXd ub = graph->_global_x_ub.segment(graph->agent_col_offset(j), d);
		const Eigen::VectorXd M = lb.cwiseAbs().cwiseMax(ub.cwiseAbs());
		for (size_t v = 0; v < problem.graphs[j].vertices.size(); ++v) {
			const auto& atoms = vertex_nonlinear[j][v];
			if (atoms.empty()) continue;
			const auto& vx = problem.graphs[j].vertices[v];
			if (!M.allFinite())
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {} has a nonlinear constraint at subgraph node {} but an "
					"unbounded state box", j, vx.sg));
			const VectorXDecisionVariable Q =
				prog.NewContinuousVariables(d, fmt::format("Q_{}_{}_{}", j, vx.sg, vx.mode));
			prog.AddBoundingBoxConstraint(lb, ub, Q);
			for (int c = 0; c < d; ++c)
				problem.nonlinear_seed.emplace_back(Q(c), x0(graph->agent_col_offset(j) + c));
			AddPerspectiveRows(prog, vertex_rows[j][v], Q.cast<Expression>(), Expression(1.0));
			const Expression unused = 1.0 - weight(j, v);
			for (int c = 0; c < d; ++c) {
				const Expression diff = problem.P[j][v](c) - Q(c);
				prog.AddLinearConstraint(diff - M(c) * unused, -kInf, 0.0);
				prog.AddLinearConstraint(-diff - M(c) * unused, -kInf, 0.0);
			}
			// An atom binds only when the vertex is used: lb <= e - s <= 0, with
			// a slack s that is 0 when used and anywhere in e's enclosure over
			// the boxes otherwise (linear big-M rows, so the nonlinear rows
			// still involve no binaries).
			const Expression w = weight(j, v);
			const bool always_used = is_constant(w) && get_constant_value(w) == 1.0;
			std::unordered_map<Variable::Id, Interval> box = copy_box;
			for (int c = 0; c < d; ++c) box[Q(c).get_id()] = {lb(c), ub(c)};
			for (const auto& atom : atoms) {
				if (atom.over.size() != d)
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: a nonlinear constraint's placeholders are {} wide, agent {}'s position {}",
						atom.over.size(), j, d));
				if (atom.ub != 0.0 || (atom.lb != 0.0 && atom.lb != -kInf))
					throw std::runtime_error("GCSWaypointMPC: a nonlinear atom's bounds must be [0 or -inf, 0]");
				Substitution sub;
				for (int c = 0; c < d; ++c) sub[atom.over(c)] = Q(c);
				Expression e = atom.e.Substitute(sub);
				if (!always_used) {
					const Interval range = Bound(e, box);
					const double s_hi = std::max(range.hi, 0.0);
					const double s_lo = atom.lb == 0.0 ? std::min(range.lo, 0.0) : 0.0;
					const Variable s_var = prog.NewContinuousVariables(1, fmt::format("S_{}_{}", j, v))(0);
					prog.AddBoundingBoxConstraint(s_lo, s_hi, s_var);
					const Expression unused = 1.0 - w;
					prog.AddLinearConstraint(s_var - s_hi * unused, -kInf, 0.0);
					prog.AddLinearConstraint(s_lo * unused - s_var, -kInf, 0.0);
					e -= s_var;
				}
				const auto constraint = std::make_shared<drake::solvers::ExpressionConstraint>(
					drake::Vector1<Expression>(e), drake::Vector1d(atom.lb),
					drake::Vector1d(atom.ub));
				prog.AddConstraint(constraint, constraint->vars());
			}
		}
	}

	// Travel-time objectives: each vertex set's centre (where arcs are
	// linearized) and bounding box over its model's columns.
	std::vector<std::vector<SetGeometry>> vertex_geom(num_agents);
	if (travel_time) {
		std::vector<int> node_of_sg(n_V, -1);
		for (int v : remaining_vertices) node_of_sg[subgraph.subgraph_id(v)] = v;
		std::vector<std::pair<const AffineRows*, int>> sets;
		for (int j = 0; j < num_agents; ++j)
			for (const AffineRows& rows : vertex_rows[j]) sets.push_back({&rows, travel_time_models[j]->dim()});
		const std::vector<SetGeometry> geom = AnalyzeSets(sets, x_range);
		size_t i = 0;
		for (int j = 0; j < num_agents; ++j) {
			const ArcTravelTimeModel& model = *travel_time_models[j];
			const int m = model.dim();
			for (size_t v = 0; v < problem.graphs[j].vertices.size(); ++v, ++i) {
				vertex_geom[j].push_back(geom[i]);
				const SetGeometry& g = geom[i];
				if (g.empty) continue;
				const auto& vx = problem.graphs[j].vertices[v];
				const std::string who = fmt::format(
					"agent {} at node {} ({})", j, node_of_sg[vx.sg],
					vx.mode == GCSModeGraph::kNoRole ? std::string("no role")
									 : fmt::format("variable {}", row_var[vx.mode]));
				const double halfwidth = ((g.hi - g.lo) / 2.0).maxCoeff();
				if (halfwidth > max_tangent_halfwidth)
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: {}: vertex set half-width {} exceeds max_tangent_halfwidth {}",
						who, halfwidth, max_tangent_halfwidth));
				const Eigen::VectorXd c = g.centre.head(m);
				if ((c.array() < model.lower.array() - 1e-9).any() || (c.array() > model.upper.array() + 1e-9).any())
					throw std::runtime_error(fmt::format(
						"GCSWaypointMPC: {}: vertex set centre lies outside the travel-time model's domain", who));
			}
		}
	}

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

		// Travel-time objectives: the model's value and gradient at every
		// non-sink arc's (tail centre, head centre), in one call. An arc at
		// an empty vertex set is unusable (its copies' perspective forces
		// Z = 0) and isn't linearized.
		Eigen::VectorXd arc_T;
		Eigen::MatrixXd arc_gX, arc_gY, arc_X, arc_Y;
		std::vector<int> arc_row(L, -1);
		if (travel_time) {
			const ArcTravelTimeModel& model = *travel_time_models[j];
			const int m = model.dim();
			if ((x0_j.head(m).array() < model.lower.array() - 1e-9).any() ||
			    (x0_j.head(m).array() > model.upper.array() + 1e-9).any())
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {}'s x0 lies outside its travel-time model's domain", j));
			int n = 0;
			for (int a = 0; a < L; ++a) {
				const auto [tail, head] = g.arcs[a];
				if (head == GCSModeGraph::kSink) continue;
				if ((tail != GCSModeGraph::kDepot && vertex_geom[j][tail].empty) || vertex_geom[j][head].empty) continue;
				arc_row[a] = n++;
			}
			arc_X.resize(n, m);
			arc_Y.resize(n, m);
			Eigen::MatrixXd arc_X_lo(n, m), arc_X_hi(n, m), arc_Y_lo(n, m), arc_Y_hi(n, m);
			for (int a = 0; a < L; ++a) {
				if (arc_row[a] < 0) continue;
				const auto [tail, head] = g.arcs[a];
				const int r = arc_row[a];
				if (tail == GCSModeGraph::kDepot) {
					arc_X.row(r) = arc_X_lo.row(r) = arc_X_hi.row(r) = x0_j.head(m).transpose();
				} else {
					arc_X.row(r) = vertex_geom[j][tail].centre.head(m).transpose();
					arc_X_lo.row(r) = vertex_geom[j][tail].lo.transpose();
					arc_X_hi.row(r) = vertex_geom[j][tail].hi.transpose();
				}
				arc_Y.row(r) = vertex_geom[j][head].centre.head(m).transpose();
				arc_Y_lo.row(r) = vertex_geom[j][head].lo.transpose();
				arc_Y_hi.row(r) = vertex_geom[j][head].hi.transpose();
			}
			std::tie(arc_T, arc_gX, arc_gY) = model.Linearize(arc_X, arc_Y);
			if (arc_T.size() != n || arc_gX.rows() != n || arc_gX.cols() != m ||
			    arc_gY.rows() != n || arc_gY.cols() != m)
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {}'s travel-time model returned the wrong shapes for {} arcs", j, n));
			if (!arc_T.allFinite() || !arc_gX.allFinite() || !arc_gY.allFinite())
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {}'s travel-time model returned non-finite values", j));
			// The tangent's maximum over the arc's bounding boxes must fit the
			// time big-M.
			const Eigen::ArrayXXd dX = arc_gX.array() * (arc_X_hi - arc_X).array();
			const Eigen::ArrayXXd dX2 = arc_gX.array() * (arc_X_lo - arc_X).array();
			const Eigen::ArrayXXd dY = arc_gY.array() * (arc_Y_hi - arc_Y).array();
			const Eigen::ArrayXXd dY2 = arc_gY.array() * (arc_Y_lo - arc_Y).array();
			const Eigen::VectorXd peak = arc_T + (dX.max(dX2) + dY.max(dY2)).rowwise().sum().matrix();
			if (n > 0 && peak.maxCoeff() > model.max_travel_time)
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {}: an arc's linearized travel time reaches {}, above its "
					"model's max_travel_time {}", j, peak.maxCoeff(), model.max_travel_time));
		}

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
				case WaypointObjective::kMinMaxTravelTime:
				case WaypointObjective::kAvgTravelTime: {
					// Perspective of T(x̄, ȳ) + gX·(x - x̄) + gY·(y - ȳ).
					const int r = arc_row[a];
					if (r < 0) break;
					dist = Expression(Z(a)) * (arc_T(r) - arc_gX.row(r).dot(arc_X.row(r)) -
								   arc_gY.row(r).dot(arc_Y.row(r)));
					for (int c = 0; c < arc_X.cols(); ++c)
						dist += arc_gX(r, c) * X[a](c) + arc_gY(r, c) * Y[a](c);
					break;
				}
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
	case WaypointObjective::kMinMaxTravelTime:
		prog.AddLinearCost(MinMaxObjectiveCost(L_ms_var(0), total_cost, length_weight));
		break;
	case WaypointObjective::kAvgL1:
	case WaypointObjective::kAvgL2:
	case WaypointObjective::kAvgTravelTime:
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
			       SubtourElimination subtour,
			       double length_weight,
			       double mip_gap,
			       bool warm_start,
			       std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models,
			       double max_tangent_halfwidth)
	: GraphWaypointMPC(graph, std::move(splines), objective),
	  _solver(solver),
	  _enforce_rigidity(enforce_rigidity),
	  _subtour(subtour),
	  _length_weight(length_weight),
	  _mip_gap(mip_gap),
	  _warm_start(warm_start),
	  _travel_time_models(std::move(travel_time_models)),
	  _max_tangent_halfwidth(max_tangent_halfwidth) {
	CheckLengthWeight(objective, length_weight);
	const bool travel_time = objective == WaypointObjective::kMinMaxTravelTime ||
				 objective == WaypointObjective::kAvgTravelTime;
	if (!travel_time && !_travel_time_models.empty())
		throw std::runtime_error("GCSWaypointMPC: travel_time_models given, but the objective isn't a travel-time one");
	if (travel_time) {
		if (static_cast<int>(_travel_time_models.size()) != graph.num_agents)
			throw std::runtime_error(fmt::format(
				"GCSWaypointMPC: travel-time objectives need one ArcTravelTimeModel per agent ({}), got {}",
				graph.num_agents, _travel_time_models.size()));
		for (int j = 0; j < graph.num_agents; ++j) {
			if (!_travel_time_models[j])
				throw std::runtime_error(fmt::format("GCSWaypointMPC: agent {}'s travel-time model is None", j));
			if (_travel_time_models[j]->dim() > graph.robot_ambient_dim(j))
				throw std::runtime_error(fmt::format(
					"GCSWaypointMPC: agent {}'s travel-time model reads {} columns, but its row has {}",
					j, _travel_time_models[j]->dim(), graph.robot_ambient_dim(j)));
		}
	}
}

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

GCSWaypointProblem GCSWaypointMPC::BuildProblem(const std::vector<int>& remaining_vertices,
						const Eigen::VectorXd& x0) const {
	GCSWaypointProblem problem = BuildGCSWaypointProblem(
		_graph, _splines, remaining_vertices, x0, _waypoints, _var_assignments,
		_enforce_rigidity, _objective, _subtour, _length_weight, _travel_time_models, _max_tangent_halfwidth);
	if (_warm_start) SetWarmStart(problem, remaining_vertices);
	return problem;
}

const ExternalMIP& GCSWaypointMPC::BeginExternalSolve(const std::vector<int>& remaining_vertices,
						const Eigen::VectorXd& x0) {
	BeginSolve(remaining_vertices, x0);
	auto pending = std::make_unique<PendingExternalSolve>();
	pending->problem = BuildProblem(remaining_vertices, x0);
	pending->remaining_vertices = remaining_vertices;
	pending->x0 = x0;
	const MathematicalProgram& prog = *pending->problem.prog;

	const auto linear = prog.GetAllLinearConstraints();
	std::set<std::string> unsupported;
	for (const auto& b : prog.GetAllConstraints()) {
		const auto* e = b.evaluator().get();
		if (!dynamic_cast<const drake::solvers::LinearConstraint*>(e) &&
		    !dynamic_cast<const drake::solvers::LorentzConeConstraint*>(e) &&
		    !dynamic_cast<const drake::solvers::ExpressionConstraint*>(e))
			unsupported.insert(drake::NiceTypeName::Get(*e));
	}
	for (const auto& b : prog.GetAllCosts())
		if (!dynamic_cast<const drake::solvers::LinearCost*>(b.evaluator().get()))
			unsupported.insert(drake::NiceTypeName::Get(*b.evaluator()));
	if (!unsupported.empty())
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC::BeginExternalSolve: only linear, Lorentz cone and expression constraints, "
			"bounding boxes and linear costs export to ExternalMIP; program has {}", fmt::join(unsupported, ", ")));

	ExternalMIP& mip = pending->mip;
	const int n = prog.num_vars();
	mip.vars = prog.decision_variables();

	for (const auto& b : prog.generic_constraints()) {
		const auto* ec = dynamic_cast<const drake::solvers::ExpressionConstraint*>(b.evaluator().get());
		if (!ec) continue;
		Substitution to_binding;
		for (int k = 0; k < ec->vars().size(); ++k)
			if (!ec->vars()(k).equal_to(b.variables()(k))) to_binding[ec->vars()(k)] = b.variables()(k);
		for (int r = 0; r < ec->num_constraints(); ++r)
			mip.nonlinear.push_back({ec->expressions()(r).Substitute(to_binding),
						 ec->lower_bound()(r), ec->upper_bound()(r)});
	}

	for (const auto& b : prog.lorentz_cone_constraints()) {
		const auto& A = b.evaluator()->A();
		const std::vector<int> idx = prog.FindDecisionVariableIndices(b.variables());
		std::vector<Eigen::Triplet<double>> cone_triplets;
		for (int k = 0; k < A.outerSize(); ++k)
			for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
				cone_triplets.emplace_back(it.row(), idx[it.col()], it.value());
		ExternalLorentzCone& cone = mip.cones.emplace_back();
		cone.A.resize(A.rows(), n);
		cone.A.setFromTriplets(cone_triplets.begin(), cone_triplets.end());
		cone.b = b.evaluator()->b();
	}

	std::vector<Eigen::Triplet<double>> triplets;
	std::vector<double> lb, ub;
	for (const auto& b : linear) {
		const auto& A = b.evaluator()->get_sparse_A();
		const std::vector<int> idx = prog.FindDecisionVariableIndices(b.variables());
		const int r0 = static_cast<int>(lb.size());
		for (int k = 0; k < A.outerSize(); ++k)
			for (Eigen::SparseMatrix<double>::InnerIterator it(A, k); it; ++it)
				triplets.emplace_back(r0 + it.row(), idx[it.col()], it.value());
		const auto& l = b.evaluator()->lower_bound();
		const auto& u = b.evaluator()->upper_bound();
		lb.insert(lb.end(), l.data(), l.data() + l.size());
		ub.insert(ub.end(), u.data(), u.data() + u.size());
	}
	mip.A.resize(static_cast<int>(lb.size()), n);
	mip.A.setFromTriplets(triplets.begin(), triplets.end());
	mip.row_lb = Eigen::Map<const Eigen::VectorXd>(lb.data(), lb.size());
	mip.row_ub = Eigen::Map<const Eigen::VectorXd>(ub.data(), ub.size());

	drake::solvers::AggregateBoundingBoxConstraints(prog, &mip.var_lb, &mip.var_ub);

	mip.c = Eigen::VectorXd::Zero(n);
	for (const auto& b : prog.linear_costs()) {
		const std::vector<int> idx = prog.FindDecisionVariableIndices(b.variables());
		const Eigen::VectorXd& a = b.evaluator()->a();
		for (int k = 0; k < a.size(); ++k) mip.c(idx[k]) += a(k);
		mip.c0 += b.evaluator()->b();
	}

	mip.is_integer.resize(n);
	for (int i = 0; i < n; ++i) {
		const auto type = prog.decision_variable(i).get_type();
		using VarType = drake::symbolic::Variable::Type;
		if (type != VarType::CONTINUOUS && type != VarType::BINARY && type != VarType::INTEGER)
			throw std::runtime_error(fmt::format(
				"GCSWaypointMPC::BeginExternalSolve: unsupported type of variable {}",
				prog.decision_variable(i).get_name()));
		mip.is_integer(i) = type != VarType::CONTINUOUS;
		if (type == VarType::BINARY) {
			mip.var_lb(i) = std::max(mip.var_lb(i), 0.0);
			mip.var_ub(i) = std::min(mip.var_ub(i), 1.0);
		}
	}
	mip.initial_guess = prog.initial_guess();
	mip.nonlinear_seed = Eigen::VectorXd::Constant(n, std::numeric_limits<double>::quiet_NaN());
	for (const auto& [v, value] : pending->problem.nonlinear_seed)
		mip.nonlinear_seed(prog.FindDecisionVariableIndex(v)) = value;

	_pending_external = std::move(pending);
	return _pending_external->mip;
}

void GCSWaypointMPC::FinishExternalSolve(const Eigen::VectorXd& x, double tol) {
	if (!_pending_external)
		throw std::runtime_error("GCSWaypointMPC::FinishExternalSolve: no BeginExternalSolve pending");
	const std::unique_ptr<PendingExternalSolve> pending = std::move(_pending_external);
	const ExternalMIP& mip = pending->mip;
	const MathematicalProgram& prog = *pending->problem.prog;
	if (x.size() != prog.num_vars())
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC::FinishExternalSolve: x has {} entries, program has {} variables",
			x.size(), prog.num_vars()));

	const Eigen::VectorXd Ax = mip.A * x;
	const Eigen::VectorXd row_violation =
		(mip.row_lb - Ax).cwiseMax(Ax - mip.row_ub).cwiseMax(0.0);
	const Eigen::VectorXd var_violation =
		(mip.var_lb - x).cwiseMax(x - mip.var_ub).cwiseMax(0.0);
	Eigen::Index worst_row = 0, worst_var = 0;
	const double max_row = row_violation.size() ? row_violation.maxCoeff(&worst_row) : 0.0;
	const double max_var = var_violation.size() ? var_violation.maxCoeff(&worst_var) : 0.0;
	double max_cone = 0.0;
	size_t worst_cone = 0;
	for (size_t k = 0; k < mip.cones.size(); ++k) {
		const Eigen::VectorXd z = mip.cones[k].A * x + mip.cones[k].b;
		const double violation = z.tail(z.size() - 1).norm() - z(0);
		if (violation > max_cone) { max_cone = violation; worst_cone = k; }
	}
	double max_nonlinear = 0.0;
	size_t worst_nonlinear = 0;
	if (!mip.nonlinear.empty()) {
		drake::symbolic::Environment env;
		for (int i = 0; i < x.size(); ++i) env.insert(mip.vars(i), x(i));
		for (size_t k = 0; k < mip.nonlinear.size(); ++k) {
			const ExternalNonlinear& nl = mip.nonlinear[k];
			const double val = nl.e.Evaluate(env);
			const double violation = std::max({nl.lb - val, val - nl.ub, 0.0});
			if (violation > max_nonlinear) { max_nonlinear = violation; worst_nonlinear = k; }
		}
	}
	if (max_row > tol || max_var > tol || max_cone > tol || max_nonlinear > tol)
		throw std::runtime_error(fmt::format(
			"GCSWaypointMPC::FinishExternalSolve: x violates the problem beyond tol {}: "
			"row {} by {}, bound on {} by {}, cone {} by {}, nonlinear row {} by {}",
			tol, worst_row, max_row, prog.decision_variable(worst_var).get_name(), max_var,
			worst_cone, max_cone, worst_nonlinear, max_nonlinear));

	MathematicalProgramResult result;
	result.set_decision_variable_index(prog.decision_variable_index());
	result.set_x_val(x);
	result.set_solution_result(drake::solvers::SolutionResult::kSolutionFound);
	result.set_optimal_cost(mip.c.dot(x) + mip.c0);

	_last_solve_time = _timer.Tick();
	StoreSolution(pending->problem, result, pending->remaining_vertices, pending->x0);
}

bool GCSWaypointMPC::SolveWith(const drake::solvers::SolverInterface& solver,
			       const std::vector<int>& remaining_vertices,
			       const Eigen::VectorXd& x0) {

	GCSWaypointProblem problem = BuildProblem(remaining_vertices, x0);
	if (!problem.prog->generic_constraints().empty())
		throw std::runtime_error(
			"GCSWaypointMPC: the graph has nonlinear node constraints, which Gurobi/Mosek can't "
			"solve here -- use SCIPWaypointMPC");

	drake::solvers::SolverOptions options;
	options.SetOption(drake::solvers::GurobiSolver::id(), "MIPGap", _mip_gap);
	options.SetOption(drake::solvers::MosekSolver::id(), "MSK_DPAR_MIO_TOL_REL_GAP", _mip_gap);

	MathematicalProgramResult result;
	solver.Solve(*problem.prog, std::nullopt, options, &result);

	if (!result.is_success()) {
		std::cerr << "GCSWaypointMPC: no feasible solution.\n";
		PrintSolverReport(*problem.prog, result, 1e-6);
		return false;
	}

	_last_solve_time = _timer.Tick();
	StoreSolution(problem, result, remaining_vertices, x0);
	return true;
}

void GCSWaypointMPC::StoreSolution(const GCSWaypointProblem& problem,
				   const MathematicalProgramResult& result,
				   const std::vector<int>& remaining_vertices,
				   const Eigen::VectorXd& x0) {
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

	std::vector<int> row_var(subgraph.num_variables(), -1);
	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) row_var[row] = var;

	// Each agent's route; its visited rows from the vertex copies, the rest
	// interpolated per block (linear spline: geodesic per block) in time
	// between its neighbouring visits, holding after the last.
	_routes.assign(_graph->num_agents, {});
	_prev_route_modes.assign(_graph->num_agents, {});
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
			const int mode = g.vertices[next].mode;
			_prev_route_modes[j].push_back(
				{node, mode == GCSModeGraph::kNoRole ? GCSModeGraph::kNoRole : row_var[mode]});
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
}

void GCSWaypointMPC::SetWarmStart(const GCSWaypointProblem& problem,
				  const std::vector<int>& remaining_vertices) const {
	if (_prev_route_modes.empty()) return;
	MathematicalProgram& prog = *problem.prog;
	const SubgraphOfConstraints& subgraph = *problem.subgraph;
	std::vector<int> node_of_sg(subgraph.num_nodes(), -1);
	for (int v : remaining_vertices) node_of_sg[subgraph.subgraph_id(v)] = v;
	std::vector<int> row_var(subgraph.num_variables(), -1);
	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) row_var[row] = var;

	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) {
		if (var >= _var_assignments.size() || _var_assignments(var) < 0) continue;
		for (int k = 0; k < _graph->num_agents; ++k)
			prog.SetInitialGuess(problem.Assignments(row, k), k == _var_assignments(var) ? 1.0 : 0.0);
	}

	for (int j = 0; j < _graph->num_agents; ++j) {
		const GCSModeGraph& g = problem.graphs[j];
		std::map<std::pair<int, int>, int> vertex_of;
		for (size_t v = 0; v < g.vertices.size(); ++v) {
			const int mode = g.vertices[v].mode;
			const int var = mode == GCSModeGraph::kNoRole ? GCSModeGraph::kNoRole : row_var[mode];
			vertex_of[{node_of_sg[g.vertices[v].sg], var}] = v;
		}

		std::vector<int> chain = {GCSModeGraph::kDepot};
		for (const auto& key : _prev_route_modes[j]) {
			const auto it = vertex_of.find(key);
			if (it != vertex_of.end()) chain.push_back(it->second);
		}
		chain.push_back(GCSModeGraph::kSink);

		std::map<std::pair<int, int>, int> arc_of;
		for (size_t a = 0; a < g.arcs.size(); ++a) arc_of[g.arcs[a]] = a;
		Eigen::VectorXd z = Eigen::VectorXd::Zero(g.arcs.size());
		bool complete = true;
		for (size_t k = 0; k + 1 < chain.size(); ++k) {
			const auto it = arc_of.find({chain[k], chain[k + 1]});
			if (it == arc_of.end()) { complete = false; break; }
			z(it->second) = 1.0;
		}
		if (complete) prog.SetInitialGuess(problem.Z[j], z);
	}
}
