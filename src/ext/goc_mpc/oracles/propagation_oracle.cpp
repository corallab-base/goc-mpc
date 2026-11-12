#include "propagation_oracle.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <drake/common/symbolic/decompose.h>

#include "../graph_of_constraints.hpp"
#include "../waypoint_common.hpp"
#include "../../utils.hpp"

namespace {

using drake::symbolic::Environment;
using drake::symbolic::Expression;
using drake::symbolic::Formula;
using drake::symbolic::FormulaKind;
using drake::symbolic::Substitution;
using drake::symbolic::Variable;
using drake::symbolic::Variables;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();
// GCSWaypointMPC's band on a static hold's endpoint rows.
constexpr double kHoldBand = 1e-3;

// (global assignable variable id, agent).
using Pair = std::pair<int, int>;

// One scalar unknown: an object's state component, an agent's
// configuration component, or a component of an agent's FK link pose
// (position, then the row-major rotation), at a subgraph node.
struct Cell {
	enum Kind { kObject, kConfig, kLink };
	Kind kind;
	int sg, owner, component;
	std::string link;
	double lb = -kInf, ub = kInf;
};

// sum coefficient * cell + constant, == 0 or <= 0.
struct Relation {
	std::vector<std::pair<int, double>> terms;
	double constant = 0.0;
	bool equality = true;
	double tolerance = 0.0;
	// False for a static hold's band: its values prove nothing.
	bool exact = true;
	std::set<Pair> deps;
	std::string where;
};

// An atom the oracle doesn't solve, only evaluates: e == 0 or e <= 0, its
// placeholders bound to cells.
struct Check {
	Expression e;
	bool equality;
	std::vector<std::pair<Variable, int>> bind;
	std::string where;
};

// A value and what it was derived from: assignment pairs, choices (by
// creation order in its run), and whether an inexact relation was used.
struct Value {
	double v = kNaN;
	bool known = false;
	bool inexact = false;
	std::set<Pair> deps;
	std::set<int> choices;
};

// A failed run: the choices and assignment pairs it was derived from.
struct Failure {
	std::set<int> conflict;
	std::set<Pair> deps;
	bool inexact = false;
	std::string what;
};

struct ChoiceInfo {
	int count = 1;
	// Its options are every possibility (an IK branch set), so exhausting
	// it proves its conflict.
	bool exhaustive = false;
};

struct Outcome {
	bool failed = false;
	Failure failure;
	std::vector<ChoiceInfo> choices;
	std::vector<std::string> log, violations;
	double violation = 0.0;
	Eigen::MatrixXd X;
	std::vector<std::vector<int>> routes;
	std::vector<CandidateSet> candidates;
};

// One placeholder resolved at a node: a cell, a constant, or neither (the
// atom is dropped: its agent doesn't visit / its variable has no row) or
// unreadable.
struct Resolved {
	enum Kind { kCell, kConstant, kSkip, kUnreadable } kind = kUnreadable;
	int cell = -1;
	double constant = kNaN;
	std::optional<Pair> dep;
};

// Flattens f's top-level conjunction.
void Atoms(const Formula& f, std::vector<Formula>* out) {
	if (f.get_kind() == FormulaKind::And) {
		for (const auto& sub : drake::symbolic::get_operands(f)) Atoms(sub, out);
	} else {
		out->push_back(f);
	}
}

double RadicalInverse(int i, int base) {
	double f = 1.0, r = 0.0;
	for (; i > 0; i /= base) {
		f /= base;
		r += f * (i % base);
	}
	return r;
}

template <typename T>
void Merge(std::set<T>* into, const std::set<T>& from) {
	into->insert(from.begin(), from.end());
}

// An assignment's relations, read once; Run propagates them for one
// setting of the choices.
class Model {
public:
	struct Context {
		const GraphOfConstraints& g;
		const SubgraphOfConstraints& sub;
		const RoutingPresolve& pre;
		const std::vector<int>& node_of;
		const std::vector<int>& rank;
		const InitializationRequest& req;
		const std::map<int, std::shared_ptr<IkSolver>>& ik;
		const CollisionChecker* checker;
		int samples;
		double tol;
	};

	Model(const Context& context, const std::map<int, int>& assignment)
		: c(context), g(context.g), sub(context.sub), asg(assignment), wd(context.g.workspace_dim),
		  n_V(context.sub.num_nodes()) {
		visits.resize(n_V);
		for (int sg = 0; sg < n_V; ++sg) {
			visits[sg] = c.pre.required_at[sg];
			for (int var : c.pre.roles_at[sg]) {
				const auto it = asg.find(var);
				if (it != asg.end()) visits[sg].insert(it->second);
			}
		}
		order.resize(n_V);
		for (int sg = 0; sg < n_V; ++sg) order[sg] = sg;
		std::sort(order.begin(), order.end(), [&](int a, int b) { return c.rank[a] < c.rank[b]; });
		for (int p = 0; p < g.num_params(); ++p) {
			const auto& vars = g._param.Vars(p);
			for (int k = 0; k < vars.size(); ++k) params[vars(k)] = g.view_param_values()(p);
		}
		for (int j = 0; j < g.num_agents; ++j) agent_ph.push_back(g._agent_q.Vars(j));
		for (int sg = 0; sg < n_V; ++sg)
			for (int j : visits[sg])
				for (int k = 0; k < g.robot_ambient_dim(j); ++k) CellId(Cell::kConfig, sg, j, k);
		for (int sg = 0; sg < n_V; ++sg)
			for (int o = 0; o < g.num_objects; ++o)
				for (int k = 0; k < g.object_ambient_dim(o); ++k) CellId(Cell::kObject, sg, o, k);
		ReadConstraints();
	}

	Outcome Run(const std::vector<int>& idx) const;

	std::vector<std::string> unresolved;

private:
	friend class Runner;
	const Context& c;
	const GraphOfConstraints& g;
	const SubgraphOfConstraints& sub;
	const std::map<int, int> asg;
	const int wd, n_V;

	std::vector<std::set<int>> visits;
	// Subgraph nodes in the order of the hard orderings.
	std::vector<int> order;
	Substitution params;
	std::vector<drake::VectorX<Variable>> agent_ph;
	std::vector<Cell> cells;
	std::map<std::tuple<int, int, int, std::string, int>, int> index;
	// Per (sg, agent): the FK links with cells.
	std::map<std::pair<int, int>, std::set<std::string>> links;
	std::vector<Relation> relations;
	std::vector<Check> checks;

	int LinkWidth() const { return wd + wd * wd; }

	int Find(Cell::Kind kind, int sg, int owner, int component, const std::string& link = "") const {
		const auto it = index.find(std::make_tuple(static_cast<int>(kind), sg, owner, link, component));
		return it == index.end() ? -1 : it->second;
	}

	int CellId(Cell::Kind kind, int sg, int owner, int component, const std::string& link = "") {
		const int found = Find(kind, sg, owner, component, link);
		if (found >= 0) return found;
		Cell cell{kind, sg, owner, component, link};
		if (kind == Cell::kObject) {
			cell.lb = g._global_x_lb(g.object_col_offset(owner) + component);
			cell.ub = g._global_x_ub(g.object_col_offset(owner) + component);
		} else if (kind == Cell::kConfig) {
			cell.lb = g._global_x_lb(g.agent_col_offset(owner) + component);
			cell.ub = g._global_x_ub(g.agent_col_offset(owner) + component);
		} else {
			links[{sg, owner}].insert(link);
		}
		cells.push_back(cell);
		const int id = static_cast<int>(cells.size()) - 1;
		index.emplace(std::make_tuple(static_cast<int>(kind), sg, owner, link, component), id);
		return id;
	}

	std::string Name(int id) const {
		const Cell& cell = cells[id];
		const int node = c.node_of[cell.sg];
		switch (cell.kind) {
		case Cell::kObject:
			return fmt::format("node {} object {}[{}]", node, cell.owner, cell.component);
		case Cell::kConfig:
			return fmt::format("node {} agent {} q[{}]", node, cell.owner, cell.component);
		default:
			return cell.component < wd
				? fmt::format("node {} agent {} {} pos[{}]", node, cell.owner, cell.link, cell.component)
				: fmt::format("node {} agent {} {} rot[{}]", node, cell.owner, cell.link, cell.component - wd);
		}
	}

	///////////////////////////////////////////////////////////////////////
	//                            Reading                                //
	///////////////////////////////////////////////////////////////////////

	Resolved Agent(int sg, int j, int k) {
		if (!visits[sg].contains(j)) return {Resolved::kSkip};
		return {Resolved::kCell, CellId(Cell::kConfig, sg, j, k)};
	}
	Resolved Link(int sg, int j, const std::string& link, int k) {
		if (!visits[sg].contains(j)) return {Resolved::kSkip};
		return {Resolved::kCell, CellId(Cell::kLink, sg, j, k, link)};
	}
	Resolved Holder(int var, const std::function<Resolved(int)>& at) {
		if (sub.subgraph_variable_id(var) < 0) return {Resolved::kSkip};
		const auto it = asg.find(var);
		if (it == asg.end()) return {Resolved::kUnreadable};
		Resolved r = at(it->second);
		r.dep = Pair{var, it->second};
		return r;
	}

	// A plain (node-side) placeholder at sg.
	Resolved Plain(const Variable& v, int sg) {
		if (const auto hit = g._object_q.Find(v))
			return {Resolved::kCell, CellId(Cell::kObject, sg, hit->first, hit->second)};
		if (const auto hit = g._agent_q.Find(v)) return Agent(sg, hit->first, hit->second);
		if (const auto hit = g._var_agent_q.Find(v))
			return Holder(hit->first, [&](int j) { return Agent(sg, j, hit->second); });
		if (const auto hit = g._agent_link_pos.Find(v))
			return Link(sg, hit->first.first, hit->first.second, hit->second);
		if (const auto hit = g._agent_link_rot.Find(v))
			return Link(sg, hit->first.first, hit->first.second, wd + hit->second);
		if (const auto hit = g._var_agent_link_pos.Find(v))
			return Holder(hit->first.first, [&](int j) { return Link(sg, j, hit->first.second, hit->second); });
		if (const auto hit = g._var_agent_link_rot.Find(v))
			return Holder(hit->first.first,
				      [&](int j) { return Link(sg, j, hit->first.second, wd + hit->second); });
		return {Resolved::kUnreadable};
	}

	// A relational edge placeholder: u_/v_ families at their endpoint, x0
	// for an endpoint before this horizon.
	Resolved Relational(const Variable& v, int sg_u, int sg_v) {
		auto agent = [&](int sg, int j, int k) -> Resolved {
			if (sg < 0) return {Resolved::kConstant, -1, c.req.x0(g.agent_col_offset(j) + k)};
			return Agent(sg, j, k);
		};
		auto object = [&](int sg, int o, int k) -> Resolved {
			if (sg < 0) return {Resolved::kConstant, -1, c.req.x0(g.object_col_offset(o) + k)};
			return {Resolved::kCell, CellId(Cell::kObject, sg, o, k)};
		};
		if (const auto hit = g._agent_q_u.Find(v)) return agent(sg_u, hit->first, hit->second);
		if (const auto hit = g._agent_q_v.Find(v)) return agent(sg_v, hit->first, hit->second);
		if (const auto hit = g._object_q_u.Find(v)) return object(sg_u, hit->first, hit->second);
		if (const auto hit = g._object_q_v.Find(v)) return object(sg_v, hit->first, hit->second);
		if (const auto hit = g._var_agent_q_u.Find(v))
			return Holder(hit->first, [&](int j) { return agent(sg_u, j, hit->second); });
		if (const auto hit = g._var_agent_q_v.Find(v))
			return Holder(hit->first, [&](int j) { return agent(sg_v, j, hit->second); });
		return {Resolved::kUnreadable};
	}

	void AddFormula(const Formula& f_raw, const std::function<Resolved(const Variable&)>& resolve, bool exact,
			const std::string& where) {
		std::vector<Formula> atoms;
		Atoms(f_raw.Substitute(params), &atoms);
		for (const Formula& atom : atoms) {
			Expression e;
			bool equality = false;
			switch (atom.get_kind()) {
			case FormulaKind::True:
				continue;
			case FormulaKind::Eq:
				e = drake::symbolic::get_lhs_expression(atom) - drake::symbolic::get_rhs_expression(atom);
				equality = true;
				break;
			case FormulaKind::Leq:
			case FormulaKind::Lt:
				e = drake::symbolic::get_lhs_expression(atom) - drake::symbolic::get_rhs_expression(atom);
				break;
			case FormulaKind::Geq:
			case FormulaKind::Gt:
				e = drake::symbolic::get_rhs_expression(atom) - drake::symbolic::get_lhs_expression(atom);
				break;
			default:
				unresolved.push_back(fmt::format("{}: {}", where, atom.to_string()));
				continue;
			}
			Substitution constants;
			std::vector<std::pair<Variable, int>> bind;
			std::set<Pair> deps;
			bool skip = false, unreadable = false;
			for (const Variable& v : e.GetVariables()) {
				const Resolved r = resolve(v);
				if (r.kind == Resolved::kSkip) { skip = true; break; }
				if (r.kind == Resolved::kUnreadable) { unreadable = true; break; }
				if (r.dep) deps.insert(*r.dep);
				if (r.kind == Resolved::kConstant) { constants[v] = r.constant; continue; }
				bind.emplace_back(v, r.cell);
			}
			if (skip) continue;
			if (unreadable) {
				unresolved.push_back(fmt::format("{}: {}", where, atom.to_string()));
				continue;
			}
			const Expression ec = constants.empty() ? e : e.Substitute(constants);
			drake::VectorX<Variable> vars(bind.size());
			for (size_t k = 0; k < bind.size(); ++k) vars(k) = bind[k].first;
			if (!drake::symbolic::IsAffine(drake::Vector1<Expression>(ec), Variables(vars))) {
				checks.push_back({ec, equality, bind, where});
				continue;
			}
			Eigen::MatrixXd A(1, vars.size());
			Eigen::VectorXd b(1);
			drake::symbolic::DecomposeAffineExpressions(drake::Vector1<Expression>(ec), vars, &A, &b);
			std::map<int, double> terms;
			for (size_t k = 0; k < bind.size(); ++k)
				if (A(0, k) != 0.0) terms[bind[k].second] += A(0, k);
			Relation rel;
			for (const auto& [cell, a] : terms)
				if (a != 0.0) rel.terms.emplace_back(cell, a);
			rel.constant = b(0);
			rel.equality = equality;
			rel.tolerance = exact ? c.tol : kHoldBand;
			rel.exact = exact;
			rel.deps = deps;
			rel.where = where;
			if (!equality && rel.terms.size() == 1) {
				Cell& cell = cells[rel.terms[0].first];
				const double a = rel.terms[0].second, bound = -rel.constant / a;
				if (a > 0.0) cell.ub = std::min(cell.ub, bound);
				else cell.lb = std::max(cell.lb, bound);
			}
			relations.push_back(std::move(rel));
		}
	}

	// The holder's end effector at sg: its "ee" FK link's position cell, or
	// a free body's leading column (zero beyond them).
	std::vector<std::pair<int, double>> EECells(int sg, int h, int k) {
		if (g.robot_symbolic_fk.contains({h, "ee"})) return {{CellId(Cell::kLink, sg, h, k, "ee"), 1.0}};
		if (k < std::min(EEPositionWidth(&g, h), g.robot_ambient_dim(h))) return {{CellId(Cell::kConfig, sg, h, k), 1.0}};
		return {};
	}
	// ... and at x0.
	double EEAtX0(int h, int k) {
		const auto it = g.robot_symbolic_fk.find({h, "ee"});
		if (it == g.robot_symbolic_fk.end())
			return k < std::min(EEPositionWidth(&g, h), g.robot_ambient_dim(h)) ? c.req.x0(g.agent_col_offset(h) + k)
											      : 0.0;
		Environment env;
		for (int q = 0; q < agent_ph[h].size(); ++q) env.insert(agent_ph[h](q), c.req.x0(g.agent_col_offset(h) + q));
		return k < it->second.position.size() ? it->second.position(k).Evaluate(env) : 0.0;
	}

	void ReadConstraints() {
		for (const auto& [phi_id, rec] : sub.get_subgraph_symbolic_ops()) {
			const int sg = sub.subgraph_id(rec.node);
			AddFormula(rec.formula, [&](const Variable& v) { return Plain(v, sg); }, true,
				   fmt::format("node constraint {} at node {}", phi_id, rec.node));
		}
		for (const auto& [phi_id, rec] : sub.get_subgraph_symbolic_edge_ops()) {
			const std::string where = fmt::format("edge constraint {} on ({}, {})", phi_id, rec.u_node, rec.v_node);
			const int sg_u = sub.subgraph_id(rec.u_node), sg_v = sub.subgraph_id(rec.v_node);
			if (rec.along_edge) {
				for (const int sg : {sg_u, sg_v})
					if (sg >= 0) AddFormula(rec.formula, [&](const Variable& v) { return Plain(v, sg); }, true, where);
			} else if (sg_v >= 0) {
				AddFormula(rec.formula, [&](const Variable& v) { return Relational(v, sg_u, sg_v); }, sg_u >= 0,
					   where);
			}
		}
		ReadHolds();
		ReadStationarity();
	}

	void ReadHolds() {
		for (const auto& [hold_id, hold] : sub.get_subgraph_hold_ops()) {
			const std::string where = fmt::format("hold {} on ({}, {})", hold_id, hold.u_node, hold.v_node);
			const int sg_v = sub.subgraph_id(hold.v_node);
			if (sg_v < 0) continue;
			const int sg_u = sub.subgraph_id(hold.u_node);
			if (hold.rigid) {
				unresolved.push_back(where + ": rigid holds aren't read");
				continue;
			}
			int h = -1;
			std::set<Pair> deps;
			if (hold.robot_ag) {
				h = *hold.robot_ag;
			} else {
				const auto it = asg.find(*hold.var_id);
				if (it == asg.end()) continue;
				h = it->second;
				deps.insert({*hold.var_id, h});
			}
			if (!visits[sg_v].contains(h) || (sg_u >= 0 && !visits[sg_u].contains(h))) continue;
			if (!g.robot_symbolic_fk.contains({h, "ee"}) && !g.robot_is_free_body(h)) {
				unresolved.push_back(fmt::format("{}: holder {} has no \"ee\" FK", where, h));
				continue;
			}
			for (int o : hold.held_point_ids) {
				const int w_o = g.object_ambient_dim(o);
				for (int k = 0; k < wd; ++k) {
					// obj_v - ee_v - (obj_u - ee_u) = 0.
					Relation rel;
					if (k < w_o) rel.terms.emplace_back(CellId(Cell::kObject, sg_v, o, k), 1.0);
					for (auto [cell, a] : EECells(sg_v, h, k)) rel.terms.emplace_back(cell, -a);
					if (sg_u >= 0) {
						if (k < w_o) rel.terms.emplace_back(CellId(Cell::kObject, sg_u, o, k), -1.0);
						for (auto [cell, a] : EECells(sg_u, h, k)) rel.terms.emplace_back(cell, a);
					} else {
						rel.constant = -(k < w_o ? c.req.x0(g.object_col_offset(o) + k) : 0.0) + EEAtX0(h, k);
					}
					rel.exact = !hold.robot_ag.has_value();
					rel.tolerance = rel.exact ? c.tol : kHoldBand;
					rel.deps = deps;
					rel.where = where;
					relations.push_back(std::move(rel));
				}
			}
		}
	}

	// AddStationaryObjectConstraints' ungated rows: an object no hold may
	// move over a structure edge (or from x0 to a source) stays put.
	void ReadStationarity() {
		std::map<int, std::vector<int>> children;
		for (auto edge : sub.structure.edges()) children[edge.u].push_back(edge.e->to);
		std::map<int, std::set<int>> reach;
		auto precedes = [&](int a, int b) {
			auto it = reach.find(a);
			if (it == reach.end()) {
				std::set<int> seen{a};
				std::vector<int> stack{a};
				while (!stack.empty()) {
					const int n = stack.back();
					stack.pop_back();
					for (int ch : children[n])
						if (seen.insert(ch).second) stack.push_back(ch);
				}
				it = reach.emplace(a, std::move(seen)).first;
			}
			return it->second.contains(b);
		};
		auto span = [&](int node_a, int node_b) {
			const int sg_a = node_a >= 0 ? sub.subgraph_id(node_a) : -1;
			const int sg_b = sub.subgraph_id(node_b);
			for (int o = 0; o < g.num_objects; ++o) {
				bool gated = false;
				for (const auto& [hold_id, hold] : g.hold_ops) {
					if (std::find(hold.held_point_ids.begin(), hold.held_point_ids.end(), o) ==
					    hold.held_point_ids.end())
						continue;
					if (sub.subgraph_id(hold.v_node) < 0) continue;
					if (sub.subgraph_id(hold.u_node) >= 0 && precedes(node_b, hold.u_node)) continue;
					if (node_a >= 0 && precedes(hold.v_node, node_a)) continue;
					gated = true;
					break;
				}
				if (gated) continue;
				for (int k = 0; k < g.object_ambient_dim(o); ++k) {
					Relation rel;
					rel.terms.emplace_back(CellId(Cell::kObject, sg_b, o, k), 1.0);
					if (sg_a >= 0) rel.terms.emplace_back(CellId(Cell::kObject, sg_a, o, k), -1.0);
					else rel.constant = -c.req.x0(g.object_col_offset(o) + k);
					rel.tolerance = c.tol;
					rel.where = node_a >= 0 ? fmt::format("object {} stationary on ({}, {})", o, node_a, node_b)
								: fmt::format("object {} stationary from x0 to {}", o, node_b);
					relations.push_back(std::move(rel));
				}
			}
		};
		for (auto edge : sub.structure.edges()) span(edge.u, edge.e->to);
		for (int v : sub.structure.sources()) span(-1, v);
	}
};

// One propagation of a Model for one setting of its choices.
class Runner {
public:
	Runner(const Model& model, const std::vector<int>& idx) : m(model), idx(idx), values(model.cells.size()) {}

	Outcome Run() {
		Propagate();
		if (!out.failed) {
			FillRest();
			Evaluate();
		}
		const auto& g = m.g;
		const auto& req = m.c.req;
		out.X = Eigen::MatrixXd::Constant(static_cast<int>(req.nodes.size()), g.total_dim, kNaN);
		for (size_t r = 0; r < req.nodes.size(); ++r) {
			const int sg = m.sub.structure.contains_node(req.nodes[r]) ? m.sub.subgraph_id(req.nodes[r]) : -1;
			if (sg < 0) continue;
			for (int o = 0; o < g.num_objects; ++o)
				for (int k = 0; k < g.object_ambient_dim(o); ++k)
					out.X(r, g.object_col_offset(o) + k) = values[m.Find(Cell::kObject, sg, o, k)].v;
			for (int j : m.visits[sg])
				for (int k = 0; k < g.robot_ambient_dim(j); ++k)
					out.X(r, g.agent_col_offset(j) + k) = values[m.Find(Cell::kConfig, sg, j, k)].v;
		}
		out.routes.assign(g.num_agents, {});
		for (int sg : m.order)
			for (int j : m.visits[sg]) out.routes[j].push_back(m.c.node_of[sg]);
		return std::move(out);
	}

private:
	const Model& m;
	const std::vector<int>& idx;
	std::vector<Value> values;
	Outcome out;

	// A derivation: what a set of known cells was derived from.
	struct Source {
		std::set<Pair> deps;
		std::set<int> choices;
		bool inexact = false;
		void Add(const Value& v) {
			Merge(&deps, v.deps);
			Merge(&choices, v.choices);
			inexact = inexact || v.inexact;
		}
	};

	void Set(int id, double v, const Source& source) {
		values[id] = {v, true, source.inexact, source.deps, source.choices};
	}

	void Fail(const Source& source, bool inexact, std::string what) {
		out.failed = true;
		out.failure = {source.choices, source.deps, source.inexact || inexact, what};
		out.log.push_back(std::move(what));
	}

	// A new choice with `count` options: the option this run takes.
	int Choose(int count, bool exhaustive, int* choice) {
		*choice = static_cast<int>(out.choices.size());
		out.choices.push_back({count, exhaustive});
		return *choice < static_cast<int>(idx.size()) ? idx[*choice] : 0;
	}

	// The agent's x0, clamped into its bounds at sg.
	Eigen::VectorXd X0Reference(int sg, int j) const {
		const int d = m.g.robot_ambient_dim(j);
		Eigen::VectorXd q = m.c.req.x0.segment(m.g.agent_col_offset(j), d);
		for (int k = 0; k < d; ++k) {
			const Cell& cell = m.cells[m.Find(Cell::kConfig, sg, j, k)];
			q(k) = std::clamp(q(k), cell.lb, std::max(cell.lb, cell.ub));
		}
		return q;
	}

	// The agent's configuration at sg it would start from: the previous
	// guess there, else x0; clamped into its bounds.
	Eigen::VectorXd Reference(int sg, int j) const {
		const auto& g = m.g;
		const int d = g.robot_ambient_dim(j), col = g.agent_col_offset(j);
		Eigen::VectorXd q = m.c.req.x0.segment(col, d);
		if (const InitialGuess* prev = m.c.req.previous) {
			const auto it = std::find(prev->nodes.begin(), prev->nodes.end(), m.c.node_of[sg]);
			if (it != prev->nodes.end() && prev->X.cols() == g.total_dim) {
				const Eigen::VectorXd p = prev->X.row(it - prev->nodes.begin()).segment(col, d).transpose();
				if (p.allFinite()) q = p;
			}
		}
		for (int k = 0; k < d; ++k) {
			const Cell& cell = m.cells[m.Find(Cell::kConfig, sg, j, k)];
			q(k) = std::clamp(q(k), cell.lb, std::max(cell.lb, cell.ub));
		}
		return q;
	}

	// An object component's sampling interval: its bounds, a little inside
	// (just its reference when unbounded).
	std::pair<double, double> Interval(int id, double reference) const {
		const Cell& cell = m.cells[id];
		if (!std::isfinite(cell.lb) || !std::isfinite(cell.ub) || cell.lb > cell.ub) return {reference, reference};
		const double inset = std::min(0.02, 0.25 * (cell.ub - cell.lb));
		return {cell.lb + inset, cell.ub - inset};
	}

	// An object component's reference: the previous guess, else x0, clamped
	// into its interval.
	double ObjectReference(int id) const {
		const Cell& cell = m.cells[id];
		const auto& g = m.g;
		const int col = g.object_col_offset(cell.owner) + cell.component;
		double v = m.c.req.x0(col);
		if (const InitialGuess* prev = m.c.req.previous) {
			const auto it = std::find(prev->nodes.begin(), prev->nodes.end(), m.c.node_of[cell.sg]);
			if (it != prev->nodes.end() && prev->X.cols() == g.total_dim &&
			    std::isfinite(prev->X(it - prev->nodes.begin(), col)))
				v = prev->X(it - prev->nodes.begin(), col);
		}
		if (std::isfinite(cell.lb) && std::isfinite(cell.ub) && cell.lb <= cell.ub) {
			const auto [lo, hi] = Interval(id, v);
			return std::clamp(v, lo, hi);
		}
		return std::clamp(v, cell.lb, std::max(cell.lb, cell.ub));
	}

	bool BlockKnown(int sg, int j) const {
		for (int k = 0; k < m.g.robot_ambient_dim(j); ++k)
			if (!values[m.Find(Cell::kConfig, sg, j, k)].known) return false;
		return true;
	}

	Eigen::VectorXd Block(int sg, int j) const {
		Eigen::VectorXd q(m.g.robot_ambient_dim(j));
		for (int k = 0; k < q.size(); ++k) q(k) = values[m.Find(Cell::kConfig, sg, j, k)].v;
		return q;
	}

	// Sets agent j's configuration at sg (its unknown components), then
	// every unknown component of its FK links there.
	void SetBlock(int sg, int j, const Eigen::VectorXd& q, const Source& source) {
		for (int k = 0; k < q.size(); ++k) {
			const int id = m.Find(Cell::kConfig, sg, j, k);
			if (!values[id].known) Set(id, q(k), source);
		}
		FillLinks(sg, j, source);
	}

	void FillLinks(int sg, int j, const Source& source) {
		const auto lit = m.links.find({sg, j});
		if (lit == m.links.end()) return;
		Environment env;
		for (int k = 0; k < m.agent_ph[j].size(); ++k)
			env.insert(m.agent_ph[j](k), values[m.Find(Cell::kConfig, sg, j, k)].v);
		for (const std::string& link : lit->second) {
			const auto fk = m.g.robot_symbolic_fk.find({j, link});
			if (fk == m.g.robot_symbolic_fk.end()) continue;
			for (int k = 0; k < m.LinkWidth(); ++k) {
				const int id = m.Find(Cell::kLink, sg, j, k, link);
				if (id < 0 || values[id].known) continue;
				const auto& part = k < m.wd ? fk->second.position : fk->second.rotation;
				const int kk = k < m.wd ? k : k - m.wd;
				if (kk < part.size()) Set(id, part(kk).Evaluate(env), source);
			}
		}
	}

	int Unknowns(const Relation& rel, int* last) const {
		int n = 0;
		for (const auto& [cell, a] : rel.terms)
			if (!values[cell].known) { ++n; *last = cell; }
		return n;
	}

	// Step 1 to a fixed point; whether anything was set.
	bool SolveEqualities() {
		bool any = false;
		for (bool changed = true; changed;) {
			changed = false;
			for (const Relation& rel : m.relations) {
				if (!rel.equality) continue;
				int cell = -1;
				if (Unknowns(rel, &cell) != 1) continue;
				double rest = rel.constant, a = 0.0;
				Source source;
				source.deps = rel.deps;
				source.inexact = !rel.exact;
				for (const auto& [id, coeff] : rel.terms) {
					if (id == cell) { a += coeff; continue; }
					rest += coeff * values[id].v;
					source.Add(values[id]);
				}
				Set(cell, -rest / a, source);
				out.log.push_back(fmt::format("{} = {:.6g} ({})", m.Name(cell), -rest / a, rel.where));
				changed = any = true;
			}
		}
		return any;
	}

	// The agents already placed at sg other than j, and what they came from.
	CollisionScene Scene(int sg, int j, Source* source) const {
		CollisionScene scene;
		scene.node = m.c.node_of[sg];
		for (int k : m.visits[sg]) {
			if (k == j || !BlockKnown(sg, k)) continue;
			scene.others.emplace_back(k, Block(sg, k));
			for (int q = 0; q < m.g.robot_ambient_dim(k); ++q) source->Add(values[m.Find(Cell::kConfig, sg, k, q)]);
		}
		return scene;
	}

	// Step 2: one IK solve. 1: set, 0: none ready, -1: failed.
	int SolveOneIK() {
		for (int sg : m.order) {
			for (int j : m.visits[sg]) {
				const auto solver = m.c.ik.find(j);
				if (solver == m.c.ik.end() || m.wd != 3) continue;
				bool all_unknown = true;
				const int d = m.g.robot_ambient_dim(j);
				for (int k = 0; k < d && all_unknown; ++k)
					all_unknown = !values[m.Find(Cell::kConfig, sg, j, k)].known;
				if (!all_unknown) continue;
				const auto lit = m.links.find({sg, j});
				if (lit == m.links.end()) continue;
				for (const std::string& link : lit->second) {
					std::vector<int> ids;
					for (int k = 0; k < m.LinkWidth(); ++k) {
						const int id = m.Find(Cell::kLink, sg, j, k, link);
						if (id < 0 || !values[id].known) break;
						ids.push_back(id);
					}
					if (static_cast<int>(ids.size()) != m.LinkWidth()) continue;
					return SolveIK(sg, j, link, ids, *solver->second);
				}
			}
		}
		return 0;
	}

	int SolveIK(int sg, int j, const std::string& link, const std::vector<int>& ids, const IkSolver& solver) {
		const int d = m.g.robot_ambient_dim(j), node = m.c.node_of[sg];
		IkRequest request;
		request.agent = j;
		request.link = link;
		Eigen::Matrix3d R;
		for (int r = 0; r < 3; ++r)
			for (int k = 0; k < 3; ++k) R(r, k) = values[ids[3 + 3 * r + k]].v;
		request.world_T_link.linear() = R;
		const Eigen::Vector3d p(values[ids[0]].v, values[ids[1]].v, values[ids[2]].v);
		request.world_T_link.translation() = p;
		request.reference = X0Reference(sg, j);
		request.lower.resize(d);
		request.upper.resize(d);
		for (int k = 0; k < d; ++k) {
			const Cell& cell = m.cells[m.Find(Cell::kConfig, sg, j, k)];
			request.lower(k) = cell.lb;
			request.upper(k) = cell.ub;
		}
		Source source;
		for (int id : ids) source.Add(values[id]);
		const bool orthonormal = (R.transpose() * R - Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff() < 1e-6;
		const std::vector<Eigen::VectorXd> q = orthonormal ? solver.Solve(request) : std::vector<Eigen::VectorXd>{};
		const std::string at = fmt::format("node {} agent {}: {} at p = [{:.4g}, {:.4g}, {:.4g}]", node, j, link, p(0),
						   p(1), p(2));
		CandidateSet cell_candidates;
		cell_candidates.node = node;
		cell_candidates.agent = j;
		cell_candidates.link = link;
		cell_candidates.world_T_link = request.world_T_link;
		cell_candidates.ik_solutions = static_cast<int>(q.size());
		if (q.empty()) {
			out.candidates.push_back(std::move(cell_candidates));
			Fail(source, !orthonormal || !solver.complete(),
			     fmt::format("{}: no IK solution{}", at, orthonormal ? "" : " (rotation not orthonormal)"));
			return -1;
		}
		for (const auto& qi : q)
			if (qi.size() != d)
				throw std::runtime_error(fmt::format(
					"PropagationOracle: agent {}'s IK solver gave a {}-wide configuration, not {}", j, qi.size(), d));
		std::vector<Eigen::VectorXd> free;
		std::vector<std::string> hits;
		if (m.c.checker != nullptr) {
			const CollisionScene scene = Scene(sg, j, &source);
			const std::vector<CollisionResult> results = m.c.checker->CheckBatch(m.g, j, q, scene);
			for (size_t b = 0; b < q.size(); ++b) {
				if (results[b].free) free.push_back(q[b]);
				else hits.push_back(results[b].hit);
			}
		} else {
			free = q;
		}
		cell_candidates.configurations = free;
		cell_candidates.rejected = hits;
		if (free.empty()) {
			out.candidates.push_back(std::move(cell_candidates));
			Fail(source, !solver.complete(),
			     fmt::format("{}: all {} IK branches collide ({})", at, q.size(), fmt::join(hits, "; ")));
			return -1;
		}
		int choice = -1;
		const int option = Choose(static_cast<int>(free.size()), solver.complete(), &choice);
		if (option >= static_cast<int>(free.size())) {
			Source exhausted;
			exhausted.choices = {choice};
			Fail(exhausted, false, fmt::format("{}: branch choice {} exhausted", at, choice));
			return -1;
		}
		source.choices.insert(choice);
		SetBlock(sg, j, free[option], source);
		cell_candidates.chosen = option;
		out.candidates.push_back(std::move(cell_candidates));
		out.log.push_back(fmt::format("{}: IK -> {} branches, {} collision-free{}; choice {} takes #{}: q = [{:.4f}]",
					      at, q.size(), free.size(),
					      hits.empty() ? "" : fmt::format(" (rejected: {})", fmt::join(hits, "; ")), choice,
					      option, fmt::join(free[option], ", ")));
		return 1;
	}

	// Step 3: one choice; false when nothing is entangled or it failed.
	bool ChooseOne() {
		std::set<int> entangled;
		for (const Relation& rel : m.relations) {
			if (!rel.equality) continue;
			int last = -1;
			if (Unknowns(rel, &last) < 2) continue;
			for (const auto& [cell, a] : rel.terms)
				if (!values[cell].known) entangled.insert(cell);
		}
		if (entangled.empty()) return false;
		auto key = [&](int id) { return std::make_tuple(m.c.rank[m.cells[id].sg], m.cells[id].owner, m.cells[id].component); };
		int best = -1;
		for (int id : entangled)
			if (m.cells[id].kind == Cell::kObject && (best < 0 || key(id) < key(best))) best = id;
		if (best >= 0) {
			// The object's entangled unknown components at that node, together.
			std::vector<int> group;
			for (int id : entangled)
				if (m.cells[id].kind == Cell::kObject && m.cells[id].sg == m.cells[best].sg &&
				    m.cells[id].owner == m.cells[best].owner)
					group.push_back(id);
			std::sort(group.begin(), group.end(), [&](int a, int b) { return m.cells[a].component < m.cells[b].component; });
			const int dims = static_cast<int>(group.size());
			Eigen::VectorXd ref(dims), lo(dims), hi(dims);
			for (int k = 0; k < dims; ++k) {
				ref(k) = ObjectReference(group[k]);
				std::tie(lo(k), hi(k)) = Interval(group[k], ref(k));
			}
			constexpr int kPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
			std::vector<Eigen::VectorXd> options = {ref};
			for (int i = 1; i < m.c.samples; ++i) {
				Eigen::VectorXd x(dims);
				for (int k = 0; k < dims; ++k) x(k) = lo(k) + RadicalInverse(i, kPrimes[k % 12]) * (hi(k) - lo(k));
				options.push_back(x);
			}
			std::stable_sort(options.begin() + 1, options.end(), [&](const auto& a, const auto& b) {
				return (a - ref).squaredNorm() < (b - ref).squaredNorm();
			});
			int choice = -1;
			const int option = Choose(static_cast<int>(options.size()), false, &choice);
			Source source;
			source.choices = {choice};
			if (option >= static_cast<int>(options.size())) {
				Fail(source, false, fmt::format("region choice {} exhausted", choice));
				return false;
			}
			std::vector<std::string> names;
			for (int k = 0; k < dims; ++k) {
				Set(group[k], options[option](k), source);
				names.push_back(fmt::format("{} = {:.6g} in [{:.6g}, {:.6g}]", m.Name(group[k]), options[option](k),
							    m.cells[group[k]].lb, m.cells[group[k]].ub));
			}
			out.log.push_back(fmt::format("choice {} (region, option {} of {}): {}", choice, option, options.size(),
						      fmt::join(names, ", ")));
			return true;
		}
		for (int id : entangled)
			if (best < 0 || m.c.rank[m.cells[id].sg] < m.c.rank[m.cells[best].sg]) best = id;
		const int sg = m.cells[best].sg, j = m.cells[best].owner;
		const Eigen::VectorXd q = Reference(sg, j);
		int choice = -1;
		Choose(1, false, &choice);
		Source source;
		source.choices = {choice};
		SetBlock(sg, j, q, source);
		out.log.push_back(fmt::format("choice {} (reference): node {} agent {}: q = [{:.4f}] (its links are "
					      "constrained but no full pose is known)",
					      choice, m.c.node_of[sg], j, fmt::join(q, ", ")));
		return true;
	}

	void Propagate() {
		while (!out.failed) {
			if (SolveEqualities()) continue;
			const int ik = SolveOneIK();
			if (ik < 0) return;
			if (ik > 0) continue;
			if (!ChooseOne()) return;
		}
	}

	// Whatever is unset takes its reference; every link follows by FK.
	void FillRest() {
		for (size_t id = 0; id < m.cells.size(); ++id) {
			if (values[id].known || m.cells[id].kind != Cell::kObject) continue;
			Set(id, ObjectReference(id), {});
		}
		for (int sg = 0; sg < m.n_V; ++sg) {
			for (int j : m.visits[sg]) {
				if (!BlockKnown(sg, j)) SetBlock(sg, j, Reference(sg, j), {});
				else FillLinks(sg, j, {});
			}
		}
	}

	void Evaluate() {
		const double tol = m.c.tol;
		auto violated = [&](double r, bool equality, double t) { return equality ? std::abs(r) > t : r > t; };
		for (const Relation& rel : m.relations) {
			double r = rel.constant;
			bool known = true;
			for (const auto& [id, a] : rel.terms) {
				known = known && values[id].known;
				r += a * values[id].v;
			}
			if (known && violated(r, rel.equality, rel.tolerance)) {
				out.violation += std::abs(r);
				out.violations.push_back(fmt::format("{}: residual {:.3g}", rel.where, r));
			}
		}
		for (size_t id = 0; id < m.cells.size(); ++id) {
			if (!values[id].known) continue;
			const double v = values[id].v;
			const double over = std::max(m.cells[id].lb - v, v - m.cells[id].ub);
			if (over > tol) {
				out.violation += over;
				out.violations.push_back(fmt::format("{} = {:.6g} outside [{:.6g}, {:.6g}]", m.Name(id), v,
								     m.cells[id].lb, m.cells[id].ub));
			}
		}
		for (const Check& check : m.checks) {
			Environment env;
			bool known = true;
			for (const auto& [v, id] : check.bind) {
				known = known && values[id].known;
				env.insert(v, values[id].v);
			}
			if (!known) continue;
			double r;
			try {
				r = check.e.Evaluate(env);
			} catch (const std::exception&) {
				continue;
			}
			if (violated(r, check.equality, std::sqrt(tol))) {
				out.violation += std::abs(r);
				out.violations.push_back(fmt::format("{} (nonlinear): residual {:.3g}", check.where, r));
			}
		}
		// Every visiting agent against the environment and the agents before it.
		if (m.c.checker == nullptr) return;
		for (int sg = 0; sg < m.n_V; ++sg) {
			CollisionScene scene;
			scene.node = m.c.node_of[sg];
			for (int j : m.visits[sg]) {
				const Eigen::VectorXd q = Block(sg, j);
				const CollisionResult result = m.c.checker->CheckBatch(m.g, j, {q}, scene).front();
				if (!result.free) {
					out.violation += 1.0;
					out.violations.push_back(fmt::format("node {}: {}", scene.node, result.hit));
				}
				scene.others.emplace_back(j, q);
			}
		}
	}
};

Outcome Model::Run(const std::vector<int>& idx) const { return Runner(*this, idx).Run(); }

struct Search {
	Outcome outcome;
	int runs = 0;
	std::vector<std::vector<Pair>> nogoods;
};

// Conflict-directed backjumping over a Model's choices.
Search Backjump(const Model& model, int propagations, int threads) {
	Search s;
	std::vector<int> idx;
	std::map<int, Failure> acc;
	auto absorb = [](Failure* into, const Failure& f, int c) {
		for (int k : f.conflict)
			if (k != c) into->conflict.insert(k);
		Merge(&into->deps, f.deps);
		into->inexact = into->inexact || f.inexact;
		into->what = f.what;
	};
	s.outcome = model.Run(idx);
	++s.runs;
	while (s.outcome.failed) {
		Failure f = s.outcome.failure;
		const std::vector<ChoiceInfo> info = s.outcome.choices;
		bool rerun = false;
		while (!rerun) {
			if (f.conflict.empty()) {
				if (!f.inexact) s.nogoods.emplace_back(f.deps.begin(), f.deps.end());
				s.outcome.failure = f;
				return s;
			}
			if (s.runs >= propagations) {
				s.outcome.failure = f;
				s.outcome.log.push_back(fmt::format("stopped after {} propagations", s.runs));
				return s;
			}
			const int c = *f.conflict.rbegin();
			absorb(&acc[c], f, c);
			const int next = (c < static_cast<int>(idx.size()) ? idx[c] : 0) + 1;
			if (next >= info[c].count) {
				// Exhausted: its accumulated conflict moves to the earlier choices.
				Failure e = acc[c];
				e.inexact = e.inexact || !info[c].exhaustive;
				acc.erase(c);
				f = e;
				continue;
			}
			for (auto it = acc.upper_bound(c); it != acc.end();) it = acc.erase(it);
			idx.resize(c + 1, 0);
			const int batch = info[c].exhaustive ? 1 : std::max(1, std::min({threads, info[c].count - next,
											     propagations - s.runs}));
			std::vector<Outcome> outcomes(batch);
			if (batch == 1) {
				idx[c] = next;
				outcomes[0] = model.Run(idx);
			} else {
				std::vector<std::future<Outcome>> futures;
				for (int b = 0; b < batch; ++b) {
					std::vector<int> idx_b = idx;
					idx_b[c] = next + b;
					futures.push_back(std::async(std::launch::async, [&model, idx_b] { return model.Run(idx_b); }));
				}
				for (int b = 0; b < batch; ++b) outcomes[b] = futures[b].get();
			}
			s.runs += batch;
			// The first option (in order) that succeeds or fails for a reason
			// other than this choice is taken; the ones before it refute c.
			for (int b = 0; b < batch; ++b) {
				const Outcome& o = outcomes[b];
				const bool refutes = o.failed && !o.failure.conflict.empty() && *o.failure.conflict.rbegin() == c;
				if (refutes && b + 1 < batch) {
					absorb(&acc[c], o.failure, c);
					continue;
				}
				idx[c] = next + b;
				s.outcome = std::move(outcomes[b]);
				rerun = true;
				break;
			}
		}
	}
	return s;
}

}  // namespace

PropagationOracle::PropagationOracle(std::map<int, std::shared_ptr<IkSolver>> ik_solvers,
				     std::shared_ptr<CollisionChecker> collision_checker, int max_assignments, int samples,
				     int propagations, int threads, double tolerance)
	: _ik_solvers(std::move(ik_solvers)), _collision_checker(std::move(collision_checker)),
	  _max_assignments(max_assignments), _samples(samples), _propagations(propagations), _threads(threads),
	  _tolerance(tolerance) {
	for (const auto& [agent, solver] : _ik_solvers)
		if (!solver) throw std::invalid_argument(fmt::format("PropagationOracle: agent {}'s IK solver is None", agent));
	if (max_assignments < 1 || samples < 1 || propagations < 1 || threads < 1)
		throw std::invalid_argument("PropagationOracle: max_assignments, samples, propagations and threads must be >= 1");
}

InitialGuess PropagationOracle::Initialize(const InitializationRequest& request) const {
	const auto start = std::chrono::steady_clock::now();
	if (request.graph == nullptr) throw std::invalid_argument("PropagationOracle: request has no graph");
	const GraphOfConstraints& graph = *request.graph;
	if (request.x0.size() != graph.total_dim)
		throw std::invalid_argument(fmt::format("PropagationOracle: x0 has {} entries, the graph's state {}",
							request.x0.size(), graph.total_dim));
	const std::vector<int>& nodes = request.nodes;
	const SubgraphOfConstraints subgraph(const_cast<GraphOfConstraints*>(&graph), nodes);
	const RoutingPresolve presolve = BuildRoutingPresolve(graph, subgraph);
	const int n_V = subgraph.num_nodes();
	std::vector<int> node_of(n_V, -1);
	for (int u : nodes)
		if (subgraph.structure.contains_node(u)) node_of[subgraph.subgraph_id(u)] = u;

	// A linear extension of the hard orderings: by number of nodes that must
	// come strictly earlier, then node id.
	std::vector<int> rank(n_V);
	{
		std::vector<std::pair<std::pair<int, int>, int>> key(n_V);
		for (int b = 0; b < n_V; ++b) {
			int preds = 0;
			for (int a = 0; a < n_V; ++a)
				if (a != b && presolve.before[a][b] && !presolve.before[b][a]) ++preds;
			key[b] = {{preds, node_of[b]}, b};
		}
		std::sort(key.begin(), key.end());
		for (int r = 0; r < n_V; ++r) rank[key[r].second] = r;
	}

	// Assignable variables in subgraph row order, each with its candidate
	// agents in preference order.
	std::vector<int> vars(subgraph.num_variables(), -1);
	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) vars[row] = var;
	std::vector<std::vector<int>> candidates(vars.size());
	for (size_t k = 0; k < vars.size(); ++k) {
		const int var = vars[k];
		std::vector<int> pool;
		const auto committed = graph.committed_assignments.find(var);
		if (committed != graph.committed_assignments.end()) {
			pool = {committed->second};
		} else {
			for (int j = 0; j < graph.num_agents; ++j) {
				const auto it = graph.var_candidates.find(var);
				if (it == graph.var_candidates.end() || it->second.contains(j)) pool.push_back(j);
			}
		}
		std::vector<int> preferred;
		if (var < request.previous_assignments.size() && request.previous_assignments(var) >= 0)
			preferred.push_back(request.previous_assignments(var));
		if (request.previous) {
			const auto it = request.previous->assignments.find(var);
			if (it != request.previous->assignments.end()) preferred.push_back(it->second);
		}
		for (int j : preferred)
			if (std::find(pool.begin(), pool.end(), j) != pool.end() &&
			    std::find(candidates[k].begin(), candidates[k].end(), j) == candidates[k].end())
				candidates[k].push_back(j);
		for (int j : pool)
			if (std::find(candidates[k].begin(), candidates[k].end(), j) == candidates[k].end())
				candidates[k].push_back(j);
	}

	const Model::Context context{graph,    subgraph,    presolve, node_of, rank, request, _ik_solvers,
				     _collision_checker.get(), _samples, _tolerance};
	InitialGuess guess;
	guess.nodes = request.nodes;
	std::vector<std::vector<Pair>> nogoods;
	std::vector<std::string> summary;
	std::optional<Outcome> best;
	std::vector<std::string> best_unresolved;
	std::map<int, int> best_asg;
	auto worse = [](const Outcome& a, const Outcome& b) {
		return std::make_tuple(a.failed, a.violations.size(), a.violation) >
		       std::make_tuple(b.failed, b.violations.size(), b.violation);
	};
	auto excluded = [&](const std::map<int, int>& asg) {
		for (const auto& ng : nogoods) {
			bool all = true;
			for (const auto& [var, j] : ng) {
				const auto it = asg.find(var);
				all = all && it != asg.end() && it->second == j;
			}
			if (all) return true;
		}
		return false;
	};
	auto consistent = [&](const std::map<int, int>& asg) {
		for (const auto& [a, b] : graph.var_ineq_pairs) {
			const auto ia = asg.find(a), ib = asg.find(b);
			if (ia != asg.end() && ib != asg.end() && ia->second == ib->second) return false;
		}
		for (int sg = 0; sg < n_V; ++sg) {
			std::set<int> holders;
			for (int var : presolve.roles_at[sg]) {
				const auto it = asg.find(var);
				if (it != asg.end() && !holders.insert(it->second).second) return false;
			}
		}
		return true;
	};
	auto named = [](const std::vector<Pair>& pairs) {
		std::vector<std::string> out;
		for (const auto& [var, j] : pairs) out.push_back(fmt::format("var {} -> agent {}", var, j));
		return fmt::format("[{}]", fmt::join(out, ", "));
	};

	std::map<int, int> asg;
	std::function<void(size_t)> enumerate = [&](size_t k) {
		if (guess.assignments_tried >= _max_assignments) return;
		if (k == vars.size()) {
			if (excluded(asg)) return;
			++guess.assignments_tried;
			const Model model(context, asg);
			Search search = Backjump(model, _propagations, _threads);
			Outcome& outcome = search.outcome;
			summary.push_back(fmt::format(
				"assignment {}: {} after {} propagations",
				named(std::vector<Pair>(asg.begin(), asg.end())),
				outcome.failed ? "failed: " + outcome.failure.what
				: outcome.violations.empty() ? std::string("ok")
							     : fmt::format("{} violations", outcome.violations.size()),
				search.runs));
			for (CandidateSet& found : outcome.candidates) {
				auto same = [&](const CandidateSet& c) {
					if (c.node != found.node || c.agent != found.agent || c.link != found.link ||
					    !c.world_T_link.isApprox(found.world_T_link, 1e-12) ||
					    c.configurations.size() != found.configurations.size() || c.chosen != found.chosen)
						return false;
					for (size_t b = 0; b < c.configurations.size(); ++b)
						if (!c.configurations[b].isApprox(found.configurations[b], 1e-12)) return false;
					return true;
				};
				const auto it = std::find_if(guess.candidates.begin(), guess.candidates.end(), same);
				if (it != guess.candidates.end()) {
					it->assignments.push_back(asg);
				} else {
					found.assignments = {asg};
					guess.candidates.push_back(std::move(found));
				}
			}
			for (const auto& ng : search.nogoods) {
				nogoods.push_back(ng);
				summary.push_back("  infeasible together: " + named(ng));
			}
			if (!outcome.failed && outcome.violations.empty())
				guess.feasible.push_back({asg, outcome.X, outcome.routes});
			if (!best || worse(*best, outcome)) {
				best = std::move(outcome);
				best_asg = asg;
				best_unresolved = model.unresolved;
			}
			return;
		}
		for (int j : candidates[k]) {
			asg[vars[k]] = j;
			if (consistent(asg)) enumerate(k + 1);
			asg.erase(vars[k]);
		}
	};
	enumerate(0);

	guess.infeasible_assignments = nogoods;
	guess.log = std::move(summary);
	if (best) {
		guess.success = !best->failed && best->violations.empty();
		guess.X = std::move(best->X);
		guess.routes = std::move(best->routes);
		guess.assignments = best_asg;
		guess.log.insert(guess.log.end(), best->log.begin(), best->log.end());
		guess.violations = std::move(best->violations);
		if (best->failed) guess.violations.push_back("failed: " + best->failure.what);
		guess.unresolved = std::move(best_unresolved);
	} else {
		guess.log.push_back("no assignment satisfies the graph's assignment restrictions");
	}
	guess.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
	return guess;
}
