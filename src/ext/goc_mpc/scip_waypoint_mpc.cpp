#include "scip_waypoint_mpc.hpp"
#include "expression_tape.hpp"
#include "local_nlp_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <map>
#include <stdexcept>
#include <unordered_map>

#include <Eigen/Dense>
#include <fmt/format.h>

#include <objscip/objeventhdlr.h>
#include <objscip/objrelax.h>
#include <scip/cons_linear.h>
#include <scip/cons_nonlinear.h>
#include <scip/scip.h>
#include <scip/scipdefplugins.h>

namespace {

#define SCIP_CHECK(call)                                                                        \
	do {                                                                                    \
		const SCIP_RETCODE rc_ = (call);                                                \
		if (rc_ != SCIP_OKAY)                                                           \
			throw std::runtime_error(fmt::format("SCIPWaypointMPC: {} returned SCIP error {}", \
							     #call, static_cast<int>(rc_)));    \
	} while (0)

struct ScipFree {
	void operator()(SCIP* scip) const { SCIPfree(&scip); }
};

double ScipBound(SCIP* scip, double v) {
	if (std::isinf(v) || std::abs(v) >= SCIPinfinity(scip)) return v > 0 ? SCIPinfinity(scip) : -SCIPinfinity(scip);
	return v;
}

std::string StatusName(SCIP_STATUS status) {
	switch (status) {
	case SCIP_STATUS_OPTIMAL: return "optimal";
	case SCIP_STATUS_INFEASIBLE: return "infeasible";
	case SCIP_STATUS_UNBOUNDED: return "unbounded";
	case SCIP_STATUS_INFORUNBD: return "infeasible or unbounded";
	case SCIP_STATUS_TIMELIMIT: return "time limit";
	case SCIP_STATUS_GAPLIMIT: return "gap limit";
	case SCIP_STATUS_NODELIMIT: return "node limit";
	case SCIP_STATUS_TOTALNODELIMIT: return "total node limit";
	case SCIP_STATUS_STALLNODELIMIT: return "stall node limit";
	case SCIP_STATUS_MEMLIMIT: return "memory limit";
	case SCIP_STATUS_SOLLIMIT: return "solution limit";
	case SCIP_STATUS_BESTSOLLIMIT: return "best solution limit";
	case SCIP_STATUS_PRIMALLIMIT: return "primal limit";
	case SCIP_STATUS_DUALLIMIT: return "dual limit";
	case SCIP_STATUS_USERINTERRUPT: return "user interrupt";
	case SCIP_STATUS_TERMINATE: return "terminated";
	case SCIP_STATUS_RESTARTLIMIT: return "restart limit";
	case SCIP_STATUS_UNKNOWN: return "unknown";
	}
	return fmt::format("status {}", static_cast<int>(status));
}

void SetParam(SCIP* scip, const std::string& name, const SCIPWaypointMPC::ParamValue& value) {
	SCIP_PARAM* param = SCIPgetParam(scip, name.c_str());
	if (!param) throw std::runtime_error(fmt::format("SCIPWaypointMPC: no SCIP parameter '{}'", name));
	auto as_double = [&]() -> double {
		if (const auto* v = std::get_if<double>(&value)) return *v;
		if (const auto* v = std::get_if<long long>(&value)) return static_cast<double>(*v);
		if (const auto* v = std::get_if<bool>(&value)) return *v ? 1.0 : 0.0;
		throw std::runtime_error(fmt::format("SCIPWaypointMPC: parameter '{}' takes a number", name));
	};
	auto as_integer = [&]() -> long long {
		const double v = as_double();
		if (v != std::floor(v))
			throw std::runtime_error(fmt::format("SCIPWaypointMPC: parameter '{}' takes an integer", name));
		return static_cast<long long>(v);
	};
	auto as_string = [&]() -> const std::string& {
		if (const auto* v = std::get_if<std::string>(&value)) return *v;
		throw std::runtime_error(fmt::format("SCIPWaypointMPC: parameter '{}' takes a string", name));
	};
	switch (SCIPparamGetType(param)) {
	case SCIP_PARAMTYPE_BOOL: SCIP_CHECK(SCIPsetBoolParam(scip, name.c_str(), as_integer() != 0)); break;
	case SCIP_PARAMTYPE_INT: SCIP_CHECK(SCIPsetIntParam(scip, name.c_str(), static_cast<int>(as_integer()))); break;
	case SCIP_PARAMTYPE_LONGINT: SCIP_CHECK(SCIPsetLongintParam(scip, name.c_str(), as_integer())); break;
	case SCIP_PARAMTYPE_REAL: SCIP_CHECK(SCIPsetRealParam(scip, name.c_str(), as_double())); break;
	case SCIP_PARAMTYPE_CHAR: {
		const std::string& s = as_string();
		if (s.size() != 1)
			throw std::runtime_error(fmt::format("SCIPWaypointMPC: parameter '{}' takes one character", name));
		SCIP_CHECK(SCIPsetCharParam(scip, name.c_str(), s[0]));
		break;
	}
	case SCIP_PARAMTYPE_STRING: SCIP_CHECK(SCIPsetStringParam(scip, name.c_str(), as_string().c_str())); break;
	}
}

// Adds row r of A (over x) with lhs <= A_r x + extra_coef * extra <= rhs.
void AddLinearRow(SCIP* scip, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A, int r,
		  const std::vector<SCIP_VAR*>& x, double lhs, double rhs, const std::string& name,
		  SCIP_VAR* extra = nullptr, double extra_coef = 0.0) {
	std::vector<SCIP_VAR*> vars;
	std::vector<double> vals;
	for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(A, r); it; ++it) {
		vars.push_back(x[it.col()]);
		vals.push_back(it.value());
	}
	if (extra) {
		vars.push_back(extra);
		vals.push_back(extra_coef);
	}
	SCIP_CONS* cons;
	SCIP_CHECK(SCIPcreateConsBasicLinear(scip, &cons, name.c_str(), static_cast<int>(vars.size()),
					     vars.data(), vals.data(), ScipBound(scip, lhs), ScipBound(scip, rhs)));
	SCIP_CHECK(SCIPaddCons(scip, cons));
	SCIP_CHECK(SCIPreleaseCons(scip, &cons));
}

using drake::symbolic::Expression;
using drake::symbolic::ExpressionKind;

using VarMap = std::unordered_map<drake::symbolic::Variable::Id, SCIP_VAR*>;

// e as a SCIP expression over var_of's variables; the caller releases it.
SCIP_EXPR* ToScipExpr(SCIP* scip, const Expression& e, const VarMap& var_of) {
	namespace sym = drake::symbolic;
	SCIP_EXPR* out = nullptr;
	auto release = [&](std::vector<SCIP_EXPR*>& children) {
		for (SCIP_EXPR*& c : children) SCIP_CHECK(SCIPreleaseExpr(scip, &c));
	};
	auto unary = [&](auto create, const Expression& arg) {
		std::vector<SCIP_EXPR*> child = {ToScipExpr(scip, arg, var_of)};
		SCIP_CHECK(create(scip, &out, child[0], nullptr, nullptr));
		release(child);
	};
	auto power = [&](const Expression& base, const Expression& exponent) {
		if (!sym::is_constant(exponent))
			throw std::runtime_error(fmt::format(
				"SCIPWaypointMPC: only constant exponents are supported, got {}", e.to_string()));
		std::vector<SCIP_EXPR*> child = {ToScipExpr(scip, base, var_of)};
		SCIP_CHECK(SCIPcreateExprPow(scip, &out, child[0], sym::get_constant_value(exponent), nullptr, nullptr));
		release(child);
	};
	switch (e.get_kind()) {
	case ExpressionKind::Constant:
		SCIP_CHECK(SCIPcreateExprValue(scip, &out, sym::get_constant_value(e), nullptr, nullptr));
		break;
	case ExpressionKind::Var: {
		const auto it = var_of.find(sym::get_variable(e).get_id());
		if (it == var_of.end())
			throw std::runtime_error(fmt::format(
				"SCIPWaypointMPC: {} isn't a decision variable", sym::get_variable(e).get_name()));
		SCIP_CHECK(SCIPcreateExprVar(scip, &out, it->second, nullptr, nullptr));
		break;
	}
	case ExpressionKind::Add: {
		std::vector<SCIP_EXPR*> children;
		std::vector<double> coefs;
		for (const auto& [term, coef] : sym::get_expr_to_coeff_map_in_addition(e)) {
			children.push_back(ToScipExpr(scip, term, var_of));
			coefs.push_back(coef);
		}
		SCIP_CHECK(SCIPcreateExprSum(scip, &out, static_cast<int>(children.size()), children.data(),
					     coefs.data(), sym::get_constant_in_addition(e), nullptr, nullptr));
		release(children);
		break;
	}
	case ExpressionKind::Mul: {
		std::vector<SCIP_EXPR*> children;
		for (const auto& [base, exponent] : sym::get_base_to_exponent_map_in_multiplication(e)) {
			if (sym::is_constant(exponent) && sym::get_constant_value(exponent) == 1.0) {
				children.push_back(ToScipExpr(scip, base, var_of));
			} else {
				power(base, exponent);
				children.push_back(out);
			}
		}
		SCIP_CHECK(SCIPcreateExprProduct(scip, &out, static_cast<int>(children.size()), children.data(),
						 sym::get_constant_in_multiplication(e), nullptr, nullptr));
		release(children);
		break;
	}
	case ExpressionKind::Div: {
		std::vector<SCIP_EXPR*> children = {ToScipExpr(scip, sym::get_first_argument(e), var_of)};
		power(sym::get_second_argument(e), Expression(-1.0));
		children.push_back(out);
		SCIP_CHECK(SCIPcreateExprProduct(scip, &out, 2, children.data(), 1.0, nullptr, nullptr));
		release(children);
		break;
	}
	case ExpressionKind::Pow: power(sym::get_first_argument(e), sym::get_second_argument(e)); break;
	case ExpressionKind::Sqrt: power(sym::get_argument(e), Expression(0.5)); break;
	case ExpressionKind::Sin: unary(SCIPcreateExprSin, sym::get_argument(e)); break;
	case ExpressionKind::Cos: unary(SCIPcreateExprCos, sym::get_argument(e)); break;
	case ExpressionKind::Exp: unary(SCIPcreateExprExp, sym::get_argument(e)); break;
	case ExpressionKind::Log: unary(SCIPcreateExprLog, sym::get_argument(e)); break;
	case ExpressionKind::Abs: unary(SCIPcreateExprAbs, sym::get_argument(e)); break;
	default:
		throw std::runtime_error(fmt::format(
			"SCIPWaypointMPC: unsupported operation in nonlinear constraint {} (constants, variables, "
			"+, *, /, constant powers, sqrt, sin, cos, exp, log, abs only)", e.to_string()));
	}
	return out;
}

// v as a * active + c (active null when v is fixed, a == 0 then).
struct Resolved {
	SCIP_VAR* active;
	double a, c;
};
Resolved Resolve(SCIP* scip, SCIP_VAR* v) {
	Resolved r{v, 1.0, 0.0};
	if (SCIPgetProbvarSum(scip, &r.active, &r.a, &r.c) != SCIP_OKAY || r.a == 0.0) r.active = nullptr;
	return r;
}

// sum_k vals[k] vars[k] over SCIP's active variables: (column, coefficient)
// pairs into *terms (col_of maps an active variable to its column), the
// constant into *constant.
void ToActive(SCIP* scip, std::vector<SCIP_VAR*> vars, std::vector<double> vals,
	      const std::unordered_map<SCIP_VAR*, int>& col_of, std::vector<std::pair<int, double>>* terms,
	      double* constant) {
	int nvars = static_cast<int>(vars.size()), required = 0;
	*constant = 0.0;
	SCIP_CHECK(SCIPgetProbvarLinearSum(scip, vars.data(), vals.data(), &nvars, nvars, constant, &required));
	if (required > nvars) {
		vars.resize(required);
		vals.resize(required);
		SCIP_CHECK(SCIPgetProbvarLinearSum(scip, vars.data(), vals.data(), &nvars, required, constant, &required));
	}
	terms->clear();
	for (int k = 0; k < nvars; ++k) terms->emplace_back(col_of.at(vars[k]), vals[k]);
}

// A transformed variable's name in the exported program (SCIP's original
// variables are x<i>, mip.vars(i)).
std::string ExportedName(const ExternalMIP& mip, SCIP_VAR* var) {
	std::string name = SCIPvarGetName(var);
	if (name.rfind("t_", 0) == 0) name = name.substr(2);
	if (name.size() > 1 && name[0] == 'x') {
		try {
			const int i = std::stoi(name.substr(1));
			if (i >= 0 && i < mip.vars.size()) return mip.vars(i).get_name();
		} catch (const std::exception&) {
		}
	}
	return name;
}

// Records every branch-and-bound node from SCIP's node events.
class NodeTrace : public scip::ObjEventhdlr {
public:
	static constexpr SCIP_EVENTTYPE kEvents = SCIP_EVENTTYPE_NODEEVENT | SCIP_EVENTTYPE_NODEDELETE;

	NodeTrace(SCIP* scip, std::function<std::string(SCIP_VAR*)> name)
		: ObjEventhdlr(scip, "goc_nodetrace", "records branch-and-bound nodes"), _name(std::move(name)) {}

	SCIP_DECL_EVENTINIT(scip_init) override {
		return SCIPcatchEvent(scip, kEvents, eventhdlr, nullptr, nullptr);
	}
	SCIP_DECL_EVENTEXIT(scip_exit) override {
		return SCIPdropEvent(scip, kEvents, eventhdlr, nullptr, -1);
	}
	SCIP_DECL_EVENTEXEC(scip_exec) override {
		SCIP_NODE* node = SCIPeventGetNode(event);
		if (node == nullptr) return SCIP_OKAY;
		SCIPNodeRecord& rec = Record(node);
		const SCIP_EVENTTYPE type = SCIPeventGetType(event);
		if (type & SCIP_EVENTTYPE_NODEFOCUSED) {
			rec.events.push_back("focused");
			rec.focus_time = SCIPgetSolvingTime(scip);
		}
		if (type & SCIP_EVENTTYPE_NODEFEASIBLE) rec.events.push_back("feasible");
		if (type & SCIP_EVENTTYPE_NODEINFEASIBLE) rec.events.push_back("infeasible");
		if (type & SCIP_EVENTTYPE_NODEBRANCHED) rec.events.push_back("branched");
		if (type & SCIP_EVENTTYPE_NODEDELETE) rec.events.push_back("deleted");
		return SCIP_OKAY;
	}

	SCIPNodeRecord& Record(SCIP_NODE* node) {
		const long long number = SCIPnodeGetNumber(node);
		auto [it, added] = nodes.try_emplace(number);
		if (added) {
			order.push_back(number);
			it->second.number = number;
			SCIP_NODE* parent = SCIPnodeGetParent(node);
			it->second.parent = parent ? SCIPnodeGetNumber(parent) : -1;
			it->second.depth = SCIPnodeGetDepth(node);
			int count = 0;
			SCIPnodeGetParentBranchings(node, nullptr, nullptr, nullptr, &count, 0);
			std::vector<SCIP_VAR*> vars(count);
			std::vector<double> bounds(count);
			std::vector<SCIP_BOUNDTYPE> types(count);
			if (count > 0)
				SCIPnodeGetParentBranchings(node, vars.data(), bounds.data(), types.data(), &count, count);
			for (int k = 0; k < count; ++k)
				it->second.branchings.push_back(fmt::format(
					"{} {} {}", _name(vars[k]), types[k] == SCIP_BOUNDTYPE_LOWER ? ">=" : "<=", bounds[k]));
		}
		return it->second;
	}

	std::map<long long, SCIPNodeRecord> nodes;
	std::vector<long long> order;

private:
	std::function<std::string(SCIP_VAR*)> _name;
};

// Each node's relaxation, in place of SCIP's LP (lp/solvefreq = -1): SCIP's
// presolved problem at the node -- its active variables within the node's
// bounds and its linear constraints -- plus the nonlinear rows and cones,
// with the integers relaxed, solved locally and treated as if globally
// optimal. Its value is the node's lower bound and its solution the one
// SCIP branches on (a fractional integer) or accepts (integral); a solve
// that doesn't reach SCIP's feasibility tolerance closes the node as
// infeasible.
//
// The nonlinear rows and cones are evaluated as exported, over the
// original variables x, which SCIP's presolve maps affinely onto its active
// ones y (x = T y + t): SCIP's own copies of them, rewritten by presolve,
// are only checked.
//
// The local solve is a LocalSolver (local_nlp_solver.hpp) from one start:
// mip.nonlinear_seed (the initialization oracle's guess, or the agents'
// current configurations), everything else 0.
class LocalNLPRelax : public scip::ObjRelax {
public:
	LocalNLPRelax(SCIP* scip, const ExternalMIP& mip, std::vector<SCIP_VAR*> x, SCIPSolveStats* stats,
		      LocalNLPSolver solver)
		: ObjRelax(scip, "goc_localnlp", "local NLP relaxation", 1, 1, TRUE),
		  _mip(mip), _x(std::move(x)), _stats(stats), _solver(MakeLocalSolver(solver)) {
		// The nonlinear rows, compiled per group of rows sharing variables
		// (e.g. one agent's FK pins at one node).
		std::unordered_map<drake::symbolic::Variable::Id, int> index;
		for (int i = 0; i < mip.vars.size(); ++i) index.emplace(mip.vars(i).get_id(), i);
		std::vector<int> parent(mip.vars.size());
		for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);
		auto find = [&](int i) {
			while (parent[i] != i) i = parent[i] = parent[parent[i]];
			return i;
		};
		const int rows = static_cast<int>(mip.nonlinear.size());
		std::vector<int> root_var(rows, -1);
		for (int r = 0; r < rows; ++r) {
			for (const auto& v : mip.nonlinear[r].e.GetVariables()) {
				const int i = find(index.at(v.get_id()));
				if (root_var[r] < 0) root_var[r] = i;
				else parent[i] = find(root_var[r]);
			}
			if (root_var[r] < 0) throw std::runtime_error("SCIPWaypointMPC: a nonlinear row has no variables");
		}
		std::map<int, std::vector<int>> by_root;
		for (int r = 0; r < rows; ++r) by_root[find(root_var[r])].push_back(r);
		for (const auto& [root, members] : by_root) {
			Group& group = _groups.emplace_back();
			group.rows = members;
			drake::VectorX<drake::symbolic::Expression> e(members.size());
			drake::symbolic::Variables vs;
			for (size_t k = 0; k < members.size(); ++k) {
				e(k) = mip.nonlinear[members[k]].e;
				vs.insert(e(k).GetVariables());
			}
			drake::VectorX<drake::symbolic::Variable> v(vs.size());
			int k = 0;
			for (const auto& var : vs) {
				v(k++) = var;
				group.x.push_back(index.at(var.get_id()));
			}
			group.tape = std::make_unique<ExpressionTape>(e, v);
		}
	}

	// The first exception a callback raised (SCIP's C callbacks can't carry it).
	std::string error;
	// The least-violated failed leaf solve: its final point over x (empty
	// before one), its violation (the nonlinear rows' and cones' max) and
	// objective. Candidates are leaves whose linear rows hold, compared by
	// violation, then objective.
	Eigen::VectorXd fallback;
	double fallback_violation = kInf, fallback_objective = kInf;

	SCIP_DECL_RELAXINITSOL(scip_initsol) override {
		_tx.resize(_x.size());
		for (size_t i = 0; i < _x.size(); ++i) SCIP_CALL(SCIPgetTransformedVar(scip, _x[i], &_tx[i]));
		return SCIP_OKAY;
	}

	SCIP_DECL_RELAXEXEC(scip_exec) override {
		try {
			return Run(scip, relax, lowerbound, result);
		} catch (const std::exception& ex) {
			if (error.empty()) error = ex.what();
			return SCIP_ERROR;
		}
	}

private:
	using SparseRows = Eigen::SparseMatrix<double, Eigen::RowMajor>;

	// Nonlinear rows sharing variables: their compiled form over the x
	// entries `x`, and their positions among the nonlinear rows.
	struct Group {
		std::unique_ptr<ExpressionTape> tape;
		std::vector<int> x, rows;
	};

	// The constraints at a node over the active variables y, beyond the
	// linear rows: the nonlinear rows on x = T y + t, then the cones on
	// z = Z y + z0.
	struct Model {
		SparseRows T;
		Eigen::VectorXd t;
		std::vector<SparseRows> Tg;  // each group's x entries: Tg y + tg
		std::vector<Eigen::VectorXd> tg;
		std::vector<SparseRows> Z;   // each cone's z = Z y + z0
		std::vector<Eigen::VectorXd> z0;
		Eigen::VectorXd scratch;  // n zeros between uses
	};

	// The nonlinear rows' and cones' values g at y, their Jacobian's values
	// into J.
	void Evaluate(Model& model, const Eigen::VectorXd& y, Eigen::VectorXd* g, SparseRows* J) const {
		const int m_nl = static_cast<int>(_mip.nonlinear.size());
		const int m_cone = static_cast<int>(_mip.cones.size());
		g->resize(m_nl + m_cone);
		// Copies the scratch row into J's row r, re-zeroing the scratch.
		Eigen::VectorXd& row = model.scratch;
		auto write = [&](int r) {
			for (SparseRows::InnerIterator it(*J, r); it; ++it) {
				it.valueRef() = row(it.col());
				row(it.col()) = 0.0;
			}
		};
		Eigen::VectorXd value;
		Eigen::MatrixXd jac;
		for (size_t k = 0; k < _groups.size(); ++k) {
			const Group& group = _groups[k];
			group.tape->Eval(model.Tg[k] * y + model.tg[k], &value, &jac);
			for (size_t i = 0; i < group.rows.size(); ++i) {
				(*g)(group.rows[i]) = value(i);
				for (int j = 0; j < model.Tg[k].rows(); ++j)
					for (SparseRows::InnerIterator it(model.Tg[k], j); it; ++it)
						row(it.col()) += jac(i, j) * it.value();
				write(group.rows[i]);
			}
		}
		for (int k = 0; k < m_cone; ++k) {
			// |z_1:| - z_0; at z_1: = 0 the subgradient 0 for the norm.
			const Eigen::VectorXd z = model.Z[k] * y + model.z0[k];
			const double norm = z.tail(z.size() - 1).norm();
			(*g)(m_nl + k) = norm - z(0);
			for (int i = 0; i < z.size(); ++i) {
				const double w = i == 0 ? -1.0 : (norm > 0 ? z(i) / norm : 0.0);
				for (SparseRows::InnerIterator it(model.Z[k], i); it; ++it) row(it.col()) += w * it.value();
			}
			write(m_nl + k);
		}
	}

	SCIP_RETCODE Run(SCIP* scip, SCIP_RELAX* relax, double* lowerbound, SCIP_RESULT* result) {
		const auto tic = std::chrono::steady_clock::now();
		using Clock = std::chrono::steady_clock;
		auto since = [](Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); };
		LocalSolveRecord record;
		SCIP_VAR** all = SCIPgetVars(scip);
		const std::vector<SCIP_VAR*> vars(all, all + SCIPgetNVars(scip));
		Model model;
		LocalNLP nlp;
		const int n = static_cast<int>(vars.size());
		std::unordered_map<SCIP_VAR*, int> col_of;
		Eigen::VectorXd& lb = nlp.lb;
		Eigen::VectorXd& ub = nlp.ub;
		Eigen::VectorXd& c = nlp.c;
		Eigen::VectorXd y = Eigen::VectorXd::Zero(n);
		lb.resize(n);
		ub.resize(n);
		c.resize(n);
		for (int k = 0; k < n; ++k) {
			col_of.emplace(vars[k], k);
			lb(k) = SCIPisInfinity(scip, -SCIPvarGetLbLocal(vars[k])) ? -kInf : SCIPvarGetLbLocal(vars[k]);
			ub(k) = SCIPisInfinity(scip, SCIPvarGetUbLocal(vars[k])) ? kInf : SCIPvarGetUbLocal(vars[k]);
			c(k) = SCIPvarGetObj(vars[k]);
		}

		// x = T y + t, and the start: x's seeds carried onto y.
		std::vector<Eigen::Triplet<double>> tT;
		model.t = Eigen::VectorXd::Zero(_x.size());
		std::vector<std::pair<int, double>> terms;
		for (size_t i = 0; i < _x.size(); ++i) {
			if (!_tx[i]) throw std::runtime_error("an original variable has no transformed one");
			double constant;
			ToActive(scip, {_tx[i]}, {1.0}, col_of, &terms, &constant);
			model.t(i) = constant;
			for (const auto& [col, a] : terms) tT.emplace_back(static_cast<int>(i), col, a);
			if (terms.size() == 1 && std::isfinite(_mip.nonlinear_seed(i)))
				y(terms[0].first) = (_mip.nonlinear_seed(i) - constant) / terms[0].second;
		}
		model.T.resize(static_cast<int>(_x.size()), n);
		model.T.setFromTriplets(tT.begin(), tT.end());
		for (const Group& group : _groups) {
			std::vector<Eigen::Triplet<double>> tg_entries;
			Eigen::VectorXd tg(group.x.size());
			for (size_t k = 0; k < group.x.size(); ++k) {
				for (SparseRows::InnerIterator it(model.T, group.x[k]); it; ++it)
					tg_entries.emplace_back(static_cast<int>(k), it.col(), it.value());
				tg(k) = model.t(group.x[k]);
			}
			SparseRows Tg(static_cast<int>(group.x.size()), n);
			Tg.setFromTriplets(tg_entries.begin(), tg_entries.end());
			model.Tg.push_back(std::move(Tg));
			model.tg.push_back(std::move(tg));
		}
		for (const auto& cone : _mip.cones) {
			model.Z.push_back(cone.A * model.T);
			model.z0.push_back(cone.A * model.t + cone.b);
		}

		// SCIP's linear constraints (its nonlinear ones, rewritten by
		// presolve, are only checked: their exported forms are evaluated).
		std::vector<Eigen::Triplet<double>> tA;
		std::vector<double> a0, l_lin, u_lin;
		SCIP_CONS** conss = SCIPgetConss(scip);
		for (int k = 0; k < SCIPgetNConss(scip); ++k) {
			SCIP_CONS* cons = conss[k];
			if (!SCIPconsIsActive(cons) || !SCIPconsIsChecked(cons)) continue;
			const std::string handler = SCIPconshdlrGetName(SCIPconsGetHdlr(cons));
			if (handler == "nonlinear") continue;
			SCIP_Bool ok_n = FALSE, ok_v = FALSE, ok_c = FALSE, ok_l = FALSE, ok_u = FALSE;
			int nv = 0;
			SCIP_CALL(SCIPgetConsNVars(scip, cons, &nv, &ok_n));
			std::vector<SCIP_VAR*> cv(std::max(nv, 1));
			std::vector<double> vals(std::max(nv, 1));
			if (ok_n) SCIP_CALL(SCIPgetConsVars(scip, cons, cv.data(), nv, &ok_v));
			if (ok_n) SCIP_CALL(SCIPgetConsVals(scip, cons, vals.data(), nv, &ok_c));
			const double lhs = SCIPconsGetLhs(scip, cons, &ok_l);
			const double rhs = SCIPconsGetRhs(scip, cons, &ok_u);
			if (!(ok_n && ok_v && ok_c && ok_l && ok_u))
				throw std::runtime_error(fmt::format(
					"constraint {} ({}) isn't linear or nonlinear; the local NLP relaxation can't read it",
					SCIPconsGetName(cons), handler));
			cv.resize(nv);
			vals.resize(nv);
			double constant;
			ToActive(scip, cv, vals, col_of, &terms, &constant);
			const int r = static_cast<int>(a0.size());
			for (const auto& [col, a] : terms) tA.emplace_back(r, col, a);
			a0.push_back(constant);
			l_lin.push_back(SCIPisInfinity(scip, -lhs) ? -kInf : lhs);
			u_lin.push_back(SCIPisInfinity(scip, rhs) ? kInf : rhs);
		}
		const int m_lin = static_cast<int>(a0.size());
		nlp.A.resize(m_lin, n);
		nlp.A.setFromTriplets(tA.begin(), tA.end());
		nlp.A.makeCompressed();
		nlp.a0 = Eigen::Map<const Eigen::VectorXd>(a0.data(), m_lin);
		nlp.l_lin = Eigen::Map<const Eigen::VectorXd>(l_lin.data(), m_lin);
		nlp.u_lin = Eigen::Map<const Eigen::VectorXd>(u_lin.data(), m_lin);
		std::vector<double> l, u;
		for (size_t r = 0; r < _mip.nonlinear.size(); ++r) {
			l.push_back(_mip.nonlinear[r].lb);
			u.push_back(_mip.nonlinear[r].ub);
		}
		for (size_t k = 0; k < _mip.cones.size(); ++k) {
			l.push_back(-kInf);
			u.push_back(0.0);
		}
		nlp.l = Eigen::Map<const Eigen::VectorXd>(l.data(), l.size());
		nlp.u = Eigen::Map<const Eigen::VectorXd>(u.data(), u.size());
		const int m = static_cast<int>(l.size());
		record.vars = n;
		record.rows = m_lin + m;

		// g's Jacobian pattern: each nonlinear row on its group's columns,
		// each cone row on its Z's columns.
		{
			auto columns = [](const SparseRows& M) {
				std::vector<int> cols;
				for (int r = 0; r < M.rows(); ++r)
					for (SparseRows::InnerIterator it(M, r); it; ++it) cols.push_back(static_cast<int>(it.col()));
				std::sort(cols.begin(), cols.end());
				cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
				return cols;
			};
			const int m_nl = static_cast<int>(_mip.nonlinear.size());
			std::vector<Eigen::Triplet<double>> tJ;
			for (size_t k = 0; k < _groups.size(); ++k) {
				const std::vector<int> cols = columns(model.Tg[k]);
				for (int r : _groups[k].rows)
					for (int col : cols) tJ.emplace_back(r, col, 0.0);
			}
			for (size_t k = 0; k < _mip.cones.size(); ++k)
				for (int col : columns(model.Z[k])) tJ.emplace_back(m_nl + static_cast<int>(k), col, 0.0);
			nlp.J.resize(m, n);
			nlp.J.setFromTriplets(tJ.begin(), tJ.end());
			nlp.J.makeCompressed();
		}
		model.scratch = Eigen::VectorXd::Zero(n);
		nlp.evaluate = [&](const Eigen::VectorXd& yy, Eigen::VectorXd* g, SparseRows* J) {
			Evaluate(model, yy, g, J);
		};
		record.build_time = since(tic);

		_solver->Solve(nlp, &y, &record);

		Eigen::VectorXd g;
		SparseRows J = nlp.J;
		Evaluate(model, y, &g, &J);
		const Eigen::VectorXd ay = nlp.A * y + nlp.a0;
		bool linear_ok = g.allFinite() && ay.allFinite();
		for (int r = 0; linear_ok && r < m_lin; ++r)
			if (!SCIPisFeasGE(scip, ay(r), nlp.l_lin(r)) || !SCIPisFeasLE(scip, ay(r), nlp.u_lin(r)))
				linear_ok = false;
		bool feasible = linear_ok;
		for (int r = 0; feasible && r < m; ++r)
			if (!SCIPisFeasGE(scip, g(r), nlp.l(r)) || !SCIPisFeasLE(scip, g(r), nlp.u(r))) feasible = false;

		++_stats->local_solves;
		if (!feasible) {
			++_stats->local_failures;
			const double viol = (nlp.l - g).cwiseMax(g - nlp.u).cwiseMax(0.0).lpNorm<Eigen::Infinity>();
			const double lin_viol =
				(nlp.l_lin - ay).cwiseMax(ay - nlp.u_lin).cwiseMax(0.0).lpNorm<Eigen::Infinity>();
			// No proof of infeasibility: with integers still free, branch
			// (no bound from this node); only a leaf is cut off.
			*result = SCIP_CUTOFF;
			for (int k = 0; k < n; ++k)
				if (SCIPvarIsIntegral(vars[k]) && SCIPvarGetLbLocal(vars[k]) < SCIPvarGetUbLocal(vars[k]) - 0.5) {
					*result = SCIP_DIDNOTRUN;
					break;
				}
			// A leaf still cut off for SCIP, kept outside it as a fallback.
			const double objective = c.dot(y);
			if (*result == SCIP_CUTOFF && linear_ok && std::isfinite(viol) &&
			    (viol < fallback_violation || (viol == fallback_violation && objective < fallback_objective))) {
				fallback = model.T * y + model.t;
				fallback_violation = viol;
				fallback_objective = objective;
			}
			_stats->last_local_failure =
				fmt::format("max violation {} on the nonlinear rows and cones, {} on the linear rows, after "
					    "{} outer iterations{}", viol, lin_viol, record.outer,
					    *result == SCIP_CUTOFF ? " (leaf)" : " (integers free)");
		} else {
			*lowerbound = c.dot(y) + SCIPgetTransObjoffset(scip);
			SCIP_CALL(SCIPsetRelaxSolVals(scip, relax, n, const_cast<SCIP_VAR**>(vars.data()), y.data(), TRUE));
			SCIP_CALL(SCIPmarkRelaxSolValid(scip, relax, TRUE));
			*result = SCIP_SUCCESS;
		}
		record.success = feasible;
		record.total_time = since(tic);
		_stats->local_solve_time += record.total_time;
		_stats->local_log.push_back(record);
		if (trace != nullptr) {
			SCIPNodeRecord& node = trace->Record(SCIPgetCurrentNode(scip));
			node.local_solves.push_back(static_cast<int>(_stats->local_log.size()) - 1);
			node.relaxation = feasible ? "feasible"
					  : *result == SCIP_CUTOFF ? "infeasible: cut off (leaf)"
								    : "infeasible: branched (integers free)";
			if (feasible) node.lower_bound = *lowerbound;
			node.assignments.clear();
			const Eigen::VectorXd x_final = model.T * y + model.t;
			for (int i = 0; i < _mip.vars.size(); ++i) {
				const std::string name = _mip.vars(i).get_name();
				if (name.rfind("Assignments", 0) == 0)
					node.assignments.push_back(fmt::format("{} = {:.3f}", name, x_final(i)));
			}
		}
		return SCIP_OKAY;
	}

	static constexpr double kInf = std::numeric_limits<double>::infinity();

public:
	// Where each local solve is filed under its node (null: not traced).
	NodeTrace* trace = nullptr;

private:
	const ExternalMIP& _mip;
	std::vector<SCIP_VAR*> _x, _tx;
	SCIPSolveStats* _stats;
	std::unique_ptr<LocalSolver> _solver;
	std::vector<Group> _groups;
};

}  // namespace

SCIPWaypointMPC::SCIPWaypointMPC(GraphOfConstraints& graph,
				 std::vector<CubicConfigurationSpline> splines,
				 bool enforce_rigidity,
				 WaypointObjective objective,
				 SubtourElimination subtour,
				 double length_weight,
				 double mip_gap,
				 bool warm_start,
				 std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models,
				 double max_tangent_halfwidth,
				 double time_limit,
				 std::map<std::string, ParamValue> params,
				 double feasibility_tol,
				 bool verbose,
				 bool fallback_to_least_violated,
				 LocalNLPSolver local_solver,
				 std::shared_ptr<InitializationOracle> initialization_oracle)
	: GCSWaypointMPC(graph, std::move(splines), WaypointSolver::kGurobi, enforce_rigidity, objective, subtour,
			 length_weight, mip_gap, warm_start, std::move(travel_time_models), max_tangent_halfwidth),
	  _time_limit(time_limit),
	  _params(std::move(params)),
	  _feasibility_tol(feasibility_tol),
	  _verbose(verbose),
	  _fallback_to_least_violated(fallback_to_least_violated),
	  _local_solver(local_solver) {
	_initialization_oracle = std::move(initialization_oracle);
}

bool SCIPWaypointMPC::Solve(const std::vector<int>& remaining_vertices,
			    const Eigen::VectorXd& x0) {
	const ExternalMIP& mip = BeginExternalSolve(remaining_vertices, x0);
	const int n = static_cast<int>(mip.c.size());

	SCIP* raw = nullptr;
	SCIP_CHECK(SCIPcreate(&raw));
	const std::unique_ptr<SCIP, ScipFree> owner(raw);
	SCIP* scip = owner.get();
	SCIP_CHECK(SCIPincludeDefaultPlugins(scip));
	SCIPsetMessagehdlrQuiet(scip, !_verbose);
	SCIP_CHECK(SCIPcreateProbBasic(scip, "gcs_waypoint"));

	// The problem holds its variables, so ours are released once added.
	std::vector<SCIP_VAR*> x(n);
	for (int i = 0; i < n; ++i) {
		const SCIP_VARTYPE type = !mip.is_integer(i) ? SCIP_VARTYPE_CONTINUOUS
					  : (mip.var_lb(i) >= 0.0 && mip.var_ub(i) <= 1.0) ? SCIP_VARTYPE_BINARY
											     : SCIP_VARTYPE_INTEGER;
		SCIP_CHECK(SCIPcreateVarBasic(scip, &x[i], fmt::format("x{}", i).c_str(),
					      ScipBound(scip, mip.var_lb(i)), ScipBound(scip, mip.var_ub(i)),
					      mip.c(i), type));
		SCIP_CHECK(SCIPaddVar(scip, x[i]));
		SCIP_VAR* ours = x[i];
		SCIP_CHECK(SCIPreleaseVar(scip, &ours));
	}
	if (mip.c0 != 0.0) SCIP_CHECK(SCIPaddOrigObjoffset(scip, mip.c0));

	for (int r = 0; r < mip.A.rows(); ++r) {
		if (mip.A.outerIndexPtr()[r] == mip.A.outerIndexPtr()[r + 1]) {
			if (mip.row_lb(r) > 0.0 || mip.row_ub(r) < 0.0)
				throw std::runtime_error(fmt::format("SCIPWaypointMPC: empty row {} requires {} <= 0 <= {}",
								     r, mip.row_lb(r), mip.row_ub(r)));
			continue;
		}
		AddLinearRow(scip, mip.A, r, x, mip.row_lb(r), mip.row_ub(r), fmt::format("r{}", r));
	}

	for (size_t k = 0; k < mip.cones.size(); ++k) {
		const ExternalLorentzCone& cone = mip.cones[k];
		std::vector<SCIP_VAR*> z(cone.A.rows());
		for (int r = 0; r < cone.A.rows(); ++r) {
			const std::string name = fmt::format("cone{}_z{}", k, r);
			SCIP_CHECK(SCIPcreateVarBasic(scip, &z[r], name.c_str(), r == 0 ? 0.0 : -SCIPinfinity(scip),
						      SCIPinfinity(scip), 0.0, SCIP_VARTYPE_CONTINUOUS));
			SCIP_CHECK(SCIPaddVar(scip, z[r]));
			// A_r x - z_r = -b_r
			AddLinearRow(scip, cone.A, r, x, -cone.b(r), -cone.b(r), fmt::format("cone{}_row{}", k, r), z[r], -1.0);
		}
		SCIP_CONS* soc;
		SCIP_CHECK(SCIPcreateConsBasicSOCNonlinear(scip, &soc, fmt::format("cone{}", k).c_str(),
							   static_cast<int>(z.size()) - 1, z.data() + 1, nullptr, nullptr,
							   0.0, z[0], 1.0, 0.0));
		// The local NLP relaxation holds the cones; SCIP only checks them.
		SCIP_CHECK(SCIPsetConsSeparated(scip, soc, FALSE));
		SCIP_CHECK(SCIPsetConsEnforced(scip, soc, FALSE));
		SCIP_CHECK(SCIPsetConsPropagated(scip, soc, FALSE));
		SCIP_CHECK(SCIPaddCons(scip, soc));
		SCIP_CHECK(SCIPreleaseCons(scip, &soc));
		for (SCIP_VAR*& v : z) SCIP_CHECK(SCIPreleaseVar(scip, &v));
	}

	_stats = SCIPSolveStats{};
	auto* trace = new NodeTrace(scip, [&mip](SCIP_VAR* v) { return ExportedName(mip, v); });
	SCIP_CHECK(SCIPincludeObjEventhdlr(scip, trace, TRUE));
	{
		// SCIP only checks the nonlinear rows (presolve carries them through
		// its reductions, and they lock their variables) and never branches,
		// separates or propagates on them: the relaxation holds them. (SCIP
		// requires them initial; the LP is never solved.)
		VarMap var_of;
		for (int i = 0; i < n; ++i) var_of.emplace(mip.vars(i).get_id(), x[i]);
		for (size_t k = 0; k < mip.nonlinear.size(); ++k) {
			const ExternalNonlinear& nl = mip.nonlinear[k];
			SCIP_EXPR* expr = ToScipExpr(scip, nl.e, var_of);
			SCIP_CONS* cons;
			SCIP_CHECK(SCIPcreateConsNonlinear(scip, &cons, fmt::format("nl{}", k).c_str(), expr,
							   ScipBound(scip, nl.lb), ScipBound(scip, nl.ub), TRUE, FALSE, FALSE,
							   TRUE, FALSE, FALSE, FALSE, FALSE, FALSE));
			SCIP_CHECK(SCIPreleaseExpr(scip, &expr));
			SCIP_CHECK(SCIPaddCons(scip, cons));
			SCIP_CHECK(SCIPreleaseCons(scip, &cons));
		}
	}
	// Every node's relaxation, in place of SCIP's LP.
	auto* local_nlp = new LocalNLPRelax(scip, mip, x, &_stats, _local_solver);
	local_nlp->trace = trace;
	SCIP_CHECK(SCIPincludeObjRelax(scip, local_nlp, TRUE));
	SCIP_CHECK(SCIPsetIntParam(scip, "lp/solvefreq", -1));

	if (mip.initial_guess.array().isFinite().any()) {
		SCIP_SOL* start;
		SCIP_CHECK(SCIPcreatePartialSol(scip, &start, nullptr));
		for (int i = 0; i < n; ++i)
			if (std::isfinite(mip.initial_guess(i)))
				SCIP_CHECK(SCIPsetSolVal(scip, start, x[i], mip.initial_guess(i)));
		SCIP_Bool stored;
		SCIP_CHECK(SCIPaddSolFree(scip, &start, &stored));
	}

	SCIP_CHECK(SCIPsetRealParam(scip, "limits/gap", _mip_gap));
	if (std::isfinite(_time_limit)) SCIP_CHECK(SCIPsetRealParam(scip, "limits/time", _time_limit));
	// 2.5-6x faster than SCIP's defaults on test_scip_waypoint_mpc.py's min-max cases.
	SCIP_CHECK(SCIPsetIntParam(scip, "separating/aggregation/freq", -1));
	SCIP_CHECK(SCIPsetIntParam(scip, "presolving/maxrestarts", 0));
	SCIP_CHECK(SCIPsetIntParam(scip, "separating/maxrounds", 1));
	for (const auto& [name, value] : _params) SetParam(scip, name, value);

	Configure(scip, x, mip);

	const SCIP_RETCODE solved = SCIPsolve(scip);
	if (!local_nlp->error.empty())
		throw std::runtime_error("SCIPWaypointMPC: local NLP: " + local_nlp->error);
	SCIP_CHECK(solved);
	if (_verbose) SCIP_CHECK(SCIPprintStatistics(scip, nullptr));

	_stats.status = StatusName(SCIPgetStatus(scip));
	_stats.nodes = SCIPgetNTotalNodes(scip);
	_stats.primal_bound = SCIPgetPrimalbound(scip);
	_stats.dual_bound = SCIPgetDualbound(scip);
	_stats.gap = SCIPgetGap(scip);
	_stats.solving_time = SCIPgetSolvingTime(scip);
	_stats.feasible_leaves = SCIPgetNFeasibleLeaves(scip);
	_stats.infeasible_leaves = SCIPgetNInfeasibleLeaves(scip);
	_stats.objlim_leaves = SCIPgetNObjlimLeaves(scip);
	for (long long number : trace->order) _stats.node_log.push_back(trace->nodes.at(number));
	std::vector<std::pair<double, double>> history;
	SCIP_SOL** sols = SCIPgetSols(scip);
	for (int k = 0; k < SCIPgetNSols(scip); ++k)
		history.emplace_back(SCIPsolGetTime(sols[k]), SCIPgetSolOrigObj(scip, sols[k]));
	std::sort(history.begin(), history.end());
	_stats.solution_times.clear();
	_stats.solution_objectives.clear();
	for (const auto& [t, obj] : history) {
		_stats.solution_times.push_back(t);
		_stats.solution_objectives.push_back(obj);
	}

	if (SCIPgetNSols(scip) == 0) {
		if (_fallback_to_least_violated && local_nlp->fallback.size()) {
			Eigen::VectorXd fallback = local_nlp->fallback;
			for (int i = 0; i < n; ++i)
				if (mip.is_integer(i)) fallback(i) = std::round(fallback(i));
			_stats.fallback = true;
			_stats.fallback_violation = local_nlp->fallback_violation;
			if (_verbose)
				std::cerr << "SCIPWaypointMPC: no feasible solution (" << _stats.status
					  << "); returning the least-violated leaf (violation "
					  << local_nlp->fallback_violation << ").\n";
			// Its nonlinear rows and cones miss by fallback_violation; the
			// linear rows held at its node.
			FinishExternalSolve(fallback, std::numeric_limits<double>::infinity());
			return true;
		}
		std::cerr << "SCIPWaypointMPC: no feasible solution (" << _stats.status << ").\n";
		return false;
	}
	SCIP_SOL* best = SCIPgetBestSol(scip);
	Eigen::VectorXd sol(n);
	for (int i = 0; i < n; ++i) sol(i) = SCIPgetSolVal(scip, best, x[i]);
	FinishExternalSolve(sol, _feasibility_tol);
	return true;
}
