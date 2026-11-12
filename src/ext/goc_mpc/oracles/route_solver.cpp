#include "route_solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "../graph_of_constraints.hpp"
#include "../waypoint_common.hpp"

using drake::solvers::MathematicalProgram;
using drake::solvers::MatrixXDecisionVariable;
using drake::solvers::VectorXDecisionVariable;
using drake::symbolic::Expression;

ArcCost DistanceArcCost(bool l1) {
	return [l1](int, const Eigen::MatrixXd& from, const Eigen::MatrixXd& to) -> Eigen::VectorXd {
		const Eigen::MatrixXd d = to - from;
		return l1 ? Eigen::VectorXd(d.cwiseAbs().rowwise().sum()) : Eigen::VectorXd(d.rowwise().norm());
	};
}

RouteSolution SolveRoutes(const GraphOfConstraints& graph, const InitialGuess& guess, const Eigen::VectorXd& x0,
			  const ArcCost& arc_cost, bool min_max, double length_weight, double mip_gap) {
	const auto t0 = std::chrono::steady_clock::now();
	constexpr double kInf = std::numeric_limits<double>::infinity();
	if (x0.size() != graph.total_dim)
		throw std::invalid_argument(fmt::format("SolveRoutes: x0 has {} entries, the graph's state {}", x0.size(),
							graph.total_dim));
	RouteSolution out;
	out.nodes = guess.nodes;
	const std::vector<int>& nodes = guess.nodes;
	const int n_A = graph.num_agents, F = static_cast<int>(guess.feasible.size());
	if (F == 0) {
		out.log.push_back("no feasible assignment in the guess");
		return out;
	}
	const SubgraphOfConstraints subgraph(const_cast<GraphOfConstraints*>(&graph), nodes);
	const RoutingPresolve presolve = BuildRoutingPresolve(graph, subgraph);
	const int n_V = subgraph.num_nodes();
	std::vector<int> node_of(n_V, -1), row_of(n_V, -1);
	for (size_t r = 0; r < nodes.size(); ++r) {
		const int sg = subgraph.subgraph_id(nodes[r]);
		if (sg >= 0) node_of[sg] = nodes[r], row_of[sg] = static_cast<int>(r);
	}
	auto agent_q = [&](const Eigen::MatrixXd& X, int r, int j) -> Eigen::VectorXd {
		return X.row(r).segment(graph.agent_col_offset(j), graph.robot_ambient_dim(j)).transpose();
	};

	// One configuration per (node, agent): the first feasible assignment's
	// that has the agent there.
	std::vector<std::vector<Eigen::VectorXd>> q(n_V, std::vector<Eigen::VectorXd>(n_A));
	std::vector<std::vector<std::vector<bool>>> visits(F, std::vector<std::vector<bool>>(n_A, std::vector<bool>(n_V)));
	for (int f = 0; f < F; ++f) {
		const AssignmentGuess& fa = guess.feasible[f];
		if (fa.X.rows() != static_cast<int>(nodes.size()) || fa.X.cols() != graph.total_dim)
			throw std::invalid_argument("SolveRoutes: a feasible guess's X doesn't match the guess's nodes");
		for (int j = 0; j < n_A; ++j)
			for (int node : fa.routes.at(j)) {
				const int sg = subgraph.subgraph_id(node);
				if (sg < 0) continue;
				visits[f][j][sg] = true;
				const Eigen::VectorXd qj = agent_q(fa.X, row_of[sg], j);
				if (q[sg][j].size() == 0) q[sg][j] = qj;
				else if (!q[sg][j].isApprox(qj, 1e-9))
					out.log.push_back(fmt::format("node {} agent {}: assignments place it differently; using the "
								      "first's configuration",
								      node_of[sg], j));
			}
	}

	MathematicalProgram prog;
	// Assignment selector, and the assignment matrix it implies.
	const VectorXDecisionVariable s = prog.NewBinaryVariables(F, "s");
	prog.AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(F), 1.0, s);
	const int n_vars = subgraph.num_variables();
	const MatrixXDecisionVariable A = prog.NewContinuousVariables(n_vars, n_A, "Assignments");
	std::vector<int> var_of_row(n_vars, -1);
	for (const auto& [var, row] : subgraph._variable_to_subgraph_variable_id) var_of_row[row] = var;
	for (int k = 0; k < n_vars; ++k)
		for (int j = 0; j < n_A; ++j) {
			Expression e = Expression(A(k, j));
			for (int f = 0; f < F; ++f) {
				const auto it = guess.feasible[f].assignments.find(var_of_row[k]);
				if (it != guess.feasible[f].assignments.end() && it->second == j) e -= Expression(s(f));
			}
			prog.AddLinearEqualityConstraint(e, 0.0);
		}
	AddRoleExactlyOneAgent(prog, subgraph, A, n_A);
	AddCommittedAssignments(prog, graph, subgraph, A);
	AddAssignmentVariableConstraints(prog, graph, subgraph, A);

	// Node timestamps (routing indices: depot 0, node sg + 1, sink n_V + 1).
	const int n_N = n_V + 2;
	const double M = n_N;
	const VectorXDecisionVariable t = prog.NewContinuousVariables(n_N, "t");
	prog.AddBoundingBoxConstraint(0.0, M, t);
	prog.AddBoundingBoxConstraint(0.0, 0.0, t(0));
	const std::vector<drake::symbolic::Variable> cond = AddConditionBinaryVars(prog, graph);
	Eigen::VectorXi previous = Eigen::VectorXi::Constant(graph.num_variables, -1);
	AddOrderingConstraints(prog, graph, subgraph, t, M, A, previous, cond);

	// Per agent: arcs between the nodes it can visit, flow = its visits.
	struct Arc {
		int j, a, b;  // routing indices
		double length;
	};
	std::vector<Arc> arcs;
	std::vector<VectorXDecisionVariable> Z(n_A);
	std::vector<Expression> agent_cost(n_A);
	for (int j = 0; j < n_A; ++j) {
		const int d = graph.robot_ambient_dim(j);
		const Eigen::VectorXd start = x0.segment(graph.agent_col_offset(j), d);
		std::vector<int> can;  // routing indices of nodes the agent may visit
		for (int sg = 0; sg < n_V; ++sg)
			if (q[sg][j].size() > 0) can.push_back(sg + 1);
		auto config = [&](int ri) -> const Eigen::VectorXd& { return ri == 0 ? start : q[ri - 1][j]; };
		std::vector<int> tails = can, heads = can;
		tails.insert(tails.begin(), 0);
		heads.push_back(n_N - 1);
		std::vector<Arc> mine;
		for (int a : tails)
			for (int b : heads)
				if (a != b && presolve.arc_allowed(j, a, b)) mine.push_back({j, a, b, 0.0});
		// Every priced arc's cost in one call.
		std::vector<int> priced;
		for (size_t k = 0; k < mine.size(); ++k)
			if (mine[k].b != n_N - 1) priced.push_back(static_cast<int>(k));
		if (!priced.empty()) {
			Eigen::MatrixXd from(priced.size(), d), to(priced.size(), d);
			for (size_t r = 0; r < priced.size(); ++r) {
				from.row(r) = config(mine[priced[r]].a).transpose();
				to.row(r) = config(mine[priced[r]].b).transpose();
			}
			const Eigen::VectorXd c = arc_cost(j, from, to);
			if (c.size() != static_cast<int>(priced.size()) || !c.allFinite())
				throw std::runtime_error(fmt::format("SolveRoutes: agent {}'s arc costs are malformed", j));
			for (size_t r = 0; r < priced.size(); ++r) mine[priced[r]].length = c(r);
		}
		Z[j] = prog.NewBinaryVariables(mine.size(), fmt::format("Z_{}", j));
		std::vector<Expression> in(n_N), outflow(n_N);
		for (size_t k = 0; k < mine.size(); ++k) {
			const Arc& arc = mine[k];
			outflow[arc.a] += Expression(Z[j](k));
			in[arc.b] += Expression(Z[j](k));
			agent_cost[j] += arc.length * Z[j](k);
			// A used arc moves strictly forward in time.
			prog.AddLinearConstraint(Expression(t(arc.b)) - Expression(t(arc.a)) - 1.0 + M * (1.0 - Z[j](k)), 0.0,
						 kInf);
		}
		prog.AddLinearEqualityConstraint(outflow[0], 1.0);
		prog.AddLinearEqualityConstraint(in[n_N - 1], 1.0);
		for (int sg = 0; sg < n_V; ++sg) {
			// Visit indicator: the selected assignment has the agent here.
			Expression visit;
			for (int f = 0; f < F; ++f)
				if (visits[f][j][sg]) visit += Expression(s(f));
			prog.AddLinearEqualityConstraint(in[sg + 1] - visit, 0.0);
			prog.AddLinearEqualityConstraint(outflow[sg + 1] - visit, 0.0);
		}
		arcs.insert(arcs.end(), mine.begin(), mine.end());
	}
	Expression total;
	for (const Expression& c : agent_cost) total += c;
	if (min_max) {
		const auto longest = prog.NewContinuousVariables<1>("longest")(0);
		for (const Expression& c : agent_cost) prog.AddLinearConstraint(Expression(longest) - c, 0.0, kInf);
		prog.AddLinearCost(Expression(longest) + length_weight * total);
	} else {
		prog.AddLinearCost(total);
	}

	drake::solvers::SolverOptions options;
	options.SetOption(drake::solvers::GurobiSolver::id(), "MIPGap", mip_gap);
	options.SetOption(drake::solvers::GurobiSolver::id(), "OutputFlag", 0);
	const auto result = drake::solvers::Solve(prog, std::nullopt, options);
	out.solver = result.get_solver_id().name();
	out.arcs = static_cast<int>(arcs.size());
	if (!result.is_success()) {
		out.log.push_back(fmt::format("{}: {}", out.solver, fmt::format("{}", result.get_solution_result())));
		out.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
		return out;
	}
	out.success = true;
	out.objective = result.get_optimal_cost();
	int chosen = 0;
	for (int f = 0; f < F; ++f)
		if (result.GetSolution(s(f)) > 0.5) chosen = f;
	out.assignments = guess.feasible[chosen].assignments;
	out.X = guess.feasible[chosen].X;
	// Each agent's route: follow its used arcs from the depot.
	out.routes.assign(n_A, {});
	out.agent_costs.assign(n_A, 0.0);
	size_t first = 0;
	for (int j = 0; j < n_A; ++j) {
		const Eigen::VectorXd z = result.GetSolution(Z[j]);
		for (int at = 0; at != n_N - 1;) {
			int next = -1;
			for (size_t k = 0; k < static_cast<size_t>(z.size()); ++k) {
				const Arc& arc = arcs[first + k];
				if (arc.a == at && z(k) > 0.5) {
					next = arc.b;
					out.agent_costs[j] += arc.length;
					break;
				}
			}
			if (next < 0) throw std::runtime_error("SolveRoutes: a route breaks off");
			if (next != n_N - 1) out.routes[j].push_back(node_of[next - 1]);
			at = next;
		}
		first += z.size();
	}
	for (double c : out.agent_costs) out.cost += c;
	out.times.assign(nodes.size(), std::numeric_limits<double>::quiet_NaN());
	for (int sg = 0; sg < n_V; ++sg) out.times[row_of[sg]] = result.GetSolution(t(sg + 1));
	std::vector<std::pair<double, int>> by_time;
	for (int sg = 0; sg < n_V; ++sg) {
		bool visited = false;
		for (int j = 0; j < n_A; ++j) visited = visited || visits[chosen][j][sg];
		if (visited) by_time.push_back({result.GetSolution(t(sg + 1)), node_of[sg]});
	}
	std::sort(by_time.begin(), by_time.end());
	for (const auto& [time, node] : by_time) out.order.push_back(node);
	std::vector<std::string> named;
	for (const auto& [var, j] : out.assignments) named.push_back(fmt::format("var {} -> agent {}", var, j));
	out.log.push_back(fmt::format("{}: {} feasible assignments, {} arcs; chose [{}], cost {:.4f}", out.solver, F,
				      out.arcs, fmt::join(named, ", "), out.cost));
	out.time = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
	return out;
}
