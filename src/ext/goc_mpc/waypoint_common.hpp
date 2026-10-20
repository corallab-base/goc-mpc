#pragma once

#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/mathematical_program_result.h>

#include "graph_of_constraints.hpp"
#include "graph_waypoint_mpc.hpp"
#include "../configuration_spline.hpp"

// Builder pieces shared by the Drake waypoint-phase formulations
// (MILPWaypointMPC, milp_waypoint_mpc.*): everything that doesn't depend on
// how routing and per-node configurations are encoded. Assignments (one row
// per subgraph variable, one column per agent), cond_binary_vars, and the
// routing-node timestamps t (indexed by RoutingPresolve's routing indices)
// are the decision variables these share.

enum class WaypointSolver { kGurobi, kMosek, kIPOPT };

// Subtour elimination. kMCF: a multi-commodity flow (one commodity per node
// an agent may visit, routed from the depot within the arcs the agent uses),
// the compact form of the generalized subtour elimination constraints.
// kNone: rely on the time-propagation rows alone, which still allow a
// detached cycle of zero-length arcs at equal times.
enum class SubtourElimination { kMCF, kNone };

// Routing graph N = {depot=0} ∪ task_nodes(1..n_V) ∪ {sink=n_V+1}: task node
// with subgraph id sg has routing index sg + 1.
struct RoutingPresolve {
	int n_V{0};
	// Which agents and roles each task node references (subgraph-indexed):
	// a node references a role/agent through its own symbolic constraints,
	// an edge constraint at either endpoint (u_/v_ placeholders at their own
	// endpoint, plain ones at both), or a hold's endpoints. required_at:
	// agents that must visit the node.
	std::vector<std::set<int>> roles_at, required_at;
	// before[u][v]: u must come no later than v (transitive closure of the
	// hard orderings over the remaining task nodes).
	std::vector<std::vector<bool>> before;

	static constexpr int depot = 0;
	int sink() const { return n_V + 1; }
	static int ri(int sg) { return sg + 1; }

	// Routing indices: depot before everything, sink after everything.
	bool precedes(int a, int b) const;

	// Arc pruning: arcs no optimal route needs, given the hard orderings.
	// Agent j's arc a -> b (routing indices) is pruned if
	//   - b must come before a, or
	//   - some node c that j must visit has to come between them (a before
	//     c before b).
	// Sound: with the orderings non-strict (t(u) <= t(v)), a route can only
	// take such an arc at equal times, over zero-length arcs, i.e. between
	// waypoints equal in every distance-measured block; visiting those in
	// the ordered sequence instead costs the same. For a chain visited by one
	// agent only the n+1 consecutive arcs remain, instead of ~n^2 binaries
	// with big-M rows that the solver would branch over.
	bool arc_allowed(int j, int a, int b) const;
};

RoutingPresolve BuildRoutingPresolve(const GraphOfConstraints& graph,
				     const SubgraphOfConstraints& subgraph);

// L1 distance between two symbolic configuration expressions, summed over a
// spline's R blocks only, encoded via one slack variable per component plus
// two linear inequalities (no big-M, LP-representable).
drake::symbolic::Expression ComputeL1Distance(
	drake::solvers::MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<drake::symbolic::Expression>& x1,
	const VecX<drake::symbolic::Expression>& x2,
	const std::string& tag);

// L2 (Euclidean) distance between two symbolic configuration expressions,
// summed over a spline's R blocks only, one nonnegative scalar per block
// constrained via a Lorentz cone.
drake::symbolic::Expression ComputeL2Distance(
	drake::solvers::MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const VecX<drake::symbolic::Expression>& x1,
	const VecX<drake::symbolic::Expression>& x2,
	const std::string& tag);

// Compile b_same: binary MILP variable = 1 iff variables r0 and r1 are
// assigned to the same agent in the current (or previous) solve.
drake::symbolic::Variable CompileVarEq(
	drake::solvers::MathematicalProgram& prog,
	int r0, int r1,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments);

// Compile a conditional ordering formula to a binary MILP variable.
// Returns a variable b such that b=1 iff the formula holds.
// Recognised atoms: VarEq (r_sym_i == r_sym_j) and BinVarEq (bv_sym == 0/1).
// Compound formulas use Drake FormulaKind::And / Or / Not.
drake::symbolic::Variable CompileConditionToBinary(
	drake::solvers::MathematicalProgram& prog,
	const drake::symbolic::Formula& f,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const SubgraphOfConstraints& subgraph,
	const Eigen::VectorXi& prev_var_assignments,
	const GraphOfConstraints* graph,
	const std::vector<drake::symbolic::Variable>& cond_binary_vars);

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: t(t_idx_w) weakly between t(t_idx_u) and t(t_idx_v)  ⟹  b = 1.
// One-directional by design: free to be 0 when the node isn't between, but
// never forced to 0 when it genuinely is between.
drake::symbolic::Variable GetOrAddBetweennessIndicator(
	drake::solvers::MathematicalProgram& prog,
	const drake::solvers::VectorXDecisionVariable& t_var,
	int t_idx_u, int t_idx_w, int t_idx_v,
	double M_time,
	std::map<std::tuple<int, int, int>, drake::symbolic::Variable>& cache);

// Returns (creating and caching if necessary) a binary MILP variable b such
// that: [t(t_idx_a1), t(t_idx_b1)] overlaps [t(t_idx_a2), t(t_idx_b2)] ⟹ b=1.
// Same one-directional soundness contract as GetOrAddBetweennessIndicator.
drake::symbolic::Variable GetOrAddIntervalOverlapIndicator(
	drake::solvers::MathematicalProgram& prog,
	const drake::solvers::VectorXDecisionVariable& t_var,
	int t_idx_a1, int t_idx_b1, int t_idx_a2, int t_idx_b2,
	double M_time,
	std::map<std::tuple<int, int, int, int>, drake::symbolic::Variable>& cache);

// One binary MILP variable per binary condition symbolic variable in the graph.
std::vector<drake::symbolic::Variable> AddConditionBinaryVars(
	drake::solvers::MathematicalProgram& prog,
	const GraphOfConstraints& graph);

// Every subgraph variable (role) goes to exactly one agent.
void AddRoleExactlyOneAgent(
	drake::solvers::MathematicalProgram& prog,
	const SubgraphOfConstraints& subgraph,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	int num_agents);

// Assignment commits (see add_variable_commit /
// GraphOfConstraints::committed_assignments).
void AddCommittedAssignments(
	drake::solvers::MathematicalProgram& prog,
	const GraphOfConstraints& graph,
	const SubgraphOfConstraints& subgraph,
	const drake::solvers::MatrixXDecisionVariable& Assignments);

// DAG ordering over the routing-node timestamps t_var: hard (structure
// edges) t(u) <= t(v), or conditional (_conditional_ordering_map), enforced
// only when its condition compiles to 1.
void AddOrderingConstraints(
	drake::solvers::MathematicalProgram& prog,
	const GraphOfConstraints& graph,
	const SubgraphOfConstraints& subgraph,
	const drake::solvers::VectorXDecisionVariable& t_var,
	double M_time,
	const drake::solvers::MatrixXDecisionVariable& Assignments,
	const Eigen::VectorXi& previous_var_assignments,
	const std::vector<drake::symbolic::Variable>& cond_binary_vars);

// Assignment-variable constraints (add_variable_constraint /
// add_variable_ineq_constraint), for variables present in this subgraph.
void AddAssignmentVariableConstraints(
	drake::solvers::MathematicalProgram& prog,
	const GraphOfConstraints& graph,
	const SubgraphOfConstraints& subgraph,
	const drake::solvers::MatrixXDecisionVariable& Assignments);

// Stationary objects: an object not currently being held must not move
// between two graph nodes. `state` is (subgraph nodes x graph.total_dim);
// only its object columns are read.
void AddStationaryObjectConstraints(
	drake::solvers::MathematicalProgram& prog,
	const GraphOfConstraints& graph,
	const SubgraphOfConstraints& subgraph,
	const drake::MatrixX<drake::symbolic::Expression>& state,
	const Eigen::VectorXd& x0,
	const drake::solvers::VectorXDecisionVariable& t_var,
	double M_time);

// kMCF rows for one agent's routing graph. `arcs` are (tail, head) vertex
// indices, tail -1 for the depot and head -2 for the sink; `Z` holds one
// binary per arc; vertex_node maps a vertex to its subgraph node and
// visit[sg] is the agent's visit indicator for node sg.
void AddMultiCommodityFlow(
	drake::solvers::MathematicalProgram& prog,
	const std::vector<std::pair<int, int>>& arcs,
	const drake::solvers::VectorXDecisionVariable& Z,
	const std::vector<int>& vertex_node,
	const std::vector<drake::symbolic::Expression>& visit,
	const std::string& tag);

// MinMax cost: makespan + length_weight * total_cost, where total_cost sums
// every agent's route cost. A small weight breaks the makespan objective's
// ties toward short routes for the agents that don't set the makespan; it
// can trade a makespan increase of about that weight for shorter routes.
drake::symbolic::Expression MinMaxObjectiveCost(
	const drake::symbolic::Variable& makespan,
	const drake::symbolic::Expression& total_cost,
	double length_weight);

// Throws unless length_weight >= 0, and 0 for the Avg objectives (which
// already minimize the total route cost).
void CheckLengthWeight(WaypointObjective objective, double length_weight);

void PrintSolverReport(const drake::solvers::MathematicalProgram& prog,
		       const drake::solvers::MathematicalProgramResult& result,
		       double tol = 1e-6);
