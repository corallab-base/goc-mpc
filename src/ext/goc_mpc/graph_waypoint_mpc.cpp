#include "graph_waypoint_mpc.hpp"

GraphWaypointMPC::GraphWaypointMPC(GraphOfConstraints& graph,
				   std::vector<CubicConfigurationSpline> splines,
				   WaypointObjective objective)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _objective(objective) {
	// Allocate persistent output buffers.
	_waypoints = Eigen::MatrixXd::Zero(_graph->structure.num_nodes(), _graph->total_dim);
	_assignments = Eigen::VectorXi::Constant(_graph->num_phis, -1);
	_var_assignments = Eigen::VectorXi::Constant(_graph->num_variables, -1);
	_first_cycle = true;
	_t_by_node_id = Eigen::VectorXd::Zero(_graph->structure.num_nodes());
}

void GraphWaypointMPC::BeginSolve(const std::vector<int>& remaining_vertices,
				  const Eigen::VectorXd& x0) {
	_timer.Start();

	// Warm start configurations
	if (_first_cycle) {
		for (int v : remaining_vertices) {
			_waypoints.row(v) = x0.transpose();
		}
		_first_cycle = false;
	}

	// Live-rebind: a node's row in `_waypoints`, once it's no longer in
	// remaining_vertices, is otherwise whatever the solver's write-back last
	// planned for it while still active -- the nominal solved target, not the
	// real state the graph actually reached. That's stale for a relational
	// edge constraint anchored at that node (e.g. a rigid-transport edge):
	// CompileSymbolicEdgeConstraint substitutes a passed endpoint's row as a
	// constant read from `_waypoints` (passed through as `previous_X` -- see
	// symbolic_constraint_compiler.cpp), so it would otherwise propagate the
	// *planned* transform forward even once the real one has diverged (e.g. a
	// grasp whose real standoff doesn't match the planned one). So the first
	// Solve() call that sees a node freshly missing from remaining_vertices
	// (vs. the previous call) overwrites that node's row with the real x0
	// given on THIS call, once, before anything reads it as previous_X -- not
	// every cycle after, since the contract is a frozen constant once a node
	// has passed, not a continuously-tracked one. Mirrors
	// EvolutionaryWaypointSolver's equivalent fix
	// (evolutionary_waypoint_solver/mpc.py's solve()).
	const std::set<int> remaining_set(remaining_vertices.begin(), remaining_vertices.end());
	if (_has_prev_remaining_set) {
		for (int v : _prev_remaining_set) {
			if (remaining_set.contains(v)) continue;
			_waypoints.row(v) = x0.transpose();
		}
	}
	_prev_remaining_set = remaining_set;
	_has_prev_remaining_set = true;
}

void GraphWaypointMPC::StoreAssignments(const SubgraphOfConstraints& subgraph,
					const drake::solvers::MatrixXDecisionVariable& Assignments,
					const drake::solvers::MathematicalProgramResult& result) {
	const int num_phis = _graph->num_phis;
	const int num_agents = _graph->num_agents;

	for (int i = 0; i < num_phis; ++i) {
		if (_graph->phi_to_variable_map.contains(i)) {
			const int variable_id = _graph->phi_to_variable_map.at(i);
			const int variable_k = subgraph.subgraph_variable_id(variable_id);
			if (variable_k >= 0 && Assignments.rows() > 0) {
				for (int j = 0; j < num_agents; ++j) {
					const double val = result.GetSolution(Assignments(variable_k, j));
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
}
