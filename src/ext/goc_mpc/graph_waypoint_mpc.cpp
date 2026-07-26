#include "graph_waypoint_mpc.hpp"

#include <stdexcept>

GraphWaypointMPC::GraphWaypointMPC(GraphOfConstraints& graph,
				   std::vector<CubicConfigurationSpline> splines,
				   WaypointObjective objective,
				   std::shared_ptr<EdgeCostFunctor> edge_cost_fn)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _objective(objective),
	  _edge_cost_fn(std::move(edge_cost_fn)) {
	// Allocate persistent output buffers.
	_waypoints = Eigen::MatrixXd::Zero(_graph->structure.num_nodes(), _graph->total_dim);
	_assignments = Eigen::VectorXi::Constant(_graph->num_phis, -1);
	_var_assignments = Eigen::VectorXi::Constant(_graph->num_variables, -1);
	_first_cycle = true;
	_t_by_node_id = Eigen::VectorXd::Zero(_graph->structure.num_nodes());
}

double GraphWaypointMPC::EvaluateEdgeCost(int agent,
					  const Eigen::VectorXd& w_a,
					  const Eigen::VectorXd& w_b) const {
	if (!_edge_cost_fn) {
		throw std::runtime_error(
			"GraphWaypointMPC::EvaluateEdgeCost: no edge cost functor "
			"configured — pass edge_cost_fn= when constructing the solver");
	}
	return _edge_cost_fn->Evaluate(agent, w_a, w_b);
}
