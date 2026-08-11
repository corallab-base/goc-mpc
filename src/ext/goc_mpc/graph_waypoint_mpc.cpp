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
