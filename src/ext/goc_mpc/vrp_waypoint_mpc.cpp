#include "vrp_waypoint_mpc.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include <fmt/format.h>

namespace {

bool IsTravelTime(WaypointObjective o) {
	return o == WaypointObjective::kMinMaxTravelTime || o == WaypointObjective::kAvgTravelTime;
}

bool IsMinMax(WaypointObjective o) {
	return o == WaypointObjective::kMinMaxL1 || o == WaypointObjective::kMinMaxL2 ||
	       o == WaypointObjective::kMinMaxTravelTime;
}

}  // namespace

VRPWaypointMPC::VRPWaypointMPC(GraphOfConstraints& graph, std::vector<CubicConfigurationSpline> splines,
			       std::shared_ptr<InitializationOracle> initialization_oracle, WaypointObjective objective,
			       std::vector<std::shared_ptr<ArcTravelTimeModel>> travel_time_models, double length_weight,
			       double mip_gap)
	: GraphWaypointMPC(graph, std::move(splines), objective),
	  _initialization_oracle(std::move(initialization_oracle)),
	  _travel_time_models(std::move(travel_time_models)), _length_weight(length_weight), _mip_gap(mip_gap) {
	if (!_initialization_oracle) throw std::invalid_argument("VRPWaypointMPC: needs an initialization oracle");
	if (IsTravelTime(objective) != !_travel_time_models.empty())
		throw std::invalid_argument("VRPWaypointMPC: travel_time_models go with (only) a travel-time objective");
	if (!_travel_time_models.empty() && static_cast<int>(_travel_time_models.size()) != graph.num_agents)
		throw std::invalid_argument(fmt::format("VRPWaypointMPC: {} travel-time models for {} agents",
							_travel_time_models.size(), graph.num_agents));
	for (const auto& model : _travel_time_models)
		if (!model) throw std::invalid_argument("VRPWaypointMPC: a travel-time model is None");
	if (length_weight < 0.0) throw std::invalid_argument("VRPWaypointMPC: length_weight must be >= 0");
}

bool VRPWaypointMPC::Solve(const std::vector<int>& remaining_vertices, const Eigen::VectorXd& x0) {
	BeginSolve(remaining_vertices, x0);
	GraphOfConstraints& g = *_graph;

	InitializationRequest request;
	request.graph = &g;
	request.nodes = remaining_vertices;
	request.x0 = x0;
	request.previous_assignments = _var_assignments;
	request.previous = _initial_guess ? &*_initial_guess : nullptr;
	InitialGuess guess = _initialization_oracle->Initialize(request);

	ArcCost arc_cost = DistanceArcCost(_objective == WaypointObjective::kMinMaxL1 ||
					   _objective == WaypointObjective::kAvgL1);
	if (IsTravelTime(_objective)) {
		arc_cost = [this](int j, const Eigen::MatrixXd& from, const Eigen::MatrixXd& to) -> Eigen::VectorXd {
			const ArcTravelTimeModel& model = *_travel_time_models[j];
			const int m = model.dim();
			for (const Eigen::MatrixXd* P : {&from, &to})
				for (int r = 0; r < P->rows(); ++r)
					if ((P->row(r).head(m).transpose().array() < model.lower.array() - 1e-9).any() ||
					    (P->row(r).head(m).transpose().array() > model.upper.array() + 1e-9).any())
						throw std::runtime_error(fmt::format(
							"VRPWaypointMPC: agent {}'s configuration lies outside its travel-time "
							"model's domain",
							j));
			return std::get<0>(model.Linearize(from.leftCols(m), to.leftCols(m)));
		};
	}
	RouteSolution routes = SolveRoutes(g, guess, x0, arc_cost, IsMinMax(_objective), _length_weight, _mip_gap);
	_initial_guess = std::move(guess);
	_route_solution = routes;
	if (!routes.success) {
		_last_solve_time = _timer.Tick();
		return false;
	}

	// Assignments.
	for (int i = 0; i < g.num_phis; ++i) {
		_assignments(i) = -1;
		const auto var = g.phi_to_variable_map.find(i);
		if (var == g.phi_to_variable_map.end()) continue;
		const auto it = routes.assignments.find(var->second);
		if (it != routes.assignments.end()) _assignments(i) = it->second;
	}
	for (const auto& [var, j] : routes.assignments) _var_assignments(var) = j;

	// Times, objects, and each agent's visited rows; its other rows
	// interpolated between its visits.
	const std::vector<int>& nodes = routes.nodes;
	auto row = [&](int node) { return static_cast<int>(std::find(nodes.begin(), nodes.end(), node) - nodes.begin()); };
	_t_by_node_id.setZero();
	for (int v : remaining_vertices) _t_by_node_id(v) = routes.times[row(v)];
	for (int v : remaining_vertices)
		for (int o = 0; o < g.num_objects; ++o) {
			const int off = g.object_col_offset(o), d = g.object_ambient_dim(o);
			_waypoints.row(v).segment(off, d) = routes.X.row(row(v)).segment(off, d);
		}
	_routes = routes.routes;
	for (int j = 0; j < g.num_agents; ++j) {
		const int off = g.agent_col_offset(j), d = g.robot_ambient_dim(j);
		std::vector<double> times = {0.0};
		std::vector<Eigen::VectorXd> pts = {x0.segment(off, d)};
		std::set<int> visited;
		for (int node : _routes[j]) {
			const Eigen::VectorXd q = routes.X.row(row(node)).segment(off, d).transpose();
			_waypoints.row(node).segment(off, d) = q;
			visited.insert(node);
			if (_t_by_node_id(node) > times.back() + 1e-9) {
				times.push_back(_t_by_node_id(node));
				pts.push_back(q);
			}
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
	_last_solve_time = _timer.Tick();
	return true;
}
