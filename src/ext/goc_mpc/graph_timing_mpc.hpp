#pragma once

#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include "drake/solvers/solve.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "graph_of_constraints.hpp"
#include "../configuration_spline.hpp"
#include "../splines.hpp"
#include "../utils.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


struct GraphOrderingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	drake::solvers::MatrixXDecisionVariable p;

	GraphOrderingProblem()
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()) {}

	GraphOrderingProblem(const GraphOrderingProblem&) = delete;
	GraphOrderingProblem& operator=(const GraphOrderingProblem&) = delete;

	GraphOrderingProblem(GraphOrderingProblem&&) = default;
	GraphOrderingProblem& operator=(GraphOrderingProblem&&) = default;
};

GraphOrderingProblem build_graph_ordering_problem(
	const Graph<py::object>& structure,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0);

struct GraphTimingProblem {
	// Necessary to use a unique_ptr for movability. Weird...
	std::unique_ptr<drake::solvers::MathematicalProgram> prog;
	std::vector<Eigen::MatrixXd> wps_list;
	std::vector<std::vector<int>> agent_nodes_list;
	std::vector<drake::solvers::MatrixXDecisionVariable> vs_list;
	std::vector<drake::solvers::VectorXDecisionVariable> time_deltas_list;

	GraphTimingProblem(int num_agents)
		: prog(std::make_unique<drake::solvers::MathematicalProgram>()),
		  wps_list(num_agents),
		  vs_list(num_agents),
		  time_deltas_list(num_agents) {}

	GraphTimingProblem(const GraphTimingProblem&) = delete;
	GraphTimingProblem& operator=(const GraphTimingProblem&) = delete;

	GraphTimingProblem(GraphTimingProblem&&) = default;
	GraphTimingProblem& operator=(GraphTimingProblem&&) = default;
};

GraphTimingProblem build_graph_timing_problem(
	const Graph<py::object>& structure,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0);


struct GraphTimingMPC {
	// Input: reference to graph of constraints
	const GraphOfConstraints* _graph;
	std::shared_ptr<std::vector<CubicConfigurationSpline>> _splines;

	// Persistent Output Buffers
	std::vector<Eigen::MatrixXd> _wps_list;
	std::vector<Eigen::MatrixXd> _vs_list;
	std::vector<Eigen::VectorXd> _time_deltas_list;
	std::vector<std::vector<int>> _agent_nodes_list;
	std::map<int, int> _agent_spline_length_map;

	// Optimization parameters
	double _time_cost;
	double _ctrl_cost;
	double _max_vel;
	double _max_acc;
	double _max_jerk;

	// Phase management
	// std::set<int> _completed_phases;
	// Eigen::VectorXd _in_degrees; // in-degrees of remaining active phases
	// py::array_t<unsigned int> back_tracking_table;
	// bool never_done = false;

	// Constructor
	GraphTimingMPC(const GraphOfConstraints& graph,
		       std::vector<CubicConfigurationSpline> splines,
		       double time_cost = 1e0,
		       double ctrl_cost = 1e0,
		       double max_vel = -1.0,
		       double max_acc = -1.0,
		       double max_jerk = -1.0);

	// Core solve routine
	bool solve(const Eigen::VectorXd& x0,
		   const Eigen::VectorXd& v0,
		   const std::vector<int>& remaining_vertices,
		   const Eigen::MatrixXd& waypoints,
		   const Eigen::VectorXi& assignments);

	int get_agent_spline_length(int agent) const;
	std::vector<int> get_agent_spline_nodes(int agent) const;

	std::set<int> set_progressed_time(double delta, double tau_cutoff);

	// Spline generator
	void fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
				const Eigen::VectorXd& x0,
				const Eigen::VectorXd& v0) const;

	// Phase tracking
	// double current_minimum_time_delta() const;
	// bool done() const;

	// Safe indexing and accessors
	// py::array_t<unsigned int> get_ordering() const;
	// py::array_t<double> get_waypoints() const;
	// py::array_t<double> get_time_deltas() const;
	// py::array_t<double> get_times() const;
	const std::vector<double> get_next_taus() const;
	const std::vector<int> get_next_nodes() const;

	const std::vector<Eigen::MatrixXd> &view_wps_list() const { return _wps_list; }
	const std::vector<Eigen::MatrixXd> &view_vs_list() const { return _vs_list; }
	const std::vector<Eigen::VectorXd> &view_time_deltas_list() const { return _time_deltas_list; }
	const std::vector<std::vector<int>> &view_agent_nodes_list() const { return _agent_nodes_list; }
	const std::map<int, int> &view_agent_spline_length_map() const { return _agent_spline_length_map; }

	// State updates
	// bool set_progressed_time(double time_delta, double time_delta_cutoff);
	// void set_updated_waypoints(const py::array_t<double>& _waypoints,
	// 			   bool set_next_waypoint_tangent);
	// void update_backtrack();
	// void update_set_phase(unsigned int phase_to);


};
