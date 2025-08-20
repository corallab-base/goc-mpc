#pragma once

#include <iostream>

#include <drake/solvers/mathematical_program.h>
#include <drake/solvers/ipopt_solver.h>
#include <drake/solvers/branch_and_bound.h>
#include <drake/solvers/mosek_solver.h>
#include <drake/solvers/gurobi_solver.h>
#include <drake/solvers/solve.h>
#include <drake/multibody/plant/multibody_plant.h>
#include <drake/multibody/tree/multibody_tree_indexes.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../graphs.hpp"
#include "../utils.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;
using drake::multibody::MultibodyPlant;
using drake::multibody::ModelInstanceIndex;
using namespace pybind11::literals;
namespace py = pybind11;


struct SubgraphOfConstraints;

enum class DeferredOpKind {
	kLinearEq,
	kLinearIneq,
	kBoundingBox,
	kQuadraticCost,
	kNonlinearConstraint,
	kOther,
	// MultiAgent
	kAgentLinearEq,
};

struct DeferredOp {
	DeferredOpKind kind;
	int id;
	int node;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const struct SubgraphOfConstraints&,
			   const int,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&)> builder;
};

struct PhiConstraint {
	using BindingC = drake::solvers::Binding<drake::solvers::Constraint>;

	struct PullX      { int x_index; };     // z[p] = x[x_index]
	struct PullAssign { int agent_i; };     // z[p] = (agent_i == assignment_phi) ? 1 : 0
	struct PullAllX   {};                   // z = x  (must match sizes)

	using Pull = std::variant<PullX, PullAssign, PullAllX>;

	BindingC binding;
	std::vector<Pull> pulls;  // either {PullAllX{}} or one entry per var

	PhiConstraint(BindingC b, std::vector<Pull> ps) : binding(b), pulls(ps) {}
	
	// Build evaluator input z from (x, assignment_phi)
	Eigen::VectorXd make_input(const Eigen::VectorXd& x, int assignment_phi) const {
		// Fast path: whole x is the input
		if (pulls.size() == 1 && std::holds_alternative<PullAllX>(pulls[0])) {
			// Sanity check: sizes must match
			DRAKE_DEMAND(x.size() == binding.variables().size());
			return x;
		}
		// General path: per-entry gather
		Eigen::VectorXd z(pulls.size());
		for (int p = 0; p < static_cast<int>(pulls.size()); ++p) {
			if (auto px = std::get_if<PullX>(&pulls[p])) {
				z[p] = x[px->x_index];
			} else if (auto pa = std::get_if<PullAssign>(&pulls[p])) {
				z[p] = (pa->agent_i == assignment_phi) ? 1.0 : 0.0;
			} else {
				throw std::logic_error("PullAllX must be the only entry if used.");
			}
		}
		return z;
	}
};


struct GraphOfConstraints {

	const MultibodyPlant<double> *plant;
	Graph<py::object> structure;
	std::map<int, int> node_to_phi_map;
	std::map<int, int> phi_to_variable_map;
	std::map<int, struct DeferredOp> ops;
	int num_phis, _num_variables, _num_total_assignables;
	int num_agents, num_objects, dim, non_robot_dim, total_dim;
	
        // For each phi, you may have one or many "phi constraints".
	std::unordered_map<int, std::vector<PhiConstraint>> _constraints_per_phi;

	// Required for big-M computation
	Eigen::VectorXd _global_x_lb;
	Eigen::VectorXd _global_x_ub;

	// Constructor
	// GraphOfConstraints(unsigned int num_agents, unsigned int dim,
	// 		   const Eigen::VectorXd& global_x_lb,
	// 		   const Eigen::VectorXd& global_x_ub);

	GraphOfConstraints(const MultibodyPlant<double> *plant,
			   const std::vector<std::string> robots,
			   const std::vector<std::string> objects,
			   const Eigen::VectorXd& global_x_lb,
			   const Eigen::VectorXd& global_x_ub);

	int add_variable();

	Graph<py::object> get_structure() const { return structure; }

	std::pair<std::vector<std::vector<int>>,
		  std::vector<std::pair<int, int>>> get_agent_paths(
			  const std::vector<int>& remaining_vertices,
			  const Eigen::VectorXi& assignments) const;

	std::vector<int> get_phi_ids(int node) const;

	bool evaluate_phi(int phi_id,
			  const Eigen::VectorXd& x,
			  int assignment_phi,
			  double tol) const;
	
	void clear_constraints_per_phi();

	// Plain Constraint Adders (typed)
	// Note: these copy the numpy array's passed to them, but they're called
	// once so it's fine.

	// lb <= x <= ub on node k
	void add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	// Ax = b on node k
	void add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	// lb <= A x <= ub on node k
	void add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub);

	// 0.5 x'Qx + b'x + c on node k
	void add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c = 0.0);

	// Multi-Agent Constraint Adders (typed)

	// Ax_i = b on node k for some agent i
	void add_assignable_linear_eq(int k, int var, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

	// Temp Plant-Based Constraint Adders

	// Sort of hard-coded
	void add_robot_above_cube_constraint(int k, int agent_i, int cube_i, double delta_z);


private:
	template <typename F>
	void _add_op(DeferredOpKind kind, int node, F&& f) {
		const int id = num_phis++;
		node_to_phi_map[node] = id;
		ops[id] = DeferredOp{kind, id, node, std::forward<F>(f)};
	}

	template <typename F>
	void _add_assignable_op(DeferredOpKind kind, int node, int var, F&& f) {
		const int id = num_phis++;
		node_to_phi_map[node] = id;
		phi_to_variable_map[id] = var;
		ops[id] = DeferredOp{kind, id, node, std::forward<F>(f)};
	}
};

/*
 * Subgraph
 */

struct SubgraphOfConstraints {
	InducedSubgraphView<py::object> structure;
	std::map<int, int> _variable_to_subgraph_variable_id; // unique ids for variables relevant to subgraph.
	std::map<int, DeferredOp> _subgraph_ops;

	SubgraphOfConstraints(GraphOfConstraints *graph, const std::vector<int>& vertices) :
		structure(graph->structure, vertices) {

		// std::map<int, int> phi_to_subgraph_node_id;
		// std::map<int, int> phi_to_subgraph_assignable_id;
		// std::map<int, int> subgraph_assignable_id_to_phi;

		int num_subgraph_variables = 0;
		
		for (int v : vertices) {
			// if there is/are phi associated with v
			if (graph->node_to_phi_map.contains(v)) {
				// Store the relevant ops so they can be applied
				const int phi_id = graph->node_to_phi_map.at(v);
				_subgraph_ops[phi_id] = graph->ops.at(phi_id);

				// Record the mapping from phi id to subgraph node and assignable var idxs.
				if (graph->phi_to_variable_map.contains(phi_id)) {
					const int variable_id = graph->phi_to_variable_map.at(phi_id);
					
					if (!_variable_to_subgraph_variable_id.contains(variable_id)) {
						_variable_to_subgraph_variable_id[variable_id] = num_subgraph_variables++;
					}
					
				}
			}
		}
	}

	// Function that returns an iterator to the beginning of the map
	const std::map<int, DeferredOp>& get_subgraph_ops() const {
		return _subgraph_ops;
	}

	int num_nodes() const {
		return structure.num_nodes();
	}

	int num_variables() const {
		return _variable_to_subgraph_variable_id.size();
	}

	int subgraph_id(int u) const {
		return structure.subgraph_id(u);
	}

	int subgraph_variable_id(int var) const {
		if (!_variable_to_subgraph_variable_id.contains(var)) { return -1; }
		return _variable_to_subgraph_variable_id.at(var);
	}
};
