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

#include "../graphs.hpp"
#include "../utils.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;
using namespace pybind11::literals;
namespace py = pybind11;


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
	size_t id;
	size_t node;
	std::function<void(drake::solvers::MathematicalProgram&,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const drake::solvers::MatrixXDecisionVariable&,
			   const std::map<size_t, size_t>&,
			   const std::map<size_t, size_t>&)> builder;
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
	Graph<py::object> structure;
	std::map<size_t, size_t> phi_map;
	std::map<size_t, struct DeferredOp> ops;
	size_t num_phis, _num_total_assignables, num_agents, dim;

        // For each phi, you may have one or many "phi constraints".
	std::unordered_map<size_t, std::vector<PhiConstraint>> _constraints_per_phi;

	// Required for big-M computation
	Eigen::VectorXd _global_x_lb;
	Eigen::VectorXd _global_x_ub;

	// Constructor
	GraphOfConstraints(unsigned int num_agents, unsigned int dim,
			   const Eigen::VectorXd& global_x_lb,
			   const Eigen::VectorXd& global_x_ub);

	Graph<py::object> get_structure() const { return structure; }

	std::pair<std::vector<std::vector<size_t>>,
		  std::vector<std::pair<size_t, size_t>>> get_agent_paths(
			  const std::vector<size_t>& remaining_vertices,
			  const Eigen::VectorXi& assignments) const;

	std::vector<size_t> get_phi_ids(size_t node) const;

	bool evaluate_phi(size_t phi_id,
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
	void add_assignable_linear_eq(int node_k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b);

private:
	template <typename F>
	void _add_op(DeferredOpKind kind, size_t node, F&& f) {
		const size_t id = num_phis++;
		phi_map[node] = id;
		ops[id] = DeferredOp{kind, id, node, std::forward<F>(f)};
	}
};


// struct SubgraphOfConstraints {
// 	const GraphOfConstraints* _g;
// 	std::vector<bool>   _mask;       // membership mask
// 	std::vector<size_t> _node_list;  // compact node list for fast outer loops

// 	// Construct from parent graph and a list of nodes in the subgraph
// 	SubgraphOfConstraints(const GraphOfConstraints& g, const std::vector<size_t>& nodes);

// 	size_t degree(size_t u) const;

// 	// Lightweight neighbor range that filters on the fly (const-only).
// 	class NeighborIter {
// 	public:
// 		using InnerIter = typename std::vector<Edge>::const_iterator;
// 		NeighborIter(const GraphT* g, const std::vector<bool>* mask, size_t u, InnerIter it, InnerIter end)
// 			: g_(g), mask_(mask), u_(u), it_(it), end_(end) { advance_to_valid(); }

// 		const Edge& operator*() const { return *it_; }
// 		const Edge* operator->() const { return &*it_; }

// 		NeighborIter& operator++() { ++it_; advance_to_valid(); return *this; }
// 		bool operator==(const NeighborIter& o) const { return it_ == o.it_; }
// 		bool operator!=(const NeighborIter& o) const { return !(*this == o); }

// 	private:
// 		void advance_to_valid() {
// 			while (it_ != end_ && !(it_->to < mask_->size() && (*mask_)[it_->to] && g_->alive(it_->to))) {
// 				++it_;
// 			}
// 		}
// 		const GraphT* g_;
// 		const std::vector<bool>* mask_;
// 		size_t u_;
// 		InnerIter it_, end_;
// 	};

// 	struct NeighborRange {
// 		NeighborIter begin_, end_;
// 		NeighborIter begin() const { return begin_; }
// 		NeighborIter end()   const { return end_; }
// 	};

// 	NeighborRange neighbors(size_t u) const {
// 		if (!contains_node(u)) {
// 			// empty range: begin == end
// 			auto it = g_->neighbors(std::min(u, g_->num_nodes() ? g_->num_nodes()-1 : 0)).end();
// 			return NeighborRange{NeighborIter(g_, &mask_, u, it, it), NeighborIter(g_, &mask_, u, it, it)};
// 		}
// 		const auto& nbrs = g_->neighbors(u);
// 		return NeighborRange{
// 			NeighborIter(g_, &mask_, u, nbrs.begin(), nbrs.end()),
// 			NeighborIter(g_, &mask_, u, nbrs.end(),   nbrs.end())
// 		};
// 	}

// 	// Iterate nodes in the subgraph
// 	const std::vector<size_t>& nodes() const { return node_list_; }

// 	// Iterate all edges in the subgraph (u, Edge&), on the fly
// 	struct EdgeRef { size_t u; const Edge* e; };

// 	class EdgeIter {
// 	public:
// 		EdgeIter(const GraphT* g, const std::vector<bool>* mask,
// 			 const std::vector<size_t>* node_list, size_t node_idx)
// 			: g_(g), mask_(mask), node_list_(node_list), node_idx_(node_idx)
// 			{
// 				advance_node();
// 			}

// 		EdgeRef operator*() const { return EdgeRef{ node_list_->at(node_idx_), &*it_ }; }
// 		EdgeIter& operator++() { ++it_; advance_edge(); return *this; }

// 		bool operator==(const EdgeIter& o) const {
// 			return node_idx_ == o.node_idx_ && (end_ ? it_ == o.it_ : true);
// 		}
// 		bool operator!=(const EdgeIter& o) const { return !(*this == o); }

// 	private:
// 		void advance_node() {
// 			// Move to first node that has at least one valid outgoing edge
// 			while (node_idx_ < node_list_->size()) {
// 				size_t u = node_list_->at(node_idx_);
// 				const auto& nbrs = g_->neighbors(u);
// 				it_ = nbrs.begin(); end_it_ = nbrs.end();
// 				advance_edge();
// 				if (it_ != end_it_) { end_ = false; return; }
// 				++node_idx_;
// 			}
// 			// end sentinel
// 			end_ = true;
// 		}

// 		void advance_edge() {
// 			while (node_idx_ < node_list_->size()) {
// 				while (it_ != end_it_) {
// 					if ((*mask_)[it_->to] && g_->alive(it_->to)) return;
// 					++it_;
// 				}
// 				++node_idx_;
// 				if (node_idx_ >= node_list_->size()) { end_ = true; return; }
// 				size_t u = node_list_->at(node_idx_);
// 				const auto& nbrs = g_->neighbors(u);
// 				it_ = nbrs.begin(); end_it_ = nbrs.end();
// 			}
// 			end_ = true;
// 		}

// 		const GraphT* g_;
// 		const std::vector<bool>* mask_;
// 		const std::vector<size_t>* node_list_;
// 		size_t node_idx_ = 0;

// 		typename std::vector<Edge>::const_iterator it_{};
// 		typename std::vector<Edge>::const_iterator end_it_{};
// 		bool end_ = false;
// 	};

// 	struct EdgeRange {
// 		EdgeIter begin_, end_;
// 		EdgeIter begin() const { return begin_; }
// 		EdgeIter end()   const { return end_; }
// 	};

// 	EdgeRange edges() const {
// 		return EdgeRange{
// 			EdgeIter(g_, &mask_, &node_list_, 0),
// 			EdgeIter(g_, &mask_, &node_list_, node_list_.size())
// 		};
// 	}

// 	const GraphT& parent() const { return *g_; }

// private:
// };
