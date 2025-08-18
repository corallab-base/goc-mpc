#include "graph_of_constraints.hpp"

using drake::solvers::Binding;
using drake::solvers::Constraint;

// Constructor
GraphOfConstraints::GraphOfConstraints(
	unsigned int num_agents, unsigned int dim,
	const Eigen::VectorXd& global_x_lb,
	const Eigen::VectorXd& global_x_ub)
	: num_phis(0),
	  _num_total_assignables(0),
	  num_agents(num_agents),
	  dim(dim),
	  _global_x_lb(global_x_lb),
	  _global_x_ub(global_x_ub) {}

std::pair<std::vector<std::vector<size_t>>,
	  std::vector<std::pair<size_t, size_t>>> GraphOfConstraints::get_agent_paths(
		  const std::vector<size_t>& remaining_vertices,
		  const Eigen::VectorXi& assignments) const {
	const InducedSubgraphView<py::object> sg = InducedSubgraphView<py::object>(
		structure, remaining_vertices);

	// This function introduces the idea now that every node has exactly one phi. That seems pretty reasonable.

	std::vector<std::vector<size_t>> agent_nodes(num_agents);
	std::vector<std::pair<size_t, size_t>> cross_agent_edges;

	sg.dfs_visit_from_sources(
		[this, assignments, &agent_nodes, &cross_agent_edges]
		(size_t node, std::optional<size_t> parent) {
			int parent_assignment = -1;
			if (parent && phi_map.contains(*parent)) {
				const int parent_phi_id = phi_map.at(*parent);
				parent_assignment = assignments(parent_phi_id);
			}

			size_t assignment = -1;
			if (phi_map.contains(node)) {
				const int phi_id = phi_map.at(node);
				assignment = assignments(phi_id);

				if (assignment == -1) {
					for (size_t ag = 0; ag < num_agents; ++ag) {
						agent_nodes[ag].push_back(node);
					}
				} else {
					agent_nodes[assignment].push_back(node);
				}
			}

			if (parent && parent_assignment != -1 && assignment != -1 &&
			    parent_assignment != assignment) {
				cross_agent_edges.emplace_back(node, *parent);
			}
		});

	return std::make_pair<std::vector<std::vector<size_t>>&,
			      std::vector<std::pair<size_t, size_t>>&>(agent_nodes, cross_agent_edges);
}

std::vector<size_t> GraphOfConstraints::get_phi_ids(size_t node) const {
	// TODO: Maybe expand if nodes in the future support multiple phi ids (probably will).
	std::vector<size_t> phi_ids = { phi_map.at(node) };
	return phi_ids;
}

bool GraphOfConstraints::evaluate_phi(size_t phi_id,
                                      const Eigen::VectorXd& x,
                                      int assignment_phi,
                                      double tol) const {
	auto it = _constraints_per_phi.find(phi_id);
	if (it == _constraints_per_phi.end()) {
		return true;
	}
	for (const auto& pc : it->second) {
		const Eigen::VectorXd z = pc.make_input(x, assignment_phi);
		if (!pc.binding.evaluator()->CheckSatisfied(z, tol)) {
			return false;
		}
	}
	return true;
}

void GraphOfConstraints::clear_constraints_per_phi() {
	return _constraints_per_phi.clear();
}


// We’ll assume x is laid out as [x_0, x_1, ..., x_{num_agents-1}], each x_i ∈ R^dim
auto make_pulls_for_agent_i(int agent_i, int dim) {
	std::vector<PhiConstraint::Pull> pulls;
	pulls.reserve(dim + 1);
	for (int j = 0; j < dim; ++j) {
		// x index for agent_i’s j-th component:
		pulls.emplace_back(PhiConstraint::PullX{agent_i * dim + j});
	}
	pulls.emplace_back(PhiConstraint::PullAssign{agent_i}); // the selector s_i
	return pulls;
}

// Joint-Agent Constraint Adders (typed)

// lb <= x <= ub on node k
void GraphOfConstraints::add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	_add_op(DeferredOpKind::kBoundingBox, k, [=, this](auto& prog,
							   const auto& X,
							   const auto&,
							   const std::map<size_t, size_t>& phi_to_id_map,
							   const std::map<size_t, size_t>&) {
		const unsigned int phi_id = phi_map.at(k);
		const unsigned int node_k = phi_to_id_map.at(phi_id);

		drake::solvers::VectorXDecisionVariable joint_config_k(num_agents * dim);
		for (int ag = 0; ag < num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		prog.AddBoundingBoxConstraint(lb, ub, joint_config_k);
	});
}

// Ax = b on node k
void GraphOfConstraints::add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	_add_op(DeferredOpKind::kLinearEq, k, [=, this](auto& prog,
							const auto& X,
							const auto&,
							const std::map<size_t, size_t>& phi_to_id_map,
							const std::map<size_t, size_t>&) {
		const unsigned int phi_id = phi_map.at(k);
		const unsigned int node_k = phi_to_id_map.at(phi_id);

		drake::solvers::VectorXDecisionVariable joint_config_k(num_agents * dim);
		for (int ag = 0; ag < num_agents; ++ag) {
			for (int i = 0; i < dim; ++i) {
				joint_config_k(ag * dim + i) = X(node_k + ag, i);
			}
		}

		auto beq = prog.AddLinearEqualityConstraint(A, b, joint_config_k);

		PhiConstraint pc(drake::solvers::Binding<drake::solvers::Constraint>(beq), // upcast
				 { PhiConstraint::PullAllX{} }); // one sentinel entry

		_constraints_per_phi[phi_id].push_back(std::move(pc));
	});
}

// lb <= A x <= ub on node k
void GraphOfConstraints::add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	_add_op(DeferredOpKind::kLinearIneq, k, [=, this](auto& prog,
							  const auto& X,
							  const auto&,
							  const std::map<size_t, size_t>& phi_to_id_map,
							  const std::map<size_t, size_t>&) {
		const unsigned int phi_id = phi_map.at(k);
		const unsigned int node_k = phi_to_id_map.at(phi_id);

		drake::solvers::VectorXDecisionVariable joint_config_k(num_agents * dim);
		for (int ag = 0; ag < num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		auto constraint = prog.AddLinearConstraint(A, lb, ub, joint_config_k);
	});
}

// 0.5 x'Qx + b'x + c on node k
void GraphOfConstraints::add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c) {
	_add_op(DeferredOpKind::kQuadraticCost, k, [=, this](auto& prog,
							     const auto& X,
							     const auto&,
							     const std::map<size_t, size_t>& phi_to_id_map,
							     const std::map<size_t, size_t>&) {
		const unsigned int phi_id = phi_map.at(k);
		const unsigned int node_k = phi_to_id_map.at(phi_id);

		drake::solvers::VectorXDecisionVariable joint_config_k(num_agents * dim);
		for (int ag = 0; ag < num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		auto constraint = prog.AddQuadraticCost(Q, b, c, joint_config_k);
	});
}

// Single-Agent Constraint Adders (typed)
// Note: these copy the numpy array's passed to them, but they're called
// once so it's fine.

// Compute max / min of c^T x over box lb<=x<=ub.
inline std::pair<double,double> max_min_ct_x_over_box(const Eigen::RowVectorXd& c,
						      const Eigen::VectorXd& lb,
						      const Eigen::VectorXd& ub) {
	DRAKE_DEMAND(c.size() == lb.size());
	double maxv = 0.0, minv = 0.0;
	for (int j = 0; j < c.size(); ++j) {
		// if c[j] is positive, maxv is maximized when x = up[j] and
		// minimized when = lb[j]. If negative, its the opposite.
		if (c[j] >= 0) { maxv += c[j] * ub[j]; minv += c[j] * lb[j]; }
		else           { maxv += c[j] * lb[j]; minv += c[j] * ub[j]; }
	}
	return {maxv, minv};
}

// A_i x_i = b on node k for some agent i
// Enforce: A * x_{k,i} = b for the unique agent i with A_(t,i) = 1.
// A.rows() == b.size(), A.cols() == d_
void GraphOfConstraints::add_assignable_linear_eq(int k,
						const Eigen::MatrixXd& A,
						const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < structure.num_nodes());
	DRAKE_DEMAND(A.cols() == dim);
	DRAKE_DEMAND(b.size() == A.rows());

	// record an increase in the total number of assignables. (could be removed).
	_num_total_assignables++;

	_add_op(DeferredOpKind::kAgentLinearEq, k, [=, this](auto& prog,
							     const auto& X,
							     const auto& Assignments,
							     const std::map<size_t, size_t>& phi_to_id_map,
							     const std::map<size_t, size_t>& phi_to_assignable_map) {
		const unsigned int phi_id = phi_map.at(k);
		const unsigned int node_k = phi_to_id_map.at(phi_id);
		const unsigned int assignable_k = phi_to_assignable_map.at(phi_id);

		for (int i = 0; i < num_agents; ++i) {
			// Variables [ x_{k,i} ; s ] with s = A(assignable_k, i)
			drake::solvers::VectorXDecisionVariable vars(dim + 1);
			const int row_ki = node_k * num_agents + i;
			for (int j = 0; j < dim; ++j) vars[j] = X(row_ki, j);
			vars[dim] = Assignments(assignable_k, i);   // <-- use A as selector

			for (int r = 0; r < A.rows(); ++r) {
				const Eigen::RowVectorXd c = A.row(r);
				const auto [max_cx, min_cx] = max_min_ct_x_over_box(
					c,
					_global_x_lb,
					_global_x_ub);

				const double rhs = b[r];
				// Pick M so that when s = 0 the constraint is loose:
				const double M_up = std::max(0.0, max_cx - rhs);  // for c^T x <= rhs
				const double M_lo = std::max(0.0, rhs - min_cx); // for c^T x >= rhs

				// Encode using constant bounds (move M*(1-s) to LHS):
				//  c^T x - M(1-s) <= rhs    ⇔  c^T x + M s <= rhs + M
				// -c^T x - M(1-s) <= -rhs   ⇔ -c^T x + M s <= -rhs + M
				Eigen::RowVectorXd a_up(dim + 1);
				a_up.head(dim) = c;    a_up[dim] = M_up;
				const double b_up = rhs + M_up;

				Eigen::RowVectorXd a_lo(dim + 1);
				a_lo.head(dim) = -c;   a_lo[dim] = M_lo;
				const double b_lo = -rhs + M_lo;

				const double ninf = -std::numeric_limits<double>::infinity();

				auto upper = prog.AddLinearConstraint(a_up, ninf, b_up, vars);
				auto lower = prog.AddLinearConstraint(a_lo, ninf, b_lo, vars);

				auto pulls = make_pulls_for_agent_i(i, dim);
				_constraints_per_phi[phi_id].push_back(PhiConstraint{upper, pulls});
				_constraints_per_phi[phi_id].push_back(PhiConstraint{lower, pulls});
			}
		}

		
	});
}

/*
 * Subgraph
 */
