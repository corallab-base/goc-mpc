#include "graph_waypoint_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


GraphWaypointProblem build_graph_waypoint_problem(
	const Eigen::MatrixXi graph,
	int num_assignables, int num_agents,
	int dim) {

	using namespace drake::solvers;

	const int num_nodes = graph.rows();

	// Create program
	GraphWaypointProblem problem;
	problem.prog = std::make_unique<drake::solvers::MathematicalProgram>();

	// a: binary assignment variables (z x m).
	// Drake exposes a direct API for binary matrices. :contentReference[oaicite:1]{index=1}
	MatrixXDecisionVariable Assignments = problem.prog->NewBinaryVariables(num_assignables, num_agents, "Assignments");
	problem.Assignments = Assignments;

	// One-hot per row (each task gets exactly one agent): sum_k a(row,k) = 1.
	for (int i = 0; i < num_assignables; ++i) {
		problem.prog->AddLinearEqualityConstraint(
			Eigen::RowVectorXd::Ones(num_agents),
			1.0, Assignments.row(i));
	}

	// x: continuous configuration variables (n x d).
	MatrixXDecisionVariable X = problem.prog->NewContinuousVariables(num_nodes * num_agents, dim, "X");
	problem.X = X;

	//
	// OBJECTIVE FUNCTION
	//

	// Add inter-waypoint costs for edges (i, j) indicated by adj(i,j) == 1.
	// You control the cost body via `add_edge_cost`.
	for (int i = 0; i < num_nodes; ++i) {
		for (int j = 0; j < num_nodes; ++j) {
			if (graph(i, j) != 0 && i != j) {
				for (int ag = 0; ag < num_agents; ++ag) {
					const int node_i_ag_idx = i * num_agents + ag;
					const int node_j_ag_idx = j * num_agents + ag;
					VectorX<Expression> diff = X.row(node_i_ag_idx) - X.row(node_j_ag_idx);
					Expression dist = diff.squaredNorm();
					problem.prog->AddQuadraticCost(dist);
				}
			}
		}
	}

	return std::move(problem);
}

/*
 * Waypoint MPC
 */

GraphWaypointMPC::GraphWaypointMPC(const Eigen::MatrixXi& graph,
				   unsigned int num_agents,
				   unsigned int dim,
				   const Eigen::VectorXd& global_x_lb,
				   const Eigen::VectorXd& global_x_ub)
	: _graph(graph),
	  _num_agents(num_agents),
	  _dim(dim),
	  _global_x_lb(global_x_lb),
	  _global_x_ub(global_x_ub) {
	// Ensure that graph is a square matrix
	DRAKE_DEMAND(_graph.rows() == _graph.cols());
	DRAKE_DEMAND(_global_x_lb.size() == _dim);
	DRAKE_DEMAND(_global_x_ub.size() == _dim);

	// Record the total number of nodes in the overall plan, but this
	_num_total_nodes = _graph.rows();

	// Phis is initially zero but is increased by adding any constraints.
	_num_phis = 0;

	// Assignables is initially zero but is increased by adding assignable constraints.
	_num_total_assignables = 0;
}

std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXi>> GraphWaypointMPC::solve(
	const std::set<unsigned int>& remaining_vertices,
	const Eigen::VectorXd& x0) {

	// TODO: use x0

	unsigned int num_subgraph_assignables = 0;
	const unsigned int num_remaining_nodes = remaining_vertices.size();
	Eigen::MatrixXi subgraph(num_remaining_nodes, num_remaining_nodes);
	_phi_to_subgraph_node_id.clear();
	_phi_to_subgraph_assignable_id.clear();

	int i = 0;
	for (auto i_it = remaining_vertices.begin(); i_it != remaining_vertices.end(); ++i_it) {
		// Record the mapping from phi id to subgraph node and assignable var idxs.
		const unsigned int phi_id = _graph_to_phi_map[*i_it];
		const DeferredOp& op = _ops[phi_id];
		_phi_to_subgraph_node_id[phi_id] = i;
		if (op.kind == DeferredOpKind::kAgentLinearEq) {
			_phi_to_subgraph_assignable_id[phi_id] = num_subgraph_assignables++;
		}

		// Compute subgraph row.
		int j = 0;
		for (auto j_it = remaining_vertices.begin(); j_it != remaining_vertices.end(); ++j_it) {
			subgraph(i, j) = _graph(*i_it, *j_it);
			++j;
		}
		++i;
	}

	std::cout << "Subgraph m:\n" << subgraph << std::endl;

	GraphWaypointProblem problem = build_graph_waypoint_problem(
		subgraph, num_subgraph_assignables, _num_agents, _dim);

	// Rebuild constraints/costs from registry
	for (const std::pair<unsigned int, DeferredOp>& pair : _ops) {
		const unsigned int phi_id = pair.first;
		const DeferredOp& op = pair.second;
		if (remaining_vertices.contains(op.node)) {
			op.builder(*(problem.prog), problem.X, problem.Assignments);
		}
	}

	// Solve
	drake::solvers::MosekSolver solver;
	const auto result = solver.Solve(*problem.prog);
	// auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		std::cout << "Success" << std::endl;

		Eigen::VectorXi assignments(_num_total_assignables);
		for (int i = 0; i < _num_total_assignables; ++i) {
			for (int j = 0; j < _num_agents; ++j) {
				const double val = result.GetSolution(problem.Assignments(i, j));
				if (val > 0.5) {
					assignments(i) = j;
					break;
				}
			}
		}

		Eigen::MatrixXd X_flat = result.GetSolution(problem.X);
		Eigen::MatrixXd X(num_remaining_nodes, _num_agents * _dim);
		for (int i = 0; i < num_remaining_nodes; ++i) {
			// take block of m rows and stack them horizontally
			Eigen::RowVectorXd row(_num_agents * _dim);
			for (int j = 0; j < _num_agents; ++j) {
				row.segment(j * _dim, _dim) = X_flat.row(i * _num_agents + j);
			}
			X.row(i) = row;
		}

		return std::make_pair(X, assignments);
	} else {
		std::cerr << "Optimization failed." << std::endl;
		return std::nullopt;
	}
}

// Joint-Agent Constraint Adders (typed)

// lb <= x <= ub on node k
void GraphWaypointMPC::add_bounding_box(int k, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	_add_op(DeferredOpKind::kBoundingBox, k, [=, this](auto& prog, const auto& X, const auto&) {
		const unsigned int phi_id = _graph_to_phi_map[k];
		const unsigned int node_k = _phi_to_subgraph_node_id[phi_id];

		drake::solvers::VectorXDecisionVariable joint_config_k(_num_agents * _dim);
		for (int ag = 0; ag < _num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		prog.AddBoundingBoxConstraint(lb, ub, joint_config_k);
	});
}

// Ax = b on node k
void GraphWaypointMPC::add_linear_eq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
	_add_op(DeferredOpKind::kLinearEq, k, [=, this](auto& prog, const auto& X, const auto&) {
		const unsigned int phi_id = _graph_to_phi_map[k];
		const unsigned int node_k = _phi_to_subgraph_node_id[phi_id];

		drake::solvers::VectorXDecisionVariable joint_config_k(_num_agents * _dim);
		for (int ag = 0; ag < _num_agents; ++ag) {
			for (int i = 0; i < _dim; ++i) {
				joint_config_k(ag * _dim + i) = X(node_k + ag, i);
			}
		}

		prog.AddLinearEqualityConstraint(A, b, joint_config_k);
	});
}

// lb <= A x <= ub on node k
void GraphWaypointMPC::add_linear_ineq(int k, const Eigen::MatrixXd& A, const Eigen::VectorXd& lb, const Eigen::VectorXd& ub) {
	_add_op(DeferredOpKind::kLinearIneq, k, [=, this](auto& prog, const auto& X, const auto&) {
		const unsigned int phi_id = _graph_to_phi_map[k];
		const unsigned int node_k = _phi_to_subgraph_node_id[phi_id];

		drake::solvers::VectorXDecisionVariable joint_config_k(_num_agents * _dim);
		for (int ag = 0; ag < _num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		prog.AddLinearConstraint(A, lb, ub, joint_config_k);
	});
}

// 0.5 x'Qx + b'x + c on node k
void GraphWaypointMPC::add_quadratic_cost_on_node(int k, const Eigen::MatrixXd& Q, const Eigen::VectorXd& b, double c) {
	_add_op(DeferredOpKind::kQuadraticCost, k, [=, this](auto& prog, const auto& X, const auto&) {
		const unsigned int phi_id = _graph_to_phi_map[k];
		const unsigned int node_k = _phi_to_subgraph_node_id[phi_id];

		drake::solvers::VectorXDecisionVariable joint_config_k(_num_agents * _dim);
		for (int ag = 0; ag < _num_agents; ++ag) {
			joint_config_k << X.row(node_k + ag);
		}

		prog.AddQuadraticCost(Q, b, c, joint_config_k);
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
void GraphWaypointMPC::add_assignable_linear_eq(int k,
						const Eigen::MatrixXd& A,
						const Eigen::VectorXd& b) {
	DRAKE_DEMAND(k >= 0 && k < _num_total_nodes);
	DRAKE_DEMAND(A.cols() == _dim);
	DRAKE_DEMAND(b.size() == A.rows());

	// record an increase in the total number of assignables. (could be removed).
	_num_total_assignables++;

	_add_op(DeferredOpKind::kAgentLinearEq, k, [=, this](auto& prog,
						   const auto& X,
						   const auto& Assignments) {
		const unsigned int phi_id = _graph_to_phi_map[k];
		const unsigned int node_k = _phi_to_subgraph_node_id[phi_id];
		const unsigned int assignable_k = _phi_to_subgraph_assignable_id[phi_id];
		
		for (int i = 0; i < _num_agents; ++i) {
			// Variables [ x_{k,i} ; s ] with s = A(assignable_k, i)
			drake::solvers::VectorXDecisionVariable vars(_dim + 1);
			const int row_ki = node_k * _num_agents + i;
			for (int j = 0; j < _dim; ++j) vars[j] = X(row_ki, j);
			vars[_dim] = Assignments(assignable_k, i);   // <-- use A as selector

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
				Eigen::RowVectorXd a_up(_dim + 1);
				a_up.head(_dim) = c;    a_up[_dim] = M_up;
				const double b_up = rhs + M_up;

				Eigen::RowVectorXd a_lo(_dim + 1);
				a_lo.head(_dim) = -c;   a_lo[_dim] = M_lo;
				const double b_lo = -rhs + M_lo;

				const double ninf = -std::numeric_limits<double>::infinity();
				prog.AddLinearConstraint(a_up, ninf, b_up, vars);
				prog.AddLinearConstraint(a_lo, ninf, b_lo, vars);
			}
		}
	});
}
