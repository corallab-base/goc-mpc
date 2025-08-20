#include "graph_timing_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


// GraphOrderingProblem build_graph_ordering_problem(
// 	const Eigen::MatrixXd& wps,
// 	const Eigen::MatrixXi& graph,
// 	const Eigen::VectorXd& x0,
// 	const Eigen::VectorXd& v0) {

// 	using namespace drake::solvers;

// 	const int K = wps.rows();
// 	const int d = wps.cols();

// 	// Create program
// 	GraphOrderingProblem problem;

// 	// Create decision variables
// 	// p(i,j) = 1 iff waypoint i is at position j
// 	MatrixXDecisionVariable p = problem.prog->NewBinaryVariables(K, K, "p");
// 	problem.p = p;

// 	// Doubly-Stochastic
// 	// Each waypoint exactly once: sum_k p(i,k) = 1
// 	for (int i = 0; i < K; ++i) {
// 		VectorX<Variable> row = p.row(i);
// 		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, row);
// 	}
// 	// Each position filled: sum_i p(i,k) = 1
// 	for (int j = 0; j < K; ++j) {
// 		VectorX<Variable> col = p.col(j);
// 		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, col);
// 	}

// 	// Precedence: i must appear before j  ==>  sum_k k*P(i,k) + 1 <= sum_k k*P(j,k)
// 	for (int i = 0; i < K; ++i) {
// 		for (int j = 0; j < K; ++j) {
// 			if (graph(i, j) == 1) {
// 				VectorX<Variable> lhs_vars(2*K);
// 				Eigen::RowVectorXd lhs_coeffs = Eigen::RowVectorXd::Zero(2*K);
// 				// pack as [P(i, 0 through n-1), P(j, 0 through n-1)]
// 				for (int k = 0; k < K; ++k) {
// 					lhs_vars(k)     = p(i, k);
// 					lhs_vars(K + k) = p(j, k);
// 					lhs_coeffs(k)       = k;       // +k * P(i,k)
// 					lhs_coeffs(K + k)   = -k;      // -k * P(j,k)
// 				}
// 				/* lb occurs when P(i,0) = 1 and P(j, K-1) = 1. Therefore k*P(i, k) - k*P(j, k) = -(K - 1). */
// 				problem.prog->AddLinearConstraint(lhs_coeffs, -(K-1), -1, lhs_vars); // enforces pos(i)+1 <= pos(j)
// 			}
// 		}
// 	}

// 	// Helpers: squared distances from x0 to xi for all i
// 	// squared distances between xi and xj for all i,j
// 	Eigen::VectorXd s2(K);
// 	Eigen::MatrixXd d2(K, K);
// 	for (int i = 0; i < K; ++i) {
// 		s2(i) = (wps.row(i).transpose() - x0).squaredNorm();
// 		for (int j = 0; j < K; ++j) {
// 			d2(i,j) = (wps.row(i) - wps.row(j)).squaredNorm();
// 		}
// 	}

// 	// Objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_{i,j} d2(i,j)*p(i,k-1)*p(j,k)

// 	// The second term is quadratic in binaries; to keep MILP, use a *linear* proxy:
// 	// e.g., sum_{k=1}^{n-1} sum_{j} (min_i D2(i,j)) * p(j,k)  (lower-bound-ish) or just sum over k of degrees.
// 	// A better linear surrogate: use a fixed "nearest predecessor" cost C(j,k) = min_i D2(i,j).
// 	Eigen::VectorXd c_min(K);
// 	for (int j = 0; j < K; ++j) {
// 		double m = std::numeric_limits<double>::infinity();
// 		for (int i = 0; i < K; ++i) if (i != j) m = std::min(m, d2(i,j));
// 		c_min(j) = std::isfinite(m) ? m : 0.0;
// 	}
// 	drake::solvers::LinearCost* obj = nullptr;
// 	// Build linear objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_j c_min(j)*p(j,k)
// 	Eigen::VectorXd coeffs(K*K);
// 	coeffs.setZero();
// 	int idx = 0;
// 	for (int i = 0; i < K; ++i) {
// 		for (int j = 0; j < K; ++j, ++idx) {
// 			double c = (j == 0) ? s2(i) : c_min(i);
// 			coeffs(idx) = c;
// 		}
// 	}
// 	drake::VectorX<Variable> allP(K*K);
// 	idx = 0;
// 	for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) allP[idx++] = p(i, j);
// 	problem.prog->AddLinearCost(coeffs, 0.0, allP);

// 	return std::move(problem);
// }

// std::set<unsigned int> GraphTimingMPC::_next_nodes() const {
// 	std::set<unsigned int> next_nodes;

// 	for (int i = 0; i < _num_nodes; ++i) {
// 		if (_in_degrees(i) == 0) {
// 			next_nodes.insert(i);
// 		}
// 	}

// 	return next_nodes;
// }

// double GraphTimingMPC::current_minimum_time_delta() const {
// 	std::set<unsigned int> next_nodes = _next_nodes();
// 	double minimum_time_delta = -1; /* negative by default. indicates no next node */

// 	auto time_deltas_u = _time_deltas.unchecked<1>();
// 	for (unsigned int n : next_nodes) {
// 		if (minimum_time_delta < 0.0 ||
// 		    time_deltas_u(n) < minimum_time_delta) {
// 			minimum_time_delta = time_deltas_u(n);
// 		}
// 	}

// 	return minimum_time_delta;
// }

// bool GraphTimingMPC::done() const {
// 	return _completed_phases.size() == _num_nodes;
// }

GraphTimingProblem build_graph_timing_problem(
	const GraphOfConstraints& graph,
	const std::vector<int>& remaining_vertices,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXi& assignments,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double time_cost,
	double time_cost2,
	double ctrl_cost,
	double max_vel,
	double max_acc,
	double max_jerk) {

	using namespace drake::solvers;

	GraphTimingProblem problem(graph.num_agents);

	/* at this point, the waypoint optimizer has decided on assignments of
	 * agents to assignable tasks and their positions for satisfying those
	 * tasks. the order may not be completely determined for an individual
	 * agent, so we also solve for an optimal ordering if necessary. */

	// TODO: use assignments to settle conditional edges.

	int dim = graph.dim;
	const auto pair = graph.get_agent_paths(remaining_vertices, assignments);
	const auto agent_nodes = pair.first;
	const auto cross_agent_edges = pair.second;

	// for (auto edge : cross_agent_edges) {
	// 	std::cout << "edge u: " << edge.first << std::endl;
	// 	std::cout << "edge v: " << edge.second << std::endl;
	// }

	// grow agent_nodes_list to accomodate the number of agents
	problem.agent_nodes_list.resize(graph.num_agents);

	for (int i = 0; i < graph.num_agents; ++i) {
		const std::vector<int> agent_i_nodes = agent_nodes[i];
		problem.agent_nodes_list[i] = agent_i_nodes;

		const int agent_spline_length = agent_i_nodes.size();

		// If there is nothing left in this spline, there's nothing to optimize.
		if (agent_spline_length > 0) {
			char time_deltas_name[32];
			char vs_name[32];

			Eigen::MatrixXd wps_i(agent_spline_length, dim);
			for (int j = 0; j < agent_spline_length; ++j) {
				int node = agent_i_nodes[j];
				for (int k = 0; k < dim; ++k) {
					wps_i(j, k) = waypoints(node, i * dim + k);
				}
			}
			problem.wps_list[i] = wps_i;

			snprintf(time_deltas_name, 32, "time_deltas_%d", i);
			snprintf(vs_name, 32, "vs_%d", i);

			// Create variables
			VectorXDecisionVariable time_deltas_i = problem.prog->NewContinuousVariables(agent_spline_length, time_deltas_name);
			for (int j = 0; j < agent_spline_length; ++j) {
				problem.prog->AddBoundingBoxConstraint(0.01, 10.0, time_deltas_i(j));
			}
			problem.time_deltas_list[i] = time_deltas_i;

			MatrixXDecisionVariable vs_i = problem.prog->NewContinuousVariables(agent_spline_length - 1, dim, vs_name);
			problem.vs_list[i] = vs_i;

			// Set initial guess
			problem.prog->SetInitialGuess(vs_i, Eigen::MatrixXd::Constant(vs_i.rows(), vs_i.cols(), 1.0));
			problem.prog->SetInitialGuess(time_deltas_i, Eigen::VectorXd::Constant(agent_spline_length, 10.0));

			// 1. Linear objective: sum(time_deltas) (OT_f in original code)
			if (time_cost > 0.0) {
				problem.prog->AddLinearCost(time_cost * Eigen::RowVectorXd::Ones(agent_spline_length), time_deltas_i);
			}

			// 2. Also, quadratic time_delta objective : sum_i time_deltas_i^2
			if (time_cost2 > 0.0) {
				const Eigen::MatrixXd Q = time_cost2 * Eigen::MatrixXd::Identity(agent_spline_length, agent_spline_length);
				const Eigen::VectorXd b = Eigen::VectorXd::Zero(agent_spline_length);
				problem.prog->AddQuadraticCost(Q, b, time_deltas_i);
			}

			// 3. Control costs
			if (ctrl_cost > 0) {
				const double s12 = std::sqrt(12.0);

				for (int j = 0; j < agent_spline_length; ++j) {
					VectorX<Expression> xJ(dim), xJm1(dim), vJ(dim), vJm1(dim);
					const Expression tau(time_deltas_i(j));
					if (j == 0 && j < agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(0, k));
							vJ(k)   = Expression(vs_i(0, k));
						}
					} else if (j > 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else if (j == 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k)   = Expression(vs_i(j, k));
						}
					}
					const VectorX<Expression> D = (xJ - xJm1) - (0.5 * tau * (vJm1 + vJ));
					const VectorX<Expression> V = (vJ - vJm1);
					const VectorX<Expression> tilD = s12 * pow(tau, -1.5) * D;
					const VectorX<Expression> tilV = pow(tau, -0.5) * V;
					const Expression c = tilD.squaredNorm() + tilV.squaredNorm();
					problem.prog->AddCost(c);
				}
			}

			// Velocity/Acceleration/Jerk Constraints
			if (max_vel > 0) {
				for (int j = 0; j < agent_spline_length; ++j) {
					VectorX<Expression> xJm1(dim), xJ(dim), vJm1(dim), vJ(dim);
					const Expression tau(time_deltas_i(j));
					const Expression inv_tau = pow(tau, -1.0);
					if (j == 0 && j < agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(0, k));
							vJ(k)   = Expression(vs_i(0, k));
						}
					} else if (j > 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else if (j == 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k)   = Expression(vs_i(j, k));
						}
					}

					// Your cubic parameterization (unscaled a,b,c as in your code)
					// c = v0
					const VectorX<Expression> c = vJm1;
					// b = 3*(x1 - x0) - tau*(v1 + 2*v0)
					const VectorX<Expression> b = 3.0*(xJ - xJm1) - tau*(vJ + 2.0*vJm1);
					// a = -2*(x1 - x0) + tau*(v1 + v0)
					const VectorX<Expression> a = -2.0*(xJ - xJm1) + tau*(vJ + vJm1);

					// Midpoint velocity surrogate: v_mid = c + (1/tau)*(b + 3/4 a)
					const VectorX<Expression> v_mid = c + inv_tau * (b + 0.75 * a);

					// Enforce |v(0)| <= vmax and |v_mid| <= vmax  (elementwise)
					Eigen::VectorXd lb = Eigen::VectorXd::Constant(dim, -max_vel);
					Eigen::VectorXd ub = Eigen::VectorXd::Constant(dim,  max_vel);
					problem.prog->AddConstraint(c,     lb, ub);   // v(0) = c
					problem.prog->AddConstraint(v_mid, lb, ub);   // v(tau/2)
				}
			}

			if (max_acc > 0) {
				for (int j = 0; j < agent_spline_length; ++j) {
					VectorX<Expression> xJm1(dim), xJ(dim), vJm1(dim), vJ(dim);
					const Expression tau(time_deltas_i(j));
					if (j == 0 && j < agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(0, k));
							vJ(k)   = Expression(vs_i(0, k));
						}
					} else if (j > 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else if (j == 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k)   = Expression(vs_i(j, k));
						}
					}

					// b2 = 2/tau^2 * ( 3*(x1 - x0) - tau*(v1 + 2*v0) )
					const Expression inv_tau2 = pow(tau, -2.0);
					const VectorX<Expression> b2 =
						2.0 * inv_tau2 * ( 3.0*(xJ - xJm1) - tau*(vJ + 2.0*vJm1) );

					// a6_tau = 6/tau^2 * ( -2*(x1 - x0) + tau*(v1 + v0) )
					const VectorX<Expression> a6_tau =
						6.0 * inv_tau2 * ( -2.0*(xJ - xJm1) + tau*(vJ + vJm1) );

					// Endpoint accelerations
					const VectorX<Expression> acc0  = b2;                 // t = 0
					const VectorX<Expression> accT  = b2 + a6_tau;        // t = tau

					// ||acc(0)||_inf <= amax and ||acc(tau)||_inf <= amax (elementwise)
					Eigen::VectorXd lb = Eigen::VectorXd::Constant(dim, -max_acc);
					Eigen::VectorXd ub = Eigen::VectorXd::Constant(dim,  max_acc);
					problem.prog->AddConstraint(acc0, lb, ub);
					problem.prog->AddConstraint(accT, lb, ub);
				}
			}

			if (max_jerk > 0) {
				for (int j = 0; j < agent_spline_length; ++j) {
					VectorX<Expression> xJ(dim), xJm1(dim), vJ(dim), vJm1(dim);
					const Expression tau(time_deltas_i(j));
					if (j == 0 && j < agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(0, k));
							vJ(k)   = Expression(vs_i(0, k));
						}
					} else if (j > 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else if (j == 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(x0(i * dim + k));
							vJm1(k) = Expression(v0(i * dim + k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k).Zero();
						}
					} else {
						for (int k = 0; k < dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							vJm1(k) = Expression(vs_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
							vJ(k)   = Expression(vs_i(j, k));
						}
					}

					// a6 = 6/tau^3 * (-2*(x1 - x0) + tau*(v1 + v0))
					const Expression coeff = 6.0 * pow(tau, -3.0);
					const VectorX<Expression> a6 = coeff * ( -2.0 * (xJ - xJm1) + tau * (vJ + vJm1) );

					// Bound a6 with symmetric bounds:
					Eigen::VectorXd lb2 = Eigen::VectorXd::Constant(dim, -max_jerk);
					Eigen::VectorXd ub2 = Eigen::VectorXd::Constant(dim,  max_jerk);
					problem.prog->AddConstraint(a6, lb2, ub2);
				}
			}
		}

		// const bool totally_ordered = true;
		// if (!totally_ordered) {
		// 	// Solve for ordering
		// } else {
		// 	ordering = arange(0, agent_subgraph);
		// }

		// with ordering, add to program a single spline optimization problem
	}

	return problem;
}

/*
 * Graph Timing MPC
 */

GraphTimingMPC::GraphTimingMPC(const GraphOfConstraints& graph,
			       double time_cost,
			       double ctrl_cost,
			       double max_vel,
			       double max_acc,
			       double max_jerk)
	: _graph(&graph),
	  _time_cost(time_cost),
	  _ctrl_cost(ctrl_cost),
	  _max_vel(max_vel),
	  _max_acc(max_acc),
	  _max_jerk(max_jerk),
	  _vs_list(graph.num_agents),
	  _time_deltas_list(graph.num_agents) {

	int num_agents = _graph->num_agents;
	int num_nodes = _graph->structure.num_nodes();
	int dim = _graph->dim;

	// A waypoint and velocity array for each agent only.
	_wps_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		_wps_list[i] = Eigen::MatrixXd::Zero(num_nodes, dim);
	}
	_vs_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		_vs_list[i] = Eigen::MatrixXd::Zero(num_nodes, dim);
	}
	_time_deltas_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		_time_deltas_list[i] = Eigen::VectorXd::Zero(num_nodes);
	}

	// Maybe: a waypoint and velocity array for all of the objects.
}

bool GraphTimingMPC::solve(
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const std::vector<int>& remaining_vertices,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXi& assignments) {

	/* after the individual agents' positions are totally ordered we
	 * construct each agent's spline. */

	// std::array<Eigen::VectorXi, _num_agents> ordering;
	// std::array<Eigen::MatrixXd, _num_agents> ordered_agent_wps;

	/* we add constraints on the timings of these splines to reflect any
	 * cross-agent constraints in the original graph. */


	/* we solve for the velocities and timings using epigraph on the final
	 * timing */

	// // for (int ag = 0; ag < _num_agents; ++ag) {
	// // 	for
	// // }

	// const unsigned int num_nodes = graph.rows();

	// struct GraphOrderingProblem ordering_problem = build_graph_ordering_problem(
	// 	waypoints, graph, x0, v0);

	// // Solve
	// drake::solvers::MosekSolver solver;
	// const auto ordering_result = solver.Solve(*ordering_problem.prog);
	// // auto ordering_result = drake::solvers::Solve(*ordering_problem.prog);

	// Eigen::VectorXi ordering;
	// if (ordering_result.is_success()) {
	// 	for (int i = 0; i < num_nodes; ++i) {
	// 		for (int j = 0; j < num_nodes; ++j) {
	// 			const double val = ordering_result.GetSolution(ordering_problem.p(i, j));
	// 			if (val > 0.5) {
	// 				ordering(i) = j;
	// 				break;
	// 			}
	// 		}
	// 	}
	// } else {
	// 	std::cerr << "Ordering Optimization failed." << std::endl;
	// 	return std::nullopt;
	// }

	// /* TODO: Change according to phase */
	// /* initialize output ordering array (n,) */
	// Eigen::MatrixXd ordered_waypoints({num_nodes, _dim});
	// for (int i = 0; i < num_nodes; ++i) {
	// 	const auto rank = ordering(i);
	// 	for (int j = 0; j < _dim; ++j) {
	// 		ordered_waypoints(i, j) = waypoints(rank, j);
	// 	}
	// }

	GraphTimingProblem problem = build_graph_timing_problem(
		*_graph, remaining_vertices, waypoints, assignments, x0, v0,
		_time_cost, 0.0, _ctrl_cost, _max_vel, _max_acc, _max_jerk);

	// Store ordered waypoints used in problem
	_wps_list = problem.wps_list;

	// Store nodes on each spline
	_agent_nodes_list = problem.agent_nodes_list;

	// Solve
	auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		for (int i = 0; i < _graph->num_agents; ++i) {
			Eigen::MatrixXd vs = result.GetSolution(problem.vs_list[i]);
			Eigen::VectorXd taus = result.GetSolution(problem.time_deltas_list[i]);

			const int agent_spline_length = taus.size() + 1;
			_agent_spline_length_map[i] = agent_spline_length;

			for (int j = 0; j < vs.rows(); ++j) {
				_vs_list[i].row(j) = vs.row(j);
			}

			for (int j = 0; j < taus.size(); ++j) {
				_time_deltas_list[i](j) = taus(j);
			}
		}
		return true;
	} else {
		return false;
	}
}

int GraphTimingMPC::get_agent_spline_length(int agent) {
	if (!_agent_spline_length_map.contains(agent)) {
		return 0;
	} else {
		return _agent_spline_length_map.at(agent);
	}
}

std::set<int> GraphTimingMPC::set_progressed_time(double delta, double tau_cutoff) {
	/* this function, instead of resolving for all the vertices and taus, as
	 * above, just updates the first tau of each remaining active spline.
	 * These changes should also update the intialization of the solver. */

	std::set<int> passed_nodes;
	
	for (int i = 0; i < _graph->num_agents; ++i) {
		// Eigen::MatrixXd vs = result.GetSolution(problem.vs_list[i]);
		// Eigen::VectorXd taus = result.GetSolution(problem.time_deltas_list[i]);

		// const int agent_spline_length = taus.size() + 1;
		if (_agent_spline_length_map[i] > 0) {
			const double tau0 = _time_deltas_list[i](0);
			if (delta < tau0) {
				; // TODO: (changing initialization of solver)
				// _time_deltas_list[i](0) -= delta;
			} else {
				// if there is another phase
				if (_agent_nodes_list[i].size() > 1) {
					// std::cout << "node should be passed!" << std::endl;
					passed_nodes.insert(_agent_nodes_list[i][0]);
				}
			}
		}

		// for (int j = 0; j < vs.rows(); ++j) {
		// 	_vs_list[i].row(j) = vs.row(j);
		// }
	}

	return passed_nodes;
}

Eigen::VectorXd cumsum_with_zero(const Eigen::VectorXd& x, int n) {
	Eigen::VectorXd y(n+1);
	double s = 0.0;
	for (int i = 0; i < n + 1; ++i) {
		y(i) = s;
		s += x(i);
	}
	return y;
}

void GraphTimingMPC::fill_cubic_splines(std::vector<CubicSpline*>& splines,
					const Eigen::VectorXd& x0,
					const Eigen::VectorXd& v0) const {

	int d = _graph->dim;

	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			Eigen::VectorXd x0_i = x0.segment(i * d, d);
			Eigen::MatrixXd wps_i(spline_length_i, d);
			wps_i.row(0) = x0_i;
			wps_i.bottomRows(spline_length_i - 1) = _wps_list[i];

			Eigen::VectorXd v0_i = v0.segment(i * d, d);
			Eigen::MatrixXd vs_i(spline_length_i, d);
			vs_i.row(0) = v0_i;
			vs_i.block(1, 0, spline_length_i - 2, d) = _vs_list[i];
			vs_i.row(spline_length_i - 1).setZero();

			Eigen::VectorXd times_i = cumsum_with_zero(_time_deltas_list[i], spline_length_i - 1);

			splines[i]->set(wps_i, vs_i, times_i);
		} else {
			// Dummy spline that stays at x0, and comes to a stop after 1 second.
			Eigen::VectorXd x0_i = x0.segment(i * d, d);
			Eigen::MatrixXd wps_i(2, d);
			wps_i.row(0) = x0_i;
			wps_i.row(1) = x0_i;

			Eigen::VectorXd v0_i = v0.segment(i * d, d);
			Eigen::MatrixXd vs_i(2, d);
			vs_i.row(0) = v0_i;
			vs_i.row(1).setZero();

			Eigen::VectorXd times_i(2);
			times_i << 0.0, 1.0;

			splines[i]->set(wps_i, vs_i, times_i);
		}
	}
}


// Safe indexing and accessors
// py::array_t<double> GraphTimingMPC::get_waypoints() const {
// 	if (done()) {
// 		return remainder_slice_2d(_waypoints, 0);
// 	} else {
// 		return remainder_slice_2d(_waypoints, 0);
// 	}
// }

// py::array_t<double> TimingMPC::get_time_deltas() const {
// 	if (done()) {
// 		// Return single time step
// 		py::array_t<double> ret(1);
// 		auto ret_mut = ret.mutable_unchecked<1>();
// 		ret_mut(0) = 0.1;
// 		return ret;
// 	}
// 	auto remaining_time_deltas = remainder_slice_1d(this->time_deltas, this->phase);
// 	return remaining_time_deltas;
// }

// py::array_t<unsigned int> GraphTimingMPC::get_ordering() const {
// 	// if (done()) {
// 	// 	// Return single time step
// 	// 	py::array_t<double> ret(1);
// 	// 	auto ret_mut = ret.mutable_unchecked<1>();
// 	// 	ret_mut(0) = 0.1;
// 	// 	return ret;
// 	// }
// 	// auto remaining_time_deltas = remainder_slice_1d(this->time_deltas, this->phase);
// 	// return integral(remaining_time_deltas);
// 	return _ordering;
// }

// py::array_t<double> GraphTimingMPC::get_times() const {
// 	if (done()) {
// 		// Return single time step
// 		py::array_t<double> ret(1);
// 		auto ret_mut = ret.mutable_unchecked<1>();
// 		ret_mut(0) = 0.1;
// 		return ret;
// 	}
// 	auto remaining_time_deltas = remainder_slice_1d(_time_deltas, 0);
// 	return integral(remaining_time_deltas);
// }


// py::array_t<double> GraphTimingMPC::get_vels() const {
// 	const int phase = 0; // static_cast<int>(this->phase);

// 	// Done: return final velocity (usually zero)
// 	if (done()) {
// 		return remainder_slice_2d(_vels, _num_nodes - 1);
// 	}

// 	int num_rows = _num_nodes - phase;  // including final appended zero row
// 	int num_cols = _dim;

// 	// Allocate final velocity array (including appended zero row)
// 	py::array_t<double> result({num_rows, num_cols});
// 	auto result_mut = result.mutable_unchecked<2>();

// 	// Directly copy from this->vels
// 	auto src = remainder_slice_2d(_vels, phase).unchecked<2>();
// 	for (int i = 0; i < num_rows - 1; ++i) {
// 		for (int j = 0; j < _dim; ++j) {
// 			result_mut(i, j) = src(i, j);
// 		}
// 	}

// 	// Append zero vector at the end
// 	for (int j = 0; j < _dim; ++j) {
// 		result_mut(num_rows - 1, j) = 0.0;
// 	}

// 	return result;
// }


/* after having optimized for a waypoint ordering and a spline going
 * through the waypoints in that order, update the spline according to
 * an amount of passed time. Also check for phase progression when it's
 * expected. */
// bool GraphTimingMPC::set_progressed_time(double time_delta, double time_delta_cutoff) {
// 	bool any_progression = false;
// 	std::set<unsigned int> next_nodes = _next_nodes();

// 	auto time_deltas_mut = _time_deltas.mutable_unchecked<1>();
// 	for (unsigned int n : next_nodes) {
// 		if (time_delta < time_deltas_mut(n)) {
// 			time_deltas_mut(n) -= time_delta;
// 		} else {

// 		}
// 	}

// 	if (time_delta < tau(phase)) { // time still within phase
// 		tau(phase) -= gap; //change initialization of timeOpt
// 		return false;
// 	}

// 	//time beyond current phase
// 	if (phase + 1 < nPhases()) { //if there exists another phase
// 		tau(phase+1) -= gap-tau(phase); //change initialization of timeOpt
// 		tau(phase) = 0.; //change initialization of timeOpt
// 	} else {
// 		if(phase+1==nPhases() && neverDone) { //stay in last phase and reinit tau=.1
// 			tau(phase)=.1+tauCutoff;
// 			return false;
// 		}
// 		tau = 0.;
// 	}

// 	phase++; //increase phase
// 	return true;
// }

// void TimingMPC::set_updated_waypoints(const py::array_t<double>& _waypoints, bool set_next_waypoint_tangent) {
// 	if (_waypoints.size() != this->waypoints.size()) { //full reset
// 		waypoints = _waypoints;
// 		tau = 10.0 * ones(waypoints.d0);
// 		vels.clear();
// 		tangents.clear();
// 	} else if (&waypoints != &_waypoints) {
// 		waypoints = _waypoints;
// 	}

// 	if (set_next_waypoint_tangent) {
// 		LOG(-1) <<"questionable";
// 		tangents.resize(waypoints.d0-1, waypoints.d1);
// 		for(uint k=1; k<waypoints.d0; k++) {
// 			tangents[k-1] = waypoints[k] - waypoints[k-1];
// 			op_normalize(tangents[k-1].noconst());
// 		}
// 	}
// }

// void TimingMPC::update_backtrack() {
// 	if (this->phase == 0) {
// 		throw std::runtime_error("Cannot backtrack from phase 0.");
// 	}

// 	/* by default, go back one */
// 	unsigned int phaseTo = this->phase - 1;

// 	/* Check if back_tracking_table is initialized and non-empty */
// 	/* if so, use that to determine where to go */
// 	if (this->back_tracking_table.size() > 0) {
// 		if (this->phase >= this->back_tracking_table.size()) {
// 			throw std::runtime_error("Phase index out of bounds in back_tracking_table.");
// 		}
// 		auto bt = this->back_tracking_table.unchecked<1>();
// 		phaseTo = bt(this->phase);
// 	}

// 	this->update_set_phase(phaseTo);
// }

// void TimingMPC::update_set_phase(unsigned int phaseTo) {
// 	std::cout << "[TimingMPC] Backtracking from phase " << this->phase
// 		  << " to " << phaseTo << " times: " << std::endl;

// 	if (phaseTo > this->phase) {
// 		throw std::runtime_error("Cannot advance phase using update_set_phase — only backward steps allowed.");
// 	}

// 	auto time_deltas_ = this->time_deltas.mutable_unchecked<1>();
// 	int N = time_deltas_.shape(0);

// 	while (this->phase > phaseTo) {
// 		if (this->phase < static_cast<unsigned int>(N)) {
// 			time_deltas_(this->phase) = std::max(1.0, time_deltas_(this->phase));
// 		}
// 		this->phase--;
// 	}

// 	time_deltas_(this->phase) = 1.0;
// }


