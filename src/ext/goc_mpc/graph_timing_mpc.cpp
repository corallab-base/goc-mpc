#include "graph_timing_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


using drake::solvers::MathematicalProgram;
using drake::solvers::MathematicalProgramResult;
using drake::solvers::IpoptSolverDetails;

static void PrintSolverReport(GraphTimingProblem* problem,
			      const MathematicalProgramResult& result,
			      double tol = 1e-6) {
	using std::cout;
	using std::endl;

	cout << "=== Drake Solve() Report ===\n";
	cout << "Solver:           " << result.get_solver_id().name() << "\n";
	// to_string(SolutionResult) is supported; if your Drake is older, cast to int.
	cout << "SolutionResult:   " << to_string(result.get_solution_result()) << "\n";
	cout << "Success?          " << (result.is_success() ? "yes" : "no") << "\n";

	// Print optimal cost if available.
	if (std::isfinite(result.get_optimal_cost())) {
		cout << "Optimal cost:     " << result.get_optimal_cost() << "\n";
	}

	// Try to obtain a decision vector to evaluate constraints.
	Eigen::VectorXd x;
	bool have_x = false;
	try {
		x = result.GetSolution(problem->prog->decision_variables());
		have_x = true;
	} catch (const std::exception&) {
		// Fall back to initial guess (may be zero) if solver did not return x.
		if (problem->prog->decision_variables().size() > 0) {
			x = problem->prog->initial_guess();
			have_x = true;
			cout << "(No returned solution vector; using initial guess for diagnostics.)\n";
		}
	}

	// Constraint violation scan.
	if (have_x) {
		try {
			auto print_binding_violation = [&](const auto& bindings, const char* kind) {
				for (const auto& b : bindings) {
					const auto& c = *b.evaluator();
					const auto& vars = b.variables();
					Eigen::VectorXd x_b = result.GetSolution(vars);

					double tol = 0.05;
					if (!c.CheckSatisfied(x_b, tol)) {
						cout << "Violation [" << kind << "]: "
						     << " (constraint: " << c.get_description() << ")\n";
					} else {
						cout << "Satisfied [" << kind << "]: "
						     << " (constraint: " << c.get_description() << ")\n";
					}
				}
			};

			cout << "--- Constraint violations (inf-norm > " << tol << ") ---\n";
			print_binding_violation(problem->prog->bounding_box_constraints(),   "BoundingBox");
			print_binding_violation(problem->prog->linear_equality_constraints(),"LinEq");
			print_binding_violation(problem->prog->linear_constraints(),         "LinIneq");
			print_binding_violation(problem->prog->lorentz_cone_constraints(),   "Lorentz");
			print_binding_violation(problem->prog->rotated_lorentz_cone_constraints(),"RotLorentz");
			print_binding_violation(problem->prog->quadratic_constraints(),      "Quadratic");
			print_binding_violation(problem->prog->exponential_cone_constraints(),"ExpCone");
			print_binding_violation(problem->prog->generic_constraints(),        "Generic");
		} catch (const std::exception& e) {
			cout << "Exception in printing binding violations: " << e.what() << endl;
		}
	} else {
		cout << "(No decision vector available to evaluate constraints.)\n";
	}

	// Try to print some solver-specific details (best-effort).
	try {
		const auto& sid = result.get_solver_id().name();
		if (sid == "Ipopt") {
			struct IpoptDetails {
				int status; int iterations; double objective; double dual_inf;
				double constr_viol; double comp_viol; double primal_step; double dual_step;
			};

			for (const std::string& s : result.GetInfeasibleConstraintNames(*problem->prog, tol)) {
				cout << "infeasible constraint: " << s << std::endl;
			}

			// If your Drake provides IpoptSolver::Details, use that exact type.
			// Example (adjust to your Drake version):
			const auto& d = result.get_solver_details<drake::solvers::IpoptSolver>();
			cout << "Ipopt status: " << d.ConvertStatusToString() << endl;
		} else if (sid == "NLopt") {
			const auto& d = result.get_solver_details<drake::solvers::NloptSolver>();
			cout << "TODO: Print nlopt information" << endl;
		} else if (sid == "OSQP") {
			// Example (adjust to your Drake version):
			// const auto& d = result.get_solver_details<OsqpSolver>();
			// cout << "OSQP status: " << d.status_val << ", iters: " << d.iter
			//      << ", obj: " << d.obj_val << "\n";
		} else if (sid == "Gurobi") {
			// const auto& d = result.get_solver_details<GurobiSolver>();
			// cout << "Gurobi status: " << d.optimization_status << ", iters: "
			//      << d.iteration_count << ", mip_gap: " << d.mip_relative_gap << "\n";
		}
	} catch (const std::exception& e) {
		cout << "(Could not print solver-specific details: " << e.what() << ")\n";
	}

	cout << "=== End Report ===\n";
}


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
	const std::vector<CubicConfigurationSpline>& splines,
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

	int ambient_dim = splines.at(0).ambient_dim();
	int tangent_dim = splines.at(0).tangent_dim();

	const auto& [parents, agent_nodes, agent_interactions] = graph.get_agent_paths(remaining_vertices, assignments);

	// for (auto edge : cross_agent_edges) {
	// 	std::cout << "edge u: " << edge.first << std::endl;
	// 	std::cout << "edge v: " << edge.second << std::endl;
	// }

	// grow agent_nodes_list to accomodate the number of agents
	problem.agent_nodes_list.resize(graph.num_agents);

	for (int i = 0; i < graph.num_agents; ++i) {
		const CubicConfigurationSpline& spline = splines.at(i);

		const std::vector<int> agent_i_nodes = agent_nodes[i];
		problem.agent_nodes_list[i] = agent_i_nodes;

		const int agent_spline_length = agent_i_nodes.size();

		// If there is nothing left in this spline, there's nothing to optimize.
		if (agent_spline_length > 0) {
			char time_deltas_name[32];
			char vs_name[32];

			Eigen::MatrixXd wps_i(agent_spline_length, ambient_dim);
			for (int j = 0; j < agent_spline_length; ++j) {
				int node = agent_i_nodes[j];
				for (int k = 0; k < ambient_dim; ++k) {
					wps_i(j, k) = waypoints(node, i * ambient_dim + k);
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

			MatrixXDecisionVariable vs_i = problem.prog->NewContinuousVariables(agent_spline_length - 1, tangent_dim, vs_name);
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
					VectorX<Expression> xJ(ambient_dim), xJm1(ambient_dim), vJ(tangent_dim), vJm1(tangent_dim);
					const Expression tau(time_deltas_i(j));
					if (j == 0 && j < agent_spline_length - 1) {
						for (int k = 0; k < ambient_dim; ++k) {
							xJm1(k) = Expression(x0(i * ambient_dim + k));
							xJ(k)   = Expression(wps_i(0, k));
						}
						for (int k = 0; k < tangent_dim; ++k) {
							vJm1(k) = Expression(v0(i * tangent_dim + k));
							vJ(k)   = Expression(vs_i(0, k));
						}
					} else if (j > 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < ambient_dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
						}
						for (int k = 0; k < tangent_dim; ++k) {
							vJm1(k) = Expression(vs_i(j-1, k));
							vJ(k) = Expression(0.0);
						}
					} else if (j == 0 && j == agent_spline_length - 1) {
						for (int k = 0; k < ambient_dim; ++k) {
							xJm1(k) = Expression(x0(i * ambient_dim + k));
							xJ(k)   = Expression(wps_i(j, k));
						}
						for (int k = 0; k < tangent_dim; ++k) {
							vJm1(k) = Expression(v0(i * tangent_dim + k));
							vJ(k) = Expression(0.0);
						}
					} else {
						for (int k = 0; k < ambient_dim; ++k) {
							xJm1(k) = Expression(wps_i(j-1, k));
							xJ(k)   = Expression(wps_i(j, k));
						}
						for (int k = 0; k < tangent_dim; ++k) {
							vJm1(k) = Expression(vs_i(j-1, k));
							vJ(k)   = Expression(vs_i(j, k));
						}
					}
					const Expression c = spline.compute_ctrl_cost<Expression>(xJ, xJm1, vJ, vJm1, tau);
					problem.prog->AddCost(c);
				}
			}

			// Velocity/Acceleration/Jerk Constraints
			// if (max_vel > 0) {
			// 	for (int j = 0; j < agent_spline_length; ++j) {
			// 		VectorX<Expression> xJm1(dim), xJ(dim), vJm1(dim), vJ(dim);
			// 		const Expression tau(time_deltas_i(j));
			// 		const Expression inv_tau = pow(tau, -1.0);
			// 		if (j == 0 && j < agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * dim + k));
			// 				vJm1(k) = Expression(v0(i * dim + k));
			// 				xJ(k)   = Expression(wps_i(0, k));
			// 				vJ(k)   = Expression(vs_i(0, k));
			// 			}
			// 		} else if (j > 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k).Zero();
			// 			}
			// 		} else if (j == 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * dim + k));
			// 				vJm1(k) = Expression(v0(i * dim + k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k).Zero();
			// 			}
			// 		} else {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k)   = Expression(vs_i(j, k));
			// 			}
			// 		}

			// 		// Your cubic parameterization (unscaled a,b,c as in your code)
			// 		// c = v0
			// 		const VectorX<Expression> c = vJm1;
			// 		// b = 3*(x1 - x0) - tau*(v1 + 2*v0)
			// 		const VectorX<Expression> b = 3.0*(xJ - xJm1) - tau*(vJ + 2.0*vJm1);
			// 		// a = -2*(x1 - x0) + tau*(v1 + v0)
			// 		const VectorX<Expression> a = -2.0*(xJ - xJm1) + tau*(vJ + vJm1);

			// 		// Midpoint velocity surrogate: v_mid = c + (1/tau)*(b + 3/4 a)
			// 		const VectorX<Expression> v_mid = c + inv_tau * (b + 0.75 * a);

			// 		// Enforce |v(0)| <= vmax and |v_mid| <= vmax  (elementwise)
			// 		Eigen::VectorXd lb = Eigen::VectorXd::Constant(dim, -max_vel);
			// 		Eigen::VectorXd ub = Eigen::VectorXd::Constant(dim,  max_vel);
			// 		problem.prog->AddConstraint(c,     lb, ub);   // v(0) = c
			// 		problem.prog->AddConstraint(v_mid, lb, ub);   // v(tau/2)
			// 	}
			// }

			// if (max_acc > 0) {
			// 	for (int j = 0; j < agent_spline_length; ++j) {
			// 		VectorX<Expression> xJ(ambient_dim), xJm1(ambient_dim), vJ(tangent_dim), vJm1(tangent_dim);
			// 		const Expression tau(time_deltas_i(j));
			// 		if (j == 0 && j < agent_spline_length - 1) {
			// 			for (int k = 0; k < ambient_dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * ambient_dim + k));
			// 				xJ(k)   = Expression(wps_i(0, k));
			// 			}
			// 			for (int k = 0; k < tangent_dim; ++k) {
			// 				vJm1(k) = Expression(v0(i * tangent_dim + k));
			// 				vJ(k)   = Expression(vs_i(0, k));
			// 			}
			// 		} else if (j > 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < ambient_dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 			}
			// 			for (int k = 0; k < tangent_dim; ++k) {
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				vJ(k) = Expression(0.0);
			// 			}
			// 		} else if (j == 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < ambient_dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * ambient_dim + k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 			}
			// 			for (int k = 0; k < tangent_dim; ++k) {
			// 				vJm1(k) = Expression(v0(i * tangent_dim + k));
			// 				vJ(k) = Expression(0.0);
			// 			}
			// 		} else {
			// 			for (int k = 0; k < ambient_dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 			}
			// 			for (int k = 0; k < tangent_dim; ++k) {
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				vJ(k)   = Expression(vs_i(j, k));
			// 			}
			// 		}

			// 		// b2 = 2/tau^2 * ( 3*(x1 - x0) - tau*(v1 + 2*v0) )
			// 		const Expression inv_tau2 = pow(tau, -2.0);
			// 		const VectorX<Expression> b2 =
			// 			2.0 * inv_tau2 * ( 3.0*(xJ - xJm1) - tau*(vJ + 2.0*vJm1) );

			// 		// a6_tau = 6/tau^2 * ( -2*(x1 - x0) + tau*(v1 + v0) )
			// 		const VectorX<Expression> a6_tau =
			// 			6.0 * inv_tau2 * ( -2.0*(xJ - xJm1) + tau*(vJ + vJm1) );

			// 		// Endpoint accelerations
			// 		const VectorX<Expression> acc0  = b2;                 // t = 0
			// 		const VectorX<Expression> accT  = b2 + a6_tau;        // t = tau

			// 		// ||acc(0)||_inf <= amax and ||acc(tau)||_inf <= amax (elementwise)
			// 		Eigen::VectorXd lb = Eigen::VectorXd::Constant(dim, -max_acc);
			// 		Eigen::VectorXd ub = Eigen::VectorXd::Constant(dim,  max_acc);
			// 		problem.prog->AddConstraint(acc0, lb, ub);
			// 		problem.prog->AddConstraint(accT, lb, ub);
			// 	}
			// }

			// if (max_jerk > 0) {
			// 	for (int j = 0; j < agent_spline_length; ++j) {
			// 		VectorX<Expression> xJ(dim), xJm1(dim), vJ(dim), vJm1(dim);
			// 		const Expression tau(time_deltas_i(j));
			// 		if (j == 0 && j < agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * dim + k));
			// 				vJm1(k) = Expression(v0(i * dim + k));
			// 				xJ(k)   = Expression(wps_i(0, k));
			// 				vJ(k)   = Expression(vs_i(0, k));
			// 			}
			// 		} else if (j > 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k).Zero();
			// 			}
			// 		} else if (j == 0 && j == agent_spline_length - 1) {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(x0(i * dim + k));
			// 				vJm1(k) = Expression(v0(i * dim + k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k).Zero();
			// 			}
			// 		} else {
			// 			for (int k = 0; k < dim; ++k) {
			// 				xJm1(k) = Expression(wps_i(j-1, k));
			// 				vJm1(k) = Expression(vs_i(j-1, k));
			// 				xJ(k)   = Expression(wps_i(j, k));
			// 				vJ(k)   = Expression(vs_i(j, k));
			// 			}
			// 		}

			// 		// a6 = 6/tau^3 * (-2*(x1 - x0) + tau*(v1 + v0))
			// 		const Expression coeff = 6.0 * pow(tau, -3.0);
			// 		const VectorX<Expression> a6 = coeff * ( -2.0 * (xJ - xJm1) + tau * (vJ + vJm1) );

			// 		// Bound a6 with symmetric bounds:
			// 		Eigen::VectorXd lb2 = Eigen::VectorXd::Constant(dim, -max_jerk);
			// 		Eigen::VectorXd ub2 = Eigen::VectorXd::Constant(dim,  max_jerk);
			// 		problem.prog->AddConstraint(a6, lb2, ub2);
			// 	}
			// }
		}

		// const bool totally_ordered = true;
		// if (!totally_ordered) {
		// 	// Solve for ordering
		// } else {
		// 	ordering = arange(0, agent_subgraph);
		// }

	}

	for (const struct AgentInteraction& p : agent_interactions) {
		if (p.type == AgentInteraction::Type::LESS_THAN) {
			// taus
			Eigen::VectorX<Expression> taus_i = problem.time_deltas_list[p.agent_i].head(p.agent_i_depth+1);
			Eigen::VectorX<Expression> taus_j = problem.time_deltas_list[p.agent_j].head(p.agent_j_depth+1);
			problem.prog->AddLinearConstraint(taus_i.sum() <= taus_j.sum());
		} else if (p.type == AgentInteraction::Type::EQUAL) {
			Eigen::VectorX<Expression> taus_i = problem.time_deltas_list[p.agent_i].head(p.agent_i_depth+1);
			Eigen::VectorX<Expression> taus_j = problem.time_deltas_list[p.agent_j].head(p.agent_j_depth+1);
			problem.prog->AddLinearConstraint(taus_i.sum() == taus_j.sum());
		} else {
			throw std::runtime_error("Not implemented");
		}
	}

	return problem;
}

/*
 * Graph Timing MPC
 */

GraphTimingMPC::GraphTimingMPC(const GraphOfConstraints& graph,
			       std::vector<CubicConfigurationSpline> splines,
			       double time_cost,
			       double time_cost2,
			       double ctrl_cost,
			       double max_vel,
			       double max_acc,
			       double max_jerk)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _time_cost(time_cost),
	  _time_cost2(time_cost2),
	  _ctrl_cost(ctrl_cost),
	  _max_vel(max_vel),
	  _max_acc(max_acc),
	  _max_jerk(max_jerk),
	  _vs_list(graph.num_agents),
	  _time_deltas_list(graph.num_agents) {

	int num_agents = _graph->num_agents;
	int num_nodes = _graph->structure.num_nodes();
	// Assuming all the same.
	int ambient_dim = _splines->at(0).ambient_dim();
	int tangent_dim = _splines->at(0).tangent_dim();

	// A waypoint and velocity array for each agent only.
	_wps_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		_wps_list[i] = Eigen::MatrixXd::Zero(num_nodes, ambient_dim);
	}
	_vs_list.resize(num_agents);
	for (int i = 0; i < num_agents; ++i) {
		_vs_list[i] = Eigen::MatrixXd::Zero(num_nodes, tangent_dim);
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

	_timer.Start();

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

	std::unique_ptr<GraphTimingProblem> problem;
	try {
		problem = std::make_unique<GraphTimingProblem>(
			build_graph_timing_problem(
				*_graph, *_splines, remaining_vertices, waypoints, assignments, x0, v0,
				_time_cost, _time_cost2, _ctrl_cost, _max_vel, _max_acc, _max_jerk));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in timing problem construction" << std::endl;
		return false;
	}



	// Store ordered waypoints used in problem
	_wps_list = problem->wps_list;

	// Store nodes on each spline
	_agent_nodes_list = problem->agent_nodes_list;

	// Warm start vs and taus per agent
	// for (int i = 0; i < _graph->num_agents; ++i) {
	// 	const int agent_spline_length = taus.size() + 1;

	// 	for (int j = 0; j < vs.rows(); ++j) {
	// 		problem->prog->SetInitialGuess(
	// 			problem->vs_list[i], _vs_list[i].row(j));
	// 	}

	// 	for (int j = 0; j < taus.size(); ++j) {
	// 		problem->prog->SetInitialGuess(
	// 			problem->time_deltas_list[i](j), _time_deltas_list[i](j));
	// 	}
	// }

	// for (int v : remaining_vertices) {
	// 	const int i = problem->subgraph->subgraph_id(v);
	// 	problem->prog->SetInitialGuess(problem->X.row(i), _waypoints.row(v));
	// }


	// Solve
	drake::solvers::IpoptSolver solver;
	// auto result = solver.Solve(*problem.prog);

	MathematicalProgramResult result;
	try {
		// result = drake::solvers::Solve(*(problem->prog));
		result = solver.Solve(*(problem->prog));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in solver" << std::endl;
		return false;
	}

	if (result.is_success()) {

		_last_solve_time = _timer.Tick();

		for (int i = 0; i < _graph->num_agents; ++i) {
			Eigen::MatrixXd vs = result.GetSolution(problem->vs_list[i]);
			Eigen::VectorXd taus = result.GetSolution(problem->time_deltas_list[i]);

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
		PrintSolverReport(problem.get(), result, 1e-6);
		return false;
	}
}

int GraphTimingMPC::get_agent_spline_length(int agent) const {
	if (!_agent_spline_length_map.contains(agent)) {
		return 0;
	} else {
		return _agent_spline_length_map.at(agent);
	}
}

std::vector<int> GraphTimingMPC::get_agent_spline_nodes(int agent) const {
	if (agent < 0 || agent > _agent_nodes_list.size()) {
		return std::vector<int>();
	} else {
		return _agent_nodes_list.at(agent);
	}
}


std::set<int> GraphTimingMPC::set_progressed_time(double delta, double tau_cutoff) {
	/* this function, instead of resolving for all the vertices and taus, as
	 * above, just updates the first tau of each remaining active spline.
	 * These changes should also update the intialization of the solver. */

	std::set<int> passed_nodes;

	for (int i = 0; i < _graph->num_agents; ++i) {
		// Eigen::MatrixXd vs = result.GetSolution(problem->vs_list[i]);
		// Eigen::VectorXd taus = result.GetSolution(problem->time_deltas_list[i]);

		// const int agent_spline_length = taus.size() + 1;
		if (_agent_spline_length_map[i] > 0) {
			const double tau0 = _time_deltas_list[i](0);
			if (delta < tau0 - tau_cutoff) {
				; // TODO: (changing initialization of solver)
				// _time_deltas_list[i](0) -= delta;
			} else {
				// if there is another phase
				if (_agent_nodes_list[i].size() > 0) {
					// std::cout << "node should be passed!" << std::endl;
					passed_nodes.insert(_agent_nodes_list[i][0]);
					std::cout << "passed " << _agent_nodes_list[i][0] << std::endl;
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

void GraphTimingMPC::fill_cubic_splines(std::vector<CubicConfigurationSpline*>& splines,
					const Eigen::VectorXd& x0,
					const Eigen::VectorXd& v0) const {

	int a_d = splines[0]->ambient_dim();
	int t_d = splines[0]->tangent_dim();

	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(spline_length_i, a_d);
			wps_i.row(0) = x0_i;
			wps_i.bottomRows(spline_length_i - 1) = _wps_list[i];

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(spline_length_i, t_d);
			vs_i.row(0) = v0_i;
			vs_i.block(1, 0, spline_length_i - 2, t_d) = _vs_list[i];
			vs_i.row(spline_length_i - 1).setZero();

			Eigen::VectorXd times_i = cumsum_with_zero(_time_deltas_list[i], spline_length_i - 1);

			splines[i]->set(wps_i, vs_i, times_i);
		} else {
			// Dummy spline that stays at x0, and comes to a stop after 1 second.
			Eigen::VectorXd x0_i = x0.segment(i * a_d, a_d);
			Eigen::MatrixXd wps_i(2, a_d);
			wps_i.row(0) = x0_i;
			wps_i.row(1) = x0_i;

			Eigen::VectorXd v0_i = v0.segment(i * t_d, t_d);
			Eigen::MatrixXd vs_i(2, t_d);
			vs_i.row(0) = v0_i;
			vs_i.row(1).setZero();

			Eigen::VectorXd times_i(2);
			times_i << 0.0, 1.0;

			splines[i]->set(wps_i, vs_i, times_i);
		}
	}
}


// Safe indexing and accessors
const std::vector<double> GraphTimingMPC::get_next_taus() const {
	std::vector<double> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			double tau = _time_deltas_list[i](0);
			result.push_back(tau);
		}
	}
	return result;
}

// Safe indexing and accessors
const std::vector<int> GraphTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1) {
			int node = _agent_nodes_list[i].at(0);
			result.push_back(node);
		}
	}
	return result;
}


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


