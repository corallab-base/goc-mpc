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
						Eigen::VectorXd y;
						c.Eval(x_b, &y);
						cout << "Violation [" << kind << "]: "
						     << " (constraint: " << c.get_description() << ")"
						     << " value=" << y.transpose()
						     << " lb=" << c.lower_bound().transpose()
						     << " ub=" << c.upper_bound().transpose() << "\n";
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

AgentSegmentVars add_agent_timing_segments(
	MathematicalProgram& prog,
	const CubicConfigurationSpline& spline,
	const Eigen::MatrixXd& wps_i,
	const Eigen::VectorXd& x0_i,
	const Eigen::VectorXd& v0_i,
	int agent_index,
	double time_cost,
	double time_cost2,
	double acceleration_cost,
	double energy_cost,
	double arclength_cost,
	double stability_cost,
	const std::vector<double>& max_vel,
	const std::vector<double>& max_acc,
	const std::vector<double>& max_jerk) {

	using namespace drake::solvers;

	const int agent_spline_length = static_cast<int>(wps_i.rows());
	const int ambient_dim = static_cast<int>(wps_i.cols());
	const int tangent_dim = spline.tangent_dim();

	char time_deltas_name[32];
	char vs_name[32];
	snprintf(time_deltas_name, 32, "time_deltas_%d", agent_index);
	snprintf(vs_name, 32, "vs_%d", agent_index);

	// Create variables
	VectorXDecisionVariable time_deltas_i = prog.NewContinuousVariables(agent_spline_length, time_deltas_name);
	for (int j = 0; j < agent_spline_length; ++j) {
		prog.AddBoundingBoxConstraint(0.01, 100.0, time_deltas_i(j))
			.evaluator()->set_description("time delta bound constraint");
	}

	MatrixXDecisionVariable vs_i = prog.NewContinuousVariables(agent_spline_length - 1, tangent_dim, vs_name);

	// Set initial guess
	prog.SetInitialGuess(vs_i, Eigen::MatrixXd::Constant(vs_i.rows(), vs_i.cols(), 1.0));
	prog.SetInitialGuess(time_deltas_i, Eigen::VectorXd::Constant(agent_spline_length, 10.0));

	// 1. Linear objective: sum(time_deltas) (OT_f in original code)
	if (time_cost > 0.0) {
		prog.AddLinearCost(time_cost * Eigen::RowVectorXd::Ones(agent_spline_length), time_deltas_i);
	}

	// 2. Also, quadratic time_delta objective : sum_i time_deltas_i^2
	if (time_cost2 > 0.0) {
		const Eigen::MatrixXd Q = time_cost2 * Eigen::MatrixXd::Identity(agent_spline_length, agent_spline_length);
		const Eigen::VectorXd b = Eigen::VectorXd::Zero(agent_spline_length);
		prog.AddQuadraticCost(Q, b, time_deltas_i);
	}

	// 3. Control costs
	for (int j = 0; j < agent_spline_length; ++j) {
		VectorX<Expression> xJ(ambient_dim), xJm1(ambient_dim), vJ(tangent_dim), vJm1(tangent_dim);
		const Expression tau(time_deltas_i(j));
		if (j == 0 && j < agent_spline_length - 1) {
			for (int k = 0; k < ambient_dim; ++k) {
				xJm1(k) = Expression(x0_i(k));
				xJ(k)   = Expression(wps_i(0, k));
			}
			for (int k = 0; k < tangent_dim; ++k) {
				vJm1(k) = Expression(v0_i(k));
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
				xJm1(k) = Expression(x0_i(k));
				xJ(k)   = Expression(wps_i(j, k));
			}
			for (int k = 0; k < tangent_dim; ++k) {
				vJm1(k) = Expression(v0_i(k));
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

		if (acceleration_cost > 0) {
			const Expression acceleration = spline.compute_ctrl_cost<Expression>(xJ, xJm1, vJ, vJm1, tau);
			prog.AddCost(acceleration_cost * acceleration);
		}

		if (energy_cost > 0) {
			const Expression energy = spline.compute_energy_cost<Expression>(xJ, xJm1, vJ, vJm1, tau);
			prog.AddCost(energy_cost * energy);
		}

		if (arclength_cost > 0) {
			const Expression arclength = spline.compute_arclength_cost<Expression>(xJ, xJm1, vJ, vJm1, tau);
			prog.AddCost(arclength_cost * arclength);
		}

		// Convex alternative to acceleration_cost -- see _stability_cost's
		// doc comment in graph_timing_mpc.hpp. xJ/xJm1 are always fixed
		// constants at this point (only vJ/vJm1 can be free decision
		// variables, for interior segments), so ||xJ - xJm1||^2 is a
		// tau-independent constant per segment -- no cross term with tau,
		// unlike acceleration_cost's coast-corrected residual. Uses
		// CubicConfigurationSpline::PositionDelta (wrap-aware per block,
		// e.g. shortest-angle for Torus), not a raw ambient subtraction --
		// the same per-block residual compute_ctrl_cost's max_acc bound
		// already uses, so a Torus block's target sitting near the +/-pi
		// wraparound doesn't register as a near-2*pi "distance" whenever
		// the real state happens to read back on the other side of the
		// branch cut.
		if (stability_cost > 0) {
			const Expression dist2 = spline.PositionDelta<Expression>(xJ, xJm1).squaredNorm();
			const Expression vel_mismatch2 = (vJ - vJm1).squaredNorm();
			const Expression stability =
				dist2 * pow(tau, -3.0) + vel_mismatch2 * pow(tau, -1.0);
			prog.AddCost(stability_cost * stability);
		}

		// Max velocity / max acceleration constraints, one block at a time
		// (replaces the old select_linear_blocks-based single-scalar-over-
		// every-R-block approach): each block gets its OWN bound (max_vel[bi]/
		// max_acc[bi], <= 0 meaning unbounded for that block, an empty vector
		// meaning unbounded for every block -- same sentinel semantics the
		// single-scalar version used), and its OWN kind of constraint --
		// R/Torus get an axis-aligned box on that block's tangent columns
		// (Torus's tangent columns are genuine angular-rate scalars, so a box
		// is still the natural bound, same as R); SO3Quat gets an isotropic
		// bound on the norm of its so(3) tangent vector instead (a box has no
		// natural axis alignment for a 3D rotation) -- expressed as a squared-
		// norm inequality rather than a Lorentz cone since this is already an
		// Ipopt NLP with nonlinear pow(tau,-2) terms elsewhere in this same
		// function, not a convex QP/SOCP. The acceleration formula's
		// displacement term uses CubicConfigurationSpline::BlockPositionDelta
		// -- the SAME per-block residual (wrap-aware for Torus, quaternion-log
		// for SO3Quat) compute_ctrl_cost's soft acceleration_cost already
		// uses, so the hard bound and the soft cost can never silently
		// diverge on that part of the math.
		for (size_t bi = 0; bi < spline.block_offsets_.size(); ++bi) {
			const auto& off = spline.block_offsets_[bi];
			const int t0 = off.tangent_offset, tN = off.tangent_size;
			const double vmax = (bi < max_vel.size()) ? max_vel[bi] : -1.0;
			const double amax = (bi < max_acc.size()) ? max_acc[bi] : -1.0;
			const std::string tag = " (block " + std::to_string(bi) + ")";

			if (vmax > 0) {
				const VectorX<Expression> vJ_blk = vJ.segment(t0, tN);
				if (off.type == CubicConfigurationSpline::Block::Type::SO3Quat) {
					VectorX<Expression> normsq(1);
					normsq(0) = vJ_blk.squaredNorm();
					prog.AddConstraint(normsq, Eigen::VectorXd::Constant(1, 0.0),
							   Eigen::VectorXd::Constant(1, vmax * vmax))
						.evaluator()->set_description("max vel constraint" + tag);
				} else if (off.type == CubicConfigurationSpline::Block::Type::SO3Mat) {
					throw std::runtime_error(
						"max_vel bound requested for an SO3Mat block; unsupported "
						"(matches compute_ctrl_cost's existing TODO stub)");
				} else { // R, Torus
					Eigen::VectorXd lb = Eigen::VectorXd::Constant(tN, -vmax);
					Eigen::VectorXd ub = Eigen::VectorXd::Constant(tN,  vmax);
					prog.AddConstraint(vJ_blk, lb, ub)
						.evaluator()->set_description("max vel constraint" + tag);
				}
			}

			if (amax > 0) {
				const VectorX<Expression> delta =
					CubicConfigurationSpline::BlockPositionDelta<Expression>(off, xJ, xJm1);
				const VectorX<Expression> vJ_blk = vJ.segment(t0, tN);
				const VectorX<Expression> vJm1_blk = vJm1.segment(t0, tN);

				// 2b = 2/tau^2 * ( 3*delta - tau*(v1 + 2*v0) )
				const Expression inv_tau2 = pow(tau, -2.0);
				const VectorX<Expression> b2 =
					2.0 * inv_tau2 * ( 3.0*delta - tau*(vJ_blk + 2.0*vJm1_blk) );

				// a6_tau = 6/tau^2 * ( -2*delta + tau*(v1 + v0) )
				const VectorX<Expression> a6_tau =
					6.0 * inv_tau2 * ( -2.0*delta + tau*(vJ_blk + vJm1_blk) );

				// Endpoint accelerations
				const VectorX<Expression> acc0 = b2;          // t = 0
				const VectorX<Expression> accT = b2 + a6_tau; // t = tau

				if (off.type == CubicConfigurationSpline::Block::Type::SO3Quat) {
					VectorX<Expression> normsq0(1), normsqT(1);
					normsq0(0) = acc0.squaredNorm();
					normsqT(0) = accT.squaredNorm();
					prog.AddConstraint(normsq0, Eigen::VectorXd::Constant(1, 0.0),
							   Eigen::VectorXd::Constant(1, amax * amax))
						.evaluator()->set_description("max acc constraint" + tag);
					prog.AddConstraint(normsqT, Eigen::VectorXd::Constant(1, 0.0),
							   Eigen::VectorXd::Constant(1, amax * amax))
						.evaluator()->set_description("max acc constraint" + tag);
				} else if (off.type == CubicConfigurationSpline::Block::Type::SO3Mat) {
					throw std::runtime_error(
						"max_acc bound requested for an SO3Mat block; unsupported "
						"(matches compute_ctrl_cost's existing TODO stub)");
				} else { // R, Torus
					Eigen::VectorXd lb = Eigen::VectorXd::Constant(tN, -amax);
					Eigen::VectorXd ub = Eigen::VectorXd::Constant(tN,  amax);
					prog.AddConstraint(acc0, lb, ub)
						.evaluator()->set_description("max acc constraint" + tag);
					prog.AddConstraint(accT, lb, ub)
						.evaluator()->set_description("max acc constraint" + tag);
				}
			}
		}

		// if (max_jerk > 0) {
		// 	// a6 = 6/tau^3 * (-2*(x1 - x0) + tau*(v1 + v0))
		// 	const Expression coeff = 6.0 * pow(tau, -3.0);
		// 	const VectorX<Expression> a6 = coeff * ( -2.0 * (xJ - xJm1) + tau * (vJ + vJm1) );

		// 	// Bound a6 with symmetric bounds:
		// 	Eigen::VectorXd lb2 = Eigen::VectorXd::Constant(dim, -max_jerk);
		// 	Eigen::VectorXd ub2 = Eigen::VectorXd::Constant(dim,  max_jerk);
		// 	prog.AddConstraint(a6, lb2, ub2);
		// }
	}

	return AgentSegmentVars{time_deltas_i, vs_i};
}

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
	double acceleration_cost,
	double energy_cost,
	double arclength_cost,
	const std::vector<double>& max_vel,
	const std::vector<double>& max_acc,
	const std::vector<double>& max_jerk,
	double stability_cost,
	// If non-empty, per-agent node lists are sorted by MILP timestamp instead of BFS order.
	const Eigen::VectorXd& t_by_node) {

	using namespace drake::solvers;

	GraphTimingProblem problem(graph.num_agents);

	/* at this point, the waypoint optimizer has decided on assignments of
	 * agents to assignable tasks and their positions for satisfying those
	 * tasks. the order may not be completely determined for an individual
	 * agent, so we also solve for an optimal ordering if necessary. */

	// TODO: use assignments to settle conditional edges.

	int ambient_dim = splines.at(0).ambient_dim();
	int tangent_dim = splines.at(0).tangent_dim();

	// Ownership resolution (per-phi, falling back to formula introspection
	// for plain literal-placeholder constraints), t_by_node-based ordering,
	// and cross-agent/same-node interaction depths are all resolved inside
	// get_agent_paths itself now -- see its definition in
	// graph_of_constraints.cpp.
	auto [parents, agent_nodes, agent_interactions] =
		graph.get_agent_paths(remaining_vertices, assignments, t_by_node);

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
			Eigen::MatrixXd wps_i(agent_spline_length, ambient_dim);
			for (int j = 0; j < agent_spline_length; ++j) {
				int node = agent_i_nodes[j];
				for (int k = 0; k < ambient_dim; ++k) {
					wps_i(j, k) = waypoints(node, i * ambient_dim + k);
				}
			}
			problem.wps_list[i] = wps_i;

			const Eigen::VectorXd x0_i = x0.segment(i * ambient_dim, ambient_dim);
			const Eigen::VectorXd v0_i = v0.segment(i * tangent_dim, tangent_dim);

			AgentSegmentVars vars = add_agent_timing_segments(
				*problem.prog, spline, wps_i, x0_i, v0_i, i,
				time_cost, time_cost2, acceleration_cost, energy_cost,
				arclength_cost, stability_cost, max_vel, max_acc, max_jerk);

			problem.time_deltas_list[i] = vars.time_deltas;
			problem.vs_list[i] = vars.vs;
		}

		// const bool totally_ordered = true;
		// if (!totally_ordered) {
		// 	// Solve for ordering
		// } else {
		// 	ordering = arange(0, agent_subgraph);
		// }

	}

	add_agent_interaction_constraints(
		*problem.prog, problem.time_deltas_list, agent_interactions, graph.edge_to_min_tau_map);

	return problem;
}

void add_agent_interaction_constraints(
	drake::solvers::MathematicalProgram& prog,
	const std::vector<drake::solvers::VectorXDecisionVariable>& time_deltas_list,
	const std::vector<AgentInteraction>& agent_interactions,
	const std::map<std::pair<int, int>, double>& edge_to_min_tau_map) {

	for (const struct AgentInteraction& p : agent_interactions) {
		if (p.type == AgentInteraction::Type::LESS_THAN) {
			double min_tau = 0.0;
			if (edge_to_min_tau_map.contains(std::make_pair(p.node_u, p.node_v))) {
				min_tau = edge_to_min_tau_map.at(std::make_pair(p.node_u, p.node_v));
			}

			// taus
			Eigen::VectorX<Expression> taus_i = time_deltas_list[p.agent_i].head(p.agent_i_depth+1);
			Eigen::VectorX<Expression> taus_j = time_deltas_list[p.agent_j].head(p.agent_j_depth+1);
			prog.AddLinearConstraint(taus_i.sum() + min_tau <= taus_j.sum())
				.evaluator()->set_description("timing less than constraint");
		} else if (p.type == AgentInteraction::Type::EQUAL) {
			Eigen::VectorX<Expression> taus_i = time_deltas_list[p.agent_i].head(p.agent_i_depth+1);
			Eigen::VectorX<Expression> taus_j = time_deltas_list[p.agent_j].head(p.agent_j_depth+1);
			prog.AddLinearConstraint(taus_i.sum() == taus_j.sum())
				.evaluator()->set_description("timing equality constraint");
		} else {
			throw std::runtime_error("Not implemented");
		}
	}
}

GraphTimingProblem build_dense_graph_timing_problem(
	const std::vector<CubicConfigurationSpline>& splines,
	const std::vector<Eigen::MatrixXd>& agent_dense_wps,
	const std::vector<std::vector<int>>& agent_dense_node_ids,
	const std::vector<AgentInteraction>& agent_interactions,
	const std::map<std::pair<int, int>, double>& edge_to_min_tau_map,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	double time_cost,
	double time_cost2,
	double acceleration_cost,
	double energy_cost,
	double arclength_cost,
	double stability_cost,
	const std::vector<double>& max_vel,
	const std::vector<double>& max_acc,
	const std::vector<double>& max_jerk) {

	const int num_agents = static_cast<int>(splines.size());
	GraphTimingProblem problem(num_agents);
	// Unlike wps_list/vs_list/time_deltas_list (sized by GraphTimingProblem's
	// own constructor), agent_nodes_list isn't -- build_graph_timing_problem
	// grows it explicitly for the same reason before its own per-agent loop.
	problem.agent_nodes_list.resize(num_agents);

	const int ambient_dim = splines.at(0).ambient_dim();
	const int tangent_dim = splines.at(0).tangent_dim();

	for (int i = 0; i < num_agents; ++i) {
		const Eigen::MatrixXd& wps_i = agent_dense_wps.at(i);
		const int agent_spline_length = static_cast<int>(wps_i.rows());

		problem.agent_nodes_list[i] = agent_dense_node_ids.at(i);

		if (agent_spline_length > 0) {
			problem.wps_list[i] = wps_i;

			const Eigen::VectorXd x0_i = x0.segment(i * ambient_dim, ambient_dim);
			const Eigen::VectorXd v0_i = v0.segment(i * tangent_dim, tangent_dim);

			AgentSegmentVars vars = add_agent_timing_segments(
				*problem.prog, splines.at(i), wps_i, x0_i, v0_i, i,
				time_cost, time_cost2, acceleration_cost, energy_cost,
				arclength_cost, stability_cost, max_vel, max_acc, max_jerk);

			problem.time_deltas_list[i] = vars.time_deltas;
			problem.vs_list[i] = vars.vs;
		}
	}

	add_agent_interaction_constraints(
		*problem.prog, problem.time_deltas_list, agent_interactions, edge_to_min_tau_map);

	return problem;
}

/*
 * Graph Timing MPC
 */

GraphTimingMPC::GraphTimingMPC(const GraphOfConstraints& graph,
			       std::vector<CubicConfigurationSpline> splines,
			       double time_cost,
			       double time_cost2,
			       double acceleration_cost,
			       double energy_cost,
			       double arclength_cost,
			       std::vector<double> max_vel,
			       std::vector<double> max_acc,
			       std::vector<double> max_jerk,
			       double stability_cost)
	: _graph(&graph),
	  _splines(std::make_shared<std::vector<CubicConfigurationSpline>>(std::move(splines))),
	  _time_cost(time_cost),
	  _time_cost2(time_cost2),
	  _acceleration_cost(acceleration_cost),
	  _energy_cost(energy_cost),
	  _arclength_cost(arclength_cost),
	  _max_vel(std::move(max_vel)),
	  _max_acc(std::move(max_acc)),
	  _max_jerk(std::move(max_jerk)),
	  _stability_cost(stability_cost),
	  _vs_list(graph.num_agents),
	  _time_deltas_list(graph.num_agents) {

	// Each bound vector is either empty (unbounded for every block) or has
	// exactly one entry per block in the spline's spec (see _max_vel's own
	// doc comment in graph_timing_mpc.hpp) -- catch a caller passing the
	// wrong count here, loudly, rather than silently reading garbage/out-of-
	// bounds indices later in add_agent_timing_segments.
	{
		const size_t num_blocks = _splines->at(0).block_offsets_.size();
		auto check = [&](const std::vector<double>& v, const char* name) {
			if (!v.empty() && v.size() != num_blocks) {
				throw std::runtime_error(
					std::string(name) + " size (" + std::to_string(v.size())
					+ ") must match the spline's block count (" + std::to_string(num_blocks) + ")");
			}
		};
		check(_max_vel, "max_vel");
		check(_max_acc, "max_acc");
		check(_max_jerk, "max_jerk");
	}

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
	const Eigen::VectorXi& assignments,
	const Eigen::VectorXd& t_by_node) {

	_timer.Start();

	// Snapshotted before `_agent_nodes_list`/`_vs_list`/`_time_deltas_list`
	// get overwritten with this cycle's own values below, so the warm-start
	// block further down can still node-id-match this cycle's decision
	// variables against last cycle's converged solution.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	std::unique_ptr<GraphTimingProblem> problem;
	try {
		problem = std::make_unique<GraphTimingProblem>(
			build_graph_timing_problem(
				*_graph, *_splines, remaining_vertices, waypoints, assignments, x0, v0,
				_time_cost, _time_cost2, _acceleration_cost, _energy_cost, _arclength_cost,
				_max_vel, _max_acc, _max_jerk, _stability_cost, t_by_node));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in timing problem construction: " << e.what() << std::endl;
		return false;
	}

	// Store ordered waypoints used in problem
	_wps_list = problem->wps_list;

	// Store nodes on each spline
	_agent_nodes_list = problem->agent_nodes_list;

	// Warm start vs and taus per agent from last cycle's converged solution,
	// node-id-matched against this cycle's (possibly shorter, if a leading
	// node just completed) node list -- `build_graph_timing_problem` above
	// otherwise resets every decision variable to the same generic guess
	// (time_deltas=10.0, vs=1.0) every single cycle. NLopt is a local
	// solver: re-solving a barely-changed problem (x0/v0 advance by one
	// `position_velocity`-tracked timestep each cycle) from that same
	// generic guess, rather than from the previous cycle's own solution,
	// lets it land in a very different local optimum cycle to cycle --
	// diagnosed via `po_goc_mpc.experiments.basic_fmm_experiment`'s
	// spline-iterations plot, which showed the planned horizon jumping
	// from ~2s to 28s over three consecutive cycles before the solve
	// started failing outright a few cycles later.
	for (int i = 0; i < _graph->num_agents; ++i) {
		if (!prev_agent_spline_length_map.contains(i)) {
			continue;  // no previous solve for this agent yet (e.g. cycle 0)
		}

		const std::vector<int>& new_nodes = problem->agent_nodes_list[i];
		const std::vector<int>& old_nodes = prev_agent_nodes_list[i];

		for (int j_new = 0; j_new < static_cast<int>(new_nodes.size()); ++j_new) {
			const auto it = std::find(old_nodes.begin(), old_nodes.end(), new_nodes[j_new]);
			if (it == old_nodes.end()) {
				continue;  // this node wasn't part of last cycle's plan
			}
			const int j_old = static_cast<int>(std::distance(old_nodes.begin(), it));

			if (j_old < prev_time_deltas_list[i].size()) {
				problem->prog->SetInitialGuess(
					problem->time_deltas_list[i](j_new), prev_time_deltas_list[i](j_old));
			}
			if (j_old < prev_vs_list[i].rows() && j_new < problem->vs_list[i].rows()) {
				for (int k = 0; k < problem->vs_list[i].cols(); ++k) {
					problem->prog->SetInitialGuess(
						problem->vs_list[i](j_new, k), prev_vs_list[i](j_old, k));
				}
			}
		}
	}

	// for (int v : remaining_vertices) {
	// 	const int i = problem->subgraph->subgraph_id(v);
	// 	problem->prog->SetInitialGuess(problem->X.row(i), _waypoints.row(v));
	// }

	// Solve
	using drake::solvers::IpoptSolver;
	// using drake::solvers::NloptSolver;
	IpoptSolver solver;

	// Choose Nlopt algorithm
	// problem->prog->SetSolverOption(NloptSolver::id(), "algorithm", "LD_SLSQP");

	MathematicalProgramResult result;
	try {
		result = solver.Solve(*(problem->prog));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in timing solver: " << e.what() << std::endl;
		std::cout << "problem: " << std::endl;
		std::cout << problem->prog->to_string() << std::endl;
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
		std::cout << "problem: " << std::endl;
		std::cout << problem->prog->to_string() << std::endl;
		return false;
	}
}

bool GraphTimingMPC::solve_dense(
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const std::vector<Eigen::MatrixXd>& agent_dense_wps,
	const std::vector<std::vector<int>>& agent_dense_node_ids,
	const std::vector<AgentInteraction>& agent_interactions) {

	_timer.Start();

	// Same warm-start snapshot purpose as solve()'s.
	const std::vector<Eigen::MatrixXd> prev_vs_list = _vs_list;
	const std::vector<Eigen::VectorXd> prev_time_deltas_list = _time_deltas_list;
	const std::vector<std::vector<int>> prev_agent_nodes_list = _agent_nodes_list;
	const std::map<int, int> prev_agent_spline_length_map = _agent_spline_length_map;

	std::unique_ptr<GraphTimingProblem> problem;
	try {
		problem = std::make_unique<GraphTimingProblem>(
			build_dense_graph_timing_problem(
				*_splines, agent_dense_wps, agent_dense_node_ids, agent_interactions,
				_graph->edge_to_min_tau_map, x0, v0,
				_time_cost, _time_cost2, _acceleration_cost, _energy_cost,
				_arclength_cost, _stability_cost, _max_vel, _max_acc, _max_jerk));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in dense timing problem construction: " << e.what() << std::endl;
		return false;
	}

	_wps_list = problem->wps_list;
	_agent_nodes_list = problem->agent_nodes_list;

	// Warm start by matching real graph-node id where available -- same
	// pattern as solve()'s, except synthetic (-1) rows are skipped: unlike
	// real graph nodes, a synthetic traced interior point can legitimately
	// change point-for-point between cycles (the traced polyline itself
	// shifts as x0 moves), so matching one arbitrary -1 to another would
	// seed the solver from an unrelated point instead of just leaving it at
	// add_agent_timing_segments' own generic guess.
	for (int i = 0; i < _graph->num_agents; ++i) {
		if (!prev_agent_spline_length_map.contains(i)) {
			continue;
		}

		const std::vector<int>& new_ids = agent_dense_node_ids[i];
		const std::vector<int>& old_ids = prev_agent_nodes_list[i];

		for (int j_new = 0; j_new < static_cast<int>(new_ids.size()); ++j_new) {
			if (new_ids[j_new] < 0) {
				continue;
			}
			const auto it = std::find(old_ids.begin(), old_ids.end(), new_ids[j_new]);
			if (it == old_ids.end()) {
				continue;
			}
			const int j_old = static_cast<int>(std::distance(old_ids.begin(), it));

			if (j_old < prev_time_deltas_list[i].size()) {
				problem->prog->SetInitialGuess(
					problem->time_deltas_list[i](j_new), prev_time_deltas_list[i](j_old));
			}
			if (j_old < prev_vs_list[i].rows() && j_new < problem->vs_list[i].rows()) {
				for (int k = 0; k < problem->vs_list[i].cols(); ++k) {
					problem->prog->SetInitialGuess(
						problem->vs_list[i](j_new, k), prev_vs_list[i](j_old, k));
				}
			}
		}
	}

	using drake::solvers::IpoptSolver;
	IpoptSolver solver;

	// TEMPORARY diagnostic only (investigating the IterationLimit-with-
	// zero-violations failure mode -- does Ipopt actually reach
	// Solve_Succeeded given more budget, or does it stay stuck regardless?
	// NOT touching acceptance criteria/tolerances, only how long it's
	// allowed to keep trying for a genuine convergence).
	problem->prog->SetSolverOption(IpoptSolver::id(), "max_iter", 20000);

	MathematicalProgramResult result;
	try {
		result = solver.Solve(*(problem->prog));
	} catch (const std::exception& e) {
		std::cout << "Caught exception in dense timing solver: " << e.what() << std::endl;
		std::cout << "problem: " << std::endl;
		std::cout << problem->prog->to_string() << std::endl;
		return false;
	}

	if (result.is_success()) {
		_last_solve_time = _timer.Tick();

		for (int i = 0; i < _graph->num_agents; ++i) {
			// Direct reassignment (not a write into a fixed-size buffer,
			// unlike solve()'s): solve()'s _vs_list/_time_deltas_list are
			// pre-sized to the graph's own num_nodes at construction time,
			// since a sparse (real-node-only) spline can never exceed that
			// -- a dense one routinely does (many synthetic rows per real
			// node), so writing rows into that buffer here would be a
			// straight out-of-bounds write.
			_vs_list[i] = result.GetSolution(problem->vs_list[i]);
			_time_deltas_list[i] = result.GetSolution(problem->time_deltas_list[i]);

			_agent_spline_length_map[i] = static_cast<int>(_time_deltas_list[i].size()) + 1;
		}
		return true;
	} else {
		PrintSolverReport(problem.get(), result, 1e-6);
		std::cout << "problem: " << std::endl;
		std::cout << problem->prog->to_string() << std::endl;
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
			// After a sparse (real-node-only) solve() this array holds one
			// real node per row, so checking just row 0 (the ORIGINAL
			// version of this loop) is already correct -- but after a
			// dense solve_dense() it holds several synthetic (-1) rows
			// ahead of each real one (see build_dense_graph_timing_problem/
			// GraphOfConstraints::get_agent_paths), and a real node several
			// rows deep can already be reachable within `delta` even though
			// row 0 isn't it (confirmed empirically while debugging why a
			// dense-solved episode's phases never advanced despite the
			// robot visibly having reached one). Walking forward,
			// accumulating tau, handles both cases identically (degenerates
			// to the original single-row check when every row is real).
			double cumulative_tau = 0.0;
			for (int j = 0; j < static_cast<int>(_agent_nodes_list[i].size()); ++j) {
				cumulative_tau += _time_deltas_list[i](j);
				if (delta < cumulative_tau - tau_cutoff) {
					// TODO: (changing initialization of solver)
					break; // haven't reached even this row yet -- neither has any later one
				}
				const int node_id = _agent_nodes_list[i][j];
				if (node_id >= 0) {
					passed_nodes.insert(node_id);
				}
				// else: a synthetic traced interior point was passed --
				// nothing further to do for it.
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


const std::vector<int> GraphTimingMPC::get_next_nodes() const {
	std::vector<int> result;
	for (int i = 0; i < _graph->num_agents; ++i) {
		const int spline_length_i = _agent_spline_length_map.at(i);

		if (spline_length_i > 1 && !_agent_nodes_list[i].empty()) {
			const int node = _agent_nodes_list[i].at(0);
			if (node >= 0) {
				result.push_back(node);
			}
			// else: first row is a synthetic traced interior point (only
			// possible after solve_dense()) -- no real "next node" to
			// report for this agent yet.
		}
	}
	return result;
}
