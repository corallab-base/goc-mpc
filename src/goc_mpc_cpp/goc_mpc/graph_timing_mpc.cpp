#include "graph_timing_mpc.hpp"

using Eigen::VectorX;
using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;


GraphOrderingProblem build_graph_ordering_problem(
	const Eigen::MatrixXd& wps,
	const Eigen::MatrixXi& graph,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0) {

	using namespace drake::solvers;

	const ssize_t K = wps.rows();
	const ssize_t d = wps.cols();

	// Create program
	GraphOrderingProblem problem;

	// Create decision variables
	// p(i,j) = 1 iff waypoint i is at position j
	MatrixXDecisionVariable p = problem.prog->NewBinaryVariables(K, K, "p");
	problem.p = p;

	// Doubly-Stochastic
	// Each waypoint exactly once: sum_k p(i,k) = 1
	for (int i = 0; i < K; ++i) {
		VectorX<Variable> row = p.row(i);
		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, row);
	}
	// Each position filled: sum_i p(i,k) = 1
	for (int j = 0; j < K; ++j) {
		VectorX<Variable> col = p.col(j);
		problem.prog->AddLinearEqualityConstraint(Eigen::RowVectorXd::Ones(K), 1.0, col);
	}

	// Precedence: i must appear before j  ==>  sum_k k*P(i,k) + 1 <= sum_k k*P(j,k)
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j) {
			if (graph(i, j) == 1) {
				VectorX<Variable> lhs_vars(2*K);
				Eigen::RowVectorXd lhs_coeffs = Eigen::RowVectorXd::Zero(2*K);
				// pack as [P(i, 0 through n-1), P(j, 0 through n-1)]
				for (int k = 0; k < K; ++k) {
					lhs_vars(k)     = p(i, k);
					lhs_vars(K + k) = p(j, k);
					lhs_coeffs(k)       = k;       // +k * P(i,k)
					lhs_coeffs(K + k)   = -k;      // -k * P(j,k)
				}
				/* lb occurs when P(i,0) = 1 and P(j, K-1) = 1. Therefore k*P(i, k) - k*P(j, k) = -(K - 1). */
				problem.prog->AddLinearConstraint(lhs_coeffs, -(K-1), -1, lhs_vars); // enforces pos(i)+1 <= pos(j)
			}
		}
	}

	// Helpers: squared distances from x0 to xi for all i
	// squared distances between xi and xj for all i,j
	Eigen::VectorXd s2(K);
	Eigen::MatrixXd d2(K, K);
	for (int i = 0; i < K; ++i) {
		s2(i) = (wps.row(i).transpose() - x0).squaredNorm();
		for (int j = 0; j < K; ++j) {
			d2(i,j) = (wps.row(i) - wps.row(j)).squaredNorm();
		}
	}

	// Objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_{i,j} d2(i,j)*p(i,k-1)*p(j,k)

	// The second term is quadratic in binaries; to keep MILP, use a *linear* proxy:
	// e.g., sum_{k=1}^{n-1} sum_{j} (min_i D2(i,j)) * p(j,k)  (lower-bound-ish) or just sum over k of degrees.
	// A better linear surrogate: use a fixed "nearest predecessor" cost C(j,k) = min_i D2(i,j).
	Eigen::VectorXd c_min(K);
	for (int j = 0; j < K; ++j) {
		double m = std::numeric_limits<double>::infinity();
		for (int i = 0; i < K; ++i) if (i != j) m = std::min(m, d2(i,j));
		c_min(j) = std::isfinite(m) ? m : 0.0;
	}
	drake::solvers::LinearCost* obj = nullptr;
	// Build linear objective: sum_i s2(i)*p(i,0) + sum_{k=1}^{n-1} sum_j c_min(j)*p(j,k)
	Eigen::VectorXd coeffs(K*K);
	coeffs.setZero();
	int idx = 0;
	for (int i = 0; i < K; ++i) {
		for (int j = 0; j < K; ++j, ++idx) {
			double c = (j == 0) ? s2(i) : c_min(i);
			coeffs(idx) = c;
		}
	}
	drake::VectorX<Variable> allP(K*K);
	idx = 0;
	for (int i = 0; i < K; ++i) for (int j = 0; j < K; ++j) allP[idx++] = p(i, j);
	problem.prog->AddLinearCost(coeffs, 0.0, allP);

	return std::move(problem);
}

/*
 * Timing MPC
 */

GraphTimingMPC::GraphTimingMPC(unsigned int num_agents,
			       unsigned int dim,
			       double time_cost,
			       double ctrl_cost)
	: _num_agents(num_agents),
	  _dim(dim),
	  _time_cost(time_cost),
	  _ctrl_cost(ctrl_cost) {
	return;
}

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

std::optional<std::pair<Eigen::MatrixXd, Eigen::VectorXd>> GraphTimingMPC::solve(
	const Eigen::MatrixXi& graph,
	const Eigen::VectorXd& x0,
	const Eigen::VectorXd& v0,
	const Eigen::MatrixXd& waypoints,
	const Eigen::VectorXi& assignments) {

	/* at this point, the waypoint optimizer has decided on assignments of
	 * agents to assignable tasks and their positions for satisfying those
	 * tasks. the order may not be completely determined for an individual
	 * agent, so we also solve for an optimal ordering if necessary. */

	// for (int i = 0; i < _num_agents; ++i) {

	// }

	// const bool totally_ordered = true;
	// if (!totally_ordered) {

	// } else {

	// }

	/* after the individual agents' positions are totally ordered we
	 * construct each agent's spline. */

	// std::array<Eigen::VectorXi, _num_agents> ordering;
	// std::array<Eigen::MatrixXd, _num_agents> ordered_agent_wps;

	/* we add constraints on the timings of these splines to reflect any
	 * cross-agent constraints in the original graph. */


	/* we solve for the velocities and timings using epigraph on the final
	 * timing */




	// for (int ag = 0; ag < _num_agents; ++ag) {
	// 	for
	// }

	const unsigned int num_nodes = graph.rows();

	struct GraphOrderingProblem ordering_problem = build_graph_ordering_problem(
		waypoints, graph, x0, v0);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto ordering_result = solver.Solve(*ordering_problem.prog);
	// auto ordering_result = drake::solvers::Solve(*ordering_problem.prog);

	Eigen::VectorXi ordering;
	if (ordering_result.is_success()) {
		for (int i = 0; i < num_nodes; ++i) {
			for (int j = 0; j < num_nodes; ++j) {
				const double val = ordering_result.GetSolution(ordering_problem.p(i, j));
				if (val > 0.5) {
					ordering(i) = j;
					break;
				}
			}
		}
	} else {
		std::cerr << "Ordering Optimization failed." << std::endl;
		return std::nullopt;
	}

	/* TODO: Change according to phase */
	/* initialize output ordering array (n,) */
	Eigen::MatrixXd ordered_waypoints({num_nodes, _dim});
	for (size_t i = 0; i < num_nodes; ++i) {
		const auto rank = ordering(i);
		for (size_t j = 0; j < _dim; ++j) {
			ordered_waypoints(i, j) = waypoints(rank, j);
		}
	}

	struct TimingProblem problem = build_timing_problem(
		ordered_waypoints,
		x0, v0,
		_time_cost, _ctrl_cost,
		true, false,
		-1.0, -1.0, -1.0,
		false, -1.0);

	// Solve
	auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		std::cout << "Success" << std::endl;

		const auto v = result.GetSolution(problem.v);
		const auto taus = result.GetSolution(problem.time_deltas);

		return std::make_pair(v, taus);
	} else {
		std::cerr << "Optimization failed." << std::endl;
		return std::nullopt;
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
// 	const ssize_t phase = 0; // static_cast<ssize_t>(this->phase);

// 	// Done: return final velocity (usually zero)
// 	if (done()) {
// 		return remainder_slice_2d(_vels, _num_nodes - 1);
// 	}

// 	ssize_t num_rows = _num_nodes - phase;  // including final appended zero row
// 	ssize_t num_cols = _dim;

// 	// Allocate final velocity array (including appended zero row)
// 	py::array_t<double> result({num_rows, num_cols});
// 	auto result_mut = result.mutable_unchecked<2>();

// 	// Directly copy from this->vels
// 	auto src = remainder_slice_2d(_vels, phase).unchecked<2>();
// 	for (ssize_t i = 0; i < num_rows - 1; ++i) {
// 		for (ssize_t j = 0; j < _dim; ++j) {
// 			result_mut(i, j) = src(i, j);
// 		}
// 	}

// 	// Append zero vector at the end
// 	for (ssize_t j = 0; j < _dim; ++j) {
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
// 	ssize_t N = time_deltas_.shape(0);

// 	while (this->phase > phaseTo) {
// 		if (this->phase < static_cast<unsigned int>(N)) {
// 			time_deltas_(this->phase) = std::max(1.0, time_deltas_(this->phase));
// 		}
// 		this->phase--;
// 	}

// 	time_deltas_(this->phase) = 1.0;
// }


// void GraphTimingMPC::fill_cubic_spline(CubicSpline& S, const py::array_t<double>& x0, const py::array_t<double>& v0) const {

// 	py::array_t<double> pts = this->get_waypoints();
// 	py::array_t<double> times = this->get_times();
// 	py::array_t<double> vels = this->get_vels();
// 	pts = prepend_row_2d(pts, x0);
// 	vels = prepend_row_2d(vels, v0);
// 	times = prepend_1d(times, 0.0);

// 	if (times.size() > 1) {
// 		S.set(pts, vels, times);
// 	}
// }
