#include "timing_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


/*
 * Timing MPC
 */


TimingMPC::TimingMPC(const Eigen::MatrixXd& _waypoints, double _time_cost, double _ctrl_cost)
  : waypoints(_waypoints),
    time_cost(_time_cost),
    ctrl_cost(_ctrl_cost) {

	ssize_t n = _waypoints.shape(0);
	ssize_t d = _waypoints.shape(1);

	this->_n = n;
	this->_d = d;

	/* initialize output times array (n,) */
	this->time_deltas = py::array_t<double>(n);
	auto time_deltas_mut = this->time_deltas.mutable_unchecked<1>();
	for (ssize_t i = 0; i < n; ++i) {
		time_deltas_mut(i) = 10.0;
	}

	/* initialize output vels array (n-1, d) */
	this->vels = py::array_t<double>({n - 1, d});
	auto vels_mut = this->vels.mutable_unchecked<2>();
	for (ssize_t i = 0; i < n - 1; ++i) {
		for (ssize_t f = 0; f < d; ++f) {
			vels_mut(i, f) = 0.0;
		}
	}

	/* initialize empty back tracking table */
	/* TODO: expose method for changing it */
	this->back_tracking_table = py::array_t<unsigned int>(0);

	/* initialize empty tangents array */
	/* TODO: expose method for changing it */
	this->tangents = py::array_t<double>(0);

	// opt .set_verbose(0)
	// .set_maxStep(1e0)
	// .set_stopTolerance(1e-4)
	// .set_damping(1e-2);
}


int TimingMPC::solve(const Eigen::VectorXd& x0,
		     const Eigen::VectorXd& v0,
		     int verbose) {

	struct TimingProblem problem = build_timing_problem(
		remainder_slice_2d(this->waypoints, this->phase),
		x0, v0,
		this->time_cost, this->ctrl_cost,
		true, false,
		-1.0, -1.0, -1.0,
		false, -1.0);

	// Solve
	auto result = drake::solvers::Solve(*problem.prog);

	if (result.is_success()) {
		// Eigen::Vector2d x_val = result.GetSolution(x);
		// std::cout << "Optimal solution:\n" << x_val << std::endl;
		// std::cout << "Optimal cost: " << result.get_optimal_cost() << std::endl;
		std::cout << "Success" << std::endl;

		const auto v = result.GetSolution(problem.v);
		const auto taus = result.GetSolution(problem.time_deltas);

		// Write velocities to output array
		auto vels_mut = this->vels.mutable_unchecked<2>();
		// subtract 1 because last vec isn't relevant
		for (size_t i = this->phase, j = 0; i < this->_n - 1; ++i, ++j) {
			for (size_t f = 0; f < this->_d; ++f) {
				vels_mut(i, f) = v(j, f);
			}
		}

		auto time_deltas_mut = this->time_deltas.mutable_unchecked<1>();
		for (size_t i = this->phase, j = 0; i < this->_n; ++i, ++j) {
			printf("Setting time_deltas(%d) = %f\n", i, taus(j));
			time_deltas_mut(i) = taus(j);
		}

	} else {
		std::cerr << "Optimization failed." << std::endl;

		std::cout << "solver: " << result.get_solver_id().name() << "\n";
		std::cout << "solution_result: " << static_cast<int>(result.get_solution_result()) << "\n";
// If IPOPT:
		if (result.get_solver_id().name() == "Ipopt") {
			const auto& d = result.get_solver_details<drake::solvers::IpoptSolver>();
			std::cout << "ipopt_status: " << d.status << "  ipopt_message: " << d.ConvertStatusToString() << "\n";
		}
// What constraints were infeasible (if any):
		// for (const auto& info : result.GetInfeasibleConstraints(*prog)) {
		// 	std::cout << "Infeasible: " << info.description << "\n";
		// }
	}

	return 0;
	


// 	tau({phase, -1}) = nlp.tau;
// 	vels({phase, -1}) = nlp.v;
// 	warmstart_dual = ret->dual;

// 	if(verbose>0) {
// 		LOG(0) <<"phase: " <<phase <<" tau: " <<tau;
// 	}
// 	return ret;
}


// Safe indexing and accessors
py::array_t<double> TimingMPC::get_waypoints() const {
	if (this->done()) {
		return remainder_slice_2d(this->waypoints, this->n_phases() - 1);
	} else {
		return remainder_slice_2d(this->waypoints, this->phase);
	}
}

py::array_t<double> TimingMPC::get_time_deltas() const {
	if (done()) {
		// Return single time step
		py::array_t<double> ret(1);
		auto ret_mut = ret.mutable_unchecked<1>();
		ret_mut(0) = 0.1;
		return ret;
	}
	auto remaining_time_deltas = remainder_slice_1d(this->time_deltas, this->phase);
	return remaining_time_deltas;
}

py::array_t<double> TimingMPC::get_times() const {
	if (done()) {
		// Return single time step
		py::array_t<double> ret(1);
		auto ret_mut = ret.mutable_unchecked<1>();
		ret_mut(0) = 0.1;
		return ret;
	}
	auto remaining_time_deltas = remainder_slice_1d(this->time_deltas, this->phase);
	return integral(remaining_time_deltas);
}


py::array_t<double> TimingMPC::get_vels() const {
	const ssize_t phase_ = static_cast<ssize_t>(this->phase);
	const ssize_t N = this->waypoints.shape(0);
	const ssize_t D = this->waypoints.shape(1);

	// Done: return final velocity (usually zero)
	if (done()) {
		return remainder_slice_2d(this->vels, N - 1);
	}

	ssize_t num_rows = N - phase_;  // including final appended zero row

	// Allocate final velocity array (including appended zero row)
	py::array_t<double> result({num_rows, D});
	auto result_ = result.mutable_unchecked<2>();

	if (this->tangents.size() > 0) {
		// Get slices
		auto speeds = remainder_slice_1d(this->vels, this->phase).unchecked<1>();
		auto tangents = remainder_slice_2d(this->tangents, this->phase).unchecked<2>();

		for (ssize_t i = 0; i < num_rows - 1; ++i) {
			double s = speeds(i);
			for (ssize_t j = 0; j < D; ++j) {
				result_(i, j) = s * tangents(i, j);
			}
		}
	} else {
		// Directly copy from this->vels
		auto src = remainder_slice_2d(this->vels, this->phase).unchecked<2>();
		for (ssize_t i = 0; i < num_rows - 1; ++i) {
			for (ssize_t j = 0; j < D; ++j) {
				result_(i, j) = src(i, j);
			}
		}
	}

	// Append zero vector at the end
	for (ssize_t j = 0; j < D; ++j) {
		result_(num_rows - 1, j) = 0.0;
	}

	return result;
}


// bool TimingMPC::set_progressed_time(double gap, double tauCutoff) {
// 	if(gap < tau(phase)) { //time still within phase
// 		tau(phase) -= gap; //change initialization of timeOpt
// 		return false;
// 	}

// 	//time beyond current phase
// 	if(phase+1 < nPhases()) { //if there exists another phase
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

void TimingMPC::update_backtrack() {
	if (this->phase == 0) {
		throw std::runtime_error("Cannot backtrack from phase 0.");
	}

	/* by default, go back one */
	unsigned int phaseTo = this->phase - 1;

	/* Check if back_tracking_table is initialized and non-empty */
	/* if so, use that to determine where to go */
	if (this->back_tracking_table.size() > 0) {
		if (this->phase >= this->back_tracking_table.size()) {
			throw std::runtime_error("Phase index out of bounds in back_tracking_table.");
		}
		auto bt = this->back_tracking_table.unchecked<1>();
		phaseTo = bt(this->phase);
	}

	this->update_set_phase(phaseTo);
}

void TimingMPC::update_set_phase(unsigned int phaseTo) {
	std::cout << "[TimingMPC] Backtracking from phase " << this->phase
		  << " to " << phaseTo << " times: " << std::endl;

	if (phaseTo > this->phase) {
		throw std::runtime_error("Cannot advance phase using update_set_phase — only backward steps allowed.");
	}

	auto time_deltas_ = this->time_deltas.mutable_unchecked<1>();
	ssize_t N = time_deltas_.shape(0);

	while (this->phase > phaseTo) {
		if (this->phase < static_cast<unsigned int>(N)) {
			time_deltas_(this->phase) = std::max(1.0, time_deltas_(this->phase));
		}
		this->phase--;
	}

	time_deltas_(this->phase) = 1.0;
}

void TimingMPC::fill_cubic_spline(CubicSpline& S, const py::array_t<double>& x0, const py::array_t<double>& v0) const {

	py::array_t<double> _pts = this->get_waypoints();
	py::array_t<double> _times = this->get_times();
	py::array_t<double> _vels = this->get_vels();
	_pts = prepend_row_2d(_pts, x0);
	_vels = prepend_row_2d(_vels, v0);
	_times = prepend_1d(_times, 0.0);

	if (_times.size() > 1) {
		S.set(_pts, _vels, _times);
	}
}
