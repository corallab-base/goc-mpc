#include "graph_timing_mpc.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


/*
 * Timing MPC
 */

GraphTimingMPC::GraphTimingMPC(const py::array_t<double>& waypoints,
			       const py::array_t<double>& graph,
			       double time_cost,
			       double ctrl_cost)
  : _waypoints(waypoints),
    _graph(graph),
    _time_cost(time_cost),
    _ctrl_cost(ctrl_cost) {

	_n = waypoints.shape(0);
	_d = waypoints.shape(1);

	/* initialize output ordering array (n,) */
	_ordering = py::array_t<unsigned int>(_n);
	auto ordering_mut = _ordering.mutable_unchecked<1>();
	for (ssize_t i = 0; i < _n; ++i) {
		ordering_mut(i) = i;
	}

	/* initialize output times array (n,) */
	_time_deltas = py::array_t<double>(_n);
	auto times_mut = _time_deltas.mutable_unchecked<1>();
	for (ssize_t i = 0; i < _n; ++i) {
		times_mut(i) = i * 10.0;
	}

	/* initialize output vels array (n-1, d) */
	_vels = py::array_t<double>({_n - 1, _d});
	auto vels_mut = _vels.mutable_unchecked<2>();
	for (ssize_t i = 0; i < _n - 1; ++i) {
		for (ssize_t f = 0; f < _d; ++f) {
			vels_mut(i, f) = 0.0;
		}
	}

	/* initialize empty back tracking table */
	/* TODO: expose method for changing it */
	// this->back_tracking_table = py::array_t<unsigned int>(0);

	// opt .set_verbose(0)
	// .set_maxStep(1e0)
	// .set_stopTolerance(1e-4)
	// .set_damping(1e-2);
}

py::array_t<double> GraphTimingMPC::_remaining_waypoints() const {
	/* TODO */
	return this->_waypoints;
}

py::array_t<double> GraphTimingMPC::_remaining_graph() const {
	/* TODO */
	return this->_graph;
}

bool GraphTimingMPC::done() const {
	return false;
}

int GraphTimingMPC::solve(const py::array_t<double>& x0,
			  const py::array_t<double>& v0,
			  int verbose) {

	const py::array_t<double> remaining_waypoints = this->_remaining_waypoints();
	const py::array_t<double> remaining_graph = this->_remaining_graph();

	struct GraphOrderingProblem ordering_problem = build_graph_ordering_problem(
		remaining_waypoints, remaining_graph, x0, v0);

	// Solve
	drake::solvers::MosekSolver solver;
	const auto ordering_result = solver.Solve(*ordering_problem.prog);
	// auto ordering_result = drake::solvers::Solve(*ordering_problem.prog);

	if (ordering_result.is_success()) {
		std::cout << "Ordering Success" << std::endl;

		auto ordering_mut = _ordering.mutable_unchecked<1>();
		for (int i = 0; i < _n; ++i) {
			for (int j = 0; j < _n; ++j) {
				const double val = ordering_result.GetSolution(ordering_problem.p(i, j));
				if (val > 0.5) {
					ordering_mut(i) = j;
					break;
				}
			}
		}
	} else {
		std::cerr << "Ordering Optimization failed." << std::endl;
		return -1;
	}

	/* initialize output ordering array (n,) */
	py::array_t<double> remaining_ordered_waypoints({_n, _d});
	auto new_wps_mut = remaining_ordered_waypoints.mutable_unchecked<2>();
	auto wps_u = _waypoints.unchecked<2>();
	auto ordering_u = _ordering.unchecked<1>();
	for (size_t i = 0; i < _n; ++i) {
		const auto rank = ordering_u(i);
		for (size_t j = 0; j < _d; ++j) {
			new_wps_mut(i, j) = wps_u(rank, j);
		}
	}

	struct TimingProblem problem = build_timing_problem(
		remaining_ordered_waypoints,
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

		// Write velocities to output array
		auto vels_mut = _vels.mutable_unchecked<2>();
		// subtract 1 because last vec isn't relevant
		for (size_t i = 0, j = 0; i < _n - 1; ++i, ++j) {
			for (size_t f = 0; f < _d; ++f) {
				vels_mut(i, f) = v(j, f);
			}
		}

		auto time_deltas_mut = _time_deltas.mutable_unchecked<1>();
		for (size_t i = 0, j = 0; i < _n; ++i, ++j) {
			time_deltas_mut(i) = taus(j);
		}
	} else {
		std::cerr << "Optimization failed." << std::endl;
		return -1;
	}

	return 0;
}

// Safe indexing and accessors
py::array_t<double> GraphTimingMPC::get_waypoints() const {
	if (done()) {
		return remainder_slice_2d(_waypoints, 0);
	} else {
		return remainder_slice_2d(_waypoints, 0);
	}
}

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

py::array_t<unsigned int> GraphTimingMPC::get_ordering() const {
	// if (done()) {
	// 	// Return single time step
	// 	py::array_t<double> ret(1);
	// 	auto ret_mut = ret.mutable_unchecked<1>();
	// 	ret_mut(0) = 0.1;
	// 	return ret;
	// }
	// auto remaining_time_deltas = remainder_slice_1d(this->time_deltas, this->phase);
	// return integral(remaining_time_deltas);
	return _ordering;
}

py::array_t<double> GraphTimingMPC::get_times() const {
	if (done()) {
		// Return single time step
		py::array_t<double> ret(1);
		auto ret_mut = ret.mutable_unchecked<1>();
		ret_mut(0) = 0.1;
		return ret;
	}
	auto remaining_time_deltas = remainder_slice_1d(_time_deltas, 0);
	return integral(remaining_time_deltas);
}


py::array_t<double> GraphTimingMPC::get_vels() const {
	const ssize_t phase = 0; // static_cast<ssize_t>(this->phase);

	// Done: return final velocity (usually zero)
	if (done()) {
		return remainder_slice_2d(_vels, _n - 1);
	}

	ssize_t num_rows = _n - phase;  // including final appended zero row
	ssize_t num_cols = _d;

	// Allocate final velocity array (including appended zero row)
	py::array_t<double> result({num_rows, num_cols});
	auto result_mut = result.mutable_unchecked<2>();

	// Directly copy from this->vels
	auto src = remainder_slice_2d(_vels, phase).unchecked<2>();
	for (ssize_t i = 0; i < num_rows - 1; ++i) {
		for (ssize_t j = 0; j < _d; ++j) {
			result_mut(i, j) = src(i, j);
		}
	}

	// Append zero vector at the end
	for (ssize_t j = 0; j < _d; ++j) {
		result_mut(num_rows - 1, j) = 0.0;
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


void GraphTimingMPC::fill_cubic_spline(CubicSpline& S, const py::array_t<double>& x0, const py::array_t<double>& v0) const {

	py::array_t<double> pts = this->get_waypoints();
	py::array_t<double> times = this->get_times();
	py::array_t<double> vels = this->get_vels();
	pts = prepend_row_2d(pts, x0);
	vels = prepend_row_2d(vels, v0);
	times = prepend_1d(times, 0.0);

	if (times.size() > 1) {
		S.set(pts, vels, times);
	}
}
