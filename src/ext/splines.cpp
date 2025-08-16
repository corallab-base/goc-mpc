/**
 * @file    Example C++ extension for Python using Pybind11 that adds two
 *          integers.
 */

#include "splines.hpp"

using namespace pybind11::literals;
namespace py = pybind11;


/*
 * CubicPiece
 */

void CubicPiece::set(const Eigen::Ref<const Eigen::VectorXd>& x0,
                     const Eigen::Ref<const Eigen::VectorXd>& v0,
                     const Eigen::Ref<const Eigen::VectorXd>& x1,
                     const Eigen::Ref<const Eigen::VectorXd>& v1,
                     double tau)
{
	if (tau <= 0.0) {
		throw std::invalid_argument("CubicPiece::set(): tau must be > 0");
	}
	const int d = static_cast<int>(x0.size());
	if (v0.size() != d || x1.size() != d || v1.size() != d) {
		throw std::invalid_argument("CubicPiece::set(): dimension mismatch");
	}

	// Allocate / resize coefficient vectors
	_a.resize(d);
	_b.resize(d);
	_c.resize(d);
	_d.resize(d);

	// Coefficients from cubic Hermite conditions
	_d = x0;          // d
	_c = v0;          // c

	const Eigen::VectorXd dx  = x1 - x0;
	const Eigen::VectorXd dv0 = v0;
	const Eigen::VectorXd dv1 = v1;

	const double tau2 = tau * tau;
	const double tau3 = tau2 * tau;

	// a = (-2*(x1-x0) + tau*(v0+v1)) / tau^3
	_a.noalias() = (-2.0 * dx + tau * (dv0 + dv1)) / tau3;

	// b = (3*(x1-x0) - tau*(2*v0 + v1)) / tau^2
	_b.noalias() = (3.0 * dx - tau * (2.0 * dv0 + dv1)) / tau2;
}

void CubicPiece::eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
                           std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
                           std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
                           double t) const
{
	const int d = dim();
	auto chk = [d](const std::optional<Eigen::Ref<Eigen::VectorXd>>& v, const char* name){
		if (v && v->size() != d) {
			throw std::invalid_argument(std::string("CubicPiece::eval_into(): ") + name + " has wrong size");
		}
	};
	chk(x, "x");
	chk(xDot, "xDot");
	chk(xDDot, "xDDot");

	const double t2 = t * t;
	const double t3 = t2 * t;

	if (x) {
		// x(t) = d + t c + t^2 b + t^3 a
		x->noalias() = _d + t * _c + t2 * _b + t3 * _a;
	}
	if (xDot) {
		// x'(t) = c + 2 t b + 3 t^2 a
		xDot->noalias() = _c + (2.0 * t) * _b + (3.0 * t2) * _a;
	}
	if (xDDot) {
		// x''(t) = 2 b + 6 t a
		xDDot->noalias() = 2.0 * _b + (6.0 * t) * _a;
	}
}

Eigen::VectorXd CubicPiece::eval(double t, unsigned int diff) const
{
	const int d = dim();
	Eigen::VectorXd out(d);

	const double t2 = t * t;

	switch (diff) {
	case 0:
		// position
		out.noalias() = _d + t * _c + (t * t) * _b + (t * t2) * _a;
		break;
	case 1:
		// velocity
		out.noalias() = _c + (2.0 * t) * _b + (3.0 * t2) * _a;
		break;
	case 2:
		// acceleration
		out.noalias() = 2.0 * _b + (6.0 * t) * _a;
		break;
	case 3:
		// jerk (third derivative): constant 6 a
		out.noalias() = 6.0 * _a;
		break;
	default:
		// higher derivatives are zero
		out.setZero();
		break;
	}
	return out;
}

/*
 * CubicSpline
 */


void CubicSpline::clear() {
	_pieces.clear();
	_times.resize(0);
	_dim = -1;
}


// ---------- Set from absolute times ----------
// pts:  (N, dim), vels: (N, dim), times: (N,) strictly increasing
void CubicSpline::set(const Eigen::Ref<const Eigen::MatrixXd>& pts,
		      const Eigen::Ref<const Eigen::MatrixXd>& vels,
		      const Eigen::Ref<const Eigen::VectorXd>& times)
{
	const int N   = static_cast<int>(times.size());
	const int Np  = static_cast<int>(pts.rows());
	const int Nv  = static_cast<int>(vels.rows());
	const int dim = static_cast<int>(pts.cols());

	if (N < 2) throw std::invalid_argument("CubicSpline::set(): need at least 2 knots");
	if (Np != N || Nv != N) throw std::invalid_argument("CubicSpline::set(): pts/vels rows must match times size");
	if (vels.cols() != dim) throw std::invalid_argument("CubicSpline::set(): pts and vels must have same number of columns");

	// Check strictly increasing times
	for (int i = 1; i < N; ++i) {
		if (!(times(i) > times(i-1))) {
			throw std::invalid_argument("CubicSpline::set(): times must be strictly increasing");
		}
	}

	// Build into temporaries for strong exception safety
	Eigen::VectorXd times_new = times;      // copy
	std::vector<CubicPiece> pieces_new;
	pieces_new.resize(N - 1);

	// Set each piece
	for (int k = 0; k < N - 1; ++k) {
		const double tau = times(k + 1) - times(k);
		pieces_new[k].set(pts.row(k).transpose(),
				  vels.row(k).transpose(),
				  pts.row(k + 1).transpose(),
				  vels.row(k + 1).transpose(),
				  tau);
	}

	// Commit
	_times.swap(times_new);
	_pieces.swap(pieces_new);
	_dim = dim;
}

// ---------- Append using durations relative to end ----------
// pts: (K_new, dim), vels: (K_new, dim), times: (K_new,)
// times is strictly increasing; interpreted as offsets from the current end time (times(0) > 0)
void CubicSpline::append(const Eigen::Ref<const Eigen::MatrixXd>& pts,
			 const Eigen::Ref<const Eigen::MatrixXd>& vels,
			 const Eigen::Ref<const Eigen::VectorXd>& times)
{
	if (!initialized()) {
		throw std::invalid_argument("CubicSpline::append(): spline must be initialized first");
	}
	const int K_new = static_cast<int>(times.size());
	if (K_new <= 0) {
		throw std::invalid_argument("CubicSpline::append(): times must have at least one element");
	}
	if (pts.rows() != K_new || vels.rows() != K_new) {
		throw std::invalid_argument("CubicSpline::append(): pts/vels rows must match times size");
	}
	if (_dim < 0) _dim = static_cast<int>(pts.cols());
	if (pts.cols() != _dim || vels.cols() != _dim) {
		throw std::invalid_argument("CubicSpline::append(): pts/vels cols must match spline dimension");
	}
	if (times(0) <= 0.0) {
		throw std::invalid_argument("CubicSpline::append(): times(0) must be > 0 (durations from end)");
	}
	for (int i = 1; i < K_new; ++i) {
		if (!(times(i) > times(i-1))) {
			throw std::invalid_argument("CubicSpline::append(): times must be strictly increasing");
		}
	}

	// Build temporaries
	const int N_old = static_cast<int>(_times.size());
	const double t_end  = _times(N_old - 1);
	const double tau_last = _times(N_old - 1) - _times(N_old - 2);

	Eigen::VectorXd times_new(N_old + K_new);
	times_new.head(N_old) = _times;
	for (int i = 0; i < K_new; ++i) times_new(N_old + i) = t_end + times(i);

	std::vector<CubicPiece> pieces_new = _pieces;
	pieces_new.resize(_pieces.size() + K_new);

	// Evaluate end state of the last piece at its end (relative time = tau_last)
	Eigen::VectorXd x(_dim), xDot(_dim);
	{
		std::optional<Eigen::Ref<Eigen::VectorXd>> X(x), Xd(xDot);
		std::optional<Eigen::Ref<Eigen::VectorXd>> Xdd = std::nullopt;
		_pieces.back().eval_into(std::move(X), std::move(Xd), std::move(Xdd), tau_last);
	}

	// First appended segment connects from (x, xDot) to (pts[0], vels[0]) in time times(0)
	{
		pieces_new[_pieces.size()].set(
			x, xDot,
			pts.row(0).transpose(),
			vels.row(0).transpose(),
			times(0)
			);
	}

	// Remaining appended segments
	for (int k = 1; k < K_new; ++k) {
		const double tau = times(k) - times(k - 1);
		if (tau <= 0.0) throw std::invalid_argument("CubicSpline::append(): non-positive segment duration computed");
		pieces_new[_pieces.size() + k].set(
			pts.row(k - 1).transpose(),
			vels.row(k - 1).transpose(),
			pts.row(k).transpose(),
			vels.row(k).transpose(),
			tau
			);
	}

	// Commit
	_times.swap(times_new);
	_pieces.swap(pieces_new);
}

// ---------- Find piece index ----------
// Returns k in [0, num_pieces()-1] such that t ∈ [ _times[k], _times[k+1] )
// If t == end(), returns last piece index.
unsigned int CubicSpline::get_piece(double t) const
{
	const int N = static_cast<int>(_times.size());
	if (N < 2) throw std::runtime_error("CubicSpline::get_piece(): spline is empty");

	if (t <= _times(0)) return 0u;
	if (t >= _times(N - 1)) return static_cast<unsigned int>(N - 2);

	// lower_bound to find first time >= t
	auto first = _times.data();
	auto last  = _times.data() + N;
	const double* it = std::lower_bound(first, last, t);

	// it points to first element >= t; segment is k-1
	int idx = static_cast<int>(it - first);
	int k = std::max(0, idx - 1);
	if (k >= N - 1) k = N - 2;
	return static_cast<unsigned int>(k);
}

// ---------- Evaluate into optional outputs ----------
void CubicSpline::eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
			    std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
			    std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
			    double t) const
{
	const int N = static_cast<int>(_times.size());
	if (N < 2) throw std::invalid_argument("CubicSpline::eval_into(): spline is empty");

	// Size checks
	if (_dim < 0) throw std::runtime_error("CubicSpline::eval_into(): unknown dimension");
	auto chk = [this](const std::optional<Eigen::Ref<Eigen::VectorXd>>& v, const char* name){
		if (v && v->size() != _dim)
			throw std::invalid_argument(std::string("CubicSpline::eval_into(): ") + name + " has wrong size");
	};
	chk(x, "x");
	chk(xDot, "xDot");
	chk(xDDot, "xDDot");

	const double t0 = _times(0);
	const double tN = _times(N - 1);

	// Before start: evaluate first piece at 0; zero acceleration if requested
	if (t <= t0) {
		_pieces.front().eval_into(x, xDot, xDDot, 0.0);
		if (xDDot) xDDot->setZero();   // boundary behavior
		return;
	}

	// After end: evaluate last piece at its end; zero acceleration if requested
	if (t >= tN) {
		const double tau_last = _times(N - 1) - _times(N - 2);
		_pieces.back().eval_into(x, xDot, xDDot, tau_last);
		if (xDDot) xDDot->setZero();
		return;
	}

	// Main case
	const unsigned int k = get_piece(t);
	const double t_rel = t - _times(static_cast<int>(k));
	_pieces[k].eval_into(x, xDot, xDDot, t_rel);
}

// ---------- Single-output evaluation ----------
Eigen::VectorXd CubicSpline::eval(double t, unsigned int diff) const
{
	if (_dim < 0 || _times.size() < 2) {
		throw std::invalid_argument("CubicSpline::eval(): spline is empty");
	}
	Eigen::VectorXd out(_dim);

	if (diff <= 2) {
		std::optional<Eigen::Ref<Eigen::VectorXd>> X, Xd, Xdd;
		if      (diff == 0) X.emplace(out);
		else if (diff == 1) Xd.emplace(out);
		else                Xdd.emplace(out);
		eval_into(std::move(X), std::move(Xd), std::move(Xdd), t);
		return out;
	}

	// diff > 2 → delegate to piece and return (uses piece.eval semantics)
	const unsigned int k = get_piece(t);
	const double t_rel = std::clamp(t, _times(0), _times(_times.size() - 1)) - _times(static_cast<int>(k));
	return _pieces[k].eval(t_rel, diff);
}

// ---------- Evaluate many times ----------
Eigen::MatrixXd CubicSpline::eval_multiple(const Eigen::Ref<const Eigen::VectorXd>& T,
					   unsigned int diff) const
{
	if (_dim < 0 || _times.size() < 2) {
		throw std::invalid_argument("CubicSpline::eval_multiple(): spline is empty");
	}
	const int M = static_cast<int>(T.size());
	Eigen::MatrixXd out(M, _dim);

	if (M == 0) return out;

	if (diff <= 2) {
		Eigen::VectorXd tmp(_dim);
		for (int i = 0; i < M; ++i) {
			std::optional<Eigen::Ref<Eigen::VectorXd>> X, Xd, Xdd;
			if      (diff == 0) X.emplace(tmp);
			else if (diff == 1) Xd.emplace(tmp);
			else                Xdd.emplace(tmp);
			eval_into(std::move(X), std::move(Xd), std::move(Xdd), T(i));
			out.row(i) = tmp.transpose();
		}
	} else {
		for (int i = 0; i < M; ++i) {
			out.row(i) = eval(T(i), diff).transpose();
		}
	}
	return out;
}

// ---------- Start / End ----------
// double CubicSpline::begin() const {
// 	if (!initialized()) throw std::runtime_error("CubicSpline::begin(): spline is empty");
// 	return _times(0);
// }

// double CubicSpline::end() const {
// 	if (!initialized()) throw std::runtime_error("CubicSpline::end(): spline is empty");
// 	return _times(_times.size() - 1);
// }


/*
 * PYBIND11 MODULE
 */

void init_submodule_splines(py::module_& m) {
        py::module_ splines = m.def_submodule("splines", "Splines module.");
        py::class_<CubicSpline>(splines, "CubicSpline")
                .def(py::init<>())
		.def("dimension", &CubicSpline::dimension)
		.def("num_pieces", &CubicSpline::num_pieces)
		.def("initialized", &CubicSpline::initialized)
                .def("clear", &CubicSpline::clear)
                .def("set", &CubicSpline::set)
                .def("append", &CubicSpline::append)
                .def("get_piece", &CubicSpline::get_piece)
                .def("eval_into", &CubicSpline::eval_into)
                .def("eval", &CubicSpline::eval)
                .def("eval_multiple", &CubicSpline::eval_multiple)
                .def("begin", &CubicSpline::begin)
                .def("end", &CubicSpline::end);
}
