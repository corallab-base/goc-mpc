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

void CubicPiece::set(const Eigen::VectorXd& x0,
                     const Eigen::VectorXd& v0,
                     const Eigen::VectorXd& x1,
                     const Eigen::VectorXd& v1,
                     double tau) {
        ssize_t dim = x0.size();
        _a = Eigen::VectorXd(dim);
	_b = Eigen::VectorXd(dim);
	_c = Eigen::VectorXd(dim);
	_d = Eigen::VectorXd(dim);

        double tau2 = tau * tau;
        double tau3 = tau * tau2;

        for (ssize_t i = 0; i < dim; ++i) {
                _d(i) = x0(i);
                _c(i) = v0(i);
                _b(i) = (3.0 * (x1(i) - x0(i)) - tau * (v1(i) + 2.0 * v0(i))) / tau2;
                _a(i) = (-2.0 * (x1(i) - x0(i)) + tau * (v1(i) + v0(i))) / tau3;
        }
}

// Fill any subset of outputs. Pass std::nullopt for ones you don't want.
void CubicPiece::eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
			   std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
			   std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
			   double t) const {
	const int n = _a.size();
	const double t2 = t * t;
	const double t3 = t2 * t;

	if (x) {
		if (x->size() != n) {
			throw std::runtime_error("x has wrong size");
		}
		// x(t) = d + t c + t^2 b + t^3 a
		x->noalias() = _d + t * _c + t2 * _b + t3 * _a;
	}

	if (xDot) {
		if (xDot->size() != n) {
			throw std::runtime_error("xDot has wrong size");
		}
		// x'(t) = c + 2 t b + 3 t^2 a
		xDot->noalias() = _c + (2.0 * t) * _b + (3.0 * t2) * _a;
	}

	if (xDDot) {
		if (xDDot->size() != n) {
			throw std::runtime_error("xDDot has wrong size");
		}
		// x''(t) = 2 b + 6 t a
		xDDot->noalias() = 2.0 * _b + (6.0 * t) * _a;
	}
}

Eigen::VectorXd CubicPiece::eval(double t, unsigned int diff) const {
        ssize_t dim = _a.size();

	Eigen::VectorXd out(dim);
        double t2 = t * t;

        if (diff == 0) {
                double t3 = t2 * t;
                for (ssize_t i = 0; i < dim; ++i)
                        out(i) = _d(i) + t * _c(i) + t2 * _b(i) + t3 * _a(i);
        } else if (diff == 1) {
                for (ssize_t i = 0; i < dim; ++i)
                        out(i) = _c(i) + 2.0 * t * _b(i) + 3.0 * t2 * _a(i);
        } else if (diff == 2) {
                for (ssize_t i = 0; i < dim; ++i)
                        out(i) = 2.0 * _b(i) + 6.0 * t * _a(i);
        } else if (diff == 3) {
                for (ssize_t i = 0; i < dim; ++i)
                        out(i) = 6.0 * _a(i);
        } else {
                throw py::value_error("CubicPiece::eval(): derivative order must be 0–3");
        }

        return out;
}

/*
 * CubicSpline
 */

void CubicSpline::set(const Eigen::MatrixXd& pts,
                      const Eigen::MatrixXd& vels,
                      const Eigen::VectorXd& times)
{
        // if (times.ndim() != 1 || times.shape(0) < 2) {
        //         throw py::value_error("CubicSpline::set() requires at least 2 time points");
        // }

        // if (pts.ndim() != 2 || vels.ndim() != 2) {
        //         throw py::value_error("pts and vels must be 2D arrays");
        // }

        // if (pts.shape(0) != vels.shape(0)) {
        //         throw py::value_error("pts and vels must have the same number of rows");
        // }

        // if (pts.shape(0) != times.shape(0)) {
        //         throw py::value_error("pts/vels rows must match time entries");
        // }

        this->_times = times;
	this->_dim = pts.cols();
        ssize_t K = pts.rows() - 1;
        this->_pieces.resize(K);

        for (ssize_t k = 0; k < K; ++k) {
		Eigen::VectorXd x0 = pts.row(k);
		Eigen::VectorXd v0 = vels.row(k);
		Eigen::VectorXd x1 = pts.row(k + 1);
		Eigen::VectorXd v1 = vels.row(k + 1);
                double tau = times(k + 1) - times(k);
                this->_pieces[k].set(x0, v0, x1, v1, tau);
        }
}

// Append K_new segments defined by (pts, vels, new_times).
// Shapes:
//   pts:  (K_new, dim)
//   vels: (K_new, dim)
//   new_times: (K_new,)   -- durations; must be positive and (strictly) increasing cumulative offsets
void CubicSpline::append(const Eigen::Ref<const Eigen::MatrixXd>& pts,
			 const Eigen::Ref<const Eigen::MatrixXd>& vels,
			 const Eigen::Ref<const Eigen::VectorXd>& new_times) {

	// Basic checks mirroring your Python code
	if (new_times.size() < 1 || new_times(0) <= 1e-6) {
		throw std::invalid_argument("CubicSpline::append(): new_times[0] must be > 0");
	}

	const int T = static_cast<int>(_times.size());
	if (T < 2) {
		throw std::invalid_argument("CubicSpline::append(): spline must be initialized first");
	}

	const int K_new = static_cast<int>(pts.rows());
	if (K_new <= 0) {
		throw std::invalid_argument("CubicSpline::append(): pts must have at least one row");
	}
	if (vels.rows() != pts.rows() || vels.cols() != pts.cols()) {
		throw std::invalid_argument("CubicSpline::append(): vels shape must match pts");
	}
	if (new_times.size() != K_new) {
		throw std::invalid_argument("CubicSpline::append(): new_times length must equal pts.rows()");
	}

	// Dimension consistency
	const int d = static_cast<int>(pts.cols());
	if (_dim < 0) _dim = d;
	if (d != _dim) {
		throw std::invalid_argument("CubicSpline::append(): pts.cols() != spline dimension");
	}

	// Ensure new_times is strictly increasing (durations increasing cumulatively)
	for (int i = 1; i < new_times.size(); ++i) {
		if (new_times(i) - new_times(i-1) <= 1e-12) {
			throw std::invalid_argument("CubicSpline::append(): new_times must be strictly increasing");
		}
	}

	// Last segment duration of the existing spline
	const double t_end  = _times(T - 1);
	const double t_prev = _times(T - 2);
	const double dt_last = t_end - t_prev;

	// Evaluate current end state at dt_last
	Eigen::VectorXd x(d), xDot(d);
	{
		Eigen::Ref<Eigen::VectorXd> x_ref(x);
		Eigen::Ref<Eigen::VectorXd> xdot_ref(xDot);
		// no acceleration requested
		_pieces.back().eval_into(x_ref, xdot_ref, std::nullopt, dt_last);
	}

	// Extend _times with the new absolute times
	Eigen::VectorXd new_times_abs(T + K_new);
	new_times_abs.head(T) = _times;
	for (int i = 0; i < K_new; ++i) {
		new_times_abs(T + i) = t_end + new_times(i);
	}
	_times.swap(new_times_abs);

	// Resize pieces to fit the K_new appended segments
	const int K_old = static_cast<int>(_pieces.size());
	_pieces.resize(K_old + K_new);

	// First appended segment connects from the current end state (x, xDot)
	{
		Eigen::VectorXd x1  = pts.row(0).transpose();
		Eigen::VectorXd v1  = vels.row(0).transpose();
		const double tau = new_times(0);
		_pieces[K_old].set(x, xDot, x1, v1, tau);
	}

	// Remaining appended segments use (x0, v0) = previous waypoint, (x1, v1) = current waypoint
	for (int k = 1; k < K_new; ++k) {
		Eigen::VectorXd x0  = pts.row(k - 1).transpose();
		Eigen::VectorXd v0  = vels.row(k - 1).transpose();
		Eigen::VectorXd x1  = pts.row(k).transpose();
		Eigen::VectorXd v1  = vels.row(k).transpose();
		const double tau = new_times(k) - new_times(k - 1);
		if (tau <= 1e-12) {
			throw std::invalid_argument("CubicSpline::append(): segment duration must be positive");
		}
		_pieces[K_old + k].set(x0, v0, x1, v1, tau);
	}
}

unsigned int CubicSpline::get_piece(double t) const {
        ssize_t n = _times.size();

        if (n < 2)
                throw py::value_error("CubicSpline is empty");

        if (t <= _times(0)) return 0;
        if (t >= _times(n - 1)) return this->_pieces.size() - 1;

        for (ssize_t k = 1; k < n; ++k) {
                if (t < _times(k)) return k - 1;
        }
        return n - 2;  // fallback
}

void CubicSpline::eval_into(std::optional<Eigen::Ref<Eigen::VectorXd>> x,
			    std::optional<Eigen::Ref<Eigen::VectorXd>> xDot,
			    std::optional<Eigen::Ref<Eigen::VectorXd>> xDDot,
			    double t) const
{
	const int N = static_cast<int>(_times.size());
	if (N < 2) {
		throw std::invalid_argument("CubicSpline is empty");
	}

	const double t0 = _times(0);
	const double tN = _times(N - 1);

	// Quick size checks (if provided)
	const int d = _dim;
	auto chk = [d](const std::optional<Eigen::Ref<Eigen::VectorXd>>& v, const char* name){
		if (v && v->size() != d) {
			throw std::invalid_argument(std::string(name) + " has wrong size");
		}
	};
	chk(x,    "x");
	chk(xDot, "xDot");
	chk(xDDot,"xDDot");

	// Before-first interval: evaluate at start; zero acceleration if requested
	if (t <= t0) {
		_pieces.front().eval_into(x, xDot, xDDot, /*t_rel=*/0.0);
		if (xDDot) {
			xDDot->setZero();  // match original: force zero acceleration at boundary
		}
		return;
	}

	// After-last interval: evaluate at end; zero acceleration if requested
	if (t >= tN) {
		const double tau_last = _times(N - 1) - _times(N - 2);
		_pieces.back().eval_into(x, xDot, xDDot, /*t_rel=*/tau_last);
		if (xDDot) {
			xDDot->setZero();
		}
		return;
	}

	// Main case: find piece k and evaluate at relative time
	const unsigned int k = get_piece(t);
	const double t_rel = t - _times(static_cast<int>(k));
	_pieces[static_cast<size_t>(k)].eval_into(x, xDot, xDDot, t_rel);
}

Eigen::VectorXd CubicSpline::eval(double t, unsigned int diff) const {
	const int N = static_cast<int>(_times.size());
	if (N < 2) {
		throw std::invalid_argument("CubicSpline::eval(): spline is empty");
	}

	// Determine dimension once (prefer stored _dim, otherwise infer)
	const int d = (_dim >= 0) ? _dim
		: static_cast<int>(/* e.g. */ _pieces.front()._d.size());
	Eigen::VectorXd out(d);

	if (diff <= 2) {
		// Prepare optionals
		std::optional<Eigen::Ref<Eigen::VectorXd>> X, Xd, Xdd;

		if (diff == 0) {
			X.emplace(out);                // position
		} else if (diff == 1) {
			Xd.emplace(out);               // velocity
		} else { // diff == 2
			Xdd.emplace(out);              // acceleration
		}

		// Reuse the spline dispatcher you already implemented
		this->eval_into(std::move(X), std::move(Xd), std::move(Xdd), t);
		return out;
	}

	// Higher derivatives: delegate to the appropriate piece (same as your original)
	const unsigned int k = this->get_piece(t);
	const double t_rel = t - _times(static_cast<int>(k));
	return _pieces[k].eval(t_rel, diff);
}

Eigen::MatrixXd CubicSpline::eval_multiple(const Eigen::Ref<const Eigen::VectorXd>& T,
                                           unsigned int diff) const {
	const int N = static_cast<int>(_times.size());
	if (N < 2) {
		throw std::invalid_argument("CubicSpline::eval_multiple(): spline is empty");
	}

	const int M = static_cast<int>(T.size());
	const int d = (_dim >= 0) ? _dim
		: static_cast<int>(/* e.g. */ _pieces.front()._a.size()); // or your stored dim

	Eigen::MatrixXd out(M, d);
	if (M == 0) return out;  // empty query → empty matrix (0 × d)

	if (diff <= 2) {
		// Use eval_into to fill each row efficiently.
		Eigen::VectorXd tmp(d);
		for (int i = 0; i < M; ++i) {
			std::optional<Eigen::Ref<Eigen::VectorXd>> X, Xd, Xdd;
			if      (diff == 0) X.emplace(tmp);
			else if (diff == 1) Xd.emplace(tmp);
			else                Xdd.emplace(tmp);  // diff == 2

			this->eval_into(std::move(X), std::move(Xd), std::move(Xdd), T(i));
			out.row(i) = tmp.transpose();
		}
	} else {
		// Higher derivatives: delegate to piece-wise eval (returns VectorXd)
		for (int i = 0; i < M; ++i) {
			out.row(i) = this->eval(T(i), diff).transpose();
		}
	}

	return out;
}

double CubicSpline::begin() const {
        return this->_times(0);
}

double CubicSpline::end() const {
        return this->_times(this->_times.size() - 1);
}


/*
 * PYBIND11 MODULE
 */

void init_submodule_splines(py::module_& m) {
        py::module_ splines = m.def_submodule("splines", "Splines module.");
        py::class_<CubicPiece>(splines, "CubicPiece")
                .def(py::init<>())
                .def("set", &CubicPiece::set)
                .def("eval_into", &CubicPiece::eval_into)
                .def("eval", &CubicPiece::eval);
        py::class_<CubicSpline>(splines, "CubicSpline")
                .def(py::init<>())
                .def("get_piece", &CubicSpline::get_piece)
                .def("set", &CubicSpline::set)
                .def("eval", &CubicSpline::eval)
                .def("eval_multiple", &CubicSpline::eval_multiple)
                .def("eval_into", &CubicSpline::eval_into)
                .def("begin", &CubicSpline::begin)
                .def("end", &CubicSpline::end);
}
