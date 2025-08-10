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

void CubicPiece::set(const py::array_t<double>& x0,
                     const py::array_t<double>& v0,
                     const py::array_t<double>& x1,
                     const py::array_t<double>& v1,
                     double tau) {
        auto x0_ = x0.unchecked<1>();
        auto v0_ = v0.unchecked<1>();
        auto x1_ = x1.unchecked<1>();
        auto v1_ = v1.unchecked<1>();

        ssize_t dim = x0_.shape(0);
        this->a = py::array_t<double>(dim);
        this->b = py::array_t<double>(dim);
        this->c = py::array_t<double>(dim);
        this->d = py::array_t<double>(dim);

        auto a_ = this->a.mutable_unchecked<1>();
        auto b_ = this->b.mutable_unchecked<1>();
        auto c_ = this->c.mutable_unchecked<1>();
        auto d_ = this->d.mutable_unchecked<1>();

        double tau2 = tau * tau;
        double tau3 = tau * tau2;

        for (ssize_t i = 0; i < dim; ++i) {
                d_(i) = x0_(i);
                c_(i) = v0_(i);
                b_(i) = (3.0 * (x1_(i) - x0_(i)) - tau * (v1_(i) + 2.0 * v0_(i))) / tau2;
                a_(i) = (-2.0 * (x1_(i) - x0_(i)) + tau * (v1_(i) + v0_(i))) / tau3;
        }
}

void CubicPiece::eval_into(py::object x,
                           py::object xDot,
                           py::object xDDot,
                           double t) const
{
        ssize_t dim = this->d.shape(0);

        auto a_ = this->a.unchecked<1>();
        auto b_ = this->b.unchecked<1>();
        auto c_ = this->c.unchecked<1>();
        auto d_ = this->d.unchecked<1>();

        double t2 = t * t;
        double t3 = t2 * t;

        // Position
        if (!x.is_none()) {
                if (!py::isinstance<py::array_t<double>>(x)) {
                        throw py::type_error("x must be a NumPy array or None");
                }

                py::array_t<double> x_arr = x.cast<py::array_t<double>>();
                if (x_arr.ndim() != 1 || x_arr.shape(0) != dim) {
                        throw py::value_error("x must have shape (dim,) matching the spline");
                }

                auto x_ = x_arr.mutable_unchecked<1>();
                for (ssize_t i = 0; i < dim; ++i) {
                        x_(i) = d_(i) + t * c_(i) + t2 * b_(i) + t3 * a_(i);
                }
        }

        // Velocity
        if (!xDot.is_none()) {
                if (!py::isinstance<py::array_t<double>>(xDot)) {
                        throw py::type_error("xDot must be a NumPy array or None");
                }

                py::array_t<double> xDot_arr = xDot.cast<py::array_t<double>>();
                if (xDot_arr.ndim() != 1 || xDot_arr.shape(0) != dim) {
                        throw py::value_error("xDot must have shape (dim,) matching the spline");
                }

                auto xDot_ = xDot_arr.mutable_unchecked<1>();
                for (ssize_t i = 0; i < dim; ++i) {
                        xDot_(i) = c_(i) + 2.0 * t * b_(i) + 3.0 * t2 * a_(i);
                }
        }

        // Acceleration
        if (!xDDot.is_none()) {
                if (!py::isinstance<py::array_t<double>>(xDDot)) {
                        throw py::type_error("xDDot must be a NumPy array or None");
                }

                py::array_t<double> xDDot_arr = xDDot.cast<py::array_t<double>>();
                if (xDDot_arr.ndim() != 1 || xDDot_arr.shape(0) != dim) {
                        throw py::value_error("xDDot must have shape (dim,) matching the spline");
                }

                auto xDDot_ = xDDot_arr.mutable_unchecked<1>();
                for (ssize_t i = 0; i < dim; ++i) {
                        xDDot_(i) = 2.0 * b_(i) + 6.0 * t * a_(i);
                }
        }
}

py::array_t<double> CubicPiece::eval(double t, unsigned int diff) const {
        ssize_t dim = this->d.shape(0);
        auto a_ = this->a.unchecked<1>();
        auto b_ = this->b.unchecked<1>();
        auto c_ = this->c.unchecked<1>();
        auto d_ = this->d.unchecked<1>();

        py::array_t<double> out(dim);
        auto out_ = out.mutable_unchecked<1>();
        double t2 = t * t;

        if (diff == 0) {
                double t3 = t2 * t;
                for (ssize_t i = 0; i < dim; ++i)
                        out_(i) = d_(i) + t * c_(i) + t2 * b_(i) + t3 * a_(i);
        } else if (diff == 1) {
                for (ssize_t i = 0; i < dim; ++i)
                        out_(i) = c_(i) + 2.0 * t * b_(i) + 3.0 * t2 * a_(i);
        } else if (diff == 2) {
                for (ssize_t i = 0; i < dim; ++i)
                        out_(i) = 2.0 * b_(i) + 6.0 * t * a_(i);
        } else if (diff == 3) {
                for (ssize_t i = 0; i < dim; ++i)
                        out_(i) = 6.0 * a_(i);
        } else {
                throw py::value_error("CubicPiece::eval(): derivative order must be 0–3");
        }

        return out;
}

/*
 * CubicSpline
 */

py::array_t<double> get_row(const py::array_t<double>& mat, ssize_t i) {
        if (mat.ndim() != 2) {
                throw py::value_error("Expected a 2D array");
        }

        ssize_t cols = mat.shape(1);
        auto mat_ = mat.unchecked<2>();

        py::array_t<double> row(cols);
        auto row_ = row.mutable_unchecked<1>();

        for (ssize_t j = 0; j < cols; ++j) {
                row_(j) = mat_(i, j);
        }

        return row;
}

void CubicSpline::set(const py::array_t<double>& pts,
                      const py::array_t<double>& vels,
                      const py::array_t<double>& times)
{
        if (times.ndim() != 1 || times.shape(0) < 2) {
                throw py::value_error("CubicSpline::set() requires at least 2 time points");
        }

        if (pts.ndim() != 2 || vels.ndim() != 2) {
                throw py::value_error("pts and vels must be 2D arrays");
        }

        if (pts.shape(0) != vels.shape(0)) {
                throw py::value_error("pts and vels must have the same number of rows");
        }

        if (pts.shape(0) != times.shape(0)) {
                throw py::value_error("pts/vels rows must match time entries");
        }

        this->times = times;
        ssize_t K = pts.shape(0) - 1;
        this->pieces.resize(K);

        auto times_ = times.unchecked<1>();

        for (ssize_t k = 0; k < K; ++k) {
                py::array_t<double> x0 = get_row(pts, k);
                py::array_t<double> v0 = get_row(vels, k);
                py::array_t<double> x1 = get_row(pts, k + 1);
                py::array_t<double> v1 = get_row(vels, k + 1);
                double tau = times_(k + 1) - times_(k);
                this->pieces[k].set(x0, v0, x1, v1, tau);
        }
}

void CubicSpline::append(const py::array_t<double>& pts,
                         const py::array_t<double>& vels,
                         const py::array_t<double>& new_times)
{
        ssize_t dim = pts.shape(1);

        if (new_times.ndim() != 1 || new_times.shape(0) < 1 || new_times.at(0) < 1e-6) {
                throw py::value_error("CubicSpline::append(): new_times[0] must be > 0");
        }

        auto times_ = this->times.unchecked<1>();
        ssize_t T = times_.shape(0);
        if (T < 2) throw py::value_error("CubicSpline::append(): spline must be initialized first");

        // Get last time segment duration
        double t_end = times_(T - 1);
        double t_prev = times_(T - 2);
        double dt_last = t_end - t_prev;

        // Evaluate current end state into two new arrays
        py::array_t<double> x = py::array_t<double>({dim});
        py::array_t<double> xDot = py::array_t<double>({dim});
        this->pieces.back().eval_into(x, xDot, py::none(), dt_last);

        // Extend time array
        std::vector<double> new_times_vec(T);
        for (ssize_t i = 0; i < T; ++i)
                new_times_vec[i] = times_(i);

        auto new_times_ = new_times.unchecked<1>();
        for (ssize_t i = 0; i < new_times.shape(0); ++i)
                new_times_vec.push_back(t_end + new_times_(i));

        this->times = py::array_t<double>(new_times_vec.size(), new_times_vec.data());

        // Resize and insert new segments
        ssize_t K_new = pts.shape(0);
        ssize_t K_old = this->pieces.size();
        this->pieces.resize(K_old + K_new);

        // First segment connects to last known (x, xDot)
        py::array_t<double> x1 = get_row(pts, 0);
        py::array_t<double> v1 = get_row(vels, 0);
        this->pieces[K_old].set(x, xDot, x1, v1, new_times_(0));

        // Remaining new segments
        for (ssize_t k = 1; k < K_new; ++k) {
                py::array_t<double> x0 = get_row(pts, k - 1);
                py::array_t<double> v0 = get_row(vels, k - 1);
                py::array_t<double> x1 = get_row(pts, k);
                py::array_t<double> v1 = get_row(vels, k);
                double tau = new_times_(k) - new_times_(k - 1);
                this->pieces[K_old + k].set(x0, v0, x1, v1, tau);
        }
}

unsigned int CubicSpline::get_piece(double t) const {
        auto times_ = this->times.unchecked<1>();
        ssize_t N = times_.shape(0);

        if (N < 2)
                throw py::value_error("CubicSpline is empty");

        if (t <= times_(0)) return 0;
        if (t >= times_(N - 1)) return this->pieces.size() - 1;

        for (ssize_t k = 1; k < N; ++k) {
                if (t < times_(k)) return k - 1;
        }
        return N - 2;  // fallback
}

void CubicSpline::eval_into(py::object x,
                            py::object xDot,
                            py::object xDDot,
                            double t) const {
        auto times_ = this->times.unchecked<1>();
        ssize_t N = times_.shape(0);

        if (N < 2)
                throw py::value_error("CubicSpline is empty");

        double t0 = times_(0);
        double tN = times_(N - 1);

        // Dispatch to first piece
        if (t <= t0) {
                this->pieces[0].eval_into(x, xDot, xDDot, 0.0);

                if (!xDDot.is_none()) {
                        py::array_t<double> xDot_arr = xDot.cast<py::array_t<double>>();
                        py::array_t<double> xDDot_arr = xDDot.cast<py::array_t<double>>();

                        if (xDDot_arr.ndim() != 1 || xDDot_arr.shape(0) != xDot_arr.shape(0)) {
                                throw py::value_error("xDDot must match xDot shape for zeroing");
                        }

                        auto acc_ = xDDot_arr.mutable_unchecked<1>();
                        for (ssize_t i = 0; i < acc_.shape(0); ++i) {
                                acc_(i) = 0.0;
                        }
                }

                return;
        }

        // Dispatch to last piece
        if (t >= tN) {
                this->pieces.back().eval_into(x, xDot, xDDot, tN - times_(N - 2));

                if (!xDDot.is_none()) {
                        py::array_t<double> xDot_arr = xDot.cast<py::array_t<double>>();
                        py::array_t<double> xDDot_arr = xDDot.cast<py::array_t<double>>();

                        if (xDDot_arr.ndim() != 1 || xDDot_arr.shape(0) != xDot_arr.shape(0)) {
                                throw py::value_error("xDDot must match xDot shape for zeroing");
                        }

                        auto acc_ = xDDot_arr.mutable_unchecked<1>();
                        for (ssize_t i = 0; i < acc_.shape(0); ++i) {
                                acc_(i) = 0.0;
                        }
                }

                return;
        }

        // Main case: evaluate at piece k
        unsigned int k = this->get_piece(t);
        double t_rel = t - times_(k);
        this->pieces[k].eval_into(x, xDot, xDDot, t_rel);
}


py::array_t<double> CubicSpline::eval(double t, unsigned int diff) const {
        // Get output shape from evaluating a single sample
        ssize_t dim = this->pieces[0].a.shape(0);
        py::array_t<double> out(dim);
        if (diff == 0)      this->eval_into(out, py::none(), py::none(), t);
        else if (diff == 1) this->eval_into(py::none(), out, py::none(), t);
        else if (diff == 2) this->eval_into(py::none(), py::none(), out, t);
        else {
                unsigned int k = this->get_piece(t);
                double t_rel = t - this->times.unchecked<1>()(k);
                out = this->pieces[k].eval(t_rel, diff);
        }
        return out;
}

py::array_t<double> CubicSpline::eval_multiple(const py::array_t<double>& T, unsigned int diff) const {
        auto T_ = T.unchecked<1>();
        ssize_t M = T.shape(0);

        // Get output shape from evaluating a single sample
        py::array_t<double> first = this->eval(T_(0), diff);
        ssize_t dim = first.shape(0);

        py::array_t<double> out({M, dim});
        auto out_ = out.mutable_unchecked<2>();

        for (ssize_t i = 0; i < M; ++i) {
                py::array_t<double> vi = this->eval(T_(i), diff);
                auto vi_ = vi.unchecked<1>();
                for (ssize_t j = 0; j < dim; ++j)
                        out_(i, j) = vi_(j);
        }

        return out;
}

double CubicSpline::begin() const {
        return this->times.unchecked<1>()(0);
}

double CubicSpline::end() const {
        return this->times.unchecked<1>()(this->times.shape(0) - 1);
}

/*
 * Costs and Constraints
 */

#include "cubic_spline_leap_cost.cpp"
// #include "cubic_spline_acc0.cpp"
// #include "cubic_spline_acc1.cpp"
// #include "cubic_spline_max_acc.cpp"
// #include "cubic_spline_max_jerk.cpp"
// #include "cubic_spline_max_vel.cpp"
// #include "cubic_spline_pos_vel_acc.cpp"

// double single_piece_cost(Eigen::VectorXd x0, Variable v0, Eigen::VectorXd x1, Variable v1, Variable tau) {
// 	Eigen::VectorXd D = (x1 - x0) - 0.5 * tau * (v0 + v1);
// 	Eigen::VectorXd V = v1 - v0;
// 	const double s12 = std::sqrt(12.0);
// 	double cost = (s12 * std::pow(tau, -1.5)) * D.norm() + std::pow(tau, -0.5) * V.norm();
// 	return cost * cost;
// };

// This struct represents a cost/constraint block for enforcing max velocity in a cubic spline segment
// Inputs: fixed waypoints x0, x1; decision velocities v0, v1; decision scalar tau
// struct CubicSplineMaxVelCost {
// 	const Eigen::VectorXd x0, x1;
// 	const double max_vel;

// 	explicit CubicSplineMaxVelCost(const Eigen::VectorXd& x0_,
// 				       const Eigen::VectorXd& x1_,
// 				       double max_vel_)
// 		: x0(x0_), x1(x1_), max_vel(max_vel_) {}

// 	// Evaluate the 4 * d-dimensional constraint vector
// 	Eigen::VectorXd operator()(const Eigen::Ref<const Eigen::VectorXd>& x) const {
// 		const int d = x0_.size();
// 		Eigen::VectorXd v0 = x.segment(0, d);
// 		Eigen::VectorXd v1 = x.segment(d, d);
// 		double tau = x(2 * d);

// 		double tau2 = tau * tau;
// 		Eigen::VectorXd c = v0;
// 		Eigen::VectorXd b = 3. * (x1 - x0) - tau * (v1 + 2. * v0);
// 		Eigen::VectorXd a = -2. * (x1 - x0) + tau * (v1 + v0);

// 		// Compute approximation of peak velocity (mid-trajectory)
// 		Eigen::VectorXd vmax = c + (1. / tau) * (b + 0.75 * a);

// 		Eigen::VectorXd y(4 * d);
// 		y.segment(0 * d, d) = v0 - max_vel * Eigen::VectorXd::Ones(d);
// 		y.segment(1 * d, d) = -v0 - max_vel * Eigen::VectorXd::Ones(d);
// 		y.segment(2 * d, d) = vmax - max_vel * Eigen::VectorXd::Ones(d);
// 		y.segment(3 * d, d) = -vmax - max_vel * Eigen::VectorXd::Ones(d);

// 		return y;
// 	}

// };


// arr CubicSplineMaxJer(const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
//   //jerk is 6a
//   double tau2 = tau*tau, tau3 = tau*tau2, tau4 = tau2*tau2;
//   arr a6 = 6./tau3 * (-2.*(x1-x0) + tau*(v1+v0));
//   if(tauJ.N) {
//     a6.J() += (36./tau4 * (x1-x0)) * tauJ;
//     a6.J() += (-12./tau3 * (v1+v0)) * tauJ;
//   }

//   uint n=x0.N;
//   arr y(2*n);
//   if(a6.jac) y.J().sparse().resize(y.N, a6.jac->d1, 0);
//   y.setVectorBlock(a6, 0*n);
//   y.setVectorBlock(-a6, 1*n);
//   return y;
// }

// arr CubicSplineAcc0(const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
//   //acceleration is 6a t + 2b
//   double tau2 = tau*tau, tau3 = tau*tau2;
//   arr b2 = 2./tau2 * (3.*(x1-x0) - tau*(v1+2.*v0));
//   if(tauJ.N) {
//     b2.J() += -12./tau3 * (x1-x0) * tauJ;
//     b2.J() -= -2./tau2 * (v1+2.*v0) * tauJ;
//   }
//   return b2;
// }

// arr CubicSplineAcc1(const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
//   //acceleration is 6a t + 2b
//   double tau2 = tau*tau, tau3 = tau*tau2;
//   //  arr d = x0;
//   //  arr c = v0;
//   arr b2 = 2./tau2 * (3.*(x1-x0) - tau*(v1+2.*v0));
//   if(tauJ.N) {
//     b2.J() += -12./tau3 * (x1-x0) * tauJ;
//     b2.J() -= -2./tau2 * (v1+2.*v0) * tauJ;
//   }
//   arr a6_tau = 6./tau2 * (-2.*(x1-x0) + tau*(v1+v0));
//   if(tauJ.N) {
//     a6_tau.J() -= -24./tau3 * (x1-x0) * tauJ;
//     a6_tau.J() += -6./tau2 * (v1+v0) * tauJ;
//   }
//   return a6_tau + b2;
// }

// arr CubicSplineMaxAcc(const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
//   //acceleration is 6a t + 2b
//   double tau2 = tau*tau, tau3 = tau*tau2;
//   //  arr d = x0;
//   //  arr c = v0;
//   arr b2 = 2./tau2 * (3.*(x1-x0) - tau*(v1+2.*v0));
//   if(tauJ.N) {
//     b2.J() += -12./tau3 * (x1.noJ()-x0.noJ()) * tauJ;
//     b2.J() -= -2./tau2 * (v1.noJ()+2.*v0.noJ()) * tauJ;
//   }
//   arr a6_tau = 6./tau2 * (-2.*(x1-x0) + tau*(v1+v0));
//   if(tauJ.N) {
//     a6_tau.J() -= -24./tau3 * (x1.noJ()-x0.noJ()) * tauJ;
//     a6_tau.J() += -6./tau2 * (v1.noJ()+v0.noJ()) * tauJ;
//   }

//   uint d=x0.N;
//   arr y(4*d);
//   if(b2.jac) y.J().sparse().resize(y.N, b2.jac->d1, 0);
//   y.setVectorBlock(b2, 0*d);
//   y.setVectorBlock(-b2, 1*d);
//   y.setVectorBlock(b2 + a6_tau, 2*d);
//   y.setVectorBlock(-b2 - a6_tau, 3*d);
//   return y;
// }


// arr CubicSplineMaxVel(const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
// 	//acc is 6a t + 2b; with root at t=-b/3a
// 	//velocity is 3a t^2 + 2b t + c; at root is -b^2 / 3a + c

// #if 1
// 	double tau2 = tau*tau;
// 	//  arr d = x0;
// 	arr c = v0;
// 	arr b = (3.*(x1-x0) - tau*(v1+2.*v0));
// 	if(tauJ.N) {
// 		b.J() -= (v1+2.*v0) * tauJ;
// 	}
// 	arr a = (-2.*(x1-x0) + tau*(v1+v0));
// 	if(tauJ.N) {
// 		a.J() += (v1+v0) * tauJ;
// 	}
// 	arr t=-tau*b.noJ()/(3.*a.noJ());
// 	//indicators for each dimension
// 	arr iv0=zeros(t.N), iv1=zeros(t.N), ivm=zeros(t.N);
// 	for(uint i=0; i<t.N; i++) {
// 		if(t(i)<=0) iv0(i)=1.;
// 		else if(t(i)>=tau) iv1(i)=1.;
// 		else ivm(i)=1.;
// 	}
// 	arr vmax = c + (1./tau) * (b + 3./4.*a);
// 	if(tauJ.N) {
// 		vmax.J() -= (1./tau2) * (b + 3./4.*a) * tauJ;
// 	}

// 	//  vmax = iv0%v0;
// 	//  vmax += iv1%v1;
// 	//  vmax += ((1./(3.*tau)) * ((ivm%b%b)/a) + c);
// 	//  if(tauJ.N){
// 	//    vmax.J() -= (1./(3.*tau2)) * ((ivm%b%b)/a) * tauJ;
// 	//  }
// #endif

// 	uint n=x0.N;
// 	arr y(4*n);
// 	y.setZero();
// 	if(v0.jac) y.J().sparse().resize(y.N, v0.jac->d1, 0);
// 	else if(vmax.jac) y.J().sparse().resize(y.N, vmax.jac->d1, 0);
// 	y.setVectorBlock(v0, 0*n);
// 	y.setVectorBlock(-v0, 1*n);
// 	y.setVectorBlock(vmax, 2*n);
// 	y.setVectorBlock(-vmax, 3*n);
// 	return y;
// }

// void CubicSplinePosVelAcc(arr& pos, arr& vel, arr& acc, double trel, const arr& x0, const arr& v0, const arr& x1, const arr& v1, double tau, const arr& tauJ) {
//   //position at time t:
//   // a t^3 + b t^2 + c t + d

//   CHECK_GE(trel, 0., "");
//   CHECK_LE(trel, 1., "");

//   double tau2 = tau*tau;
//   double tau3 = tau2*tau;
//   arr d = x0;
//   arr c = v0;
// #if 0
//   arr b = 1./tau2 * (3.*(x1-x0) - tau*(v1+2.*v0));
//   if(tauJ.N) {
//     b.J() += -6./tau3 * (x1.noJ()-x0.noJ()) * tauJ;
//     b.J() -= -1./tau2 * (v1.noJ()+2.*v0.noJ()) * tauJ;
//   }
//   arr a = 1./tau3 * (-2.*(x1-x0) + tau*(v1+v0));
//   if(tauJ.N) {
//     a.J() -= -6./tau4 * (x1.noJ()-x0.noJ()) * tauJ;
//     a.J() += -2./tau3 * (v1.noJ()+v0.noJ()) * tauJ;
//   }
// #endif
//   arr c_tau = tau*c;
//   if(tauJ.N) { if(c_tau.jac) c_tau.J() += (c) * tauJ; else c_tau.J() = c*tauJ; }

//   arr b_tau2 = (3.*(x1-x0) - tau*(v1+2.*v0));
//   if(tauJ.N) b_tau2.J() -= (v1.noJ()+2.*v0.noJ()) * tauJ;

//   arr b_tau1 = 1./tau * b_tau2;
//   if(tauJ.N) b_tau1.J() += (-1./tau2) * b_tau2.noJ() * tauJ;

//   arr b_tau0 = 1./tau2 * b_tau2;
//   if(tauJ.N) b_tau0.J() += (-2./tau3) * b_tau2.noJ() * tauJ;

//   arr a_tau3 = (-2.*(x1-x0) + tau*(v1+v0));
//   if(tauJ.N) a_tau3.J() += (v1+v0) * tauJ;

//   arr a_tau2 = 1./tau * a_tau3;
//   if(tauJ.N) a_tau2.J() += (-1./tau2) * a_tau3.noJ() * tauJ;

//   arr a_tau1 = 1./tau2 * a_tau3;
//   if(tauJ.N) a_tau1.J() += (-2./tau3) * a_tau3.noJ() * tauJ;

//   //arr a_tau0 = 1./tau3 * a_tau3;
//   //if(tauJ.N) a_tau0.J() += (-3./tau4) * a_tau3.noJ() * tauJ;

//   if(!!pos) pos = (trel*trel*trel)*a_tau3 + (trel*trel)*b_tau2 + trel*c_tau + d;
//   if(!!vel) vel = (3.*trel*trel)*a_tau2 + (2.*trel)*b_tau1 + c;
//   if(!!acc) acc = (6.*trel)*a_tau1 + (2.)*b_tau0;
// #if 0
//   if(!!pos) pos = (t*t*t)*a + (t*t)*b + t*c + d;
//   if(!!vel) vel = (3.*t*t)*a + (2.*t)*b + c;
//   if(!!acc) acc = (6.*t)*a + (2.)*b;
// #endif
// }



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
