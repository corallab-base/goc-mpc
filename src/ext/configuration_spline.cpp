#include "configuration_spline.hpp"

namespace py = pybind11;
using drake::symbolic::Expression;
template <typename T> using VecX = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T, int N> using Vec = Eigen::Matrix<T, N, 1>;


Expression CubicConfigurationSpline::compute_ctrl_cost(
    const VecX<Expression>& xJ,
    const VecX<Expression>& xJm1,
    const VecX<Expression>& vJ,
    const VecX<Expression>& vJm1,
    const Expression& tau) const {

	// Stable scaling (equivalent to || s12 tau^-1.5 D ||^2 + || tau^-0.5 V ||^2):
	const Expression inv_tau  = Expression(1.0) / tau;
	const Expression inv_tau2 = inv_tau * inv_tau;
	const Expression inv_tau3 = inv_tau2 * inv_tau;

	Expression total(0.0);

	for (const struct BlockOffset off : block_offsets_) {
		const int a0 = off.ambient_offset, aN = off.ambient_size;
		const int t0 = off.tangent_offset, tN = off.tangent_size;

		switch (off.type) {
		case Block::Type::R: {
			// Euclidean block: ambient == tangent == aN == tN
			const auto xj   = xJ.segment(a0, aN);
			const auto xjm1 = xJm1.segment(a0, aN);
			const auto vj   = vJ.segment(t0, tN);
			const auto vjm1 = vJm1.segment(t0, tN);

			const VecX<Expression> D = (xj - xjm1) - Expression(0.5) * tau * (vjm1 + vj);
			const VecX<Expression> V = (vj - vjm1);

			total += Expression(12.0) * inv_tau3 * D.squaredNorm()
				+ inv_tau * V.squaredNorm();
			break;
		}

		case Block::Type::Torus: {
			// Torus block: ambient == tangent == aN == tN
			// Use wrapped angle difference for the position residual (componentwise).
			const auto xj   = xJ.segment(a0, aN);
			const auto xjm1 = xJm1.segment(a0, aN);
			const auto vj   = vJ.segment(t0, tN);
			const auto vjm1 = vJm1.segment(t0, tN);

			VecX<Expression> Dw(aN);
			for (int k = 0; k < aN; ++k) {
				Dw[k] = wrap_to_pi(xj[k] - xjm1[k]);
			}
			const VecX<Expression> D = Dw - Expression(0.5) * tau * (vjm1 + vj);
			const VecX<Expression> V = (vj - vjm1);

			total += Expression(12.0) * inv_tau3 * D.squaredNorm()
				+ inv_tau * V.squaredNorm();
			break;
		}

		case Block::Type::SO3Quat: {
			// SO(3) quaternion block:
			// ambient = 4 (wxyz) in x*, tangent = 3 (angular velocity in chosen frame) in v*
			DRAKE_DEMAND(aN == 4 && tN == 3);

			const Eigen::Matrix<Expression,4,1> qjm1 = xJm1.segment(a0, 4);
			const Eigen::Matrix<Expression,4,1> qj   = xJ.segment(a0, 4);
			const Vec<Expression,3> wjm1 = vJm1.segment(t0, 3);
			const Vec<Expression,3> wj   = vJ.segment(t0, 3);

			const auto Rjm1 = RotFromQuatWxyz(qjm1);
			const auto Rj   = RotFromQuatWxyz(qj);
			const auto Rrel = Rjm1.transpose() * Rj;

			const Vec<Expression,3> dphi = so3_log(Rrel);
			const Vec<Expression,3> D = dphi - Expression(0.5) * tau * (wjm1 + wj);
			const Vec<Expression,3> V = (wj - wjm1);

			total += Expression(12.0) * inv_tau3 * D.squaredNorm()
				+ inv_tau * V.squaredNorm();
			break;
		}
		default:
			// Future: SE(3) block, etc.
			DRAKE_UNREACHABLE();
		}
	}

	return total;
}


/*
 * PYBIND11 MODULE
 */


void init_submodule_configuration_spline(py::module_& m) {
	py::module_ q_spline =
		m.def_submodule("configuration_spline", "General configuration splines module.");

	// Expose Block + Spec (std::vector<Block>) so you can pass a spec from Python
	py::class_<CubicConfigurationSpline::Block>(q_spline, "Block")
		.def_static("R",      &CubicConfigurationSpline::Block::R,      py::arg("k"))
		.def_static("Torus",  &CubicConfigurationSpline::Block::Torus,  py::arg("k"))
		.def_static("SO3",    &CubicConfigurationSpline::Block::SO3);

	py::class_<CubicConfigurationSpline::BlockOffset>(q_spline, "BlockOffset")
		.def_readonly("ambient_offset", &CubicConfigurationSpline::BlockOffset::ambient_offset)
		.def_readonly("tangent_offset", &CubicConfigurationSpline::BlockOffset::tangent_offset)
		.def_readonly("ambient_size", &CubicConfigurationSpline::BlockOffset::ambient_size)
		.def_readonly("tangent_size", &CubicConfigurationSpline::BlockOffset::tangent_size);

	py::class_<CubicConfigurationSpline>(q_spline, "CubicConfigurationSpline")
		.def(py::init<>())
		.def(py::init<CubicConfigurationSpline::Spec>(), py::arg("spec"))
		.def_readonly("offsets", &CubicConfigurationSpline::block_offsets_)
		.def("num_pieces",   &CubicConfigurationSpline::num_pieces)
		.def("initialized",  &CubicConfigurationSpline::initialized)
		.def("clear",        &CubicConfigurationSpline::clear)
		.def("begin",        &CubicConfigurationSpline::begin)
		.def("end",          &CubicConfigurationSpline::end)
		// ---- Shim for the templated set(...) ----
		.def("set",
		     [](CubicConfigurationSpline& self,
			const Eigen::Ref<const Eigen::MatrixXd>& pts,
			const Eigen::Ref<const Eigen::MatrixXd>& vels,
			const Eigen::Ref<const Eigen::VectorXd>& times) {
			     self.set(pts, vels, times);
		     },
		     py::arg("pts"), py::arg("vels"), py::arg("times"))
		// Handy eval that returns (q, v, a) as numpy arrays
		.def("eval",
		     [](const CubicConfigurationSpline& self, double t) {
			     auto ev = self.eval(t);
			     return py::make_tuple(std::move(ev.q_ambient),
						   std::move(ev.v_tangent),
						   std::move(ev.a_tangent));
		     },
		     py::arg("t"))
		.def("eval_multiple", &CubicConfigurationSpline::eval_multiple);
}
