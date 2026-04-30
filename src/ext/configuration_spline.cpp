#include "configuration_spline.hpp"

namespace py = pybind11;
using drake::symbolic::Expression;


/*
 * PYBIND11 MODULE
 */


void init_submodule_configuration_spline(py::module_& m) {
	py::module_ q_spline =
		m.def_submodule("configuration_spline", "General configuration splines module.");

	// Expose Block + Spec (std::vector<Block>) so you can pass a spec from Python
	py::class_<CubicConfigurationSpline::Block>(q_spline, "Block")
		.def_static("R",        &CubicConfigurationSpline::Block::R,      py::arg("k"))
		.def_static("Torus",    &CubicConfigurationSpline::Block::Torus,  py::arg("k"))
		.def_static("SO3Quat",  &CubicConfigurationSpline::Block::SO3Quat)
		.def_static("SO3Mat",   &CubicConfigurationSpline::Block::SO3Mat);

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
		.def("eval_multiple", &CubicConfigurationSpline::eval_multiple)
		.def("compute_ctrl_cost",
		     [](const CubicConfigurationSpline& self,
			const Eigen::Ref<const Eigen::VectorXd>& xJ,
			const Eigen::Ref<const Eigen::VectorXd>& xJm1,
			const Eigen::Ref<const Eigen::VectorXd>& vJ,
			const Eigen::Ref<const Eigen::VectorXd>& vJm1,
			double tau) {
			     return self.compute_ctrl_cost<double>(
				     xJ, xJm1, vJ, vJm1, tau);
		     },
		     py::arg("xJ"),
		     py::arg("xJm1"),
		     py::arg("vJ"),
		     py::arg("vJm1"),
		     py::arg("tau"));
}
