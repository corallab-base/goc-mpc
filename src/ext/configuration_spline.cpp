#include "configuration_spline.hpp"

namespace py = pybind11;
using drake::symbolic::Expression;


/*
 * PYBIND11 MODULE
 */


void init_submodule_configuration_spline(py::module_& m) {
	py::module_ q_spline =
		m.def_submodule("configuration_spline", "General configuration splines module.");

	// BlockType + readonly .type/.size on Block, and .type on BlockOffset:
	// lets Python code (e.g. po_goc_mpc's per-block max_vel/max_acc bounds
	// helper) introspect a spec's block composition instead of assuming/
	// guessing block order -- previously Block was write-only (static
	// constructors only) and BlockOffset was missing .type from its
	// otherwise-complete set of readonly fields.
	py::enum_<CubicConfigurationSpline::Block::Type>(q_spline, "BlockType")
		.value("R",       CubicConfigurationSpline::Block::Type::R)
		.value("Torus",   CubicConfigurationSpline::Block::Type::Torus)
		.value("SO3Quat", CubicConfigurationSpline::Block::Type::SO3Quat)
		.value("SO3Mat",  CubicConfigurationSpline::Block::Type::SO3Mat);

	// Expose Block + Spec (std::vector<Block>) so you can pass a spec from Python
	py::class_<CubicConfigurationSpline::Block>(q_spline, "Block")
		.def_static("R",        &CubicConfigurationSpline::Block::R,      py::arg("k"))
		.def_static("Torus",    &CubicConfigurationSpline::Block::Torus,  py::arg("k"))
		.def_static("SO3Quat",  &CubicConfigurationSpline::Block::SO3Quat)
		.def_static("SO3Mat",   &CubicConfigurationSpline::Block::SO3Mat)
		.def_readonly("type", &CubicConfigurationSpline::Block::type)
		.def_readonly("size", &CubicConfigurationSpline::Block::size);

	py::class_<CubicConfigurationSpline::BlockOffset>(q_spline, "BlockOffset")
		.def_readonly("ambient_offset", &CubicConfigurationSpline::BlockOffset::ambient_offset)
		.def_readonly("tangent_offset", &CubicConfigurationSpline::BlockOffset::tangent_offset)
		.def_readonly("ambient_size", &CubicConfigurationSpline::BlockOffset::ambient_size)
		.def_readonly("tangent_size", &CubicConfigurationSpline::BlockOffset::tangent_size)
		.def_readonly("type", &CubicConfigurationSpline::BlockOffset::type);

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
		.def("set_linear",    &CubicConfigurationSpline::set_linear,  py::arg("linear"))
		.def("is_linear",     &CubicConfigurationSpline::is_linear)
		// Per-(knot, block) knot mask consulted by the next set() -- a 0 at
		// (i, b) bridges block b straight across knot i (see
		// knot_block_active_'s own doc comment). Pass an empty array to
		// restore "every knot active".
		.def("set_block_active_mask", &CubicConfigurationSpline::set_block_active_mask, py::arg("mask"))
		.def("block_active_mask",     &CubicConfigurationSpline::block_active_mask)
		.def("eval_multiple", &CubicConfigurationSpline::eval_multiple)
		.def("ambient_dim",   &CubicConfigurationSpline::ambient_dim)
		.def("tangent_dim",   &CubicConfigurationSpline::tangent_dim)
		.def("position_delta",
		     [](const CubicConfigurationSpline& self,
			const Eigen::Ref<const Eigen::VectorXd>& xJ,
			const Eigen::Ref<const Eigen::VectorXd>& xJm1) {
			     return self.PositionDelta<double>(xJ, xJm1);
		     },
		     py::arg("xJ"),
		     py::arg("xJm1"))
		.def("retract",
		     [](const CubicConfigurationSpline& self,
			const Eigen::Ref<const Eigen::VectorXd>& xJm1,
			const Eigen::Ref<const Eigen::VectorXd>& delta) {
			     return self.Retract<double>(xJm1, delta);
		     },
		     py::arg("xJm1"),
		     py::arg("delta"))
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
