#pragma once

#include <drake/solvers/mathematical_program.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using drake::symbolic::Expression;
using drake::symbolic::Variable;
using namespace pybind11::literals;
namespace py = pybind11;

py::array_t<double> remainder_slice_1d(const py::array_t<double>& arr, unsigned int i);
py::array_t<double> remainder_slice_2d(const py::array_t<double>& arr, unsigned int i);
py::array_t<double> integral(const py::array_t<double>& x);
py::array_t<double> prepend_1d(const py::array_t<double>& a, double v);
py::array_t<double> prepend_row_2d(const py::array_t<double>& a, const py::array_t<double>& row);


// Branchless quaternion-to-rotation matrix.
// Works for any scalar type T (double, AutoDiffXd, symbolic::Expression, …).
template <typename T>
Eigen::Matrix<T,3,3>
RotFromQuatBranchless(const T& w,
                      const T& x,
                      const T& y,
                      const T& z) {
	// We assume caller enforces unit quaternion (|q| = 1).
	const T n2 = static_cast<T>(1);   // = w*w + x*x + y*y + z*z
	const T s2 = static_cast<T>(2);   // = 2 / n2
	// const T n2 = w*w + x*x + y*y + z*z;
	// const T s2 = 2.0 / n2;

	Eigen::Matrix<T,3,3> R;
	R(0,0) = 1 - s2*(y*y + z*z);
	R(0,1) =     s2*(x*y - w*z);
	R(0,2) =     s2*(x*z + w*y);

	R(1,0) =     s2*(x*y + w*z);
	R(1,1) = 1 - s2*(x*x + z*z);
	R(1,2) =     s2*(y*z - w*x);

	R(2,0) =     s2*(x*z - w*y);
	R(2,1) =     s2*(y*z + w*x);
	R(2,2) = 1 - s2*(x*x + y*y);

	return R;
}

// Extract pose (position + quaternion rotation) from a row vector.
// `row` must contain at least 7 elements starting at `robot_offset`:
// [px, py, pz, qw, qx, qy, qz].
template <typename T>
void PoseFromRow_FreeBody(const Eigen::Matrix<T,Eigen::Dynamic,1>& row,
                          int robot_offset,
                          Eigen::Matrix<T,3,1>* p_WE,
                          Eigen::Matrix<T,3,3>* R_WE) {
	*p_WE = row.template segment<3>(robot_offset + 0);
	const T& w = row(robot_offset + 3);
	const T& x = row(robot_offset + 4);
	const T& y = row(robot_offset + 5);
	const T& z = row(robot_offset + 6);
	*R_WE = RotFromQuatBranchless<T>(w,x,y,z);
}

// Templated row→Expression helper (safe for Eigen blocks).
template <class Derived>
Eigen::RowVectorX<Expression> AsExprRow(const Eigen::MatrixBase<Derived>& row) {
	return row.template cast<Expression>();
}

// World position of a "point" from a row of X (first 3 coords of the object block).
template <class DerivedRow>
Eigen::Vector3<Expression> PointWorldFromRow(
	const Eigen::MatrixBase<DerivedRow>& row_in,
	int objs_start, int non_robot_dim, int obj_id) {

	Eigen::RowVectorX<Expression> row = AsExprRow(row_in);
	return row.segment(objs_start + obj_id * non_robot_dim, 3).transpose();
}

// World position of a "point" from x0 (as Expression).
Eigen::Vector3<Expression> PointWorldFromX0(
	const Eigen::VectorXd& x0,
	int objs_start, int non_robot_dim, int obj_id);
