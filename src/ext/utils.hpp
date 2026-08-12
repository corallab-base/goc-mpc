#pragma once

#include <drake/solvers/mathematical_program.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "goc_mpc/graph_of_constraints.hpp"

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

template <typename T>
std::pair<Eigen::Matrix<T,3,1>, Eigen::Matrix<T,3,3>>
PoseFromRow_PosQuat(const Eigen::Matrix<T,Eigen::Dynamic,1>& row,
		   int robot_offset) {
	Eigen::Matrix<T,3,1> p_WE = row.template segment<3>(robot_offset + 0);
	const T& w = row(robot_offset + 3);
	const T& x = row(robot_offset + 4);
	const T& y = row(robot_offset + 5);
	const T& z = row(robot_offset + 6);
	Eigen::Matrix<T,3,3> R_WE = RotFromQuatBranchless<T>(w,x,y,z);
	return std::make_pair(p_WE, R_WE);
}

template <typename T>
std::pair<Eigen::Matrix<T,3,1>, Eigen::Matrix<T,3,3>>
PoseFromRow_PosRotMatrix(const Eigen::Matrix<T,Eigen::Dynamic,1>& row,
			int robot_offset) {
	Eigen::Matrix<T,3,1> p_WE = row.template segment<3>(robot_offset + 0);
	Eigen::Matrix<T,3,3> R_WE;
	R_WE << row(robot_offset + 3), row(robot_offset + 4), row(robot_offset + 5),
		row(robot_offset + 6), row(robot_offset + 7), row(robot_offset + 8),
		row(robot_offset + 9), row(robot_offset + 10), row(robot_offset + 11);
	return std::make_pair(p_WE, R_WE);
}

// `workspace_dim` (2 or 3) is the robot's ambient Cartesian workspace --
// see GraphOfConstraints::workspace_dim. Position is the row's leading
// workspace_dim-wide block; rotation is the workspace_dim x workspace_dim
// identity (a point mass has no orientation).
template <typename T>
std::pair<Eigen::Matrix<T,Eigen::Dynamic,1>, Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>>
PoseFromRow_PointMass(const Eigen::Matrix<T,Eigen::Dynamic,1>& row,
		      int robot_offset, int workspace_dim) {
	Eigen::Matrix<T,Eigen::Dynamic,1> p_WE = row.segment(robot_offset, workspace_dim);
	Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> R_WE =
		Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>::Zero(workspace_dim, workspace_dim);
	for (int i = 0; i < workspace_dim; ++i) R_WE(i, i) = T(1.0);
	return std::make_pair(p_WE, R_WE);
}

// State layout: [x, y, θ] (workspace_dim == 2, R² × T¹) or [x, y, z, θ]
// (workspace_dim == 3, R³ × T¹) -- yaw = rotation about the (2D) plane or
// (3D) world Z axis, immediately following the position block either way.
template <typename T>
std::pair<Eigen::Matrix<T,Eigen::Dynamic,1>, Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>>
PoseFromRow_PosYaw(const Eigen::Matrix<T,Eigen::Dynamic,1>& row,
		   int robot_offset, int workspace_dim) {
	using std::cos; using std::sin;
	DRAKE_DEMAND(workspace_dim == 2 || workspace_dim == 3);
	Eigen::Matrix<T,Eigen::Dynamic,1> p_WE = row.segment(robot_offset, workspace_dim);
	const T theta = row(robot_offset + workspace_dim);
	const T c = cos(theta);
	const T s = sin(theta);
	Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic> R_WE(workspace_dim, workspace_dim);
	if (workspace_dim == 2) {
		R_WE << c, -s,
			s,  c;
	} else {
		R_WE <<    c,  -s, T(0),
			   s,   c, T(0),
			T(0), T(0), T(1);
	}
	return std::make_pair(p_WE, R_WE);
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

// Returns a Vector3 regardless of the object's own declared width
// (graph->non_robot_dim) -- every caller of this (and of PoseFromRow, which
// this pairs with) expects a fixed 3D point. For a workspace_dim==3 graph
// (or any object whose non_robot_dim >= 3) this reads the same 3 components
// as before; for a genuinely 2D object (non_robot_dim < 3, e.g. a
// PyRoboGym-style planar item with no z/yaw of its own) it reads only the
// object's own real components and leaves the rest at 0 (a z==0 point in
// the workspace's plane), instead of reading past the end of that object's
// own slice in q into whatever happens to sit next in memory.
template <typename T>
Eigen::Vector3<T>
PointPosFromRow(const struct GraphOfConstraints* graph,
		const int point_index,
		const Eigen::Matrix<T,Eigen::Dynamic,1>& q) {
	const int point_pos_offset = graph->num_agents * graph->dim + point_index * graph->non_robot_dim;
	const int n = std::min(graph->non_robot_dim, 3);
	Eigen::Vector3<T> p_WC = Eigen::Vector3<T>::Zero();
	p_WC.segment(0, n) = q.segment(point_pos_offset, n);
	return p_WC;
}

// Return size follows graph->workspace_dim (2 or 3 -- see GraphOfConstraints'
// own doc comment on that member). kPointMass/kPosYaw are workspace_dim-aware
// (PoseFromRow_PointMass/PoseFromRow_PosYaw above); kPosQuat/kPosRotMat are
// inherently 3D rotation representations with no 2D reduction in this
// codebase, so they DEMAND workspace_dim == 3 rather than silently
// misbehave. Every caller (structured-binding `auto [p, R] = PoseFromRow(...)`
// throughout graph_of_constraints.cpp's constraint builders) is unaffected
// by the switch to a dynamic-size return: those callers only ever run
// against the default workspace_dim == 3, where the runtime values are
// identical to before, just held in a dynamic- rather than fixed-size
// Eigen type.
template <typename T>
std::pair<Eigen::Matrix<T,Eigen::Dynamic,1>, Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic>>
PoseFromRow(const struct GraphOfConstraints* graph,
	    const int agent_index,
	    const std::string ee_frame_name,
	    const Eigen::Matrix<T,Eigen::Dynamic,1>& q) {

	const int agent_config_offset = agent_index * graph->dim;
	const int workspace_dim = graph->workspace_dim;

	switch (graph->robot_kind(agent_index)) {
	case RobotKind::kPosQuat:
		if (workspace_dim != 3)
			throw std::runtime_error("PoseFromRow: kPosQuat requires workspace_dim == 3 "
						 "(quaternion rotation has no 2D reduction).");
		return PoseFromRow_PosQuat(q, agent_config_offset);
	case RobotKind::kPosRotMat:
		if (workspace_dim != 3)
			throw std::runtime_error("PoseFromRow: kPosRotMat requires workspace_dim == 3 "
						 "(3x3 rotation matrix has no 2D reduction).");
		return PoseFromRow_PosRotMatrix(q, agent_config_offset);
	case RobotKind::kPointMass:
		return PoseFromRow_PointMass(q, agent_config_offset, workspace_dim);
	case RobotKind::kPosYaw:
		return PoseFromRow_PosYaw(q, agent_config_offset, workspace_dim);
	case RobotKind::kArticulated:
		throw std::runtime_error("PoseFromRow: articulated robots require FK, not yet supported.");
	}
}
