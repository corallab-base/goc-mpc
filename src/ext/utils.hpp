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

template <typename T>
Eigen::Vector3<T>
CubePosFromRow(const struct GraphOfConstraints* graph,
	       const int cube_index,
	       const Eigen::Matrix<T,Eigen::Dynamic,1>& q) {
	const int cube_pos_offset = graph->num_agents * graph->dim + cube_index * graph->non_robot_dim;
	return q.segment(cube_pos_offset, 3);
}

template <typename T>
std::pair<Eigen::Matrix<T,3,1>, Eigen::Matrix<T,3,3>>
PoseFromRow(const struct GraphOfConstraints* graph,
	    const int agent_index,
	    const std::string ee_frame_name,
	    const Eigen::Matrix<T,Eigen::Dynamic,1>& q) {

	const int agent_config_offset = agent_index * graph->dim;

	if (graph->robot_is_pos_quat(agent_index)) {
		return PoseFromRow_PosQuat(q, agent_config_offset);
	} else if (graph->robot_is_pos_rot_mat(agent_index)) {
		return PoseFromRow_PosRotMatrix(q, agent_config_offset);
	} else {
		// const MultibodyPlant<T> *plant;

		// if constexpr (std::is_same_v<T, Expression>) {
		// 	plant = graph->_plant.get();
		// } else {
		// 	plant = graph->_double_plant.get();
		// }

		// auto ctx = plant->CreateDefaultContext();
		// graph->set_configuration(ctx, q);

		// const auto& W = plant->world_frame();
		// const auto& E = plant->GetFrameByName(
		// 	ee_frame_name,
		// 	plant->GetModelInstanceByName(
		// 		graph->_robot_names.at(agent_index)));

		// const auto X_WE = plant->CalcRelativeTransform(*ctx, W, E);

		// return std::make_pair(X_WE.translation(), X_WE.rotation().matrix());
		throw std::runtime_error("Only supporting 'pos_quat' and 'pos_rot_mat' robots.");
	}
}
