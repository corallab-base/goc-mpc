#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

using namespace pybind11::literals;
namespace py = pybind11;

py::array_t<double> remainder_slice_1d(const py::array_t<double>& arr, unsigned int i);
py::array_t<double> remainder_slice_2d(const py::array_t<double>& arr, unsigned int i);
py::array_t<double> integral(const py::array_t<double>& x);
py::array_t<double> prepend_1d(const py::array_t<double>& a, double v);
py::array_t<double> prepend_row_2d(const py::array_t<double>& a, const py::array_t<double>& row);
