/*  ------------------------------------------------------------------
    Copyright (c) 2025 Anastasios Manganaris
    --------------------------------------------------------------  */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void init_submodule_splines(py::module_&);
void init_submodule_graphs(py::module_&);
// void init_submodule_sec_mpc(py::module_&);
void init_submodule_goc_mpc(py::module_&);

PYBIND11_MODULE(_goc_mpc_cpp, m) {
    m.doc() = "Main GoC-MPC CPP library module.";
    init_submodule_splines(m);
    init_submodule_graphs(m);
    // init_submodule_sec_mpc(m);
    init_submodule_goc_mpc(m);
}
