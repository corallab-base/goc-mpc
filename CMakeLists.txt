cmake_minimum_required(VERSION 3.17)
project(goc_mpc VERSION 0.0.1)
set(PY_VERSION_SUFFIX "")  # e.g. b3.dev1, see PEP 440
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find the pybind11 library
set(PYBIND11_USE_CROSSCOMPILING On) # https://github.com/pybind/pybind11/pull/5083
# Specifying version 2.13.6 to match the patched version of pybind11 used by drake
find_package(pybind11 2.13.6 REQUIRED)

#
# drake dependency
#

find_package(drake CONFIG REQUIRED)

# Point-cloud obstacle nearest-neighbor queries (ObstacleSet::
# query_point_cloud_radius, see obstacle_set.cpp) -- header-only, system-
# installed (libnanoflann-dev), not vendored.
find_package(nanoflann REQUIRED)

# qpOASES (robotpkg install, already on this machine's CMAKE_PREFIX_PATH via
# .envrc's PATH/openrobots setup) -- GraphTimingMPC's (graph_timing_mpc.cpp)
# QP subproblem solver in its trust-region Gauss-Newton outer loop: every QP
# subproblem there (PSD Gauss-Newton Hessian + exactly-linear constraints,
# see gn_timing::AssembleObjective/BuildProblemLayout) is solvable by
# construction, so a stalled/failed step degrades to a smaller trust region
# rather than a hard infeasibility/restoration-failure stop -- see
# graph_timing_mpc.hpp's own doc comment. No exported CMake target ships
# with this package, only plain qpOASES_INCLUDE_DIR/qpOASES_LIBRARY
# variables.
find_package(qpOASES REQUIRED)

#
# _ext module stuff
#

# Compile the example Python module
pybind11_add_module(_ext MODULE
    "src/ext/goc_mpc.cpp"
    "src/ext/splines.cpp"
    "src/ext/configuration_spline.cpp"
    "src/ext/graphs.cpp"
    "src/ext/utils.cpp"
    "src/ext/sec_mpc/timing_mpc_problem.cpp"
    "src/ext/goc_mpc/goc_mpc.cpp"
    "src/ext/goc_mpc/graph_of_constraints.cpp"
    "src/ext/goc_mpc/symbolic_constraint_compiler.cpp"
    "src/ext/goc_mpc/graph_waypoint_mpc.cpp"
    "src/ext/goc_mpc/milp_waypoint_mpc.cpp"
    "src/ext/goc_mpc/graph_timing_mpc.cpp"
    "src/ext/goc_mpc/timing_gn_layout.cpp"
    "src/ext/goc_mpc/graph_short_path_mpc.cpp"
    "src/ext/goc_mpc/obstacle_set.cpp"
    "src/ext/goc_mpc/admm_short_path_mpc.cpp"
    "src/ext/goc_mpc/sqp_short_path_layout.cpp"
    "src/ext/goc_mpc/sqp_short_path_mpc.cpp"
)
target_link_libraries(_ext PRIVATE drake::drake nanoflann::nanoflann ${qpOASES_LIBRARY})
target_include_directories(_ext PRIVATE ${qpOASES_INCLUDE_DIR})
target_link_directories(_ext PRIVATE ${qpOASES_LIBRARY_DIR})
target_compile_definitions(_ext PRIVATE
    MODULE_NAME=_ext
    VERSION_INFO="${PY_FULL_VERSION}"
)

# Install the Python module
set(PYTHON_MODULE_DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})
install(TARGETS _ext
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PYTHON_MODULE_DESTINATION})

#
# stubs stuff
#

# Generate stubs for the Python module
set(PY_STUBS_INSTALLDIR ""
    CACHE PATH "Installation directory for Python stubs (relative to Python package)")

set(WITH_PY_STUBS_DEFAULT On)

if (CMAKE_CROSSCOMPILING)
    set(WITH_PY_STUBS_DEFAULT Off)
endif()

option(WITH_PY_STUBS
    "Generate Python stub files (.pyi) for the Python module."
    ${WITH_PY_STUBS_DEFAULT})

if (WITH_PY_STUBS)
    include(cmake/Pybind11Stubgen.cmake)
    pybind11_stubgen(_add_module
        PACKAGE ${PYTHON_MODULE_DESTINATION}
        DESTINATION ${PY_STUBS_INSTALLDIR}
        COMPONENT python_stubs
    )
endif()
