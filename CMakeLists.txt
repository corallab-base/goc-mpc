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

# ProxQP (proxsuite, robotpkg/openrobots install, on CMAKE_PREFIX_PATH via
# .envrc) -- the sparse QP subproblem solver in GraphShortPathMPC's
# (graph_short_path_mpc.cpp) and GraphTimingMPC's (graph_timing_mpc.cpp)
# trust-region SQP outer loops. H is block-diagonal per agent and every
# constraint row is sparse, so a sparse solver scales with agent count.
# Header-only (proxsuite::proxsuite is an INTERFACE target); needs simde,
# which its own package config pulls in.
find_package(proxsuite REQUIRED)

# SCIP (SCIP Optimization Suite built from source into
# ~/phd/software/scip-install, on CMAKE_PREFIX_PATH via .envrc) --
# SCIPWaypointMPC's (scip_waypoint_mpc.cpp) branch-and-bound solver.
find_package(SCIP CONFIG REQUIRED)

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
    "src/ext/goc_mpc/waypoint_common.cpp"
    "src/ext/goc_mpc/milp_waypoint_mpc.cpp"
    "src/ext/goc_mpc/gcs_waypoint_mpc.cpp"
    "src/ext/goc_mpc/scip_waypoint_mpc.cpp"
    "src/ext/goc_mpc/expression_tape.cpp"
    "src/ext/goc_mpc/local_nlp_solver.cpp"
    "src/ext/goc_mpc/graph_timing_mpc.cpp"
    "src/ext/goc_mpc/timing_gn_layout.cpp"
    "src/ext/goc_mpc/obstacle_set.cpp"
    "src/ext/goc_mpc/agent_collision_model.cpp"
    "src/ext/goc_mpc/sqp_short_path_layout.cpp"
    "src/ext/goc_mpc/graph_short_path_mpc.cpp"
)
target_link_libraries(_ext PRIVATE drake::drake nanoflann::nanoflann proxsuite::proxsuite ${SCIP_LIBRARIES})
target_include_directories(_ext PRIVATE ${SCIP_INCLUDE_DIRS})
# Find libscip at runtime without LD_LIBRARY_PATH (e.g. from other venvs).
get_filename_component(SCIP_LIBRARY_DIR "${SCIP_DIR}/../.." ABSOLUTE)
set_target_properties(_ext PROPERTIES BUILD_RPATH "${SCIP_LIBRARY_DIR}" INSTALL_RPATH "${SCIP_LIBRARY_DIR}")
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
