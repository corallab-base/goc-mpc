cmake_minimum_required(VERSION 3.17)
project(goc_mpc VERSION 0.0.1)
set(PY_VERSION_SUFFIX "")  # e.g. b3.dev1, see PEP 440
set(PY_FULL_VERSION ${PROJECT_VERSION}${PY_VERSION_SUFFIX})

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Find the pybind11 library
set(PYBIND11_USE_CROSSCOMPILING On) # https://github.com/pybind/pybind11/pull/5083
find_package(pybind11 REQUIRED)

#
# drake dependency
#

find_package(drake CONFIG REQUIRED)

#
# _ext module stuff
#

# Compile the example Python module
pybind11_add_module(_ext MODULE
    "src/ext/goc_mpc.cpp"
    "src/ext/splines.cpp"
    "src/ext/graphs.cpp"
    "src/ext/utils.cpp"
    "src/ext/sec_mpc/timing_mpc_problem.cpp"
    "src/ext/goc_mpc/goc_mpc.cpp"
    "src/ext/goc_mpc/graph_of_constraints.cpp"
    "src/ext/goc_mpc/graph_timing_mpc.cpp"
    "src/ext/goc_mpc/graph_waypoint_mpc.cpp"
    "src/ext/goc_mpc/graph_short_path_mpc.cpp"
)
target_link_libraries(_ext PRIVATE drake::drake)
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
