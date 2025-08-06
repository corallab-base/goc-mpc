cmake_minimum_required(VERSION 3.17)
project(goc_mpc)

# Find the Python development files
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

# Add the module to compile
Python3_add_library(_goc_mpc_cpp MODULE "src/goc_mpc_cpp/example.c" WITH_SOABI)

# Install the module
install(TARGETS _goc_mpc_cpp
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})

# Install stubs to get autocomplete and type hints
install(FILES "src/goc_mpc_cpp/_example.pyi"
        EXCLUDE_FROM_ALL
        COMPONENT python_modules
        DESTINATION ${PY_BUILD_CMAKE_IMPORT_NAME})
