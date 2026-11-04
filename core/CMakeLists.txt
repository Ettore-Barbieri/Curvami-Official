# Curvami core: generic meshfree layer and nonlinear solvers, no plate physics.
#
# Reuse from another project (e.g. Curvami as a git submodule in extern/):
#
#     add_subdirectory(extern/Curvami/core curvami_core)
#     target_link_libraries(my_app PRIVATE Curvami::solver)   # or Curvami::core
#
# Headers are included with their prefix: #include "curvami/GlobalRKPM.h",
# #include "curvami/solver/ContinuationSolver.h".
cmake_minimum_required(VERSION 3.18)
project(curvami_core LANGUAGES CXX)

include(${CMAKE_CURRENT_LIST_DIR}/cmake/CurvamiDeps.cmake)

# Meshfree layer: Gmsh reader, quadrature, neighbour search, RKPM shape
# functions, discontinuity enrichment, restart files. Header-only.
add_library(curvami_core INTERFACE)
add_library(Curvami::core ALIAS curvami_core)
target_include_directories(curvami_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(curvami_core INTERFACE Eigen3::Eigen OpenMP::OpenMP_CXX)
target_compile_features(curvami_core INTERFACE cxx_std_17)

# Nonlinear solvers: Problem interface, LineSearch / ArcLength, persistent MUMPS
# KKT solver, constraint projector, step control. Separate so the meshfree layer
# does not pull in MUMPS.
add_library(curvami_solver INTERFACE)
add_library(Curvami::solver ALIAS curvami_solver)
target_include_directories(curvami_solver INTERFACE
    ${CURVAMI_MUMPS_INCLUDE_DIR}
    ${CURVAMI_MUMPS_MPISEQ_INCLUDE_DIR}
)
target_link_libraries(curvami_solver INTERFACE
    curvami_core
    ${CURVAMI_DMUMPS_LIB}
    ${CURVAMI_MUMPS_COMMON_LIB}
    ${CURVAMI_PORD_LIB}
    ${CURVAMI_MPISEQ_LIB}
)
