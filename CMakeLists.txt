cmake_minimum_required(VERSION 3.18)
project(Curvami_KL_CXX LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Eigen, OpenMP and sequential MUMPS: the core's own dependency module, so the
# plate solver and a downstream user of core/ find them the same way.
include(${CMAKE_CURRENT_SOURCE_DIR}/core/cmake/CurvamiDeps.cmake)

# Plate-solver dependencies. Defaults are the Homebrew layout; override with -D.
set(HDF5_ROOT "/opt/homebrew/opt/hdf5" CACHE PATH "HDF5 prefix")
set(CURVAMI_SUITESPARSE_ROOT "/opt/homebrew/opt/suite-sparse" CACHE PATH "SuiteSparse prefix (UMFPACK)")
set(CURVAMI_EXTRA_PREFIX "/opt/homebrew" CACHE PATH
    "General package prefix: its include/, include/eigen3 and lib/ are searched")

find_package(HDF5 REQUIRED COMPONENTS CXX)

# ==========================================
# Locate Apple's Accelerate Framework
# ==========================================
if(APPLE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
endif()

# Stamp the binary with the commit it was built from, so every .h5 it writes can
# be checked against the build. A stale output from a pre-fix binary produced a
# convincing false diagnosis once already (the straight control "failure"), and
# a skip-if-exists run loop makes that easy to repeat.
#
# The stamp is taken at every build, not at configure time: a configure-time stamp
# stayed at the commit the build folder was configured on (build/ stamped 0e51a8c
# long after). cmake/BuildStamp.cmake rewrites the header only when the stamp
# changes, so an unchanged tree recompiles nothing.
set(CURVAMI_STAMP_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/curvami_build_stamp.h)
add_custom_target(curvami_build_stamp ALL
    COMMAND ${CMAKE_COMMAND} -DSRC=${CMAKE_CURRENT_SOURCE_DIR} -DOUT=${CURVAMI_STAMP_HEADER}
            -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildStamp.cmake
    BYPRODUCTS ${CURVAMI_STAMP_HEADER}
    COMMENT "Checking the build stamp"
    VERBATIM)

# Generic core (curvami_core, curvami_solver): see core/CMakeLists.txt.
add_subdirectory(core)

add_executable(solver
    main.cpp
    core/include/curvami/MeshfreeData.h
    core/include/curvami/FRNN.h
    core/include/curvami/GmshReader.h
    core/include/curvami/Quadrature.h
    core/include/curvami/Enrichment.h
    core/include/curvami/MeshUtils.h
    core/include/curvami/GlobalRKPM.h
    core/include/curvami/RestartManager.h
    core/include/curvami/solver/MumpsSymmetricSolver.h
    core/include/curvami/solver/ConstraintProjector.h
    core/include/curvami/solver/StepController.h
    core/include/curvami/solver/Problem.h
    core/include/curvami/solver/ContinuationSolver.h
    plate/PlateData.h
    plate/PlateProblem.h
    plate/Kinematics.h
    plate/Constitutive.h
    plate/Assembly.h
    plate/Constraints.h
    plate/Meshless.h
    plate/NonLinearSolver.h
    plate/Exporter.h
    plate/FoldAngleUtil.h
    plate/SetupHelper.h
)

# The MUMPS include dirs repeat curvami_solver's on purpose, to keep them ahead of
# ${CURVAMI_EXTRA_PREFIX}/include: on Homebrew that directory also holds Open MPI's
# mpi.h, which would shadow the sequential MUMPS stub if anything included it.
target_include_directories(solver PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/plate
    ${HDF5_INCLUDE_DIRS}
    ${CURVAMI_MUMPS_INCLUDE_DIR}
    ${CURVAMI_MUMPS_MPISEQ_INCLUDE_DIR}
    ${CURVAMI_EXTRA_PREFIX}/include
    ${CURVAMI_EXTRA_PREFIX}/include/eigen3
    ${CURVAMI_EXTRA_PREFIX}/include/eigen3/unsupported
    ${CURVAMI_SUITESPARSE_ROOT}/include
    ${CURVAMI_SUITESPARSE_ROOT}/include/suitesparse
)

target_link_directories(solver PRIVATE
    ${CURVAMI_EXTRA_PREFIX}/lib
    ${CURVAMI_SUITESPARSE_ROOT}/lib
)

target_link_libraries(solver PRIVATE
    curvami_solver
    curvami_core
    Eigen3::Eigen
    OpenMP::OpenMP_CXX
    ${HDF5_CXX_LIBRARIES}
    ${CURVAMI_DMUMPS_LIB}
    ${CURVAMI_MUMPS_COMMON_LIB}
    ${CURVAMI_PORD_LIB}
    ${CURVAMI_MPISEQ_LIB}
    umfpack
    amd
    cholmod
    colamd
    suitesparseconfig
)

if(APPLE)
    target_link_libraries(solver PRIVATE ${ACCELERATE_FRAMEWORK})
endif()

add_dependencies(solver curvami_build_stamp)
target_include_directories(solver PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/generated)

target_compile_definitions(solver PRIVATE
    NDEBUG
    EIGEN_NO_DEBUG
    EIGEN_USE_BLAS
)

target_compile_options(solver PRIVATE
    $<$<CONFIG:Release>:-O2>
)

# Set architecture target (Change to apple-m1 or apple-m2 if needed)
if(CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    target_compile_options(solver PRIVATE $<$<CONFIG:Release>:-mcpu=apple-m4>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(solver PRIVATE $<$<CONFIG:Release>:-march=native>)
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported)
    set_property(TARGET solver PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()