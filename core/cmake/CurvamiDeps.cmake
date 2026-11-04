# Dependencies of the Curvami core: Eigen, OpenMP, sequential MUMPS.
#
# Included by core/CMakeLists.txt, and by the Curvami top level before it adds
# core/. The guard makes the second include a no-op; imported targets and cache
# variables found by the first include are visible to the second.
#
# Every location is a cache option whose default is the Homebrew layout this was
# developed on; override with -D on another machine.
include_guard(GLOBAL)

set(CURVAMI_LIBOMP_ROOT "/opt/homebrew/opt/libomp" CACHE PATH
    "libomp prefix (used with AppleClang only, which ships no OpenMP runtime)")
set(CURVAMI_MUMPS_ROOT "/opt/homebrew/opt/mumps-seq" CACHE PATH
    "Sequential MUMPS prefix (include/ and lib/)")

if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

if(NOT TARGET OpenMP::OpenMP_CXX)
    # AppleClang needs explicit paths to libomp.
    if(APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I${CURVAMI_LIBOMP_ROOT}/include")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${CURVAMI_LIBOMP_ROOT}/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${CURVAMI_LIBOMP_ROOT}/lib/libomp.dylib)
    endif()
    find_package(OpenMP REQUIRED)
endif()

# MUMPS: HINTS, not PATHS. PATHS is searched after the system prefixes, and on
# Homebrew /opt/homebrew/include holds Open MPI's mpi.h: the old PATHS search for
# the sequential mpi.h stub silently returned that directory instead.
find_path(CURVAMI_MUMPS_INCLUDE_DIR NAMES dmumps_c.h
    HINTS
    ${CURVAMI_MUMPS_ROOT}/include/coin-or/mumps
    ${CURVAMI_MUMPS_ROOT}/include/mumps
    ${CURVAMI_MUMPS_ROOT}/include
    REQUIRED
)
find_path(CURVAMI_MUMPS_MPISEQ_INCLUDE_DIR NAMES mpi.h
    HINTS
    ${CURVAMI_MUMPS_ROOT}/include/coin-or/mumps
    ${CURVAMI_MUMPS_ROOT}/include/mumps_seq
    ${CURVAMI_MUMPS_ROOT}/include
    NO_DEFAULT_PATH
    REQUIRED
)
find_library(CURVAMI_DMUMPS_LIB       NAMES dmumps coinmumps       HINTS ${CURVAMI_MUMPS_ROOT}/lib REQUIRED)
find_library(CURVAMI_MUMPS_COMMON_LIB NAMES mumps_common coinmumps HINTS ${CURVAMI_MUMPS_ROOT}/lib REQUIRED)
find_library(CURVAMI_PORD_LIB         NAMES pord coinmumps         HINTS ${CURVAMI_MUMPS_ROOT}/lib REQUIRED)
find_library(CURVAMI_MPISEQ_LIB       NAMES mpiseq coinmumps       HINTS ${CURVAMI_MUMPS_ROOT}/lib REQUIRED)
