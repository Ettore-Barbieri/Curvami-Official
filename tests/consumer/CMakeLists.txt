# Minimal downstream project: proves the Curvami core builds and runs through
# add_subdirectory, with no plate code, no HDF5 and no Curvami top level.
#
#     cmake -S tests/consumer -B build/consumer && cmake --build build/consumer
#     build/consumer/consumer
cmake_minimum_required(VERSION 3.18)
project(curvami_consumer LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# A real consumer would point at its submodule, e.g. extern/Curvami/core.
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../core curvami_core)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE Curvami::solver)
