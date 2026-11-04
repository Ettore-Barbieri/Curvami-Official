# Writes the build stamp header. Run at EVERY build (custom target in the top-level
# CMakeLists.txt), not at configure time: a stamp taken at configure time stayed at
# whatever commit the build folder was configured on, whatever was checked out later.
#
# Inputs: SRC (source tree), OUT (header to write).
# The stamp is the short commit hash, with "-dirty" when anything that goes into
# the binary differs from that commit: tracked changes or untracked files under
# main.cpp, CMakeLists.txt, cmake/, core/ and plate/. Changes elsewhere
# (Postprocessing/, docs/, Simulations/) do not make the binary dirty.
# The header is rewritten only when its content changes, so a build with an
# unchanged stamp recompiles nothing.

execute_process(COMMAND git rev-parse --short HEAD
                WORKING_DIRECTORY ${SRC}
                OUTPUT_VARIABLE stamp
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
execute_process(COMMAND git status --porcelain --untracked-files=all --
                        main.cpp CMakeLists.txt cmake core plate
                WORKING_DIRECTORY ${SRC}
                OUTPUT_VARIABLE dirty
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET)
if(NOT stamp)
    set(stamp "unknown")
elseif(dirty)
    set(stamp "${stamp}-dirty")
endif()

set(content "#pragma once\n// Generated at build time by cmake/BuildStamp.cmake. Do not edit.\n#define CURVAMI_GIT_COMMIT \"${stamp}\"\n")

set(old "")
if(EXISTS ${OUT})
    file(READ ${OUT} old)
endif()
if(NOT old STREQUAL content)
    file(WRITE ${OUT} "${content}")
    message(STATUS "Curvami build stamp: ${stamp}")
endif()
