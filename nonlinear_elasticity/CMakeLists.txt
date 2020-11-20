##
#  CMake script for the dealii-adapter:
##

# Set the name of the project and target:
SET(TARGET "nonlinear_elasticity")

SET(TARGET_SRC
  ${TARGET}.cc
  )

# To make the dimension switch available via cmake
IF (NOT DEFINED DIM)
        SET(DIM 2)
ENDIF()
ADD_DEFINITIONS(-DDIM=${DIM})

# Change default build type for tutorial cases
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release"
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)
MESSAGE(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# To cleanup result files
SET(CLEAN_UP_FILES
  # a custom list of globs, e.g. *.log *.vtk
  *.vtk
)
# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

FIND_PACKAGE(deal.II 9.2.0 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Store current source directory
SET(TMP ${CMAKE_SOURCE_DIR})
SET(CMAKE_SOURCE_DIR "..")

# Query git information
DEAL_II_QUERY_GIT_INFORMATION()

# Restore source dir
SET(CMAKE_SOURCE_DIR ${TMP})

PROJECT(${TARGET} LANGUAGES CXX)

DEAL_II_INVOKE_AUTOPILOT()

FIND_PACKAGE(precice REQUIRED)
TARGET_LINK_LIBRARIES(${TARGET} precice::precice)

SET_PROPERTY(TARGET  ${TARGET}  APPEND  PROPERTY  COMPILE_DEFINITIONS
            GIT_BRANCH="${GIT_BRANCH}"
            GIT_REVISION="${GIT_REVISION}"
            GIT_SHORTREV="${GIT_SHORTREV}")
