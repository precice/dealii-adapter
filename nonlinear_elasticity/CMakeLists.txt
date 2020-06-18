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

PROJECT(${TARGET} LANGUAGES CXX)

DEAL_II_INVOKE_AUTOPILOT()

FIND_PACKAGE(precice REQUIRED)
TARGET_LINK_LIBRARIES(${TARGET} precice::precice)
