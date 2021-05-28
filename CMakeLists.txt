CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.1 QUIET
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

# Enable a switchable dimension choice
IF (NOT DEFINED DIM)
        SET(DIM 2)
ENDIF()
ADD_DEFINITIONS(-DDIM=${DIM})

# Set the target and the target source
SET( TARGET "elasticity" )
SET( TARGET_SRC ${TARGET}.cc include/adapter/parameters.cc)

DEAL_II_INITIALIZE_CACHED_VARIABLES()

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "RELEASE")
  MESSAGE(STATUS "No build type specified. Building in ${CMAKE_BUILD_TYPE} mode.")
ENDIF()

# Set the include directory and the name of the project
PROJECT(${TARGET})
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/include)

# Set the executable
ADD_EXECUTABLE(${TARGET} ${TARGET_SRC})

# Use deal.II macros for setting up the target
DEAL_II_SETUP_TARGET(${TARGET})
DEAL_II_INITIALIZE_CACHED_VARIABLES()

# Query the git information and set it in the source
DEAL_II_QUERY_GIT_INFORMATION()
SET_PROPERTY(TARGET  ${TARGET}  APPEND  PROPERTY  COMPILE_DEFINITIONS
            GIT_BRANCH="${GIT_BRANCH}"
            GIT_REVISION="${GIT_REVISION}"
            GIT_SHORTREV="${GIT_SHORTREV}")


FIND_PACKAGE(precice REQUIRED
    HINTS ${PRECICE_DIR} $ENV{precice_DIR})
TARGET_LINK_LIBRARIES(${TARGET} precice::precice)

MESSAGE(STATUS "Using the preCICE version found at ${precice_CONFIG}")

# add the individual solver
ADD_SUBDIRECTORY(source)

# ...and link them
TARGET_LINK_LIBRARIES(${TARGET}  nonlinear_elasticity linear_elasticity heat_equation)


#
# Custom "debug" and "release" make targets:
#
ADD_CUSTOM_TARGET(debug
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
)

ADD_CUSTOM_TARGET(release
COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
COMMENT "Switch CMAKE_BUILD_TYPE to Release"
)
