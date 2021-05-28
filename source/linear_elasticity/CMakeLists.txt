##
#  CMake script for the dealii-adapter:
##

# Set the name of the project and target:
SET(TARGET "linear_elasticity")

SET(_src
    ${TARGET}.cc
)

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

ADD_LIBRARY(${TARGET} OBJECT ${_src})
FIND_PACKAGE(precice REQUIRED)

DEAL_II_SETUP_TARGET(${TARGET})
