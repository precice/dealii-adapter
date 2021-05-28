##
#  CMake script for the dealii-adapter:
##

# Set the name of the project and target:
SET(TARGET "nonlinear_elasticity")

SET(_src
    ${TARGET}.cc
)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8.12)

ADD_LIBRARY(${TARGET} OBJECT ${_src})
FIND_PACKAGE(precice REQUIRED)

DEAL_II_SETUP_TARGET(${TARGET})
