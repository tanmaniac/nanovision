# Builds one pybind11 extension per impl. NV_IMPL selects which C++ sources compile:
#   student  (default) -> top-level so3.cpp/se3.cpp (the holes)    -> module _a14_0_student
#   solution           -> solution/so3.cpp/se3.cpp (the answer key) -> module _a14_0_solution
# The conftest / _impl.py drives this; you do not call cmake by hand.
cmake_minimum_required(VERSION 3.18)
project(a14_0_lie_se3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Eigen is header-only; find its include dir (conda env first, then system).
find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Dense
  PATHS $ENV{CONDA_PREFIX}/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3
  REQUIRED)

if(NOT DEFINED NV_IMPL)
  set(NV_IMPL student)
endif()

if(NV_IMPL STREQUAL "solution")
  set(MODNAME _a14_0_solution)
  set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/solution/so3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/solution/se3.cpp)
else()
  set(MODNAME _a14_0_student)
  set(SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/so3.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/se3.cpp)
endif()

pybind11_add_module(${MODNAME} ${SRC} ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp)
target_include_directories(${MODNAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(${MODNAME} PRIVATE NV_MODULE_NAME=${MODNAME})
