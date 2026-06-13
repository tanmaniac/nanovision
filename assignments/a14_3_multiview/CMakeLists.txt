# Builds one pybind11 extension per impl. models.cpp (the provided SE(3) retraction and
# pinhole helpers) is compiled in both; NV_IMPL selects which multiview source compiles:
#   student  (default) -> multiview.cpp (the holes)            -> module _a14_3_student
#   solution           -> solution/multiview.cpp (answer key)  -> module _a14_3_solution
# The conftest / _impl.py drives this; you do not call cmake by hand.
cmake_minimum_required(VERSION 3.18)
project(a14_3_multiview LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)

find_path(EIGEN3_INCLUDE_DIR
  NAMES Eigen/Dense
  PATHS $ENV{CONDA_PREFIX}/include/eigen3 /usr/include/eigen3 /usr/local/include/eigen3
  REQUIRED)

if(NOT DEFINED NV_IMPL)
  set(NV_IMPL student)
endif()

if(NV_IMPL STREQUAL "solution")
  set(MODNAME _a14_3_solution)
  set(IMPL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/solution/multiview.cpp)
else()
  set(MODNAME _a14_3_student)
  set(IMPL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/multiview.cpp)
endif()

pybind11_add_module(${MODNAME}
  ${IMPL_SRC}
  ${CMAKE_CURRENT_SOURCE_DIR}/models.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/bindings.cpp)
target_include_directories(${MODNAME} PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})
target_compile_definitions(${MODNAME} PRIVATE NV_MODULE_NAME=${MODNAME})
