// pybind11 wrapper exposing the ICP functions (and the provided primitives) to Python.
// Provided, not a hole. std::pair/std::tuple returns become Python tuples and std::vector<int>
// a list (stl.h); std::string converts too; Eigen types convert to/from numpy (eigen.h).
// Module name set by CMake via NV_MODULE_NAME.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "icp.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_4 ICP: point-to-point (Umeyama), point-to-plane, and the outer loop";

  // Provided primitives (models.cpp).
  m.def("hat3", &hat3);
  m.def("se3_exp", &se3_exp);
  m.def("nearest_neighbors", &nearest_neighbors);

  // The holes (icp.cpp).
  m.def("align_point_to_point", &align_point_to_point);
  m.def("point_to_plane_step", &point_to_plane_step);
  m.def("icp", &icp);
}
