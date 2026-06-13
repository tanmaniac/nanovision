// pybind11 wrapper exposing the factor-graph functions (and the provided SE(3) primitives) to
// Python. Provided, not a hole. std::pair/std::tuple returns become Python tuples and
// std::vector becomes a list (stl.h); Eigen types convert to/from numpy (eigen.h). Module name
// set by CMake via NV_MODULE_NAME.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "factor_graph.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_5 factor graphs: pose-graph residuals/Jacobians, GN, and the BA Schur complement";

  // Provided primitives (models.cpp).
  m.def("hat3", &hat3);
  m.def("se3_exp", &se3_exp);
  m.def("se3_log", &se3_log);
  m.def("se3_adjoint", &se3_adjoint);
  m.def("se3_right_jacobian_inv", &se3_right_jacobian_inv);

  // The holes (factor_graph.cpp).
  m.def("between_residual", &between_residual);
  m.def("between_jacobians", &between_jacobians);
  m.def("optimize_pose_graph", &optimize_pose_graph);
  m.def("schur_solve", &schur_solve);
}
