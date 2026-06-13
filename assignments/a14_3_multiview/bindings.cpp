// pybind11 wrapper exposing the multi-view estimators (and the provided primitives) to
// Python. Provided, not a hole. std::pair/std::tuple returns become Python tuples (stl.h);
// std::vector<int> becomes a list; Eigen types convert to/from numpy (eigen.h). Module name
// set by CMake via NV_MODULE_NAME.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "multiview.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_3 multi-view geometry: triangulation, eight-point, PnP, RANSAC, front-end";

  // Provided primitives (models.cpp).
  m.def("hat3", &hat3);
  m.def("se3_exp", &se3_exp);
  m.def("pinhole_project", &pinhole_project);
  m.def("pinhole_jacobian", &pinhole_jacobian);

  // The holes (multiview.cpp).
  m.def("triangulate_dlt", &triangulate_dlt);
  m.def("triangulate_refine", &triangulate_refine);
  m.def("eight_point", &eight_point);
  m.def("decompose_essential", &decompose_essential);
  m.def("recover_pose", &recover_pose);
  m.def("pnp_dlt", &pnp_dlt);
  m.def("pnp_refine", &pnp_refine);
  m.def("sampson_distance", &sampson_distance);
  m.def("ransac_fundamental", &ransac_fundamental);
  m.def("two_view_relative_pose", &two_view_relative_pose);
}
