// pybind11 wrapper exposing the SO(3)/SE(3) functions to Python. Provided, not a hole.
// The module name (_a14_0_student or _a14_0_solution) is set by CMake via NV_MODULE_NAME,
// so the student and solution builds produce distinct importable modules.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "se3.hpp"
#include "so3.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_0 Lie groups: SO(3)/SE(3) exp/log, Jacobians, adjoint, box-plus";

  m.def("hat3", &hat3);
  m.def("vee3", &vee3);
  m.def("so3_exp", &so3_exp);
  m.def("so3_log", &so3_log);
  m.def("so3_left_jacobian", &so3_left_jacobian);
  m.def("so3_left_jacobian_inv", &so3_left_jacobian_inv);
  m.def("so3_right_jacobian", &so3_right_jacobian);
  m.def("so3_right_jacobian_inv", &so3_right_jacobian_inv);

  m.def("hat6", &hat6);
  m.def("vee6", &vee6);
  m.def("se3_exp", &se3_exp);
  m.def("se3_log", &se3_log);
  m.def("se3_adjoint", &se3_adjoint);
  m.def("se3_boxplus", &se3_boxplus);
  m.def("se3_boxminus", &se3_boxminus);
}
