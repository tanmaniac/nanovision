// pybind11 wrapper exposing the Gaussian-filter functions to Python. Provided, not
// a hole. std::pair / std::tuple returns become Python tuples (needs stl.h); Eigen
// types convert to/from numpy (needs eigen.h). The module name (_a14_1_student or
// _a14_1_solution) is set by CMake via NV_MODULE_NAME.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "kalman.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_1 Gaussian filters: linear KF, EKF, UKF, and the information form";

  m.def("wrap_angle", &wrap_angle);

  m.def("kf_predict", &kf_predict);
  m.def("kf_update", &kf_update);

  m.def("ekf_f", &ekf_f);
  m.def("ekf_F_x", &ekf_F_x);
  m.def("ekf_h", &ekf_h);
  m.def("ekf_H", &ekf_H);
  m.def("ekf_predict", &ekf_predict);
  m.def("ekf_update", &ekf_update);

  m.def("ukf_sigma_points", &ukf_sigma_points);
  m.def("ukf_unscented_transform", &ukf_unscented_transform);
  m.def("ukf_cross_covariance", &ukf_cross_covariance);

  m.def("moments_to_information", &moments_to_information);
  m.def("information_to_moments", &information_to_moments);
  m.def("information_update", &information_update);
}
