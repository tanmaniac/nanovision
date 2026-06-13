// pybind11 wrapper exposing the EKF-SLAM functions (and the provided primitives) to
// Python. Provided, not a hole. std::pair returns become Python tuples (stl.h); Eigen
// types convert to/from numpy (eigen.h). Module name set by CMake via NV_MODULE_NAME.
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ekf_slam.hpp"

namespace py = pybind11;

PYBIND11_MODULE(NV_MODULE_NAME, m) {
  m.doc() = "a14_2 EKF-SLAM: joint robot+map filter, augmentation, data association";

  m.def("wrap_angle", &wrap_angle);

  // Provided primitives (models.cpp).
  m.def("robot_f", &robot_f);
  m.def("robot_F_x", &robot_F_x);
  m.def("range_bearing", &range_bearing);
  m.def("range_bearing_H_robot", &range_bearing_H_robot);
  m.def("range_bearing_H_land", &range_bearing_H_land);

  // The holes (ekf_slam.cpp).
  m.def("slam_predict", &slam_predict);
  m.def("slam_add_landmark", &slam_add_landmark);
  m.def("slam_update", &slam_update);
  m.def("slam_associate", &slam_associate);
}
