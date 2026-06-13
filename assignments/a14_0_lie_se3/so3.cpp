// SO(3) - YOUR CODE. Fill each hole; read the contract in the comment, the math is
// in the README. Run `make test A=a14_0_lie_se3`. The reference is in solution/so3.cpp.
#include "so3.hpp"

#include <cmath>
#include <stdexcept>

namespace {
constexpr double kEps = 1e-8;  // small-angle threshold for the Taylor branches
}  // namespace

Eigen::Matrix3d hat3(const Eigen::Vector3d& w) {
  // Return the 3x3 skew-symmetric matrix [w]_x such that [w]_x v = w x v.
  throw std::logic_error("NOT_IMPLEMENTED: hat3");
}

Eigen::Vector3d vee3(const Eigen::Matrix3d& W) {
  // Inverse of hat3: read the axis vector back out of the skew matrix.
  throw std::logic_error("NOT_IMPLEMENTED: vee3");
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  // Rodrigues: R = I + (sin th / th) K + ((1 - cos th)/th^2) K^2, K = hat3(w),
  // th = |w|. Use the small-angle Taylor branch (I + K + K^2/2) when th < kEps.
  throw std::logic_error("NOT_IMPLEMENTED: so3_exp");
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  // Inverse of so3_exp. th = acos((tr R - 1)/2); w = (th / 2 sin th) vee(R - R^T).
  // Handle th ~ 0 (w = vee(R - R^T)/2) and th ~ pi (extract axis from R + I).
  throw std::logic_error("NOT_IMPLEMENTED: so3_log");
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  // J_l = I + ((1 - cos th)/th^2) K + ((th - sin th)/th^3) K^2, K = hat3(w).
  // Small-angle branch: I + K/2 + K^2/6.
  throw std::logic_error("NOT_IMPLEMENTED: so3_left_jacobian");
}

Eigen::Matrix3d so3_left_jacobian_inv(const Eigen::Vector3d& w) {
  // J_l^-1 = I - K/2 + (1/th^2)(1 - (th/2) cot(th/2)) K^2.
  // Small-angle branch: I - K/2 + K^2/12.
  throw std::logic_error("NOT_IMPLEMENTED: so3_left_jacobian_inv");
}

Eigen::Matrix3d so3_right_jacobian(const Eigen::Vector3d& w) {
  // J_r(w) = J_l(-w). One line once so3_left_jacobian is done.
  throw std::logic_error("NOT_IMPLEMENTED: so3_right_jacobian");
}

Eigen::Matrix3d so3_right_jacobian_inv(const Eigen::Vector3d& w) {
  // J_r^-1(w) = J_l^-1(-w).
  throw std::logic_error("NOT_IMPLEMENTED: so3_right_jacobian_inv");
}
