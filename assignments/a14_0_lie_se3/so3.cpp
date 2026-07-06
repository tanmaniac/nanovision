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
  // Rotation vector -> rotation matrix (SO(3) exponential, Rodrigues). th = |w|; handle the
  // small-angle case (th < kEps) for numerical stability. See the README (exp/log).
  throw std::logic_error("NOT_IMPLEMENTED: so3_exp");
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  // Rotation matrix -> rotation vector (SO(3) log), the inverse of so3_exp. Handle the
  // th ~ 0 and th ~ pi cases (both make the naive form ill-conditioned). See the README (exp/log).
  throw std::logic_error("NOT_IMPLEMENTED: so3_log");
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  // SO(3) left Jacobian of the exponential map. Handle the small-angle case. See the README
  // (the Jacobians section).
  throw std::logic_error("NOT_IMPLEMENTED: so3_left_jacobian");
}

Eigen::Matrix3d so3_left_jacobian_inv(const Eigen::Vector3d& w) {
  // Inverse of so3_left_jacobian. Handle the small-angle case. See the README, Jacobians section.
  throw std::logic_error("NOT_IMPLEMENTED: so3_left_jacobian_inv");
}

Eigen::Matrix3d so3_right_jacobian(const Eigen::Vector3d& w) {
  // SO(3) right Jacobian of the exponential map (the module's downstream convention).
  // See the README (the Jacobians section).
  throw std::logic_error("NOT_IMPLEMENTED: so3_right_jacobian");
}

Eigen::Matrix3d so3_right_jacobian_inv(const Eigen::Vector3d& w) {
  // Inverse of the SO(3) right Jacobian. See the README (the Jacobians section).
  throw std::logic_error("NOT_IMPLEMENTED: so3_right_jacobian_inv");
}
