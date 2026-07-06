// SE(3) - YOUR CODE. Fill each hole; the twist ordering is xi = [rho; theta]
// (translation first). Reference in solution/se3.cpp.
#include "se3.hpp"

#include "so3.hpp"

#include <stdexcept>

Eigen::Matrix4d hat6(const Vector6d& xi) {
  // xi = [rho; theta]. Return [[hat3(theta), rho],[0, 0, 0, 0]].
  throw std::logic_error("NOT_IMPLEMENTED: hat6");
}

Vector6d vee6(const Eigen::Matrix4d& Xi) {
  // Inverse of hat6: rho from the top-right column, theta = vee3 of the top-left block.
  throw std::logic_error("NOT_IMPLEMENTED: vee6");
}

Eigen::Matrix4d se3_exp(const Vector6d& xi) {
  // Twist -> rigid transform (SE(3) exponential). xi = [rho; theta]; the translation is not
  // just rho - it couples through the rotation. See the README (exp/log).
  throw std::logic_error("NOT_IMPLEMENTED: se3_exp");
}

Vector6d se3_log(const Eigen::Matrix4d& T) {
  // Rigid transform -> twist (SE(3) log), the inverse of se3_exp. Return [rho; theta].
  // See the README (exp/log).
  throw std::logic_error("NOT_IMPLEMENTED: se3_log");
}

Matrix6d se3_adjoint(const Eigen::Matrix4d& T) {
  // SE(3) adjoint Ad_T: remaps a twist between frames, defined by T exp(xi) = exp(Ad_T xi) T,
  // for the [rho; theta] ordering. See the README (the adjoint).
  throw std::logic_error("NOT_IMPLEMENTED: se3_adjoint");
}

Eigen::Matrix4d se3_boxplus(const Eigen::Matrix4d& T, const Vector6d& xi) {
  // Retraction: apply the tangent perturbation xi to T in the right (body) frame.
  // See the README (box-plus / box-minus).
  throw std::logic_error("NOT_IMPLEMENTED: se3_boxplus");
}

Vector6d se3_boxminus(const Eigen::Matrix4d& T2, const Eigen::Matrix4d& T1) {
  // Inverse of boxplus: the tangent vector taking T1 to T2 in T1's right (body) frame.
  // See the README (box-plus / box-minus).
  throw std::logic_error("NOT_IMPLEMENTED: se3_boxminus");
}
