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
  // T = [[R, t],[0, 1]] with R = so3_exp(theta) and t = V rho,
  // V = so3_left_jacobian(theta).
  throw std::logic_error("NOT_IMPLEMENTED: se3_exp");
}

Vector6d se3_log(const Eigen::Matrix4d& T) {
  // theta = so3_log(R); rho = so3_left_jacobian_inv(theta) * t. Return [rho; theta].
  throw std::logic_error("NOT_IMPLEMENTED: se3_log");
}

Matrix6d se3_adjoint(const Eigen::Matrix4d& T) {
  // Ad_T = [[R, hat3(t) R],[0, R]] for the [rho; theta] ordering.
  throw std::logic_error("NOT_IMPLEMENTED: se3_adjoint");
}

Eigen::Matrix4d se3_boxplus(const Eigen::Matrix4d& T, const Vector6d& xi) {
  // Right perturbation: T * se3_exp(xi).
  throw std::logic_error("NOT_IMPLEMENTED: se3_boxplus");
}

Vector6d se3_boxminus(const Eigen::Matrix4d& T2, const Eigen::Matrix4d& T1) {
  // Inverse of boxplus: se3_log(T1^-1 * T2).
  throw std::logic_error("NOT_IMPLEMENTED: se3_boxminus");
}
