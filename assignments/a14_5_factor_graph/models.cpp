// Provided primitives, compiled into both builds. The SE(3) Lie-group machinery is carried
// over from the Lie-group assignment (the micro-Lie-theory layer): the exponential and
// logarithm, the adjoint, and the 6x6 inverse right Jacobian (Barfoot's closed form). These
// are not the pose-graph lesson - the holes assemble the factors and the normal equations on
// top of them - so they are given rather than re-derived here.
#include "factor_graph.hpp"

#include <cmath>

namespace {
constexpr double kEps = 1e-8;

Eigen::Matrix3d skew(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  W <<     0, -w.z(),  w.y(),
       w.z(),      0, -w.x(),
      -w.y(),  w.x(),      0;
  return W;
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = skew(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + K + 0.5 * K * K;
  return I + (std::sin(th) / th) * K + ((1.0 - std::cos(th)) / (th * th)) * K * K;
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  double c = 0.5 * (R.trace() - 1.0);
  c = std::max(-1.0, std::min(1.0, c));
  const double th = std::acos(c);
  if (th < kEps) return 0.5 * Eigen::Vector3d(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0),
                                              R(1, 0) - R(0, 1));
  const Eigen::Vector3d v(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (th / (2.0 * std::sin(th))) * v;
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = skew(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + 0.5 * K + (1.0 / 6.0) * K * K;
  const double th2 = th * th;
  return I + ((1.0 - std::cos(th)) / th2) * K + ((th - std::sin(th)) / (th2 * th)) * K * K;
}

Eigen::Matrix3d so3_left_jacobian_inv(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = skew(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I - 0.5 * K + (1.0 / 12.0) * K * K;
  const double half = 0.5 * th;
  const double c = (1.0 / (th * th)) * (1.0 - half * std::cos(half) / std::sin(half));
  return I - 0.5 * K + c * K * K;
}

// Barfoot's Q matrix (State Estimation for Robotics, eq. 7.86b), the off-diagonal block of
// the SE(3) left Jacobian for xi = [rho; theta].
Eigen::Matrix3d se3_Q(const Eigen::Vector3d& rho, const Eigen::Vector3d& theta) {
  const Eigen::Matrix3d P = skew(rho);
  const Eigen::Matrix3d W = skew(theta);
  const double th = theta.norm();
  if (th < kEps) return 0.5 * P;
  const double th2 = th * th, th4 = th2 * th2, th5 = th4 * th;
  const double c1 = (th - std::sin(th)) / (th2 * th);
  const double term_a = (1.0 - 0.5 * th2 - std::cos(th)) / th4;
  const double term_b = (th - std::sin(th) - th2 * th / 6.0) / th5;
  const double c2 = -term_a;
  const double c3 = -0.5 * (term_a - 3.0 * term_b);
  return 0.5 * P
       + c1 * (W * P + P * W + W * P * W)
       + c2 * (W * W * P + P * W * W - 3.0 * W * P * W)
       + c3 * (W * P * W * W + W * W * P * W);
}

Matrix6d se3_left_jacobian(const Vector6d& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  const Eigen::Matrix3d Jl = so3_left_jacobian(theta);
  Matrix6d J = Matrix6d::Zero();
  J.block<3, 3>(0, 0) = Jl;
  J.block<3, 3>(3, 3) = Jl;
  J.block<3, 3>(0, 3) = se3_Q(rho, theta);
  return J;
}
}  // namespace

Eigen::Matrix3d hat3(const Eigen::Vector3d& w) { return skew(w); }

Eigen::Matrix4d se3_exp(const Vector6d& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = so3_exp(theta);
  T.block<3, 1>(0, 3) = so3_left_jacobian(theta) * rho;
  return T;
}

Vector6d se3_log(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  const Eigen::Vector3d theta = so3_log(R);
  Vector6d xi;
  xi.head<3>() = so3_left_jacobian_inv(theta) * t;
  xi.tail<3>() = theta;
  return xi;
}

Matrix6d se3_adjoint(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Matrix6d Ad = Matrix6d::Zero();
  Ad.block<3, 3>(0, 0) = R;
  Ad.block<3, 3>(0, 3) = skew(t) * R;
  Ad.block<3, 3>(3, 3) = R;
  return Ad;
}

Matrix6d se3_right_jacobian_inv(const Vector6d& xi) {
  // J_r(xi) = J_l(-xi); invert the 6x6.
  return se3_left_jacobian(-xi).inverse();
}
