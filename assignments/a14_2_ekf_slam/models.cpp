// Provided motion and measurement primitives, carried over from the Kalman assignment
// (the unicycle f / F_x and the range-bearing h and its Jacobians). Compiled in both the
// student and solution builds; not a hole. EKF-SLAM is about the JOINT filter over robot
// + map, so these per-pose pieces are given and you assemble the SLAM machinery on top.
#include "ekf_slam.hpp"

#include <cmath>

Eigen::Vector3d robot_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  const double v = u(0), omega = u(1), th = x(2);
  Eigen::Vector3d xn;
  xn(0) = x(0) + v * dt * std::cos(th);
  xn(1) = x(1) + v * dt * std::sin(th);
  xn(2) = wrap_angle(th + omega * dt);
  return xn;
}

Eigen::Matrix3d robot_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  const double v = u(0), th = x(2);
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  F(0, 2) = -v * dt * std::sin(th);
  F(1, 2) = v * dt * std::cos(th);
  return F;
}

Eigen::Vector2d range_bearing(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm) {
  const double dx = lm(0) - robot(0);
  const double dy = lm(1) - robot(1);
  const double r = std::sqrt(dx * dx + dy * dy);
  const double phi = wrap_angle(std::atan2(dy, dx) - robot(2));
  return Eigen::Vector2d(r, phi);
}

Matrix23d range_bearing_H_robot(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm) {
  const double dx = lm(0) - robot(0);
  const double dy = lm(1) - robot(1);
  const double q = dx * dx + dy * dy;
  const double r = std::sqrt(q);
  Matrix23d H;
  H << -dx / r, -dy / r, 0.0,
       dy / q, -dx / q, -1.0;
  return H;
}

Eigen::Matrix2d range_bearing_H_land(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm) {
  const double dx = lm(0) - robot(0);
  const double dy = lm(1) - robot(1);
  const double q = dx * dx + dy * dy;
  const double r = std::sqrt(q);
  Eigen::Matrix2d H;
  H << dx / r, dy / r,
       -dy / q, dx / q;
  return H;
}
