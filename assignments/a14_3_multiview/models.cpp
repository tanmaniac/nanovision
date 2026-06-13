// Provided primitives, compiled into both the student and solution builds. These are
// carried over, not holes: the SO(3) skew operator and the SE(3) retraction come from the
// Lie-group assignment (compact closed forms, same right-perturbation convention), and the
// pinhole projection and its Jacobian come from the camera-geometry assignment's OpenCV
// model (+x right, +y down, +z forward; u = fx X/Z + cx). PnP refinement calls se3_exp to
// step the pose on the manifold and pinhole_jacobian for the measurement Jacobian.
#include "multiview.hpp"

#include <cmath>

namespace {
constexpr double kEps = 1e-8;
}  // namespace

Eigen::Matrix3d hat3(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  W <<      0, -w.z(),  w.y(),
        w.z(),      0, -w.x(),
       -w.y(),  w.x(),      0;
  return W;
}

namespace {
// SO(3) exponential and left Jacobian, the two pieces se3_exp needs.
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = hat3(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + K + 0.5 * K * K;
  const double a = std::sin(th) / th;
  const double b = (1.0 - std::cos(th)) / (th * th);
  return I + a * K + b * K * K;
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = hat3(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + 0.5 * K + (1.0 / 6.0) * K * K;
  const double th2 = th * th;
  const double b = (1.0 - std::cos(th)) / th2;
  const double c = (th - std::sin(th)) / (th2 * th);
  return I + b * K + c * K * K;
}
}  // namespace

Eigen::Matrix4d se3_exp(const Eigen::Matrix<double, 6, 1>& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = so3_exp(theta);
  T.block<3, 1>(0, 3) = so3_left_jacobian(theta) * rho;
  return T;
}

Eigen::Vector2d pinhole_project(const Eigen::Matrix3d& K, const Eigen::Vector3d& X_cam) {
  const double fx = K(0, 0), fy = K(1, 1), cx = K(0, 2), cy = K(1, 2);
  const double z = X_cam.z();
  return Eigen::Vector2d(fx * X_cam.x() / z + cx, fy * X_cam.y() / z + cy);
}

Eigen::Matrix<double, 2, 3> pinhole_jacobian(const Eigen::Matrix3d& K,
                                             const Eigen::Vector3d& X_cam) {
  const double fx = K(0, 0), fy = K(1, 1);
  const double x = X_cam.x(), y = X_cam.y(), z = X_cam.z();
  Eigen::Matrix<double, 2, 3> J;
  J << fx / z, 0.0, -fx * x / (z * z),
       0.0, fy / z, -fy * y / (z * z);
  return J;
}
