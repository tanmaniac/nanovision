#include "so3.hpp"

#include <cmath>

namespace {
constexpr double kEps = 1e-8;  // small-angle threshold for the Taylor branches
}  // namespace

Eigen::Matrix3d hat3(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  // clang-format off
  W <<      0, -w.z(),  w.y(),
        w.z(),      0, -w.x(),
       -w.y(),  w.x(),      0;
  // clang-format on
  return W;
}

Eigen::Vector3d vee3(const Eigen::Matrix3d& W) {
  return Eigen::Vector3d(W(2, 1), W(0, 2), W(1, 0));
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = hat3(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) {
    // sin(th)/th -> 1, (1-cos th)/th^2 -> 1/2 as th -> 0, with K = hat(w).
    return I + K + 0.5 * K * K;
  }
  const double a = std::sin(th) / th;
  const double b = (1.0 - std::cos(th)) / (th * th);
  return I + a * K + b * K * K;
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d& R) {
  double cos_th = 0.5 * (R.trace() - 1.0);
  cos_th = std::max(-1.0, std::min(1.0, cos_th));
  const double th = std::acos(cos_th);
  if (th < kEps) {
    // R ~ I + hat(w); w = vee(R - R^T)/2.
    return 0.5 * vee3(R - R.transpose());
  }
  if (M_PI - th < kEps) {
    // Near pi: sin(th) ~ 0, so the (th/2sin th) form is ill-conditioned. Use
    // R + I = 2 a a^T to extract the axis, then disambiguate the sign.
    const Eigen::Matrix3d A = R + Eigen::Matrix3d::Identity();
    int best = 0;
    double best_norm = A.col(0).norm();
    for (int i = 1; i < 3; ++i) {
      const double n = A.col(i).norm();
      if (n > best_norm) {
        best_norm = n;
        best = i;
      }
    }
    Eigen::Vector3d axis = A.col(best) / A.col(best).norm();
    if (axis.dot(vee3(R - R.transpose())) < 0.0) axis = -axis;
    return th * axis;
  }
  return (th / (2.0 * std::sin(th))) * vee3(R - R.transpose());
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = hat3(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) {
    return I + 0.5 * K + (1.0 / 6.0) * K * K;
  }
  const double th2 = th * th;
  const double b = (1.0 - std::cos(th)) / th2;
  const double c = (th - std::sin(th)) / (th2 * th);
  return I + b * K + c * K * K;
}

Eigen::Matrix3d so3_left_jacobian_inv(const Eigen::Vector3d& w) {
  const double th = w.norm();
  const Eigen::Matrix3d K = hat3(w);
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) {
    return I - 0.5 * K + (1.0 / 12.0) * K * K;
  }
  const double half = 0.5 * th;
  const double c = (1.0 / (th * th)) * (1.0 - half * std::cos(half) / std::sin(half));
  return I - 0.5 * K + c * K * K;
}

Eigen::Matrix3d so3_right_jacobian(const Eigen::Vector3d& w) {
  return so3_left_jacobian(-w);
}

Eigen::Matrix3d so3_right_jacobian_inv(const Eigen::Vector3d& w) {
  return so3_left_jacobian_inv(-w);
}
