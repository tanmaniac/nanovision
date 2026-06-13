// Provided primitives, compiled into both the student and solution builds. The SO(3) skew
// operator and the SE(3) retraction are carried over from the Lie-group assignment (compact
// closed forms, same twist ordering xi = [rho; theta]). nearest_neighbors is the
// correspondence search: a brute-force scan that stands in for a kd-tree, because building a
// balanced kd-tree is a separate exercise, not the registration lesson.
#include "icp.hpp"

#include <cmath>
#include <limits>

namespace {
constexpr double kEps = 1e-8;

Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w) {
  const double th = w.norm();
  Eigen::Matrix3d K;
  K <<     0, -w.z(),  w.y(),
       w.z(),      0, -w.x(),
      -w.y(),  w.x(),      0;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + K + 0.5 * K * K;
  const double a = std::sin(th) / th;
  const double b = (1.0 - std::cos(th)) / (th * th);
  return I + a * K + b * K * K;
}

Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w) {
  const double th = w.norm();
  Eigen::Matrix3d K;
  K <<     0, -w.z(),  w.y(),
       w.z(),      0, -w.x(),
      -w.y(),  w.x(),      0;
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  if (th < kEps) return I + 0.5 * K + (1.0 / 6.0) * K * K;
  const double th2 = th * th;
  const double b = (1.0 - std::cos(th)) / th2;
  const double c = (th - std::sin(th)) / (th2 * th);
  return I + b * K + c * K * K;
}
}  // namespace

Eigen::Matrix3d hat3(const Eigen::Vector3d& w) {
  Eigen::Matrix3d W;
  W <<     0, -w.z(),  w.y(),
       w.z(),      0, -w.x(),
      -w.y(),  w.x(),      0;
  return W;
}

Eigen::Matrix4d se3_exp(const Vector6d& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = so3_exp(theta);
  T.block<3, 1>(0, 3) = so3_left_jacobian(theta) * rho;
  return T;
}

std::pair<std::vector<int>, Eigen::VectorXd> nearest_neighbors(const Eigen::MatrixXd& src,
                                                               const Eigen::MatrixXd& dst) {
  const int n = static_cast<int>(src.rows());
  const int m = static_cast<int>(dst.rows());
  std::vector<int> idx(n);
  Eigen::VectorXd dist(n);
  for (int i = 0; i < n; ++i) {
    double best = std::numeric_limits<double>::max();
    int best_j = -1;
    for (int j = 0; j < m; ++j) {
      const double d = (src.row(i) - dst.row(j)).squaredNorm();
      if (d < best) {
        best = d;
        best_j = j;
      }
    }
    idx[i] = best_j;
    dist(i) = std::sqrt(best);
  }
  return {idx, dist};
}
