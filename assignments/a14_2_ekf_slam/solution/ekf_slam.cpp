#include "ekf_slam.hpp"

#include <cmath>

namespace {
int num_landmarks(const Eigen::VectorXd& mu) {
  return static_cast<int>((mu.size() - 3) / 2);
}

// The 2 x (3+2N) measurement Jacobian for landmark j: nonzero only in the robot block
// and the landmark-j block. Used by both the update and the association gate.
Eigen::MatrixXd measurement_jacobian(const Eigen::VectorXd& mu, int j) {
  const int m = static_cast<int>(mu.size());
  const Eigen::Vector3d robot = mu.head<3>();
  const Eigen::Vector2d lm = mu.segment<2>(3 + 2 * j);
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(2, m);
  H.block<2, 3>(0, 0) = range_bearing_H_robot(robot, lm);
  H.block<2, 2>(0, 3 + 2 * j) = range_bearing_H_land(robot, lm);
  return H;
}
}  // namespace

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q) {
  const int m = static_cast<int>(mu.size());
  const Eigen::Vector3d robot = mu.head<3>();
  const Eigen::Matrix3d F = robot_F_x(robot, u, dt);

  Eigen::VectorXd mu_new = mu;
  mu_new.head<3>() = robot_f(robot, u, dt);

  Eigen::MatrixXd P_new = P;
  // Robot-robot block gets F P_rr F^T + Q.
  P_new.topLeftCorner<3, 3>() = F * P.topLeftCorner<3, 3>() * F.transpose() + Q;
  // Robot-map cross blocks rotate through F (the map block is left untouched).
  if (m > 3) {
    P_new.block(0, 3, 3, m - 3) = F * P.block(0, 3, 3, m - 3);
    P_new.block(3, 0, m - 3, 3) = P_new.block(0, 3, 3, m - 3).transpose();
  }
  return {mu_new, P_new};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_add_landmark(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    const Eigen::Matrix2d& R) {
  const int m = static_cast<int>(mu.size());
  const Eigen::Vector3d robot = mu.head<3>();
  const double r = z(0), phi = z(1), th = robot(2);
  const double c = std::cos(th + phi), s = std::sin(th + phi);

  // Inverse measurement model: place the landmark in the world.
  Eigen::Vector2d lm(robot(0) + r * c, robot(1) + r * s);

  // Jacobians of the inverse model w.r.t. the robot pose and the measurement.
  Eigen::Matrix<double, 2, 3> Gr;
  Gr << 1.0, 0.0, -r * s,
        0.0, 1.0, r * c;
  Eigen::Matrix2d Gz;
  Gz << c, -r * s,
        s, r * c;

  const Eigen::Matrix3d P_rr = P.topLeftCorner<3, 3>();
  const Eigen::Matrix2d P_LL = Gr * P_rr * Gr.transpose() + Gz * R * Gz.transpose();
  const Eigen::MatrixXd P_Lx = Gr * P.topRows<3>();  // 2 x m, new lm vs all existing state

  Eigen::VectorXd mu_new(m + 2);
  mu_new.head(m) = mu;
  mu_new.tail<2>() = lm;

  Eigen::MatrixXd P_new(m + 2, m + 2);
  P_new.topLeftCorner(m, m) = P;
  P_new.bottomLeftCorner(2, m) = P_Lx;
  P_new.topRightCorner(m, 2) = P_Lx.transpose();
  P_new.bottomRightCorner<2, 2>() = P_LL;
  return {mu_new, P_new};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    int j, const Eigen::Matrix2d& R) {
  const int m = static_cast<int>(mu.size());
  const Eigen::Vector3d robot = mu.head<3>();
  const Eigen::Vector2d lm = mu.segment<2>(3 + 2 * j);

  Eigen::Vector2d y = z - range_bearing(robot, lm);
  y(1) = wrap_angle(y(1));

  const Eigen::MatrixXd H = measurement_jacobian(mu, j);
  const Eigen::Matrix2d S = H * P * H.transpose() + R;
  const Eigen::MatrixXd K = P * H.transpose() * S.inverse();  // m x 2

  Eigen::VectorXd mu_new = mu + K * y;
  mu_new(2) = wrap_angle(mu_new(2));

  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(m, m);
  const Eigen::MatrixXd IKH = I - K * H;
  Eigen::MatrixXd P_new = IKH * P * IKH.transpose() + K * R * K.transpose();
  return {mu_new, P_new};
}

int slam_associate(const Eigen::VectorXd& mu, const Eigen::MatrixXd& P,
                   const Eigen::Vector2d& z, const Eigen::Matrix2d& R, double gate) {
  const int N = num_landmarks(mu);
  const Eigen::Vector3d robot = mu.head<3>();
  int best = -1;
  double best_d = gate;  // only associate when strictly inside the gate
  for (int j = 0; j < N; ++j) {
    const Eigen::Vector2d lm = mu.segment<2>(3 + 2 * j);
    Eigen::Vector2d y = z - range_bearing(robot, lm);
    y(1) = wrap_angle(y(1));
    const Eigen::MatrixXd H = measurement_jacobian(mu, j);
    const Eigen::Matrix2d S = H * P * H.transpose() + R;
    const double d2 = y.transpose() * S.inverse() * y;
    if (d2 < best_d) {
      best_d = d2;
      best = j;
    }
  }
  return best;
}
