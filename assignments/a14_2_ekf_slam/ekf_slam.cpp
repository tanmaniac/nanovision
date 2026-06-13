// EKF-SLAM - YOUR CODE. Fill each hole; read the contract in the comment and in
// ekf_slam.hpp, the math is in the README. Run `make test A=a14_2_ekf_slam`. The motion
// and measurement primitives (robot_f, robot_F_x, range_bearing, range_bearing_H_robot,
// range_bearing_H_land) are provided in models.cpp - call them, do not reimplement them.
// The reference is in solution/ekf_slam.cpp.
#include "ekf_slam.hpp"

#include <cmath>
#include <stdexcept>

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q) {
  // Move the robot: mu.head<3>() = robot_f(robot, u, dt). For the covariance, with
  // F = robot_F_x(robot, u, dt) (3x3), update the robot block and the robot-map cross:
  //   P_rr' = F P_rr F^T + Q       (top-left 3x3)
  //   P_rm' = F P_rm               (rows 0..2, cols 3..)  and P_mr' = (P_rm')^T
  //   P_mm   unchanged             (the map-map block)
  // The map-map block not changing while the cross blocks do is the SLAM coupling:
  // moving the robot correlates it with every landmark it has seen.
  throw std::logic_error("NOT_IMPLEMENTED: slam_predict");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_add_landmark(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    const Eigen::Matrix2d& R) {
  // Inverse measurement model. With z = [r, phi], robot = mu.head<3>(), and
  // beta = theta + phi: the landmark is at (px + r cos beta, py + r sin beta).
  // Augment mu by these 2 entries, and grow P from m x m to (m+2) x (m+2):
  //   Gr = d(landmark)/d(robot)   = [[1, 0, -r sin beta], [0, 1, r cos beta]]   (2x3)
  //   Gz = d(landmark)/d(z)       = [[cos beta, -r sin beta], [sin beta, r cos beta]] (2x2)
  //   new landmark block  P_LL = Gr P_rr Gr^T + Gz R Gz^T          (P_rr = top-left 3x3)
  //   new cross block     P_Lx = Gr * P.topRows<3>()               (2 x m, lm vs all)
  // Place P in the top-left m x m, P_LL in the bottom-right 2x2, P_Lx in the bottom-left
  // 2 x m, and its transpose in the top-right m x 2.
  throw std::logic_error("NOT_IMPLEMENTED: slam_add_landmark");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    int j, const Eigen::Matrix2d& R) {
  // EKF update for landmark j. robot = mu.head<3>(), lm = mu.segment<2>(3 + 2j).
  // Innovation y = z - range_bearing(robot, lm), with the BEARING y(1) wrapped.
  // Build H (2 x m), zero except H.block<2,3>(0,0) = range_bearing_H_robot(robot, lm)
  // and H.block<2,2>(0, 3 + 2j) = range_bearing_H_land(robot, lm). Then the standard
  // EKF update over the FULL state: S = H P H^T + R, K = P H^T S^-1, mu' = mu + K y
  // (wrap mu'(2)), and the Joseph form P' = (I-KH) P (I-KH)^T + K R K^T.
  throw std::logic_error("NOT_IMPLEMENTED: slam_update");
}

int slam_associate(const Eigen::VectorXd& mu, const Eigen::MatrixXd& P,
                   const Eigen::Vector2d& z, const Eigen::Matrix2d& R, double gate) {
  // For each mapped landmark j, form the predicted measurement and its innovation
  // y = z - range_bearing(robot, lm) (wrap y(1)), the innovation covariance
  // S = H P H^T + R with the same H as the update, and the squared Mahalanobis distance
  // d^2 = y^T S^-1 y. Return the j with the smallest d^2 that is below `gate`, else -1
  // (a new landmark). Return -1 on an empty map.
  throw std::logic_error("NOT_IMPLEMENTED: slam_associate");
}
