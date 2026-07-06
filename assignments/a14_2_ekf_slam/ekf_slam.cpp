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
  // Predict step: move the robot with robot_f and propagate the covariance with F = robot_F_x.
  // Only the robot pose changes, so update the robot block (with process noise Q) and the
  // robot-map cross blocks, and leave the map-map block alone. Propagating the cross blocks is
  // the SLAM coupling that keeps the robot correlated with every landmark it has seen; dropping
  // it silently decorrelates the map and breaks loop closure. See the README (predict).
  throw std::logic_error("NOT_IMPLEMENTED: slam_predict");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_add_landmark(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    const Eigen::Matrix2d& R) {
  // Initialize a new landmark from measurement z = [r, phi]: place it in the world with the
  // inverse measurement model from the current pose, append its 2 coordinates to the mean, and
  // grow P from m x m to (m+2) x (m+2). The augmentation needs the Jacobians of the inverse
  // model with respect to the pose and the measurement to form the new landmark's own block and
  // its cross covariance against the whole existing state (a new landmark is born correlated
  // with the robot and, through it, the rest of the map). See the README (initializing a
  // landmark).
  throw std::logic_error("NOT_IMPLEMENTED: slam_add_landmark");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    int j, const Eigen::Matrix2d& R) {
  // EKF measurement update for the observed landmark j. Form the innovation against the
  // predicted range_bearing (wrap the BEARING component), and build the sparse 2 x m measurement
  // Jacobian H: zero except the robot columns (range_bearing_H_robot) and landmark j's columns
  // (range_bearing_H_land). Then run the standard EKF update over the FULL joint state with the
  // Joseph form, and wrap the updated heading. See the README (the measurement update).
  throw std::logic_error("NOT_IMPLEMENTED: slam_update");
}

int slam_associate(const Eigen::VectorXd& mu, const Eigen::MatrixXd& P,
                   const Eigen::Vector2d& z, const Eigen::Matrix2d& R, double gate) {
  // Data association: for each mapped landmark, form the innovation against its predicted
  // measurement (wrap the bearing) and its covariance with the same H as the update, and score
  // it by the squared Mahalanobis distance. Return the landmark with the smallest distance that
  // is below `gate`, else -1 (a new landmark). Return -1 on an empty map. See the README (data
  // association).
  throw std::logic_error("NOT_IMPLEMENTED: slam_associate");
}
