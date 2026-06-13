// EKF-SLAM: a single extended Kalman filter over the joint state of the robot pose
// and all mapped landmarks. Declarations shared by the top-level (holed) and solution
// implementations; only ekf_slam.cpp differs. models.cpp (the motion and measurement
// primitives carried over from the Kalman assignment) is provided and compiled in both.
//
// Joint state layout (a dynamic vector that GROWS as landmarks are added):
//   mu = [ px, py, theta,  l0x, l0y,  l1x, l1y,  ... ]
//        |---- robot ----| |- lm 0 -| |- lm 1 -|
// Landmark j occupies indices 3 + 2j and 3 + 2j + 1. The joint covariance P is the
// matching (3 + 2N) x (3 + 2N) matrix; its off-diagonal blocks are the whole point of
// SLAM (they couple the robot and the landmarks, and the landmarks to each other).
//
// Conventions match the Kalman assignment: forward-Euler unicycle motion, range-bearing
// measurements, and the Joseph-form covariance update. The bearing residual is wrapped.
#pragma once
#include <Eigen/Dense>

#include <utility>

using Matrix23d = Eigen::Matrix<double, 2, 3>;

inline double wrap_angle(double a) {
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a <= 0.0) a += 2.0 * M_PI;
  return a - M_PI;
}

// ---- Provided primitives (models.cpp, compiled in both builds) -------------
// The unicycle motion model and its Jacobian, and the range-bearing measurement model
// and its Jacobians, all carried over from the Kalman assignment. You call these; you
// do not reimplement them here. robot is [px, py, theta]; lm is [lx, ly].
Eigen::Vector3d robot_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt);
Eigen::Matrix3d robot_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt);
Eigen::Vector2d range_bearing(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm);
Matrix23d range_bearing_H_robot(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm);
Eigen::Matrix2d range_bearing_H_land(const Eigen::Vector3d& robot, const Eigen::Vector2d& lm);

// ---- The holes (ekf_slam.cpp) ----------------------------------------------
// Predict: move the robot through the motion model and propagate the covariance. Only
// the robot block sees process noise Q (3x3), but the robot-landmark cross-covariance
// blocks must also be updated, or the map silently decorrelates from the robot.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q);

// Initialize a new landmark from its first observation and grow the state by 2. Uses the
// inverse measurement model (place the landmark in the world from the robot pose and the
// range-bearing reading) and augments mu and P with the right Jacobian blocks.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_add_landmark(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    const Eigen::Matrix2d& R);

// Measurement update for the observation z of the already-mapped landmark j. The
// measurement Jacobian H is 2 x (3+2N), nonzero only in the robot block and the
// landmark-j block; the EKF update over the FULL joint state is what spreads the
// correction into every correlated landmark. Joseph form; wrap the bearing residual.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> slam_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::Vector2d& z,
    int j, const Eigen::Matrix2d& R);

// Data association: which mapped landmark does measurement z belong to? Returns the
// index of the nearest landmark by Mahalanobis distance of the innovation,
// d^2 = y^T S^-1 y with S = H P H^T + R, if that distance is below the chi-square gate;
// otherwise -1 (treat z as a new landmark). Returns -1 on an empty map.
int slam_associate(const Eigen::VectorXd& mu, const Eigen::MatrixXd& P,
                   const Eigen::Vector2d& z, const Eigen::Matrix2d& R, double gate);
