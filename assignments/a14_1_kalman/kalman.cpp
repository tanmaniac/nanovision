// Gaussian filters - YOUR CODE. Fill each hole; read the contract in the comment,
// the math is in the README. Run `make test A=a14_1_kalman`. The reference is in
// solution/kalman.cpp.
#include "kalman.hpp"

#include <cmath>
#include <stdexcept>

// ---- Linear Kalman filter --------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& B, const Eigen::VectorXd& u, const Eigen::MatrixXd& Q) {
  // Linear KF predict: push the mean through the linear motion model (F, B, u) and inflate the
  // covariance with process noise Q. See the README (the linear Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: kf_predict");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  // Linear KF update: form the innovation from measurement z (model H, noise R) and the Kalman
  // gain, then correct the mean and covariance. Use the JOSEPH form for the covariance (it stays
  // symmetric and SPD under round-off; the short (I-KH)P form does not). See the README (the
  // linear Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: kf_update");
}

// ---- EKF nonlinear models and their Jacobians ------------------------------

Eigen::Vector3d ekf_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  // Forward-Euler unicycle motion model. x = [px, py, theta], u = [v, omega]. Wrap the updated
  // heading to (-pi, pi] with wrap_angle. See the README (the extended Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_f");
}

Eigen::Matrix3d ekf_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  // Jacobian of ekf_f with respect to the state x (3x3). See the README (the extended Kalman
  // filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_F_x");
}

Eigen::Vector2d ekf_h(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  // Range-bearing measurement to the landmark: return [range, bearing], the bearing measured
  // relative to the robot heading and wrapped to (-pi, pi]. See the README (the extended Kalman
  // filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_h");
}

Matrix23d ekf_H(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  // Jacobian of ekf_h with respect to the state x (2x3). See the README (the extended Kalman
  // filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_H");
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_predict(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q) {
  // EKF predict: push the mean through the nonlinear model ekf_f and propagate the covariance
  // through its Jacobian ekf_F_x, adding process noise Q. See the README (the extended Kalman
  // filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_predict");
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_update(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& z,
    const Eigen::Vector2d& landmark, const Eigen::Matrix2d& R) {
  // EKF update against the known landmark. Wrap the BEARING component of the innovation to
  // (-pi, pi] (a measurement at +179 deg and a prediction at -179 deg are 2 deg apart, not 358),
  // run the linear KF update with H = ekf_H(mu) and the Joseph form, and wrap the updated
  // heading. See the README (the extended Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: ekf_update");
}

// ---- UKF unscented-transform primitives ------------------------------------

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> ukf_sigma_points(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, double alpha, double beta,
    double kappa) {
  // Build the 2n+1 sigma points (rows of a (2n+1) x n matrix, row 0 the mean point) and their
  // mean/covariance weights (Wm, Wc) for the belief (mu, P) with parameters alpha, beta, kappa.
  // The center covariance weight Wc[0] carries an extra (1 - alpha^2 + beta) term (beta = 2 is
  // Gaussian-optimal); dropping it is a quiet bug that corrupts the covariance, not the mean.
  // Return (sigmas, Wm, Wc). See the README (the unscented Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: ukf_sigma_points");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ukf_unscented_transform(
    const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& Wm, const Eigen::VectorXd& Wc,
    const Eigen::MatrixXd& noise_cov) {
  // Unscented transform: given propagated sigma points (one per row) with weights Wm, Wc and an
  // additive noise_cov, return the weighted mean and covariance. See the README (the unscented
  // Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: ukf_unscented_transform");
}

Eigen::MatrixXd ukf_cross_covariance(
    const Eigen::MatrixXd& sigmas_x, const Eigen::VectorXd& x_mean,
    const Eigen::MatrixXd& sigmas_z, const Eigen::VectorXd& z_mean,
    const Eigen::VectorXd& Wc) {
  // State-measurement cross-covariance from the two sigma-point sets, their means, and the
  // covariance weights Wc. Shape is (state dim) x (measurement dim). See the README (the
  // unscented Kalman filter).
  throw std::logic_error("NOT_IMPLEMENTED: ukf_cross_covariance");
}

// ---- Information (canonical) form ------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> moments_to_information(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P) {
  // Convert the moment form (mu, P) to the information form (the information matrix and vector,
  // the dual parameterization). See the README (the information form).
  throw std::logic_error("NOT_IMPLEMENTED: moments_to_information");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_to_moments(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega) {
  // Convert the information form (eta, Omega) back to the moment form (mu, P), the inverse of
  // moments_to_information. See the README (the information form).
  throw std::logic_error("NOT_IMPLEMENTED: information_to_moments");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_update(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  // Fold the measurement z (model H, noise R) into (eta, Omega). The update is purely additive:
  // no inverse of a state-sized matrix, which is the appeal of the form. See the README (the
  // information form).
  throw std::logic_error("NOT_IMPLEMENTED: information_update");
}
