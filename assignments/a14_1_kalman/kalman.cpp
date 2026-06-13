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
  // Push the mean through the linear motion model and inflate the covariance:
  //   mu' = F mu + B u,   P' = F P F^T + Q.
  throw std::logic_error("NOT_IMPLEMENTED: kf_predict");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  // Innovation y = z - H mu; innovation covariance S = H P H^T + R; gain
  // K = P H^T S^-1; mean mu' = mu + K y. Use the JOSEPH form for the covariance:
  //   P' = (I - K H) P (I - K H)^T + K R K^T,
  // which stays symmetric and SPD under round-off (the short form (I-KH)P does not).
  throw std::logic_error("NOT_IMPLEMENTED: kf_update");
}

// ---- EKF nonlinear models and their Jacobians ------------------------------

Eigen::Vector3d ekf_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  // Forward-Euler unicycle. x = [px, py, theta], u = [v, omega].
  //   px' = px + v dt cos(theta), py' = py + v dt sin(theta), theta' = theta + omega dt.
  // Wrap theta' to (-pi, pi] with wrap_angle.
  throw std::logic_error("NOT_IMPLEMENTED: ekf_f");
}

Eigen::Matrix3d ekf_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  // Jacobian of ekf_f w.r.t. x. Only the heading column is nontrivial:
  //   F_x = [[1, 0, -v dt sin(theta)], [0, 1, v dt cos(theta)], [0, 0, 1]].
  throw std::logic_error("NOT_IMPLEMENTED: ekf_F_x");
}

Eigen::Vector2d ekf_h(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  // Range-bearing to the landmark. dx = lx - px, dy = ly - py.
  //   r = sqrt(dx^2 + dy^2),  phi = wrap_angle(atan2(dy, dx) - theta).
  // Return [r, phi].
  throw std::logic_error("NOT_IMPLEMENTED: ekf_h");
}

Matrix23d ekf_H(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  // Jacobian of ekf_h w.r.t. x. With dx, dy as above, r^2 = dx^2 + dy^2:
  //   H = [[-dx/r, -dy/r,  0],
  //        [ dy/r^2, -dx/r^2, -1]].
  throw std::logic_error("NOT_IMPLEMENTED: ekf_H");
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_predict(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q) {
  // EKF predict: mean through the nonlinear model, covariance through its Jacobian.
  //   mu' = ekf_f(mu, u, dt),  F = ekf_F_x(mu, u, dt),  P' = F P F^T + Q.
  throw std::logic_error("NOT_IMPLEMENTED: ekf_predict");
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_update(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& z,
    const Eigen::Vector2d& landmark, const Eigen::Matrix2d& R) {
  // EKF update: H = ekf_H(mu); innovation y = z - ekf_h(mu), with the BEARING
  // component y[1] wrapped to (-pi, pi] (a measurement at +179 deg and a prediction
  // at -179 deg are 2 deg apart, not 358). Then the linear KF update with this H,
  // Joseph form for P, and wrap the updated heading mu'[2].
  throw std::logic_error("NOT_IMPLEMENTED: ekf_update");
}

// ---- UKF unscented-transform primitives ------------------------------------

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> ukf_sigma_points(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, double alpha, double beta,
    double kappa) {
  // n = mu.size(); lambda = alpha^2 (n + kappa) - n. Build the matrix square root
  // L of (n + lambda) P with a Cholesky (Eigen: ((n+lambda)*P).llt().matrixL()).
  // Sigma points (rows of a (2n+1) x n matrix): row 0 = mu; rows 1..n = mu + L.col(i);
  // rows n+1..2n = mu - L.col(i). Weights: Wm[0] = lambda/(n+lambda),
  // Wc[0] = Wm[0] + (1 - alpha^2 + beta); all other Wm[i] = Wc[i] = 1/(2(n+lambda)).
  throw std::logic_error("NOT_IMPLEMENTED: ukf_sigma_points");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ukf_unscented_transform(
    const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& Wm, const Eigen::VectorXd& Wc,
    const Eigen::MatrixXd& noise_cov) {
  // sigmas: one propagated point per row. mean = sum_i Wm[i] sigmas.row(i).
  // cov = sum_i Wc[i] d_i d_i^T + noise_cov, with d_i = sigmas.row(i) - mean.
  throw std::logic_error("NOT_IMPLEMENTED: ukf_unscented_transform");
}

Eigen::MatrixXd ukf_cross_covariance(
    const Eigen::MatrixXd& sigmas_x, const Eigen::VectorXd& x_mean,
    const Eigen::MatrixXd& sigmas_z, const Eigen::VectorXd& z_mean,
    const Eigen::VectorXd& Wc) {
  // P_xz = sum_i Wc[i] (sigmas_x.row(i) - x_mean) (sigmas_z.row(i) - z_mean)^T.
  // Shape is (state dim) x (measurement dim).
  throw std::logic_error("NOT_IMPLEMENTED: ukf_cross_covariance");
}

// ---- Information (canonical) form ------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> moments_to_information(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P) {
  // Omega = P^-1 (use P.inverse() or, better, an SPD solve), eta = Omega mu.
  throw std::logic_error("NOT_IMPLEMENTED: moments_to_information");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_to_moments(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega) {
  // P = Omega^-1, mu = P eta.
  throw std::logic_error("NOT_IMPLEMENTED: information_to_moments");
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_update(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  // The additive measurement update: with Ri = R^-1,
  //   Omega' = Omega + H^T Ri H,   eta' = eta + H^T Ri z.
  // No matrix inverse of the state size here - that is the appeal of the form.
  throw std::logic_error("NOT_IMPLEMENTED: information_update");
}
