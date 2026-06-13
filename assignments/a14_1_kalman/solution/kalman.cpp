#include "kalman.hpp"

#include <cmath>

// ---- Linear Kalman filter --------------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& B, const Eigen::VectorXd& u, const Eigen::MatrixXd& Q) {
  const Eigen::VectorXd mu_pred = F * mu + B * u;
  const Eigen::MatrixXd P_pred = F * P * F.transpose() + Q;
  return {mu_pred, P_pred};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  const Eigen::VectorXd y = z - H * mu;
  const Eigen::MatrixXd S = H * P * H.transpose() + R;
  const Eigen::MatrixXd K = P * H.transpose() * S.inverse();
  const Eigen::VectorXd mu_upd = mu + K * y;
  const Eigen::MatrixXd I = Eigen::MatrixXd::Identity(P.rows(), P.cols());
  const Eigen::MatrixXd IKH = I - K * H;
  const Eigen::MatrixXd P_upd = IKH * P * IKH.transpose() + K * R * K.transpose();
  return {mu_upd, P_upd};
}

// ---- EKF nonlinear models and their Jacobians ------------------------------

Eigen::Vector3d ekf_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  const double v = u(0), omega = u(1), th = x(2);
  Eigen::Vector3d xn;
  xn(0) = x(0) + v * dt * std::cos(th);
  xn(1) = x(1) + v * dt * std::sin(th);
  xn(2) = wrap_angle(th + omega * dt);
  return xn;
}

Eigen::Matrix3d ekf_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt) {
  const double v = u(0), th = x(2);
  Eigen::Matrix3d F = Eigen::Matrix3d::Identity();
  F(0, 2) = -v * dt * std::sin(th);
  F(1, 2) = v * dt * std::cos(th);
  return F;
}

Eigen::Vector2d ekf_h(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  const double dx = landmark(0) - x(0);
  const double dy = landmark(1) - x(1);
  const double r = std::sqrt(dx * dx + dy * dy);
  const double phi = wrap_angle(std::atan2(dy, dx) - x(2));
  return Eigen::Vector2d(r, phi);
}

Matrix23d ekf_H(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark) {
  const double dx = landmark(0) - x(0);
  const double dy = landmark(1) - x(1);
  const double q = dx * dx + dy * dy;
  const double r = std::sqrt(q);
  Matrix23d H;
  H << -dx / r, -dy / r, 0.0,
       dy / q, -dx / q, -1.0;
  return H;
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_predict(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q) {
  const Eigen::Vector3d mu_pred = ekf_f(mu, u, dt);
  const Eigen::Matrix3d F = ekf_F_x(mu, u, dt);
  const Eigen::Matrix3d P_pred = F * P * F.transpose() + Q;
  return {mu_pred, P_pred};
}

std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_update(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& z,
    const Eigen::Vector2d& landmark, const Eigen::Matrix2d& R) {
  const Matrix23d H = ekf_H(mu, landmark);
  Eigen::Vector2d y = z - ekf_h(mu, landmark);
  y(1) = wrap_angle(y(1));
  const Eigen::Matrix2d S = H * P * H.transpose() + R;
  const Eigen::Matrix<double, 3, 2> K = P * H.transpose() * S.inverse();
  Eigen::Vector3d mu_upd = mu + K * y;
  mu_upd(2) = wrap_angle(mu_upd(2));
  const Eigen::Matrix3d I = Eigen::Matrix3d::Identity();
  const Eigen::Matrix3d IKH = I - K * H;
  const Eigen::Matrix3d P_upd = IKH * P * IKH.transpose() + K * R * K.transpose();
  return {mu_upd, P_upd};
}

// ---- UKF unscented-transform primitives ------------------------------------

std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> ukf_sigma_points(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, double alpha, double beta,
    double kappa) {
  const int n = static_cast<int>(mu.size());
  const double lambda = alpha * alpha * (n + kappa) - n;
  const double scale = n + lambda;

  // Matrix square root of scale * P via Cholesky: L L^T = scale * P.
  const Eigen::MatrixXd L = (scale * P).llt().matrixL();

  Eigen::MatrixXd sigmas(2 * n + 1, n);
  sigmas.row(0) = mu.transpose();
  for (int i = 0; i < n; ++i) {
    sigmas.row(1 + i) = (mu + L.col(i)).transpose();
    sigmas.row(1 + n + i) = (mu - L.col(i)).transpose();
  }

  Eigen::VectorXd Wm(2 * n + 1), Wc(2 * n + 1);
  Wm(0) = lambda / scale;
  Wc(0) = Wm(0) + (1.0 - alpha * alpha + beta);
  const double w = 1.0 / (2.0 * scale);
  for (int i = 1; i < 2 * n + 1; ++i) {
    Wm(i) = w;
    Wc(i) = w;
  }
  return {sigmas, Wm, Wc};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> ukf_unscented_transform(
    const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& Wm, const Eigen::VectorXd& Wc,
    const Eigen::MatrixXd& noise_cov) {
  const int m = static_cast<int>(sigmas.cols());
  Eigen::VectorXd mean = Eigen::VectorXd::Zero(m);
  for (int i = 0; i < sigmas.rows(); ++i) {
    mean += Wm(i) * sigmas.row(i).transpose();
  }
  Eigen::MatrixXd cov = noise_cov;
  for (int i = 0; i < sigmas.rows(); ++i) {
    const Eigen::VectorXd d = sigmas.row(i).transpose() - mean;
    cov += Wc(i) * d * d.transpose();
  }
  return {mean, cov};
}

Eigen::MatrixXd ukf_cross_covariance(
    const Eigen::MatrixXd& sigmas_x, const Eigen::VectorXd& x_mean,
    const Eigen::MatrixXd& sigmas_z, const Eigen::VectorXd& z_mean,
    const Eigen::VectorXd& Wc) {
  Eigen::MatrixXd Pxz = Eigen::MatrixXd::Zero(x_mean.size(), z_mean.size());
  for (int i = 0; i < sigmas_x.rows(); ++i) {
    const Eigen::VectorXd dx = sigmas_x.row(i).transpose() - x_mean;
    const Eigen::VectorXd dz = sigmas_z.row(i).transpose() - z_mean;
    Pxz += Wc(i) * dx * dz.transpose();
  }
  return Pxz;
}

// ---- Information (canonical) form ------------------------------------------

std::pair<Eigen::VectorXd, Eigen::MatrixXd> moments_to_information(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P) {
  const Eigen::MatrixXd Omega = P.inverse();
  const Eigen::VectorXd eta = Omega * mu;
  return {eta, Omega};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_to_moments(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega) {
  const Eigen::MatrixXd P = Omega.inverse();
  const Eigen::VectorXd mu = P * eta;
  return {mu, P};
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_update(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
  const Eigen::MatrixXd Ri = R.inverse();
  const Eigen::MatrixXd Omega_upd = Omega + H.transpose() * Ri * H;
  const Eigen::VectorXd eta_upd = eta + H.transpose() * Ri * z;
  return {eta_upd, Omega_upd};
}
