// Gaussian filters: linear KF, EKF, UKF, and the information form. Declarations
// shared by the top-level (holed) and solution implementations; only the .cpp
// bodies differ.
//
// Conventions used throughout (and inherited from a14_0's estimation conventions):
//   - A Gaussian belief is a mean vector mu and a covariance P (symmetric, SPD).
//   - The covariance update uses the Joseph form, which stays symmetric and SPD
//     under round-off, unlike the textbook P = (I - K H) P.
//   - The EKF demo state is x = [px, py, theta] (a planar pose), control is
//     u = [v, omega] (body velocity and turn rate), and the measurement is a
//     range-bearing reading [r, phi] of a known landmark. The bearing phi is an
//     angle, so its residual is wrapped to (-pi, pi]; wrap_angle does that.
#pragma once
#include <Eigen/Dense>

#include <tuple>
#include <utility>

using Matrix23d = Eigen::Matrix<double, 2, 3>;

// Wrap an angle to (-pi, pi]. Provided (not a hole): the EKF residual and the
// state's heading both need it, and getting it wrong silently breaks the filter.
inline double wrap_angle(double a) {
  a = std::fmod(a + M_PI, 2.0 * M_PI);
  if (a <= 0.0) a += 2.0 * M_PI;
  return a - M_PI;
}

// ---- Linear Kalman filter --------------------------------------------------
// predict: mu' = F mu + B u,  P' = F P F^T + Q.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_predict(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::MatrixXd& F,
    const Eigen::MatrixXd& B, const Eigen::VectorXd& u, const Eigen::MatrixXd& Q);

// update: innovation y = z - H mu, S = H P H^T + R, gain K = P H^T S^-1,
// mu' = mu + K y, and the Joseph-form covariance P' = (I-KH) P (I-KH)^T + K R K^T.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> kf_update(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);

// ---- EKF nonlinear models and their Jacobians ------------------------------
// Motion model f(x, u, dt): forward-Euler unicycle.
//   px' = px + v dt cos(theta),  py' = py + v dt sin(theta),  theta' = theta + omega dt.
Eigen::Vector3d ekf_f(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt);
// Its Jacobian F_x = df/dx (3x3).
Eigen::Matrix3d ekf_F_x(const Eigen::Vector3d& x, const Eigen::Vector2d& u, double dt);

// Measurement model h(x, landmark): range and bearing to a known 2D landmark.
//   r = ||landmark - p||,  phi = wrap(atan2(ly-py, lx-px) - theta).
Eigen::Vector2d ekf_h(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark);
// Its Jacobian H = dh/dx (2x3).
Matrix23d ekf_H(const Eigen::Vector3d& x, const Eigen::Vector2d& landmark);

// EKF predict / update: the linear KF equations with F = F_x(mu) and H = H(mu),
// the mean propagated through the nonlinear f / h, and the bearing residual wrapped.
std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_predict(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& u,
    double dt, const Eigen::Matrix3d& Q);
std::pair<Eigen::Vector3d, Eigen::Matrix3d> ekf_update(
    const Eigen::Vector3d& mu, const Eigen::Matrix3d& P, const Eigen::Vector2d& z,
    const Eigen::Vector2d& landmark, const Eigen::Matrix2d& R);

// ---- UKF unscented-transform primitives ------------------------------------
// Sigma points and weights for the unscented transform. n = mu.size().
// lambda = alpha^2 (n + kappa) - n. The 2n+1 points are mu and mu +/- the columns
// of the matrix square root of (n + lambda) P (Cholesky). Returns:
//   sigmas: (2n+1) x n, one point per row (row 0 is the mean point);
//   Wm: 2n+1 mean weights, Wm[0] = lambda/(n+lambda), rest 1/(2(n+lambda));
//   Wc: 2n+1 covariance weights, Wc[0] = Wm[0] + (1 - alpha^2 + beta), rest as Wm.
// The (1 - alpha^2 + beta) term on Wc[0] (beta = 2 is Gaussian-optimal) is easy to
// drop and a real bug when dropped: it corrects the covariance, not the mean.
std::tuple<Eigen::MatrixXd, Eigen::VectorXd, Eigen::VectorXd> ukf_sigma_points(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P, double alpha, double beta,
    double kappa);

// Recover a Gaussian from propagated sigma points: weighted mean and covariance,
// plus an additive process/measurement noise term.
//   mean = sum_i Wm[i] y_i,  cov = sum_i Wc[i] (y_i - mean)(y_i - mean)^T + noise_cov.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> ukf_unscented_transform(
    const Eigen::MatrixXd& sigmas, const Eigen::VectorXd& Wm, const Eigen::VectorXd& Wc,
    const Eigen::MatrixXd& noise_cov);

// Cross-covariance between the state sigma points and the measurement sigma points,
// P_xz = sum_i Wc[i] (x_i - x_mean)(z_i - z_mean)^T. The UKF gain is K = P_xz S^-1.
Eigen::MatrixXd ukf_cross_covariance(
    const Eigen::MatrixXd& sigmas_x, const Eigen::VectorXd& x_mean,
    const Eigen::MatrixXd& sigmas_z, const Eigen::VectorXd& z_mean,
    const Eigen::VectorXd& Wc);

// ---- Information (canonical) form ------------------------------------------
// The dual of the moment form: Omega = P^-1 (information matrix), eta = Omega mu.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> moments_to_information(
    const Eigen::VectorXd& mu, const Eigen::MatrixXd& P);
std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_to_moments(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega);

// The measurement update is purely additive in information form:
//   Omega' = Omega + H^T R^-1 H,  eta' = eta + H^T R^-1 z.
std::pair<Eigen::VectorXd, Eigen::MatrixXd> information_update(
    const Eigen::VectorXd& eta, const Eigen::MatrixXd& Omega, const Eigen::VectorXd& z,
    const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
