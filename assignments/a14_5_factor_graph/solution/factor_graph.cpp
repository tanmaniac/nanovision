#include "factor_graph.hpp"

Vector6d between_residual(const Eigen::Matrix4d& Ti, const Eigen::Matrix4d& Tj,
                          const Eigen::Matrix4d& T_meas) {
  return se3_log(T_meas.inverse() * (Ti.inverse() * Tj));
}

std::pair<Matrix6d, Matrix6d> between_jacobians(const Eigen::Matrix4d& Ti,
                                               const Eigen::Matrix4d& Tj,
                                               const Eigen::Matrix4d& T_meas) {
  const Vector6d r = between_residual(Ti, Tj, T_meas);
  const Matrix6d Jr_inv = se3_right_jacobian_inv(r);
  const Matrix6d Ji = -Jr_inv * se3_adjoint(Tj.inverse() * Ti);
  const Matrix6d Jj = Jr_inv;
  return {Ji, Jj};
}

std::pair<std::vector<Eigen::Matrix4d>, double> optimize_pose_graph(
    const std::vector<Eigen::Matrix4d>& poses_in, const std::vector<int>& edge_i,
    const std::vector<int>& edge_j, const std::vector<Eigen::Matrix4d>& meas,
    const std::vector<Matrix6d>& info, int num_iters) {
  std::vector<Eigen::Matrix4d> poses = poses_in;
  const int N = static_cast<int>(poses.size());
  const int ne = static_cast<int>(edge_i.size());
  double cost = 0.0;

  for (int iter = 0; iter < num_iters; ++iter) {
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(6 * N, 6 * N);
    Eigen::VectorXd g = Eigen::VectorXd::Zero(6 * N);
    cost = 0.0;

    for (int k = 0; k < ne; ++k) {
      const int i = edge_i[k], j = edge_j[k];
      const Vector6d r = between_residual(poses[i], poses[j], meas[k]);
      Matrix6d Ji, Jj;
      std::tie(Ji, Jj) = between_jacobians(poses[i], poses[j], meas[k]);
      const Matrix6d& Om = info[k];
      cost += r.transpose() * Om * r;

      H.block<6, 6>(6 * i, 6 * i) += Ji.transpose() * Om * Ji;
      H.block<6, 6>(6 * j, 6 * j) += Jj.transpose() * Om * Jj;
      H.block<6, 6>(6 * i, 6 * j) += Ji.transpose() * Om * Jj;
      H.block<6, 6>(6 * j, 6 * i) += Jj.transpose() * Om * Ji;
      g.segment<6>(6 * i) += Ji.transpose() * Om * r;
      g.segment<6>(6 * j) += Jj.transpose() * Om * r;
    }

    // Anchor pose 0: solve only the free block (poses 1..N-1).
    const int nf = 6 * (N - 1);
    const Eigen::MatrixXd Hf = H.bottomRightCorner(nf, nf);
    const Eigen::VectorXd gf = g.tail(nf);
    const Eigen::VectorXd delta = Hf.ldlt().solve(-gf);

    for (int i = 1; i < N; ++i)
      poses[i] = poses[i] * se3_exp(delta.segment<6>(6 * (i - 1)));
  }

  // Final cost at the converged configuration.
  cost = 0.0;
  for (int k = 0; k < ne; ++k) {
    const Vector6d r = between_residual(poses[edge_i[k]], poses[edge_j[k]], meas[k]);
    cost += r.transpose() * info[k] * r;
  }
  return {poses, cost};
}

Eigen::VectorXd schur_solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose,
                            int lm_block) {
  const int n = static_cast<int>(H.rows());
  const int nl = n - n_pose;

  const Eigen::MatrixXd Hpp = H.topLeftCorner(n_pose, n_pose);
  const Eigen::MatrixXd Hpl = H.topRightCorner(n_pose, nl);
  const Eigen::MatrixXd Hlp = H.bottomLeftCorner(nl, n_pose);
  const Eigen::MatrixXd Hll = H.bottomRightCorner(nl, nl);
  const Eigen::VectorXd bp = b.head(n_pose);
  const Eigen::VectorXd bl = b.tail(nl);

  // Invert the block-diagonal landmark block, one lm_block x lm_block block at a time.
  Eigen::MatrixXd Hll_inv = Eigen::MatrixXd::Zero(nl, nl);
  for (int s = 0; s < nl; s += lm_block)
    Hll_inv.block(s, s, lm_block, lm_block) =
        Hll.block(s, s, lm_block, lm_block).inverse();

  const Eigen::MatrixXd Hred = Hpp - Hpl * Hll_inv * Hlp;
  const Eigen::VectorXd bred = bp - Hpl * Hll_inv * bl;
  const Eigen::VectorXd dp = Hred.ldlt().solve(bred);
  const Eigen::VectorXd dl = Hll_inv * (bl - Hlp * dp);

  Eigen::VectorXd delta(n);
  delta.head(n_pose) = dp;
  delta.tail(nl) = dl;
  return delta;
}
