// Factor graphs - YOUR CODE. Fill each hole; read the contract in the comment and in
// factor_graph.hpp, the math is in the README. Run `make test A=a14_5_factor_graph`. The SE(3)
// machinery (se3_exp, se3_log, se3_adjoint, se3_right_jacobian_inv, hat3) is provided in
// models.cpp - call it, do not reimplement it. The reference is in solution/factor_graph.cpp.
#include "factor_graph.hpp"

#include <stdexcept>

Vector6d between_residual(const Eigen::Matrix4d& Ti, const Eigen::Matrix4d& Tj,
                          const Eigen::Matrix4d& T_meas) {
  // r = Log(T_meas^-1 * (T_i^-1 * T_j)). Use se3_log. Zero when the estimated relative pose
  // T_i^-1 T_j matches the measurement.
  throw std::logic_error("NOT_IMPLEMENTED: between_residual");
}

std::pair<Matrix6d, Matrix6d> between_jacobians(const Eigen::Matrix4d& Ti,
                                               const Eigen::Matrix4d& Tj,
                                               const Eigen::Matrix4d& T_meas) {
  // With r = between_residual(Ti, Tj, T_meas) and Jr_inv = se3_right_jacobian_inv(r):
  //   J_i = -Jr_inv * se3_adjoint(T_j^-1 * T_i),   J_j = Jr_inv.
  // Return (J_i, J_j). The minus sign and the adjoint argument T_j^-1 T_i are the part to get
  // right; the numerical-vs-analytic test checks them.
  throw std::logic_error("NOT_IMPLEMENTED: between_jacobians");
}

std::pair<std::vector<Eigen::Matrix4d>, double> optimize_pose_graph(
    const std::vector<Eigen::Matrix4d>& poses, const std::vector<int>& edge_i,
    const std::vector<int>& edge_j, const std::vector<Eigen::Matrix4d>& meas,
    const std::vector<Matrix6d>& info, int num_iters) {
  // Gauss-Newton. Each iteration, with the current poses:
  //  1. Zero H (6N x 6N) and g (6N). For every edge k = (i, j): r = between_residual,
  //     (J_i, J_j) = between_jacobians, Omega = info[k], and accumulate the four H blocks
  //     H_ii += J_i^T Omega J_i, H_ij += J_i^T Omega J_j, H_ji = H_ij^T, H_jj += J_j^T Omega
  //     J_j, and g_i += J_i^T Omega r, g_j += J_j^T Omega r.
  //  2. Anchor pose 0 to fix the gauge: hold its 6 DOF fixed by solving only the free block
  //     (rows/cols 6..6N). Solve H_free delta_free = -g_free (Eigen LDLT).
  //  3. Retract each free pose: poses[i] <- poses[i] * se3_exp(delta_i).
  // Return the optimized poses and the final total weighted squared residual sum_k r^T Omega r.
  throw std::logic_error("NOT_IMPLEMENTED: optimize_pose_graph");
}

Eigen::VectorXd schur_solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose,
                            int lm_block) {
  // Solve H delta = b by marginalizing the landmarks. Partition H into pose/landmark blocks
  // H_pp (n_pose x n_pose), H_pl, H_lp, H_ll. H_ll is block-diagonal in lm_block-sized blocks
  // (one per landmark), so invert it block by block. Then
  //   H_red = H_pp - H_pl H_ll^-1 H_lp,   b_red = b_p - H_pl H_ll^-1 b_l,
  // solve H_red delta_p = b_red, back-substitute delta_l = H_ll^-1 (b_l - H_lp delta_p), and
  // return delta = [delta_p; delta_l]. The result equals the dense solve of H delta = b.
  throw std::logic_error("NOT_IMPLEMENTED: schur_solve");
}
