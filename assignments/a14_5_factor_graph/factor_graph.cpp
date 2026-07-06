// Factor graphs - YOUR CODE. Fill each hole; read the contract in the comment and in
// factor_graph.hpp, the math is in the README. Run `make test A=a14_5_factor_graph`. The SE(3)
// machinery (se3_exp, se3_log, se3_adjoint, se3_right_jacobian_inv, hat3) is provided in
// models.cpp - call it, do not reimplement it. The reference is in solution/factor_graph.cpp.
#include "factor_graph.hpp"

#include <stdexcept>

Vector6d between_residual(const Eigen::Matrix4d& Ti, const Eigen::Matrix4d& Tj,
                          const Eigen::Matrix4d& T_meas) {
  // Between-factor residual: the SE(3) tangent-space difference (via se3_log) between the
  // measured relative pose T_meas and the one the estimates predict. Zero when the estimated
  // relative pose T_i^-1 T_j matches the measurement. See the README (the pose graph and the
  // between-factor).
  throw std::logic_error("NOT_IMPLEMENTED: between_residual");
}

std::pair<Matrix6d, Matrix6d> between_jacobians(const Eigen::Matrix4d& Ti,
                                               const Eigen::Matrix4d& Tj,
                                               const Eigen::Matrix4d& T_meas) {
  // The two analytic edge Jacobians of between_residual with respect to the right perturbations
  // of T_i and T_j, using the inverse right Jacobian of SE(3) (provided) and the adjoint. Return
  // (J_i, J_j). The minus sign on the i-side and the adjoint's argument are the part to get
  // right; the numerical-vs-analytic test checks them. See the README (the pose graph and the
  // between-factor).
  throw std::logic_error("NOT_IMPLEMENTED: between_jacobians");
}

std::pair<std::vector<Eigen::Matrix4d>, double> optimize_pose_graph(
    const std::vector<Eigen::Matrix4d>& poses, const std::vector<int>& edge_i,
    const std::vector<int>& edge_j, const std::vector<Eigen::Matrix4d>& meas,
    const std::vector<Matrix6d>& info, int num_iters) {
  // Gauss-Newton pose-graph optimization. Each iteration, with the current poses: accumulate the
  // sparse normal equations H, g over all edges from each edge's residual, Jacobians, and
  // information matrix info[k] (each edge writes into its (i,i), (i,j), (j,i), (j,j) blocks);
  // anchor pose 0 to fix the gauge by solving only the free block; solve for the increment; and
  // retract each free pose with poses[i] <- poses[i] * se3_exp(delta_i). Return the optimized
  // poses and the final total weighted squared residual. See the README (Gauss-Newton, the
  // normal equations, and the gauge).
  throw std::logic_error("NOT_IMPLEMENTED: optimize_pose_graph");
}

Eigen::VectorXd schur_solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose,
                            int lm_block) {
  // Solve H delta = b by marginalizing the landmarks (the Schur complement). Partition H into
  // pose and landmark blocks; the landmark block H_ll is block-diagonal in lm_block-sized blocks
  // (one per landmark), so invert it block by block. Form the reduced pose system, solve it,
  // back-substitute for the landmarks, and return delta = [delta_p; delta_l]. The result equals
  // the dense solve of H delta = b. See the README (bundle adjustment and the Schur complement).
  throw std::logic_error("NOT_IMPLEMENTED: schur_solve");
}
