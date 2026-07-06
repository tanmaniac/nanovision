// Factor-graph optimization: pose-graph SLAM and the bundle-adjustment Schur complement, the
// smoothing back-end that replaced the EKF. Declarations shared by the top-level (holed) and
// solution implementations; only factor_graph.cpp differs. models.cpp (the SE(3) Lie-group
// machinery carried over from the Lie-group assignment, including the 6x6 right-Jacobian
// inverse needed for the analytic edge Jacobians) is provided and compiled in both.
//
// CONVENTIONS (right perturbation, matching the Lie-group assignment):
//   - A pose is T = [[R, t],[0,1]] (T_world_body: body-to-world). A twist is xi = [rho; theta]
//     with the translation part first. The retraction is the right perturbation
//     T |+ xi = T * se3_exp(xi).
//   - A between-edge measures the relative pose from i to j. Under the module's T_b_a naming
//     the measurement is T_meas = T_i_j (frame j expressed in frame i), and the estimated
//     relative pose is T_i_j = T_i^-1 T_j; the residual compares the two on the manifold.
//   - An information matrix Omega (6x6, the inverse measurement covariance) weights each edge.
#pragma once
#include <Eigen/Dense>

#include <tuple>
#include <utility>
#include <vector>

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// ---- Provided primitives (models.cpp, compiled in both builds) -------------
// Carried over from the Lie-group assignment (the micro-Lie-theory layer). hat3 is the SO(3)
// skew; se3_exp / se3_log are the SE(3) exponential and logarithm (xi = [rho; theta]);
// se3_adjoint is Ad_T = [[R, [t]_x R],[0, R]]; se3_right_jacobian_inv is the 6x6 inverse right
// Jacobian J_r^-1(xi) (the Barfoot closed form), the piece the analytic edge Jacobians need.
Eigen::Matrix3d hat3(const Eigen::Vector3d& w);
Eigen::Matrix4d se3_exp(const Vector6d& xi);
Vector6d se3_log(const Eigen::Matrix4d& T);
Matrix6d se3_adjoint(const Eigen::Matrix4d& T);
Matrix6d se3_right_jacobian_inv(const Vector6d& xi);

// ---- The holes (factor_graph.cpp) ------------------------------------------

// The between-factor residual for the edge (i, j) with measured relative pose T_meas = T_i_j:
// the SE(3) tangent-space difference between the measurement and the estimated relative pose
// (a 6-vector, zero when the estimate matches).
Vector6d between_residual(const Eigen::Matrix4d& Ti, const Eigen::Matrix4d& Tj,
                          const Eigen::Matrix4d& T_meas);

// The analytic Jacobians of that residual w.r.t. the right perturbations of pose i and pose j,
// built from the inverse right Jacobian of SE(3) and the adjoint (the published Barfoot / GTSAM
// BetweenFactor form). The minus sign on the i-side and the adjoint's argument are the classic
// hand-rolled pose-graph bug; the numerical-vs-analytic test gates them. Returns (J_i, J_j).
std::pair<Matrix6d, Matrix6d> between_jacobians(const Eigen::Matrix4d& Ti,
                                               const Eigen::Matrix4d& Tj,
                                               const Eigen::Matrix4d& T_meas);

// Gauss-Newton pose-graph optimization. The graph is `poses` (the current estimates) and a set
// of between-edges given as parallel arrays: edge k connects edge_i[k] -> edge_j[k] with
// measurement meas[k] and information info[k]. Each iteration assembles the sparse normal
// equations over the 6 DOF per pose from every edge's residual, Jacobians, and information,
// fixes the gauge by anchoring pose 0 (its block is held fixed), solves for the free-pose
// increment, and retracts T <- T * se3_exp(delta). Returns the optimized poses and the final
// total weighted squared residual.
std::pair<std::vector<Eigen::Matrix4d>, double> optimize_pose_graph(
    const std::vector<Eigen::Matrix4d>& poses, const std::vector<int>& edge_i,
    const std::vector<int>& edge_j, const std::vector<Eigen::Matrix4d>& meas,
    const std::vector<Matrix6d>& info, int num_iters);

// The Schur complement for bundle-adjustment landmark marginalization. The assembled system
// H delta = b is partitioned into pose and landmark blocks, where the landmark block H_ll is
// block-diagonal (one `lm_block` x `lm_block` block per landmark - the structure that makes BA
// cheap). With n_pose the number of leading pose columns, marginalize the landmarks: form the
// reduced pose system, solve it for the pose increment, then back-substitute for the landmarks.
// Return the full delta = [delta_p; delta_l]. Must equal the dense solve of H delta = b.
Eigen::VectorXd schur_solve(const Eigen::MatrixXd& H, const Eigen::VectorXd& b, int n_pose,
                            int lm_block);
