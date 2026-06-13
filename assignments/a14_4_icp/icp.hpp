// Iterative closest point (ICP): register a source point cloud onto a target cloud by
// estimating the rigid transform between them. Declarations shared by the top-level (holed)
// and solution implementations; only icp.cpp differs. models.cpp (the SE(3) retraction
// carried over from the Lie-group assignment, the skew operator, and the brute-force
// nearest-neighbor search that stands in for a kd-tree) is provided and compiled in both.
//
// CONVENTIONS:
//   - A transform T = [[R, t],[0,1]] maps the SOURCE cloud onto the TARGET cloud:
//     q ~ R p + t. In the module's T_b_a source-to-target naming, the estimate is
//     T_target_source. align_point_to_point(P, Q) returns the transform mapping P onto Q.
//   - Clouds are row-major N x 3 matrices. Normals (for point-to-plane) are N x 3 unit rows
//     defined on the TARGET cloud.
//   - The twist ordering is xi = [rho (translation); theta (rotation)], and the outer loop
//     composes each increment in the world frame as T <- se3_exp(xi) * T (a left update),
//     using the se3_exp retraction from the Lie-group assignment.
#pragma once
#include <Eigen/Dense>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// ---- Provided primitives (models.cpp, compiled in both builds) -------------
// Carried over from the Lie-group assignment, plus the correspondence search. hat3 is the
// SO(3) skew operator; se3_exp is the SE(3) retraction (xi = [rho; theta]). nearest_neighbors
// returns, for each row of `src`, the index of the closest row in `dst` and that distance
// (brute force - a real system uses a kd-tree, which is provided here because building a
// balanced kd-tree is not the registration lesson).
Eigen::Matrix3d hat3(const Eigen::Vector3d& w);
Eigen::Matrix4d se3_exp(const Vector6d& xi);
std::pair<std::vector<int>, Eigen::VectorXd> nearest_neighbors(const Eigen::MatrixXd& src,
                                                               const Eigen::MatrixXd& dst);

// ---- The holes (icp.cpp) ---------------------------------------------------

// Point-to-point alignment in closed form (the Umeyama / Kabsch solution to the orthogonal
// Procrustes problem). For matched clouds P, Q (row i of P corresponds to row i of Q),
// center both, form H = sum (p_i - p_bar)(q_i - q_bar)^T, take H = U S V^T, and
//   R = V diag(1, 1, det(V U^T)) U^T,   t = q_bar - R p_bar.
// The det(V U^T) correction on the last singular direction prevents R from being a reflection
// on planar or otherwise degenerate data. Returns T = [[R, t],[0,1]] mapping P onto Q.
Eigen::Matrix4d align_point_to_point(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q);

// One Gauss-Newton step of point-to-plane alignment. For matched current source points P,
// target points Q, and target unit normals N, minimize sum_i (n_i . (R p_i + t - q_i))^2
// over the small motion xi = [rho; theta]. Linearizing the retraction gives, per point,
//   a_i = [n_i; p_i x n_i] (6-vector),   b_i = -(p_i - q_i) . n_i,
// and the normal equations (sum a_i a_i^T) xi = sum a_i b_i. Return the incremental
// transform se3_exp(xi) (to be composed onto the current pose).
Eigen::Matrix4d point_to_plane_step(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& N);

// The ICP outer loop. Starting from T_init, iterate: transform the source by the current T,
// find nearest-neighbor correspondences in the target, reject pairs farther than
// max_corr_dist, solve for the incremental motion (point-to-point if mode == "point",
// point-to-plane if mode == "plane"), compose it onto T, and repeat until the update is tiny
// or max_iter is reached. Returns (the refined T_target_source, the iterations run, the final
// RMS correspondence distance).
std::tuple<Eigen::Matrix4d, int, double> icp(
    const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
    const Eigen::MatrixXd& target_normals, const Eigen::Matrix4d& T_init, int max_iter,
    double max_corr_dist, const std::string& mode);
