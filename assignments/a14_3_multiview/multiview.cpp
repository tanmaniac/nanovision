// Multi-view geometry - YOUR CODE. Fill each hole; read the contract in the comment and in
// multiview.hpp, the math is in the README. Run `make test A=a14_3_multiview`. The SE(3)
// retraction (se3_exp), the skew operator (hat3), and the pinhole helpers (pinhole_project,
// pinhole_jacobian) are provided in models.cpp - call them, do not reimplement them. The
// reference is in solution/multiview.cpp.
#include "multiview.hpp"

#include <cmath>
#include <stdexcept>

Eigen::Vector3d triangulate_dlt(const Matrix34d& P1, const Matrix34d& P2,
                                const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) {
  // Each view gives two rows of the homogeneous system A X = 0 (X homogeneous, 4-vector):
  //   x * P.row(2) - P.row(0) = 0,   y * P.row(2) - P.row(1) = 0,
  // for image point (x, y). Stack the four rows (two per view) into a 4x4 A, take the
  // right singular vector of the smallest singular value (last column of V), and
  // dehomogenize (divide by the 4th entry) to get the 3D point.
  throw std::logic_error("NOT_IMPLEMENTED: triangulate_dlt");
}

Eigen::Vector3d triangulate_refine(const Matrix34d& P1, const Matrix34d& P2,
                                   const Eigen::Vector2d& x1, const Eigen::Vector2d& x2,
                                   const Eigen::Vector3d& X0) {
  // Gauss-Newton on the 4-vector reprojection residual (both views) over the 3 unknowns of
  // X. For a projection P, the predicted image point is the first two of P [X;1] divided by
  // the third; its 2x3 Jacobian w.r.t. X follows from the quotient rule. Stack both views
  // into a 4x3 J and a 4x1 residual r, solve J^T J dX = J^T r, step X <- X + dX, iterate a
  // few times. Start from X0 (the DLT point).
  throw std::logic_error("NOT_IMPLEMENTED: triangulate_refine");
}

Eigen::Matrix3d eight_point(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2) {
  // Normalized eight-point algorithm (N >= 8 rows in each of p1, p2):
  //  1. Hartley-normalize each set: translate the centroid to the origin and scale so the
  //     mean distance to the origin is sqrt(2). Call the 3x3 similarity transforms T1, T2,
  //     so x_hat = T x in homogeneous coordinates.
  //  2. For each correspondence ((u,v) in 1, (u',v') in 2) build the row
  //     [u'u, u'v, u', v'u, v'v, v', u, v, 1] (normalized coords), forming an N x 9 matrix.
  //  3. The matrix's vec is the right singular vector of the smallest singular value;
  //     reshape it row-major into a 3x3 M_hat.
  //  4. Enforce rank 2: SVD M_hat = U S V^T, zero the smallest singular value, rebuild.
  //  5. Denormalize: M = T2^T M_hat T1. Return M.
  throw std::logic_error("NOT_IMPLEMENTED: eight_point");
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d> decompose_essential(
    const Eigen::Matrix3d& E) {
  // Split E by its SVD E = U S V^T (a true essential matrix has S = diag(1,1,0), so only the
  // singular vectors are needed); if det(U) < 0 negate U's last column (and likewise V) so
  // both are proper rotations. With
  // W = [[0,-1,0],[1,0,0],[0,0,1]]: R1 = U W V^T, R2 = U W^T V^T, t = U.col(2) (unit).
  // Return (R1, R2, t); the caller forms the four candidates (R1,+t),(R1,-t),(R2,+t),(R2,-t).
  throw std::logic_error("NOT_IMPLEMENTED: decompose_essential");
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> recover_pose(const Eigen::Matrix3d& E,
                                                         const Eigen::MatrixXd& x1,
                                                         const Eigen::MatrixXd& x2) {
  // Cheirality test. Camera 1 is P1 = [I | 0], camera 2 is P2 = [R | t] for a candidate.
  // For each of the four (R, t) candidates from decompose_essential, triangulate every
  // correspondence (x1, x2 are normalized rays) and count the points with positive depth
  // (z > 0) in BOTH cameras. Return the (R, t) with the highest count: that is T_2_1.
  throw std::logic_error("NOT_IMPLEMENTED: recover_pose");
}

Eigen::Matrix4d pnp_dlt(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                        const Eigen::Matrix3d& K) {
  // DLT pose from N >= 6 world points X (N x 3) and pixels u (N x 2). Convert pixels to
  // normalized rays x = K^-1 [u;1]. Each correspondence contributes two rows to a 2N x 12
  // homogeneous system in the 12 entries of the 3x4 matrix M = [R | t] (the relation is
  // x ~ M [X;1], i.e. the cross product x_hat x (M [X;1]) = 0; use its first two rows).
  // The solution is the smallest right singular vector, reshaped row-major into 3x4. Then:
  // pull R0 = M(:, :3), t0 = M(:, 3); project R0 onto SO(3) via SVD (R = U V^T, fix
  // det = +1); recover the scale s = 1 / mean(singular values of R0) and set t = s t0,
  // flipping the sign of (R, t) if the points end up behind the camera (mean depth < 0).
  // Return T = [[R, t],[0,1]].
  throw std::logic_error("NOT_IMPLEMENTED: pnp_dlt");
}

Eigen::Matrix4d pnp_refine(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                           const Eigen::Matrix3d& K, const Eigen::Matrix4d& T0) {
  // Gauss-Newton over SE(3) minimizing summed reprojection error, starting from T0. With
  // R, t the current pose, X_cam = R X_i + t and the residual e_i = u_i - pinhole_project.
  // Under the right perturbation T <- T se3_exp(xi), xi = [rho; theta], the 3x6 Jacobian of
  // X_cam is [R | -R hat3(X_i)], and the 2x6 measurement Jacobian is
  //   A_i = pinhole_jacobian(K, X_cam) * [R | -R hat3(X_i)].
  // Accumulate H = sum A_i^T A_i and g = sum A_i^T e_i, solve H dxi = g, update
  // T <- T * se3_exp(dxi), iterate a few times. Return the refined T.
  throw std::logic_error("NOT_IMPLEMENTED: pnp_refine");
}

Eigen::VectorXd sampson_distance(const Eigen::Matrix3d& M, const Eigen::MatrixXd& p1,
                                 const Eigen::MatrixXd& p2) {
  // First-order geometric error for each correspondence. With x1 = [u,v,1], x2 = [u',v',1],
  // a = M x1 (3-vector), b = M^T x2 (3-vector), the algebraic residual is r = x2^T M x1, and
  //   d = |r| / sqrt(a(0)^2 + a(1)^2 + b(0)^2 + b(1)^2).
  // Return the N-vector of distances. (This metric error, not the raw |r|, is what RANSAC
  // thresholds, because |r| has no units.)
  throw std::logic_error("NOT_IMPLEMENTED: sampson_distance");
}

std::pair<Eigen::Matrix3d, std::vector<int>> ransac_fundamental(
    const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2, double threshold, int iters,
    unsigned int seed) {
  // Robust F. Seed std::mt19937 with `seed`. For `iters` rounds: draw 8 distinct random
  // indices, fit F with eight_point on that minimal sample, score ALL correspondences with
  // sampson_distance, and collect the inliers (distance < threshold). Keep the inlier set
  // of the largest model seen. After the loop, refit F with eight_point on the full winning
  // inlier set and return (F, sorted inlier indices).
  throw std::logic_error("NOT_IMPLEMENTED: ransac_fundamental");
}

std::tuple<Eigen::Matrix4d, Eigen::MatrixXd, std::vector<int>> two_view_relative_pose(
    const Eigen::Matrix3d& K, const Eigen::MatrixXd& u1, const Eigen::MatrixXd& u2,
    double threshold, int iters, unsigned int seed) {
  // The VO front-end. Run ransac_fundamental on the pixels (u1, u2) to get F and the inlier
  // set. Form E = K^T F K (K1 = K2 = K). Convert the inlier pixels to normalized rays
  // (x = K^-1 [u;1], keep the first two coords) and call recover_pose to get (R, t) = T_2_1
  // by cheirality. Triangulate the inlier rays with P1 = [I|0], P2 = [R|t] into 3D points
  // in frame 1. Return (T_2_1 as a 4x4, the M x 3 triangulated points, the inlier indices).
  throw std::logic_error("NOT_IMPLEMENTED: two_view_relative_pose");
}
