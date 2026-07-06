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
  // Linear (DLT) triangulation: each view contributes two rows to a homogeneous 4x4 system
  // A X = 0 from the relation x ~ P X. Solve for the null vector (the right singular vector of
  // the smallest singular value) and dehomogenize to the 3D point. See the README
  // (triangulation).
  throw std::logic_error("NOT_IMPLEMENTED: triangulate_dlt");
}

Eigen::Vector3d triangulate_refine(const Matrix34d& P1, const Matrix34d& P2,
                                   const Eigen::Vector2d& x1, const Eigen::Vector2d& x2,
                                   const Eigen::Vector3d& X0) {
  // Nonlinear refinement of the DLT point: Gauss-Newton on the two-view reprojection residual
  // over the 3 unknowns of X, starting from X0 (the DLT estimate). See the README
  // (triangulation).
  throw std::logic_error("NOT_IMPLEMENTED: triangulate_refine");
}

Eigen::Matrix3d eight_point(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2) {
  // Normalized eight-point algorithm (N >= 8 correspondences in each of p1, p2). Hartley-
  // normalize each image's points (centroid to the origin, mean distance to the origin sqrt(2)),
  // build the N x 9 homogeneous system and take its null vector as the matrix, enforce rank 2 by
  // zeroing the smallest singular value, then denormalize. Fed pixels it returns F; fed
  // normalized rays, E. See the README (epipolar geometry and the eight-point algorithm).
  throw std::logic_error("NOT_IMPLEMENTED: eight_point");
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d> decompose_essential(
    const Eigen::Matrix3d& E) {
  // Decompose an essential matrix into its two rotation candidates and a translation direction
  // via the SVD of E, with U and V made proper rotations. Return (R1, R2, t) with t unit length;
  // the caller forms the four candidates (R1,+t), (R1,-t), (R2,+t), (R2,-t). See the README
  // (from the essential matrix to a pose).
  throw std::logic_error("NOT_IMPLEMENTED: decompose_essential");
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> recover_pose(const Eigen::Matrix3d& E,
                                                         const Eigen::MatrixXd& x1,
                                                         const Eigen::MatrixXd& x2) {
  // Cheirality test: of the four (R, t) candidates from decompose_essential, pick the one that
  // triangulates the most correspondences (x1, x2 are normalized rays) with positive depth in
  // BOTH cameras. Return that (R, t) = T_2_1. See the README (from the essential matrix to a
  // pose).
  throw std::logic_error("NOT_IMPLEMENTED: recover_pose");
}

Eigen::Matrix4d pnp_dlt(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                        const Eigen::Matrix3d& K) {
  // DLT pose from N >= 6 world-point / pixel correspondences (X is N x 3, u is N x 2, calibrated
  // by K). Build the homogeneous 2N x 12 system in the entries of the 3x4 matrix [R | t] from
  // x ~ M [X;1] on normalized rays, solve for the null vector, then project the rotation block
  // onto SO(3) and recover the metric scale and sign (points must end up in front of the
  // camera). Return T = [[R, t],[0,1]]. See the README (PnP).
  throw std::logic_error("NOT_IMPLEMENTED: pnp_dlt");
}

Eigen::Matrix4d pnp_refine(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                           const Eigen::Matrix3d& K, const Eigen::Matrix4d& T0) {
  // Nonlinear PnP: Gauss-Newton over SE(3) minimizing the summed reprojection error, starting
  // from T0, updating by the right perturbation T <- T * se3_exp(xi) with xi = [rho; theta]. Use
  // the provided pinhole_jacobian for the measurement Jacobian. Return the refined T. See the
  // README (PnP).
  throw std::logic_error("NOT_IMPLEMENTED: pnp_refine");
}

Eigen::VectorXd sampson_distance(const Eigen::Matrix3d& M, const Eigen::MatrixXd& p1,
                                 const Eigen::MatrixXd& p2) {
  // Sampson distance for each correspondence: the first-order approximation to the geometric
  // reprojection error induced by M. It has pixel units, unlike the raw algebraic residual
  // x2^T M x1, which is why RANSAC thresholds it. Return the N-vector of distances. See the
  // README (RANSAC and the Sampson distance).
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
  // The VO front-end. Run ransac_fundamental on the pixels (u1, u2) to get F and the inlier set,
  // form the essential matrix from F and K, convert the inlier pixels to normalized rays and
  // call recover_pose for (R, t) = T_2_1 by cheirality, then triangulate the inlier rays into 3D
  // points in frame 1. Return (T_2_1 as a 4x4, the M x 3 triangulated points, the inlier
  // indices). See the README (the composed front-end).
  throw std::logic_error("NOT_IMPLEMENTED: two_view_relative_pose");
}
