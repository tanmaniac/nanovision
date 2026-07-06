// Multi-view geometry estimators: triangulation, the normalized eight-point algorithm,
// essential-matrix decomposition with cheirality, PnP, the Sampson distance, RANSAC, and a
// composed two-view relative-pose front-end. Declarations shared by the top-level (holed)
// and solution implementations; only multiview.cpp differs. models.cpp (the SE(3)
// retraction carried over from the Lie-group assignment, plus the pinhole helpers) is
// provided and compiled in both builds.
//
// CONVENTIONS (each is a classic silent transpose / sign trap; pin them before coding):
//   - The UNPRIMED image is the first / reference image - the frame the points start in.
//     The PRIMED image is the second. Subscripts 1 and 2 here mean first and second.
//   - (R, t) = T_2_1 is the transform taking a frame-1 coordinate into frame-2:
//     X_2 = R X_1 + t. The essential matrix is E = [t]_x R.
//   - The essential constraint x2^T E x1 = 0 holds for NORMALIZED rays x = K^-1 u
//     (homogeneous, z = 1). The fundamental constraint x2^T F x1 = 0 holds for PIXELS u,
//     with F = K2^-T E K1^-1. Feeding pixels to the eight-point solver yields F; feeding
//     normalized rays yields E (up to the rank-2 / essential cleanup).
//   - A camera pose used for projection and PnP is T_cam_world (world-to-camera): a world
//     point X projects as u = pi(K (R X + t)), the same OpenCV pinhole as the camera-
//     geometry assignment (+x right, +y down, +z forward; u = fx X/Z + cx).
//
// Point sets are passed as row-major (N x 2) pixel/image matrices and (N x 3) world
// matrices (numpy-friendly). Projection matrices are 3x4.
#pragma once
#include <Eigen/Dense>

#include <tuple>
#include <utility>
#include <vector>

using Matrix34d = Eigen::Matrix<double, 3, 4>;

// ---- Provided primitives (models.cpp, compiled in both builds) -------------
// Carried over from the Lie-group assignment and the pinhole camera model; call them, do
// not reimplement. hat3 is the SO(3) skew operator; se3_exp is the SE(3) retraction used
// to update a pose on the manifold; pinhole_project maps a camera-frame point to a pixel
// and pinhole_jacobian is its 2x3 derivative d(u)/d(X_cam).
Eigen::Matrix3d hat3(const Eigen::Vector3d& w);
Eigen::Matrix4d se3_exp(const Eigen::Matrix<double, 6, 1>& xi);  // xi = [rho; theta]
Eigen::Vector2d pinhole_project(const Eigen::Matrix3d& K, const Eigen::Vector3d& X_cam);
Eigen::Matrix<double, 2, 3> pinhole_jacobian(const Eigen::Matrix3d& K,
                                             const Eigen::Vector3d& X_cam);

// ---- The holes (multiview.cpp) ---------------------------------------------

// Linear triangulation (DLT). Given two 3x4 projection matrices and one corresponding
// image point in each (consistent with each P: pixels if P includes K, normalized rays if
// P = [R|t]), solve the homogeneous A x = 0 system for the 3D point and dehomogenize.
Eigen::Vector3d triangulate_dlt(const Matrix34d& P1, const Matrix34d& P2,
                                const Eigen::Vector2d& x1, const Eigen::Vector2d& x2);

// Gauss-Newton refinement of a triangulated point: minimize the summed reprojection error
// in both cameras over the 3 coordinates of X, starting from X0 (the DLT estimate).
Eigen::Vector3d triangulate_refine(const Matrix34d& P1, const Matrix34d& P2,
                                   const Eigen::Vector2d& x1, const Eigen::Vector2d& x2,
                                   const Eigen::Vector3d& X0);

// The normalized eight-point algorithm. Hartley-normalize each point set (centroid to the
// origin, mean distance sqrt(2)), solve the N x 9 homogeneous system for the matrix's null
// vector, enforce rank 2 by zeroing the smallest singular value, then denormalize. Fed
// pixels it returns F; fed normalized rays it returns the (rank-2) essential matrix.
Eigen::Matrix3d eight_point(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2);

// Decompose an essential matrix into its two rotation candidates and the translation
// direction, from the SVD of E with both singular-vector matrices made proper rotations.
// Returns (R1, R2, t) with t unit length; the four physical candidates are (R1, +t),
// (R1, -t), (R2, +t), (R2, -t).
std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d> decompose_essential(
    const Eigen::Matrix3d& E);

// Recover the physical relative pose from an essential matrix and the normalized-ray
// correspondences (x1, x2 are N x 2 normalized image coordinates). Triangulate against
// each of the four decompose_essential candidates and keep the one with the most points in
// front of BOTH cameras (cheirality). Returns (R, t) = T_2_1 with t unit length.
std::pair<Eigen::Matrix3d, Eigen::Vector3d> recover_pose(const Eigen::Matrix3d& E,
                                                         const Eigen::MatrixXd& x1,
                                                         const Eigen::MatrixXd& x2);

// PnP by the Direct Linear Transform. Given N>=6 world points X (N x 3), their pixels u
// (N x 2), and the intrinsics K, solve the linear system for the 3x4 camera matrix, peel
// off R and t, project R onto SO(3) (SVD), fix the scale and sign (positive depth), and
// return the pose T = [[R, t],[0,1]] = T_cam_world.
Eigen::Matrix4d pnp_dlt(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                        const Eigen::Matrix3d& K);

// Nonlinear PnP refinement: Gauss-Newton on the summed reprojection error over SE(3),
// starting from T0. Update the pose on the manifold by the right perturbation
// T <- T * se3_exp(dxi). Returns the refined T_cam_world.
Eigen::Matrix4d pnp_refine(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                           const Eigen::Matrix3d& K, const Eigen::Matrix4d& T0);

// Per-correspondence Sampson distance for a fundamental (or essential) matrix M: the
// first-order geometric error (pixel units), NOT the raw algebraic residual x2^T M x1.
// p1, p2 are N x 2 (pixels for F, normalized rays for E); returns an N-vector of distances.
Eigen::VectorXd sampson_distance(const Eigen::Matrix3d& M, const Eigen::MatrixXd& p1,
                                 const Eigen::MatrixXd& p2);

// RANSAC for the fundamental matrix. Repeatedly draw a minimal sample of 8 correspondences,
// fit F with the eight-point algorithm, score all correspondences by the Sampson distance,
// and keep the model with the largest consensus set (distance below `threshold` pixels).
// Refit F on the final inliers. `seed` drives a deterministic RNG. Returns (F, the sorted
// inlier indices).
std::pair<Eigen::Matrix3d, std::vector<int>> ransac_fundamental(
    const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2, double threshold, int iters,
    unsigned int seed);

// The composed two-view relative-pose front-end: RANSAC F on the pixel correspondences,
// form the essential matrix from F and K (here K1 = K2 = K), recover (R, t) by cheirality on
// the inlier rays, and triangulate the inliers into 3D points expressed in frame 1. Returns the estimated
// T_2_1 (4x4), the triangulated points (M x 3 for M inliers), and the inlier indices. The
// translation scale is the inherent monocular gauge (|t| = 1).
std::tuple<Eigen::Matrix4d, Eigen::MatrixXd, std::vector<int>> two_view_relative_pose(
    const Eigen::Matrix3d& K, const Eigen::MatrixXd& u1, const Eigen::MatrixXd& u2,
    double threshold, int iters, unsigned int seed);
