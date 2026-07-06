// ICP - YOUR CODE. Fill each hole; read the contract in the comment and in icp.hpp, the math
// is in the README. Run `make test A=a14_4_icp`. The SE(3) retraction (se3_exp), the skew
// operator (hat3), and the nearest-neighbor search (nearest_neighbors) are provided in
// models.cpp - call them, do not reimplement them. The reference is in solution/icp.cpp.
#include "icp.hpp"

#include <cmath>
#include <stdexcept>

Eigen::Matrix4d align_point_to_point(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q) {
  // Closed-form point-to-point alignment (orthogonal Procrustes, Umeyama/Kabsch): the rigid
  // transform that best maps matched cloud P onto Q, from the SVD of the centroid-centered
  // cross-covariance. Include the determinant correction that keeps the rotation proper
  // (det +1) on planar or degenerate data, where the naive closest orthogonal matrix can be a
  // reflection. Return T = [[R, t],[0,1]] mapping P onto Q. See the README (point-to-point).
  throw std::logic_error("NOT_IMPLEMENTED: align_point_to_point");
}

Eigen::Matrix4d point_to_plane_step(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& N) {
  // One Gauss-Newton step of the point-to-plane cost (the residual is the distance along the
  // target normals N) over the twist xi = [rho; theta]. Assemble and solve the 6x6 normal
  // equations from the per-correspondence rows, and return the incremental transform
  // se3_exp(xi). See the README (point-to-plane).
  throw std::logic_error("NOT_IMPLEMENTED: point_to_plane_step");
}

std::tuple<Eigen::Matrix4d, int, double> icp(
    const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
    const Eigen::MatrixXd& target_normals, const Eigen::Matrix4d& T_init, int max_iter,
    double max_corr_dist, const std::string& mode) {
  // The outer loop. With T = T_init, repeat up to max_iter times:
  //  1. Transform the source by T: p'_i = R source_i + t.
  //  2. nearest_neighbors(transformed_source, target) gives, per source point, its closest
  //     target index and distance. Keep only pairs with distance <= max_corr_dist.
  //  3. Gather the matched current source points P, target points Q (and target normals N
  //     for plane mode) into N_match x 3 blocks.
  //  4. Solve the incremental transform: align_point_to_point(P, Q) for mode == "point", or
  //     point_to_plane_step(P, Q, N) for mode == "plane".
  //  5. Compose it onto the pose: T <- delta * T. Stop when the increment is tiny (e.g. its
  //     translation and rotation are below ~1e-10), tracking the RMS correspondence distance.
  // Return (T, iterations_run, final_rms_distance).
  throw std::logic_error("NOT_IMPLEMENTED: icp");
}
