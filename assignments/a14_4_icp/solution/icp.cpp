#include "icp.hpp"

#include <cmath>

Eigen::Matrix4d align_point_to_point(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q) {
  const Eigen::RowVector3d p_bar = P.colwise().mean();
  const Eigen::RowVector3d q_bar = Q.colwise().mean();
  const Eigen::MatrixXd Pc = P.rowwise() - p_bar;
  const Eigen::MatrixXd Qc = Q.rowwise() - q_bar;

  const Eigen::Matrix3d H = Pc.transpose() * Qc;  // sum (p - p_bar)(q - q_bar)^T
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
  const Eigen::Matrix3d U = svd.matrixU();
  const Eigen::Matrix3d V = svd.matrixV();

  Eigen::Vector3d d(1.0, 1.0, (V * U.transpose()).determinant());
  const Eigen::Matrix3d R = V * d.asDiagonal() * U.transpose();
  const Eigen::Vector3d t = q_bar.transpose() - R * p_bar.transpose();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

Eigen::Matrix4d point_to_plane_step(const Eigen::MatrixXd& P, const Eigen::MatrixXd& Q,
                                    const Eigen::MatrixXd& N) {
  const int n = static_cast<int>(P.rows());
  Matrix6d A = Matrix6d::Zero();
  Vector6d g = Vector6d::Zero();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d p = P.row(i).transpose();
    const Eigen::Vector3d q = Q.row(i).transpose();
    const Eigen::Vector3d nrm = N.row(i).transpose();
    Vector6d a;
    a.head<3>() = nrm;
    a.tail<3>() = p.cross(nrm);
    const double b = -(p - q).dot(nrm);
    A += a * a.transpose();
    g += a * b;
  }
  const Vector6d xi = A.ldlt().solve(g);
  return se3_exp(xi);
}

std::tuple<Eigen::Matrix4d, int, double> icp(
    const Eigen::MatrixXd& source, const Eigen::MatrixXd& target,
    const Eigen::MatrixXd& target_normals, const Eigen::Matrix4d& T_init, int max_iter,
    double max_corr_dist, const std::string& mode) {
  Eigen::Matrix4d T = T_init;
  const int ns = static_cast<int>(source.rows());
  int it = 0;
  double rms = 0.0;

  for (it = 0; it < max_iter; ++it) {
    // Transform the source by the current pose.
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::MatrixXd Pt = (source * R.transpose()).rowwise() + t.transpose();

    std::vector<int> idx;
    Eigen::VectorXd dist;
    std::tie(idx, dist) = nearest_neighbors(Pt, target);

    // Reject pairs beyond the gate; gather the consensus correspondences.
    std::vector<int> keep;
    keep.reserve(ns);
    for (int i = 0; i < ns; ++i)
      if (dist(i) <= max_corr_dist) keep.push_back(i);

    const int nk = static_cast<int>(keep.size());
    if (nk < 3) break;  // not enough correspondences to solve

    Eigen::MatrixXd Pm(nk, 3), Qm(nk, 3), Nm(nk, 3);
    double sse = 0.0;
    for (int k = 0; k < nk; ++k) {
      Pm.row(k) = Pt.row(keep[k]);
      Qm.row(k) = target.row(idx[keep[k]]);
      if (target_normals.rows() == target.rows()) Nm.row(k) = target_normals.row(idx[keep[k]]);
      sse += dist(keep[k]) * dist(keep[k]);
    }
    rms = std::sqrt(sse / nk);

    Eigen::Matrix4d delta = (mode == "plane") ? point_to_plane_step(Pm, Qm, Nm)
                                              : align_point_to_point(Pm, Qm);
    T = delta * T;

    // Convergence: a negligible incremental motion.
    const double trans_mag = delta.block<3, 1>(0, 3).norm();
    const double rot_mag = (delta.block<3, 3>(0, 0) - Eigen::Matrix3d::Identity()).norm();
    if (trans_mag < 1e-10 && rot_mag < 1e-10) {
      ++it;
      break;
    }
  }
  return {T, it, rms};
}
