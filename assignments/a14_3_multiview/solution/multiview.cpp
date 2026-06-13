#include "multiview.hpp"

#include <algorithm>
#include <cmath>
#include <random>

namespace {

// Hartley normalization of an N x 2 point set: translate the centroid to the origin and
// scale so the mean distance to the origin is sqrt(2). Returns the 3x3 similarity T (with
// x_hat = T x in homogeneous coords) and writes the normalized points into `out`.
Eigen::Matrix3d hartley_normalize(const Eigen::MatrixXd& p, Eigen::MatrixXd& out) {
  const int n = static_cast<int>(p.rows());
  const Eigen::RowVector2d c = p.colwise().mean();
  double mean_dist = 0.0;
  for (int i = 0; i < n; ++i) mean_dist += (p.row(i) - c).norm();
  mean_dist /= n;
  const double s = (mean_dist > 0.0) ? std::sqrt(2.0) / mean_dist : 1.0;

  Eigen::Matrix3d T;
  T << s, 0.0, -s * c.x(),
       0.0, s, -s * c.y(),
       0.0, 0.0, 1.0;
  out.resize(n, 2);
  for (int i = 0; i < n; ++i) out.row(i) = s * (p.row(i) - c);
  return T;
}

// Smallest-singular-value right vector (last column of V) of a matrix.
Eigen::VectorXd smallest_right_vector(const Eigen::MatrixXd& A) {
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
  return svd.matrixV().col(svd.matrixV().cols() - 1);
}

Matrix34d make_P(const Eigen::Matrix3d& R, const Eigen::Vector3d& t) {
  Matrix34d P;
  P.leftCols<3>() = R;
  P.col(3) = t;
  return P;
}

}  // namespace

Eigen::Vector3d triangulate_dlt(const Matrix34d& P1, const Matrix34d& P2,
                                const Eigen::Vector2d& x1, const Eigen::Vector2d& x2) {
  Eigen::Matrix4d A;
  A.row(0) = x1.x() * P1.row(2) - P1.row(0);
  A.row(1) = x1.y() * P1.row(2) - P1.row(1);
  A.row(2) = x2.x() * P2.row(2) - P2.row(0);
  A.row(3) = x2.y() * P2.row(2) - P2.row(1);
  const Eigen::Vector4d Xh = smallest_right_vector(A);
  return Xh.head<3>() / Xh(3);
}

Eigen::Vector3d triangulate_refine(const Matrix34d& P1, const Matrix34d& P2,
                                   const Eigen::Vector2d& x1, const Eigen::Vector2d& x2,
                                   const Eigen::Vector3d& X0) {
  Eigen::Vector3d X = X0;
  const Matrix34d Ps[2] = {P1, P2};
  const Eigen::Vector2d xs[2] = {x1, x2};
  for (int it = 0; it < 10; ++it) {
    Eigen::Matrix<double, 4, 3> J;
    Eigen::Vector4d r;
    for (int v = 0; v < 2; ++v) {
      const Eigen::Vector4d Xh(X.x(), X.y(), X.z(), 1.0);
      const Eigen::Vector3d p = Ps[v] * Xh;  // [a, b, w]
      const double w = p(2);
      const Eigen::Vector2d pred(p(0) / w, p(1) / w);
      r.segment<2>(2 * v) = xs[v] - pred;
      // d(pred)/dX = (1/w) (P_top - pred * P_bottom)(:, :3).
      for (int k = 0; k < 2; ++k)
        J.row(2 * v + k) = (Ps[v].row(k).head<3>() - pred(k) * Ps[v].row(2).head<3>()) / w;
    }
    const Eigen::Vector3d dX = (J.transpose() * J).ldlt().solve(J.transpose() * r);
    X += dX;
    if (dX.norm() < 1e-12) break;
  }
  return X;
}

Eigen::Matrix3d eight_point(const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2) {
  Eigen::MatrixXd q1, q2;
  const Eigen::Matrix3d T1 = hartley_normalize(p1, q1);
  const Eigen::Matrix3d T2 = hartley_normalize(p2, q2);
  const int n = static_cast<int>(q1.rows());

  Eigen::MatrixXd A(n, 9);
  for (int i = 0; i < n; ++i) {
    const double u = q1(i, 0), v = q1(i, 1);
    const double up = q2(i, 0), vp = q2(i, 1);
    A.row(i) << up * u, up * v, up, vp * u, vp * v, vp, u, v, 1.0;
  }
  const Eigen::VectorXd f = smallest_right_vector(A);
  Eigen::Matrix3d M_hat;
  M_hat << f(0), f(1), f(2),
           f(3), f(4), f(5),
           f(6), f(7), f(8);

  // Enforce rank 2: zero the smallest singular value.
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(M_hat, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Vector3d s = svd.singularValues();
  s(2) = 0.0;
  M_hat = svd.matrixU() * s.asDiagonal() * svd.matrixV().transpose();

  return T2.transpose() * M_hat * T1;  // denormalize
}

std::tuple<Eigen::Matrix3d, Eigen::Matrix3d, Eigen::Vector3d> decompose_essential(
    const Eigen::Matrix3d& E) {
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d U = svd.matrixU();
  Eigen::Matrix3d V = svd.matrixV();
  if (U.determinant() < 0) U.col(2) *= -1.0;
  if (V.determinant() < 0) V.col(2) *= -1.0;

  Eigen::Matrix3d W;
  W << 0, -1, 0,
       1, 0, 0,
       0, 0, 1;
  const Eigen::Matrix3d R1 = U * W * V.transpose();
  const Eigen::Matrix3d R2 = U * W.transpose() * V.transpose();
  const Eigen::Vector3d t = U.col(2).normalized();
  return {R1, R2, t};
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> recover_pose(const Eigen::Matrix3d& E,
                                                         const Eigen::MatrixXd& x1,
                                                         const Eigen::MatrixXd& x2) {
  Eigen::Matrix3d R1, R2;
  Eigen::Vector3d t;
  std::tie(R1, R2, t) = decompose_essential(E);

  const Eigen::Matrix3d Rs[4] = {R1, R1, R2, R2};
  const Eigen::Vector3d ts[4] = {t, -t, t, -t};
  const Matrix34d P1 = make_P(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  const int n = static_cast<int>(x1.rows());

  int best = 0, best_count = -1;
  for (int c = 0; c < 4; ++c) {
    const Matrix34d P2 = make_P(Rs[c], ts[c]);
    int count = 0;
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d X = triangulate_dlt(P1, P2, x1.row(i).transpose(),
                                                x2.row(i).transpose());
      const double z2 = (Rs[c] * X + ts[c]).z();
      if (X.z() > 0.0 && z2 > 0.0) ++count;
    }
    if (count > best_count) {
      best_count = count;
      best = c;
    }
  }
  return {Rs[best], ts[best]};
}

Eigen::Matrix4d pnp_dlt(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                        const Eigen::Matrix3d& K) {
  const int n = static_cast<int>(X.rows());
  const Eigen::Matrix3d Kinv = K.inverse();

  Eigen::MatrixXd A(2 * n, 12);
  A.setZero();
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d ray = Kinv * Eigen::Vector3d(u(i, 0), u(i, 1), 1.0);
    const double rx = ray.x() / ray.z(), ry = ray.y() / ray.z();
    Eigen::Vector4d Xh(X(i, 0), X(i, 1), X(i, 2), 1.0);
    // ry * (m3 . Xh) - (m2 . Xh) = 0
    A.block<1, 4>(2 * i, 4) = -Xh.transpose();
    A.block<1, 4>(2 * i, 8) = ry * Xh.transpose();
    // (m1 . Xh) - rx * (m3 . Xh) = 0
    A.block<1, 4>(2 * i + 1, 0) = Xh.transpose();
    A.block<1, 4>(2 * i + 1, 8) = -rx * Xh.transpose();
  }
  const Eigen::VectorXd m = smallest_right_vector(A);
  Eigen::Matrix<double, 3, 4> M;
  M.row(0) = m.segment<4>(0).transpose();
  M.row(1) = m.segment<4>(4).transpose();
  M.row(2) = m.segment<4>(8).transpose();

  const Eigen::Matrix3d R0 = M.leftCols<3>();
  const double s = std::cbrt(R0.determinant());  // sign-preserving scale
  const Eigen::Matrix3d Rs = R0 / s;
  const Eigen::Vector3d t = M.col(3) / s;

  // Project Rs onto SO(3).
  Eigen::JacobiSVD<Eigen::Matrix3d> svd(Rs, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d R = svd.matrixU() * svd.matrixV().transpose();
  if (R.determinant() < 0) {
    Eigen::Matrix3d Uf = svd.matrixU();
    Uf.col(2) *= -1.0;
    R = Uf * svd.matrixV().transpose();
  }

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return T;
}

Eigen::Matrix4d pnp_refine(const Eigen::MatrixXd& X, const Eigen::MatrixXd& u,
                           const Eigen::Matrix3d& K, const Eigen::Matrix4d& T0) {
  const int n = static_cast<int>(X.rows());
  Eigen::Matrix4d T = T0;
  for (int it = 0; it < 15; ++it) {
    const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    const Eigen::Vector3d t = T.block<3, 1>(0, 3);
    Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
    Eigen::Matrix<double, 6, 1> g = Eigen::Matrix<double, 6, 1>::Zero();
    for (int i = 0; i < n; ++i) {
      const Eigen::Vector3d Xw(X(i, 0), X(i, 1), X(i, 2));
      const Eigen::Vector3d Xc = R * Xw + t;
      const Eigen::Vector2d e = Eigen::Vector2d(u(i, 0), u(i, 1)) - pinhole_project(K, Xc);
      Eigen::Matrix<double, 3, 6> dXc;
      dXc.leftCols<3>() = R;
      dXc.rightCols<3>() = -R * hat3(Xw);
      const Eigen::Matrix<double, 2, 6> Ai = pinhole_jacobian(K, Xc) * dXc;
      H += Ai.transpose() * Ai;
      g += Ai.transpose() * e;
    }
    const Eigen::Matrix<double, 6, 1> dxi = H.ldlt().solve(g);
    T = T * se3_exp(dxi);
    if (dxi.norm() < 1e-12) break;
  }
  return T;
}

Eigen::VectorXd sampson_distance(const Eigen::Matrix3d& M, const Eigen::MatrixXd& p1,
                                 const Eigen::MatrixXd& p2) {
  const int n = static_cast<int>(p1.rows());
  Eigen::VectorXd d(n);
  for (int i = 0; i < n; ++i) {
    const Eigen::Vector3d x1(p1(i, 0), p1(i, 1), 1.0);
    const Eigen::Vector3d x2(p2(i, 0), p2(i, 1), 1.0);
    const Eigen::Vector3d a = M * x1;
    const Eigen::Vector3d b = M.transpose() * x2;
    const double r = x2.dot(a);
    const double denom = std::sqrt(a(0) * a(0) + a(1) * a(1) + b(0) * b(0) + b(1) * b(1));
    d(i) = (denom > 0.0) ? std::abs(r) / denom : 0.0;
  }
  return d;
}

std::pair<Eigen::Matrix3d, std::vector<int>> ransac_fundamental(
    const Eigen::MatrixXd& p1, const Eigen::MatrixXd& p2, double threshold, int iters,
    unsigned int seed) {
  const int n = static_cast<int>(p1.rows());
  std::mt19937 gen(seed);
  std::uniform_int_distribution<int> pick(0, n - 1);

  std::vector<int> best_inliers;
  for (int it = 0; it < iters; ++it) {
    // Draw 8 distinct indices.
    std::vector<int> idx;
    while (static_cast<int>(idx.size()) < 8) {
      const int j = pick(gen);
      if (std::find(idx.begin(), idx.end(), j) == idx.end()) idx.push_back(j);
    }
    Eigen::MatrixXd s1(8, 2), s2(8, 2);
    for (int k = 0; k < 8; ++k) {
      s1.row(k) = p1.row(idx[k]);
      s2.row(k) = p2.row(idx[k]);
    }
    const Eigen::Matrix3d F = eight_point(s1, s2);
    const Eigen::VectorXd d = sampson_distance(F, p1, p2);
    std::vector<int> inliers;
    for (int i = 0; i < n; ++i)
      if (d(i) < threshold) inliers.push_back(i);
    if (inliers.size() > best_inliers.size()) best_inliers = std::move(inliers);
  }

  // Refit on the consensus set.
  Eigen::Matrix3d F;
  if (best_inliers.size() >= 8) {
    Eigen::MatrixXd q1(best_inliers.size(), 2), q2(best_inliers.size(), 2);
    for (size_t k = 0; k < best_inliers.size(); ++k) {
      q1.row(k) = p1.row(best_inliers[k]);
      q2.row(k) = p2.row(best_inliers[k]);
    }
    F = eight_point(q1, q2);
  } else {
    F = eight_point(p1, p2);
  }
  return {F, best_inliers};
}

std::tuple<Eigen::Matrix4d, Eigen::MatrixXd, std::vector<int>> two_view_relative_pose(
    const Eigen::Matrix3d& K, const Eigen::MatrixXd& u1, const Eigen::MatrixXd& u2,
    double threshold, int iters, unsigned int seed) {
  Eigen::Matrix3d F;
  std::vector<int> inliers;
  std::tie(F, inliers) = ransac_fundamental(u1, u2, threshold, iters, seed);

  const Eigen::Matrix3d E = K.transpose() * F * K;
  const Eigen::Matrix3d Kinv = K.inverse();
  const int m = static_cast<int>(inliers.size());

  // Inlier normalized rays.
  Eigen::MatrixXd x1(m, 2), x2(m, 2);
  for (int k = 0; k < m; ++k) {
    const Eigen::Vector3d r1 = Kinv * Eigen::Vector3d(u1(inliers[k], 0), u1(inliers[k], 1), 1.0);
    const Eigen::Vector3d r2 = Kinv * Eigen::Vector3d(u2(inliers[k], 0), u2(inliers[k], 1), 1.0);
    x1.row(k) << r1.x() / r1.z(), r1.y() / r1.z();
    x2.row(k) << r2.x() / r2.z(), r2.y() / r2.z();
  }

  Eigen::Matrix3d R;
  Eigen::Vector3d t;
  std::tie(R, t) = recover_pose(E, x1, x2);

  // Triangulate inliers in frame 1.
  const Matrix34d P1 = make_P(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
  const Matrix34d P2 = make_P(R, t);
  Eigen::MatrixXd pts(m, 3);
  for (int k = 0; k < m; ++k)
    pts.row(k) = triangulate_dlt(P1, P2, x1.row(k).transpose(),
                                 x2.row(k).transpose()).transpose();

  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = t;
  return {T, pts, inliers};
}
