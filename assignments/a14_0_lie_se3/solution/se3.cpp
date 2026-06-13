#include "se3.hpp"

#include "so3.hpp"

Eigen::Matrix4d hat6(const Vector6d& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  Eigen::Matrix4d X = Eigen::Matrix4d::Zero();
  X.block<3, 3>(0, 0) = hat3(theta);
  X.block<3, 1>(0, 3) = rho;
  return X;
}

Vector6d vee6(const Eigen::Matrix4d& Xi) {
  Vector6d xi;
  xi.head<3>() = Xi.block<3, 1>(0, 3);
  xi.tail<3>() = vee3(Xi.block<3, 3>(0, 0));
  return xi;
}

Eigen::Matrix4d se3_exp(const Vector6d& xi) {
  const Eigen::Vector3d rho = xi.head<3>();
  const Eigen::Vector3d theta = xi.tail<3>();
  Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
  T.block<3, 3>(0, 0) = so3_exp(theta);
  T.block<3, 1>(0, 3) = so3_left_jacobian(theta) * rho;
  return T;
}

Vector6d se3_log(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  const Eigen::Vector3d theta = so3_log(R);
  Vector6d xi;
  xi.head<3>() = so3_left_jacobian_inv(theta) * t;
  xi.tail<3>() = theta;
  return xi;
}

Matrix6d se3_adjoint(const Eigen::Matrix4d& T) {
  const Eigen::Matrix3d R = T.block<3, 3>(0, 0);
  const Eigen::Vector3d t = T.block<3, 1>(0, 3);
  Matrix6d Ad = Matrix6d::Zero();
  Ad.block<3, 3>(0, 0) = R;
  Ad.block<3, 3>(0, 3) = hat3(t) * R;
  Ad.block<3, 3>(3, 3) = R;
  return Ad;
}

Eigen::Matrix4d se3_boxplus(const Eigen::Matrix4d& T, const Vector6d& xi) {
  return T * se3_exp(xi);
}

Vector6d se3_boxminus(const Eigen::Matrix4d& T2, const Eigen::Matrix4d& T1) {
  return se3_log(T1.inverse() * T2);
}
