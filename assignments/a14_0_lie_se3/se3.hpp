// SE(3): rigid-body transforms. Declarations shared by the top-level (holed) and
// solution implementations; only the .cpp bodies differ.
//
// A transform is a 4x4 matrix T = [[R, t],[0, 1]], R in SO(3), t a translation.
// A twist is a 6-vector xi = [rho; theta] with the translation part rho FIRST and
// the rotation part theta SECOND. The adjoint and Jacobians below all assume this
// ordering, so a downstream sign/argument error usually traces back to swapping it.
#pragma once
#include <Eigen/Dense>

using Vector6d = Eigen::Matrix<double, 6, 1>;
using Matrix6d = Eigen::Matrix<double, 6, 6>;

// se(3) hat / vee for xi = [rho; theta].
Eigen::Matrix4d hat6(const Vector6d& xi);
Vector6d vee6(const Eigen::Matrix4d& Xi);

// Exponential / logarithm: T = exp(hat6(xi)), xi = log(T). See the README (exp/log) for
// how the translation part couples through the rotation.
Eigen::Matrix4d se3_exp(const Vector6d& xi);
Vector6d se3_log(const Eigen::Matrix4d& T);

// Adjoint: Ad_T maps a body/right twist to a spatial/left one,
// T * exp(xi) = exp(Ad_T xi) * T. See the README (the adjoint) for the closed form.
Matrix6d se3_adjoint(const Eigen::Matrix4d& T);

// Manifold retraction (right perturbation) and its inverse.
Eigen::Matrix4d se3_boxplus(const Eigen::Matrix4d& T, const Vector6d& xi);  // T * exp(xi)
Vector6d se3_boxminus(const Eigen::Matrix4d& T2, const Eigen::Matrix4d& T1);  // log(T1^-1 T2)
