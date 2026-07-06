// SO(3): the rotation group. Declarations shared by the top-level (holed) and
// solution implementations; only the .cpp bodies differ.
//
// Convention (used by the whole a14 module): a rotation is a 3x3 matrix R with
// R^T R = I, det R = +1. A rotation vector w (axis-angle, |w| = angle in radians)
// maps to R via the exponential. The right perturbation T |+ xi = T * exp(xi) is
// the module-wide choice, so so3_right_jacobian is the Jacobian used downstream.
#pragma once
#include <Eigen/Dense>

// Skew-symmetric (hat) and its inverse (vee): hat3(w) v = w x v.
Eigen::Matrix3d hat3(const Eigen::Vector3d& w);
Eigen::Vector3d vee3(const Eigen::Matrix3d& W);

// Exponential / logarithm: R = exp([w]_x), w = log(R).
Eigen::Matrix3d so3_exp(const Eigen::Vector3d& w);
Eigen::Vector3d so3_log(const Eigen::Matrix3d& R);

// Left and right Jacobians of SO(3) and their inverses.
// Defining property of the right Jacobian: exp(w + dw) ~ exp(w) * exp(J_r(w) dw).
// See the README (the Jacobians section) for the closed forms and their relations.
Eigen::Matrix3d so3_left_jacobian(const Eigen::Vector3d& w);
Eigen::Matrix3d so3_left_jacobian_inv(const Eigen::Vector3d& w);
Eigen::Matrix3d so3_right_jacobian(const Eigen::Vector3d& w);
Eigen::Matrix3d so3_right_jacobian_inv(const Eigen::Vector3d& w);
