#pragma once

// Friction compensation models for torque-controlled robots.
//
// sigmoidal (3-param): Based on the Franka Emika Panda dynamics model by Cognetti.
//   https://github.com/marcocognetti/FrankaEmikaPandaDynModel
//
// sigmoidal_viscous (5-param): Based on the UR10 identified model by Petrone,
//   Ferrentino, Chiacchio, "The Dynamic Model of the UR10 Robot and its ROS2
//   Integration", arXiv:2502.11940, IEEE Trans. Industrial Informatics, 2025.
//   Code: https://github.com/unisa-acg/inverse-dynamics-solver

#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Core>

enum class FrictionModelType { kSignoidal, kSigmoidalViscous };

// 3-param sigmoidal model (Franka default).
// f(dq) = fp1/(1+exp(-fp2*(dq+fp3))) - fp1/(1+exp(-fp2*fp3))
// Forces f(0)=0 by construction.
inline Eigen::VectorXd get_friction(
  const Eigen::VectorXd & dq, Eigen::VectorXd fp1, Eigen::VectorXd fp2, Eigen::VectorXd fp3) {
  return (fp1.array() /
            (Eigen::VectorXd::Ones(dq.size()).array() +
             (-fp2.array() * (dq.array() + fp3.array())).exp()) -
          fp1.array() /
            (Eigen::VectorXd::Ones(dq.size()).array() + (-fp2.array() * fp3.array()).exp()))
    .matrix();
}

// 5-param sigmoidal+viscous model (UR10).
// f(dq) = f_v*dq + f_o + f_c/(1+exp(-alpha*(dq+ni)))
// Does NOT force f(0)=0 -- the offset f_o captures static friction asymmetry.
// Parameters must be in torque domain (Nm). If identified in current domain,
// multiply by the per-joint motor gain K_i before passing them here.
inline Eigen::VectorXd get_friction_sigmoidal_viscous(
  const Eigen::VectorXd & dq,
  const Eigen::VectorXd & f_v,
  const Eigen::VectorXd & f_o,
  const Eigen::VectorXd & f_c,
  const Eigen::VectorXd & alpha,
  const Eigen::VectorXd & ni) {
  return (f_v.array() * dq.array() + f_o.array() +
          f_c.array() /
            (Eigen::VectorXd::Ones(dq.size()).array() +
             (-alpha.array() * (dq.array() + ni.array())).exp()))
    .matrix();
}
