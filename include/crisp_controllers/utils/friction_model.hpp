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
//
// stribeck (4-param): Closed-form Stribeck with tanh-smoothed sign for
//   chatter-free runtime evaluation. Captures the breakaway peak (f_s)
//   distinct from kinetic Coulomb (f_c) plus viscous (f_v). Per-joint
//   smoothing sharpness `k` so the tanh transition stays inside the
//   Stribeck width v_s — a fixed k washes out the dip on small-v_s joints.

#include <Eigen/src/Core/GlobalFunctions.h>
#include <Eigen/src/Core/Matrix.h>
#include <Eigen/Core>

enum class FrictionModelType { kSignoidal, kSigmoidalViscous, kStribeck };

// All three model functions write into the caller-supplied output buffer
// to avoid heap allocation in the realtime loop. `out` must already be
// sized to match `dq`. `out.noalias() = ...` skips Eigen's aliasing check
// (no aliasing risk: `out` doesn't appear on the right-hand side).

// 3-param sigmoidal model (Franka default).
// f(dq) = fp1/(1+exp(-fp2*(dq+fp3))) - fp1/(1+exp(-fp2*fp3))
// Forces f(0)=0 by construction.
inline void get_friction(
  const Eigen::Ref<const Eigen::VectorXd> & dq,
  const Eigen::Ref<const Eigen::VectorXd> & fp1,
  const Eigen::Ref<const Eigen::VectorXd> & fp2,
  const Eigen::Ref<const Eigen::VectorXd> & fp3,
  Eigen::Ref<Eigen::VectorXd> out) {
  out.array() = fp1.array()
                  / (1.0 + (-fp2.array() * (dq.array() + fp3.array())).exp())
              - fp1.array()
                  / (1.0 + (-fp2.array() * fp3.array()).exp());
}

// 5-param sigmoidal+viscous model (UR10).
// f(dq) = f_v*dq + f_o + f_c/(1+exp(-alpha*(dq+ni)))
// Does NOT force f(0)=0 -- the offset f_o captures static friction asymmetry.
// Parameters must be in torque domain (Nm). If identified in current domain,
// multiply by the per-joint motor gain K_i before passing them here.
inline void get_friction_sigmoidal_viscous(
  const Eigen::Ref<const Eigen::VectorXd> & dq,
  const Eigen::Ref<const Eigen::VectorXd> & f_v,
  const Eigen::Ref<const Eigen::VectorXd> & f_o,
  const Eigen::Ref<const Eigen::VectorXd> & f_c,
  const Eigen::Ref<const Eigen::VectorXd> & alpha,
  const Eigen::Ref<const Eigen::VectorXd> & ni,
  Eigen::Ref<Eigen::VectorXd> out) {
  out.array() = f_v.array() * dq.array() + f_o.array()
              + f_c.array()
                  / (1.0 + (-alpha.array() * (dq.array() + ni.array())).exp());
}

// 4-param Stribeck with per-joint tanh smoothing (chatter-free at zero
// crossings). Forces f(0)=0 by construction (tanh(0)=0 + viscous term=0).
//
//   f(dq)_j = tanh(k_j · dq_j) · [f_c_j + (f_s_j − f_c_j) · exp(−(dq_j/v_s_j)²)]
//             + f_v_j · dq_j
//
//   f_c = kinetic Coulomb [Nm]
//   f_s = breakaway / static-friction asymptote [Nm]
//   v_s = Stribeck velocity (width of the dip) [rad/s]
//   f_v = viscous friction [Nm·s/rad]
//   k   = tanh sharpness [1/(rad/s)]; recommend k_j ≈ 5/v_s_j so the
//         smoothing band 1/k_j is much narrower than the Stribeck width
//         v_s_j (otherwise the smoothing washes out the dip).
inline void get_friction_stribeck(
  const Eigen::Ref<const Eigen::VectorXd> & dq,
  const Eigen::Ref<const Eigen::VectorXd> & f_c,
  const Eigen::Ref<const Eigen::VectorXd> & f_s,
  const Eigen::Ref<const Eigen::VectorXd> & v_s,
  const Eigen::Ref<const Eigen::VectorXd> & f_v,
  const Eigen::Ref<const Eigen::VectorXd> & k,
  Eigen::Ref<Eigen::VectorXd> out) {
  out.array() =
      (k.array() * dq.array()).tanh()
        * (f_c.array()
           + (f_s.array() - f_c.array())
               * (-(dq.array() / v_s.array()).square()).exp())
    + f_v.array() * dq.array();
}
