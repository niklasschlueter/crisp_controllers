#pragma once

// Shared friction-model loader + dispatcher for the three crisp_controllers.
//
// Each controller (cartesian, cartesian_admittance, torque_feedback) has its
// own auto-generated `Params::Friction` struct (different parent class, but
// the friction sub-struct fields are identical across all three). This
// header templates the loading code over `FrictionParams` so the per-
// controller .cpp files don't repeat ~30 lines each of string-matching +
// array-validation + Eigen::Map.
//
// Usage in on_configure():
//
//   FrictionState friction_state_;          // member of the controller
//   std::string err;
//   if (!load_friction_state(params_.friction, model_.nv,
//                            friction_state_, err)) {
//     RCLCPP_ERROR(get_node()->get_logger(),
//                  "Friction config invalid: %s", err.c_str());
//     return CallbackReturn::ERROR;   // RT loop has no NaN guard
//   }
//
// Usage in the realtime loop:
//
//   tau_friction = compute_friction(friction_state_, dq);

#include <Eigen/Core>
#include <cstddef>
#include <string>
#include <vector>

#include "crisp_controllers/utils/friction_model.hpp"

namespace crisp_controllers {

struct FrictionState {
  FrictionModelType type = FrictionModelType::kSignoidal;

  // 3-param sigmoidal (kSignoidal).
  Eigen::VectorXd fp1, fp2, fp3;

  // 5-param sigmoidal_viscous (kSigmoidalViscous).
  Eigen::VectorXd f_v, f_o, f_c, alpha, ni;

  // 4-param Stribeck (kStribeck). `s_` prefix to avoid shadowing the
  // sigmoidal_viscous fields above.
  Eigen::VectorXd s_f_c, s_f_s, s_v_s, s_f_v, s_k;
};

namespace detail {

inline Eigen::VectorXd to_eigen(const std::vector<double> & v) {
  return Eigen::Map<const Eigen::VectorXd>(v.data(), static_cast<int>(v.size()));
}

// Returns true iff `v` has at least nv entries; otherwise sets error_msg.
inline bool require_size(const std::vector<double> & v, int nv,
                         const char * name, std::string & error_msg) {
  if (static_cast<int>(v.size()) < nv) {
    error_msg = std::string{name} + " must have size >= " + std::to_string(nv)
                + " (got " + std::to_string(v.size()) + ")";
    return false;
  }
  return true;
}

}  // namespace detail

template <typename FrictionParams>
inline bool load_friction_state(const FrictionParams & p, int nv,
                                FrictionState & out, std::string & error_msg) {
  using detail::to_eigen;
  using detail::require_size;

  const std::string & mt = p.model_type;

  // Each branch validates fully against the source std::vector<double>
  // BEFORE writing to `out`. A rejected update returns early and leaves
  // `out` unchanged. The Eigen assignments later in each branch are
  // alloc-free in steady state because the caller's model_.nv (and thus
  // the existing vector size) is invariant after on_configure — that
  // invariant is what makes this safe to call from the RT loop, not
  // some structural exception-safety guarantee.
  if (mt == "sigmoidal") {
    if (!require_size(p.fp1, nv, "friction.fp1", error_msg)) return false;
    if (!require_size(p.fp2, nv, "friction.fp2", error_msg)) return false;
    if (!require_size(p.fp3, nv, "friction.fp3", error_msg)) return false;
    // All validation passed — atomic write.
    out.type = FrictionModelType::kSignoidal;
    out.fp1 = to_eigen(p.fp1).head(nv);
    out.fp2 = to_eigen(p.fp2).head(nv);
    out.fp3 = to_eigen(p.fp3).head(nv);
    return true;
  }

  if (mt == "sigmoidal_viscous") {
    if (!require_size(p.f_v, nv, "friction.f_v", error_msg)) return false;
    if (!require_size(p.f_o, nv, "friction.f_o", error_msg)) return false;
    if (!require_size(p.f_c, nv, "friction.f_c", error_msg)) return false;
    if (!require_size(p.alpha, nv, "friction.alpha", error_msg)) return false;
    if (!require_size(p.ni, nv, "friction.ni", error_msg)) return false;
    // All validation passed — atomic write.
    out.type = FrictionModelType::kSigmoidalViscous;
    out.f_v = to_eigen(p.f_v).head(nv);
    out.f_o = to_eigen(p.f_o).head(nv);
    out.f_c = to_eigen(p.f_c).head(nv);
    out.alpha = to_eigen(p.alpha).head(nv);
    out.ni = to_eigen(p.ni).head(nv);
    return true;
  }

  if (mt == "stribeck") {
    if (!require_size(p.stribeck.f_c, nv, "friction.stribeck.f_c", error_msg)) return false;
    if (!require_size(p.stribeck.f_s, nv, "friction.stribeck.f_s", error_msg)) return false;
    if (!require_size(p.stribeck.v_s, nv, "friction.stribeck.v_s", error_msg)) return false;
    if (!require_size(p.stribeck.f_v, nv, "friction.stribeck.f_v", error_msg)) return false;
    // v_s must be strictly positive (divisor in the Stribeck exp term).
    for (int i = 0; i < nv; ++i) {
      if (p.stribeck.v_s[i] <= 0.0) {
        error_msg = "friction.stribeck.v_s must be strictly positive";
        return false;
      }
    }
    // If `k` is provided, validate size + positivity against the source.
    // k <= 0 produces tanh(0·dq) = 0 (no Stribeck shape) or sign-flipped
    // friction (negative damping at zero) — both are unsafe.
    const bool k_default = p.stribeck.k.empty();
    if (!k_default) {
      if (!require_size(p.stribeck.k, nv, "friction.stribeck.k", error_msg)) return false;
      for (int i = 0; i < nv; ++i) {
        if (p.stribeck.k[i] <= 0.0) {
          error_msg = "friction.stribeck.k must be strictly positive (per-joint).";
          return false;
        }
      }
    }
    // All validation passed — atomic write.
    out.type = FrictionModelType::kStribeck;
    out.s_f_c = to_eigen(p.stribeck.f_c).head(nv);
    out.s_f_s = to_eigen(p.stribeck.f_s).head(nv);
    out.s_v_s = to_eigen(p.stribeck.v_s).head(nv);
    out.s_f_v = to_eigen(p.stribeck.f_v).head(nv);
    // Default k_j = 5/v_s_j: smoothing band 1/5 of Stribeck width, keeps
    // the dip from being washed out by tanh. v_s positivity already verified.
    if (k_default) {
      out.s_k = (5.0 / out.s_v_s.array()).matrix();
    } else {
      out.s_k = to_eigen(p.stribeck.k).head(nv);
    }
    return true;
  }

  error_msg = "unknown friction.model_type \"" + mt
              + "\"; expected one of \"sigmoidal\", \"sigmoidal_viscous\", \"stribeck\"";
  return false;
}

// RT-loop variant of load_friction_state: enforces a lock on
// `model_type`. Use this from the controller's `is_old()` block; use
// the un-locked overload from on_configure() to capture the initial
// model. Rationale for the lock is semantic, not RT-allocation: a
// silent mid-flight switch from e.g. stribeck to sigmoidal would drop
// the f_v term and discontinuously change the friction-torque output
// at 500 Hz — dangerous on a real arm. Better to reject loudly so the
// operator notices the no-op.
template <typename FrictionParams>
inline bool reload_friction_state(const FrictionParams & p, int nv,
                                  const std::string & locked_model_type,
                                  FrictionState & out,
                                  std::string & error_msg) {
  if (p.model_type != locked_model_type) {
    error_msg = "friction.model_type change ('" + locked_model_type
              + "' -> '" + p.model_type + "') rejected — restart the "
                "controller to switch friction models";
    return false;
  }
  return load_friction_state(p, nv, out, error_msg);
}

// Writes the per-joint friction torque into `out` (must already be sized
// to match `dq`). Zero heap allocation: the dispatched friction-model
// function writes directly into `out`'s storage. Caller pattern in the
// realtime loop:
//
//   compute_friction(friction_state_, dq, tau_friction);   // tau_friction preallocated
inline void compute_friction(const FrictionState & s,
                             const Eigen::Ref<const Eigen::VectorXd> & dq,
                             Eigen::Ref<Eigen::VectorXd> out) {
  switch (s.type) {
    case FrictionModelType::kSignoidal:
      get_friction(dq, s.fp1, s.fp2, s.fp3, out);
      return;
    case FrictionModelType::kSigmoidalViscous:
      get_friction_sigmoidal_viscous(dq, s.f_v, s.f_o, s.f_c, s.alpha, s.ni, out);
      return;
    case FrictionModelType::kStribeck:
      get_friction_stribeck(dq, s.s_f_c, s.s_f_s, s.s_v_s, s.s_f_v, s.s_k, out);
      return;
  }
  // Unreachable under -Werror=switch (every enumerator handled above).
  out.setZero();
}

}  // namespace crisp_controllers
