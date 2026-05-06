#include <gtest/gtest.h>
#include <Eigen/Core>
#include <string>
#include <vector>
#include "crisp_controllers/utils/friction_model.hpp"
#include "crisp_controllers/utils/friction_loader.hpp"

using crisp_controllers::FrictionState;
using crisp_controllers::load_friction_state;

// --- sigmoidal (3-param, Franka) model ---

TEST(FrictionModelTest, SigmoidalZeroVelocityReturnsZero) {
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd fp1(3);
  fp1 << 0.5, 1.0, 0.3;
  Eigen::VectorXd fp2(3);
  fp2 << 5.0, 10.0, 8.0;
  Eigen::VectorXd fp3(3);
  fp3 << 0.04, 0.03, -0.05;

  Eigen::VectorXd result(3);
  get_friction(dq, fp1, fp2, fp3, result);
  EXPECT_TRUE(result.isApprox(Eigen::VectorXd::Zero(3), 1e-10));
}

TEST(FrictionModelTest, SigmoidalNonzeroVelocity) {
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp1 = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp2 = Eigen::VectorXd::Constant(1, 10.0);
  Eigen::VectorXd fp3 = Eigen::VectorXd::Zero(1);

  Eigen::VectorXd result(1);
  get_friction(dq, fp1, fp2, fp3, result);
  // f(1) = 1/(1+exp(-10)) - 1/(1+1) ≈ 1.0 - 0.5 = 0.5
  EXPECT_NEAR(result(0), 0.5, 0.001);
}

// --- sigmoidal_viscous (5-param, UR10) model ---

TEST(FrictionModelTest, SigmoidalViscousZeroVelocityNonzero) {
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(2);
  Eigen::VectorXd f_v(2);
  f_v << 1.0, 2.0;
  Eigen::VectorXd f_o(2);
  f_o << 0.5, -0.3;
  Eigen::VectorXd f_c(2);
  f_c << 1.0, 1.0;
  Eigen::VectorXd alpha(2);
  alpha << 10.0, 10.0;
  Eigen::VectorXd ni(2);
  ni << 0.0, 0.0;

  Eigen::VectorXd result(2);
  get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni, result);
  // At dq=0, ni=0: f = f_o + f_c/(1+exp(0)) = f_o + f_c/2
  EXPECT_NEAR(result(0), 0.5 + 0.5, 1e-10);
  EXPECT_NEAR(result(1), -0.3 + 0.5, 1e-10);
}

TEST(FrictionModelTest, SigmoidalViscousPositiveVelocity) {
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 2.0);
  Eigen::VectorXd f_v = Eigen::VectorXd::Constant(1, 1.5);
  Eigen::VectorXd f_o = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd f_c = Eigen::VectorXd::Constant(1, 3.0);
  Eigen::VectorXd alpha = Eigen::VectorXd::Constant(1, 50.0);
  Eigen::VectorXd ni = Eigen::VectorXd::Zero(1);

  Eigen::VectorXd result(1);
  get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni, result);
  // f = 1.5*2 + 0 + 3/(1+exp(-100)) ≈ 3 + 3 = 6
  EXPECT_NEAR(result(0), 6.0, 0.01);
}

TEST(FrictionModelTest, SigmoidalViscousNegativeVelocity) {
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, -2.0);
  Eigen::VectorXd f_v = Eigen::VectorXd::Constant(1, 1.5);
  Eigen::VectorXd f_o = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd f_c = Eigen::VectorXd::Constant(1, 3.0);
  Eigen::VectorXd alpha = Eigen::VectorXd::Constant(1, 50.0);
  Eigen::VectorXd ni = Eigen::VectorXd::Zero(1);

  Eigen::VectorXd result(1);
  get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni, result);
  // f = 1.5*(-2) + 0 + 3/(1+exp(100)) ≈ -3 + 0 = -3
  EXPECT_NEAR(result(0), -3.0, 0.01);
}

TEST(FrictionModelTest, SingleJointWorks) {
  // Smoke test: each function writes into a single-element output buffer
  // without crashing.
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 0.5);
  Eigen::VectorXd out(1);

  Eigen::VectorXd fp1 = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp2 = Eigen::VectorXd::Constant(1, 5.0);
  Eigen::VectorXd fp3 = Eigen::VectorXd::Zero(1);
  get_friction(dq, fp1, fp2, fp3, out);
  EXPECT_EQ(out.size(), 1);

  Eigen::VectorXd sv_f_v = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd sv_f_o = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd sv_f_c = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd sv_alpha = Eigen::VectorXd::Constant(1, 5.0);
  Eigen::VectorXd sv_ni = Eigen::VectorXd::Zero(1);
  get_friction_sigmoidal_viscous(dq, sv_f_v, sv_f_o, sv_f_c, sv_alpha, sv_ni, out);
  EXPECT_EQ(out.size(), 1);
}

// --- Stribeck (4-param + per-joint k) model ---

TEST(FrictionModelTest, StribeckZeroVelocityReturnsZero) {
  // tanh(k·0) = 0 ⇒ Stribeck shape contributes 0; viscous term f_v·0 = 0.
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd f_c(3); f_c << 2.5, 5.0, 10.0;
  Eigen::VectorXd f_s(3); f_s << 5.0, 8.0, 12.0;
  Eigen::VectorXd v_s(3); v_s << 0.005, 0.01, 0.05;
  Eigen::VectorXd f_v(3); f_v << 4.0, 25.0, 50.0;
  Eigen::VectorXd k(3);   k   << 1000.0, 500.0, 100.0;

  Eigen::VectorXd result(3);
  get_friction_stribeck(dq, f_c, f_s, v_s, f_v, k, result);
  EXPECT_TRUE(result.isApprox(Eigen::VectorXd::Zero(3), 1e-12));
}

TEST(FrictionModelTest, StribeckHighVelocityApproachesCoulombPlusViscous) {
  // At |dq| ≫ v_s: exp(-(dq/v_s)^2) → 0, tanh(k·dq) → ±1.
  // Result should be ±f_c + f_v·dq.
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd f_c = Eigen::VectorXd::Constant(1, 3.0);
  Eigen::VectorXd f_s = Eigen::VectorXd::Constant(1, 5.0);  // ignored at high v
  Eigen::VectorXd v_s = Eigen::VectorXd::Constant(1, 0.01);
  Eigen::VectorXd f_v = Eigen::VectorXd::Constant(1, 2.0);
  Eigen::VectorXd k   = Eigen::VectorXd::Constant(1, 500.0);

  Eigen::VectorXd result(1);
  get_friction_stribeck(dq, f_c, f_s, v_s, f_v, k, result);
  // Expected: 1·(3 + 2·exp(-10000)) + 2·1 ≈ 3 + 2 = 5
  EXPECT_NEAR(result(0), 5.0, 1e-6);
}

TEST(FrictionModelTest, StribeckDipAtLowVelocity) {
  // The breakaway peak (f_s) must dominate at small dq inside the dip
  // when k is sized so that the tanh smoothing band is much narrower
  // than the Stribeck width: k = 5/v_s.
  // At dq = 0.5·v_s: tanh(5·0.5) = tanh(2.5) ≈ 0.987 (already saturated).
  //                  exp(-(0.5)^2) = exp(-0.25) ≈ 0.779.
  // Friction contribution = 0.987·(f_c + 0.779·(f_s − f_c)) + f_v·dq.
  Eigen::VectorXd v_s = Eigen::VectorXd::Constant(1, 0.005);
  Eigen::VectorXd dq  = Eigen::VectorXd::Constant(1, 0.5 * 0.005);
  Eigen::VectorXd f_c = Eigen::VectorXd::Constant(1, 2.5);
  Eigen::VectorXd f_s = Eigen::VectorXd::Constant(1, 5.0);   // dip = 2.5 Nm
  Eigen::VectorXd f_v = Eigen::VectorXd::Constant(1, 4.0);
  Eigen::VectorXd k   = 5.0 / v_s.array();

  Eigen::VectorXd result(1);
  get_friction_stribeck(dq, f_c, f_s, v_s, f_v, k, result);
  // The Stribeck shape contribution alone (without viscous) should
  // exceed kinetic Coulomb f_c — the dip is captured.
  const double viscous = f_v(0) * dq(0);
  const double stribeck_contrib = result(0) - viscous;
  EXPECT_GT(stribeck_contrib, f_c(0))
    << "Stribeck dip washed out by smoothing — k may be too small relative to v_s.";
}

TEST(FrictionModelTest, StribeckOddSymmetry) {
  // f(-dq) ≈ -f(dq). tanh, viscous, exp((dq/v_s)^2) are all even or odd
  // appropriately ⇒ overall function is odd in dq.
  Eigen::VectorXd dq_pos(3); dq_pos << 0.1, 0.5, 1.0;
  Eigen::VectorXd dq_neg = -dq_pos;
  Eigen::VectorXd f_c(3); f_c << 2.5, 10.0, 16.0;
  Eigen::VectorXd f_s(3); f_s << 5.0, 12.0, 17.0;
  Eigen::VectorXd v_s(3); v_s << 0.005, 0.05, 0.01;
  Eigen::VectorXd f_v(3); f_v << 4.0, 25.0, 56.0;
  Eigen::VectorXd k = 5.0 / v_s.array();

  Eigen::VectorXd r_pos(3), r_neg(3);
  get_friction_stribeck(dq_pos, f_c, f_s, v_s, f_v, k, r_pos);
  get_friction_stribeck(dq_neg, f_c, f_s, v_s, f_v, k, r_neg);
  EXPECT_TRUE(r_neg.isApprox(-r_pos, 1e-10));
}

// --- Loader: validation behavior ---

namespace {
// Mirror of the auto-generated Params::Friction shape. Only the fields the
// loader reads are present; the loader is templated on `auto&` so this
// duck-typed struct is sufficient.
struct FakeFrictionParams {
  std::string model_type;
  std::vector<double> fp1, fp2, fp3;
  std::vector<double> f_v, f_o, f_c, alpha, ni;
  struct Stribeck {
    std::vector<double> f_c, f_s, v_s, f_v, k;
  } stribeck;
};
}  // namespace

TEST(FrictionLoaderTest, StribeckSizeMismatchRejected) {
  FakeFrictionParams p;
  p.model_type = "stribeck";
  p.stribeck.f_c = {2.5, 5.0, 10.0};      // size 3
  p.stribeck.f_s = {5.0, 8.0};             // size 2 — mismatch
  p.stribeck.v_s = {0.005, 0.01, 0.05};
  p.stribeck.f_v = {4.0, 25.0, 50.0};

  FrictionState state;
  std::string err;
  EXPECT_FALSE(load_friction_state(p, /*nv=*/3, state, err));
  EXPECT_FALSE(err.empty()) << "Loader should populate error_msg on failure";
}

TEST(FrictionLoaderTest, StribeckRejectsNonPositiveK) {
  // k <= 0 silently breaks the Stribeck shape (tanh(0·dq) = 0 for k=0,
  // sign-inverted friction for k<0). Loader must reject.
  for (double bad_k : {0.0, -1.0}) {
    FakeFrictionParams p;
    p.model_type = "stribeck";
    p.stribeck.f_c = {2.5, 5.0};
    p.stribeck.f_s = {5.0, 8.0};
    p.stribeck.v_s = {0.005, 0.05};
    p.stribeck.f_v = {4.0, 25.0};
    p.stribeck.k   = {1000.0, bad_k};

    FrictionState state;
    std::string err;
    EXPECT_FALSE(load_friction_state(p, /*nv=*/2, state, err))
      << "Loader should reject stribeck.k = " << bad_k;
    EXPECT_FALSE(err.empty());
  }
}

TEST(FrictionLoaderTest, StribeckAutoDefaultsK) {
  // If k is left empty, loader should default to k_j = 5/v_s_j.
  FakeFrictionParams p;
  p.model_type = "stribeck";
  p.stribeck.f_c = {2.5, 5.0};
  p.stribeck.f_s = {5.0, 8.0};
  p.stribeck.v_s = {0.005, 0.05};
  p.stribeck.f_v = {4.0, 25.0};
  // p.stribeck.k empty.

  FrictionState state;
  std::string err;
  ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;
  EXPECT_NEAR(state.s_k(0), 5.0 / 0.005, 1e-9);
  EXPECT_NEAR(state.s_k(1), 5.0 / 0.05,  1e-9);
}

// --- Loader: atomicity + runtime-reload safety ---
//
// The RT-loop param-refresh hook calls load_friction_state on every
// is_old() trigger. A rejected update must leave FrictionState unchanged.

TEST(FrictionLoaderTest, AtomicityOnInvalidVsLeavesPreviousState) {
  // First load a valid stribeck config, capture the resulting state.
  FakeFrictionParams good;
  good.model_type = "stribeck";
  good.stribeck.f_c = {2.5, 5.0};
  good.stribeck.f_s = {5.0, 8.0};
  good.stribeck.v_s = {0.005, 0.05};
  good.stribeck.f_v = {4.0, 25.0};

  FrictionState state;
  std::string err;
  ASSERT_TRUE(load_friction_state(good, /*nv=*/2, state, err)) << err;
  const Eigen::VectorXd expected_f_c = state.s_f_c;
  const Eigen::VectorXd expected_v_s = state.s_v_s;

  // Now attempt a load with negative v_s — must be rejected, state unchanged.
  FakeFrictionParams bad = good;
  bad.stribeck.v_s = {-0.01, 0.05};
  EXPECT_FALSE(load_friction_state(bad, /*nv=*/2, state, err));
  EXPECT_TRUE(state.s_f_c.isApprox(expected_f_c))
    << "Rejected update must leave s_f_c unchanged";
  EXPECT_TRUE(state.s_v_s.isApprox(expected_v_s))
    << "Rejected update must leave s_v_s unchanged";
}

TEST(FrictionLoaderTest, RuntimeParamUpdateAppliesNewValues) {
  // Simulate the RT-loop reload path: load valid stribeck twice with
  // different per-joint values; the second load must produce the new
  // values (and must not allocate beyond the first call's storage).
  FakeFrictionParams p;
  p.model_type = "stribeck";
  p.stribeck.f_c = {2.5, 5.0};
  p.stribeck.f_s = {5.0, 8.0};
  p.stribeck.v_s = {0.005, 0.05};
  p.stribeck.f_v = {4.0, 25.0};

  FrictionState state;
  std::string err;
  ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;

  // Bump f_c — what the user does at runtime when tweaking comp magnitude.
  p.stribeck.f_c = {16.0, 5.0};
  ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;
  EXPECT_NEAR(state.s_f_c(0), 16.0, 1e-9);
  EXPECT_NEAR(state.s_f_c(1),  5.0, 1e-9);
}

TEST(FrictionLoaderTest, ReloadRejectsModelTypeChange) {
  // reload_friction_state is the RT-loop variant that enforces a lock
  // on model_type. Switching model_type at runtime would silently change
  // the friction-torque output at 500 Hz; reject loudly instead.
  FakeFrictionParams p;
  p.model_type = "stribeck";
  p.stribeck.f_c = {2.5, 5.0};
  p.stribeck.f_s = {5.0, 8.0};
  p.stribeck.v_s = {0.005, 0.05};
  p.stribeck.f_v = {4.0, 25.0};

  FrictionState state;
  std::string err;
  ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;
  const Eigen::VectorXd expected_f_c = state.s_f_c;

  // User attempts to switch to sigmoidal at runtime — must be rejected.
  p.model_type = "sigmoidal";
  p.fp1 = {1.0, 2.0};
  p.fp2 = {10.0, 20.0};
  p.fp3 = {0.1, 0.2};
  EXPECT_FALSE(crisp_controllers::reload_friction_state(
    p, /*nv=*/2, /*locked_model_type=*/"stribeck", state, err));
  EXPECT_NE(err.find("model_type change"), std::string::npos);
  // State must still hold the original stribeck values.
  EXPECT_EQ(state.type, FrictionModelType::kStribeck);
  EXPECT_TRUE(state.s_f_c.isApprox(expected_f_c));

  // Same model_type, valid values — must succeed.
  p.model_type = "stribeck";
  p.stribeck.f_c = {16.0, 5.0};
  ASSERT_TRUE(crisp_controllers::reload_friction_state(
    p, /*nv=*/2, /*locked_model_type=*/"stribeck", state, err)) << err;
  EXPECT_NEAR(state.s_f_c(0), 16.0, 1e-9);
}

TEST(FrictionLoaderTest, RepeatedReloadsIdempotent) {
  // Loading the same valid config N times must yield the same state.
  FakeFrictionParams p;
  p.model_type = "stribeck";
  p.stribeck.f_c = {2.5, 5.0};
  p.stribeck.f_s = {5.0, 8.0};
  p.stribeck.v_s = {0.005, 0.05};
  p.stribeck.f_v = {4.0, 25.0};

  FrictionState state;
  std::string err;
  ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;
  const Eigen::VectorXd ref_f_c = state.s_f_c;
  const Eigen::VectorXd ref_k   = state.s_k;

  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(load_friction_state(p, /*nv=*/2, state, err)) << err;
    EXPECT_TRUE(state.s_f_c.isApprox(ref_f_c)) << "drift at iter " << i;
    EXPECT_TRUE(state.s_k.isApprox(ref_k))     << "drift at iter " << i;
  }
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
