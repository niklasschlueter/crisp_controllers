#include <gtest/gtest.h>
#include <Eigen/Core>
#include "crisp_controllers/utils/friction_model.hpp"

// --- sigmoidal (3-param, Franka) model ---

TEST(FrictionModelTest, SigmoidalZeroVelocityReturnsZero) {
  Eigen::VectorXd dq = Eigen::VectorXd::Zero(3);
  Eigen::VectorXd fp1(3);
  fp1 << 0.5, 1.0, 0.3;
  Eigen::VectorXd fp2(3);
  fp2 << 5.0, 10.0, 8.0;
  Eigen::VectorXd fp3(3);
  fp3 << 0.04, 0.03, -0.05;

  Eigen::VectorXd result = get_friction(dq, fp1, fp2, fp3);
  EXPECT_TRUE(result.isApprox(Eigen::VectorXd::Zero(3), 1e-10));
}

TEST(FrictionModelTest, SigmoidalNonzeroVelocity) {
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp1 = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp2 = Eigen::VectorXd::Constant(1, 10.0);
  Eigen::VectorXd fp3 = Eigen::VectorXd::Zero(1);

  Eigen::VectorXd result = get_friction(dq, fp1, fp2, fp3);
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

  Eigen::VectorXd result = get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni);
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

  Eigen::VectorXd result = get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni);
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

  Eigen::VectorXd result = get_friction_sigmoidal_viscous(dq, f_v, f_o, f_c, alpha, ni);
  // f = 1.5*(-2) + 0 + 3/(1+exp(100)) ≈ -3 + 0 = -3
  EXPECT_NEAR(result(0), -3.0, 0.01);
}

TEST(FrictionModelTest, SingleJointWorks) {
  Eigen::VectorXd dq = Eigen::VectorXd::Constant(1, 0.5);

  Eigen::VectorXd fp1 = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd fp2 = Eigen::VectorXd::Constant(1, 5.0);
  Eigen::VectorXd fp3 = Eigen::VectorXd::Zero(1);
  EXPECT_EQ(get_friction(dq, fp1, fp2, fp3).size(), 1);

  Eigen::VectorXd sv_f_v = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd sv_f_o = Eigen::VectorXd::Zero(1);
  Eigen::VectorXd sv_f_c = Eigen::VectorXd::Constant(1, 1.0);
  Eigen::VectorXd sv_alpha = Eigen::VectorXd::Constant(1, 5.0);
  Eigen::VectorXd sv_ni = Eigen::VectorXd::Zero(1);
  EXPECT_EQ(get_friction_sigmoidal_viscous(dq, sv_f_v, sv_f_o, sv_f_c, sv_alpha, sv_ni).size(), 1);
}

int main(int argc, char ** argv) {
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
