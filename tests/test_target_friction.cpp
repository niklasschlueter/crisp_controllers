#include <gtest/gtest.h>
#include <Eigen/Core>
#include "crisp_controllers/utils/target_friction.hpp"

namespace {
const Eigen::MatrixXd kI = Eigen::MatrixXd::Identity(2, 2);
Eigen::VectorXd vec(double a, double b) {
  Eigen::VectorXd v(2);
  v << a, b;
  return v;
}
}  // namespace

TEST(TargetFrictionTest, ZeroWhenTargetRests) {
  const Eigen::VectorXd tau = target_friction_torque(
    vec(0, 0), vec(0, 0), kI, vec(1.0, 0.5), vec(2.0, 1.0), 0.02, 0.06, 1.0);
  EXPECT_TRUE(tau.isZero());
}

TEST(TargetFrictionTest, KineticWhileSlidingWithTargetSign) {
  // Joints already sliding fast: the stick part has decayed, only Fk remains.
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, -1.0), vec(1.0, -1.0), kI, vec(1.0, 0.5), vec(2.0, 1.0), 0.02, 0.06, 1.0);
  EXPECT_NEAR(tau[0], 1.0, 1e-6);
  EXPECT_NEAR(tau[1], -0.5, 1e-6);
}

TEST(TargetFrictionTest, BreakawayWhileStuckAndTargetMoves) {
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, -1.0), vec(0, 0), kI, vec(1.0, 0.5), vec(2.0, 1.0), 0.02, 0.06, 1.0);
  EXPECT_NEAR(tau[0], 2.0, 1e-6);
  EXPECT_NEAR(tau[1], -1.0, 1e-6);
}

TEST(TargetFrictionTest, NoStickPartWithoutBreakaway) {
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, 1.0), vec(0, 0), kI, vec(1.0, 0.5), Eigen::VectorXd(), 0.02, 0.06, 1.0);
  EXPECT_NEAR(tau[0], 1.0, 1e-6);
  EXPECT_NEAR(tau[1], 0.5, 1e-6);
}

TEST(TargetFrictionTest, BreakawayBelowKineticAddsNothing) {
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, 1.0), vec(0, 0), kI, vec(1.0, 0.5), vec(0.5, 0.5), 0.02, 0.06, 1.0);
  EXPECT_NEAR(tau[0], 1.0, 1e-6);
}

TEST(TargetFrictionTest, ScaleMultipliesEverything) {
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, 1.0), vec(0, 0), kI, vec(1.0, 0.5), vec(2.0, 1.0), 0.02, 0.06, 0.5);
  EXPECT_NEAR(tau[0], 1.0, 1e-6);
  EXPECT_NEAR(tau[1], 0.5, 1e-6);
}

TEST(TargetFrictionTest, CoupledDriveMapsBackOntoBothSerialJoints) {
  // DynaArm-style belt: drive 2 turns with serial joints 1 + 2. Only serial joint 1 is targeted
  // to move, so drive 1 and drive 2 both see +v; drive 2's torque lands on serial 1 and 2.
  Eigen::MatrixXd coupling = kI;
  coupling(1, 0) = 1.0;
  const Eigen::VectorXd tau = target_friction_torque(
    vec(1.0, 0.0), vec(1.0, 0.0), coupling, vec(0.3, 0.7), Eigen::VectorXd(), 0.02, 0.06, 1.0);
  EXPECT_NEAR(tau[0], 0.3 + 0.7, 1e-6);
  EXPECT_NEAR(tau[1], 0.7, 1e-6);
}
