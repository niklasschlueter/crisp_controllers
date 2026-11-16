#include <gtest/gtest.h>

#include <crisp_controllers/utils/joint_ordering.hpp>
#include <pinocchio/multibody/joint/joints.hpp>

using crisp_controllers::check_joint_order;

namespace {

// Serial chain j1 -> j2 -> j3, so Pinocchio's order is [j1, j2, j3].
pinocchio::Model make_chain() {
  pinocchio::Model model;
  pinocchio::JointIndex parent = 0;
  for (const std::string name : {"j1", "j2", "j3"}) {
    parent = model.addJoint(parent, pinocchio::JointModelRZ(), pinocchio::SE3::Identity(), name);
  }
  return model;
}

}  // namespace

TEST(JointOrdering, AcceptsPinocchioOrder) {
  EXPECT_TRUE(check_joint_order(make_chain(), {"j1", "j2", "j3"}).empty());
}

TEST(JointOrdering, RejectsOtherOrderAndNamesRequiredOrder) {
  const auto error = check_joint_order(make_chain(), {"j1", "j3", "j2"});
  EXPECT_NE(error.find("[j1, j2, j3]"), std::string::npos) << error;
}

TEST(JointOrdering, RejectsWrongCountUnknownAndDuplicateNames) {
  const auto model = make_chain();
  EXPECT_FALSE(check_joint_order(model, {"j1", "j2"}).empty());
  EXPECT_FALSE(check_joint_order(model, {"j1", "j2", "j4"}).empty());
  EXPECT_FALSE(check_joint_order(model, {"j1", "j2", "j2"}).empty());
}
