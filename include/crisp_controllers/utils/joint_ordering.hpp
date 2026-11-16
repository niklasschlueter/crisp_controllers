#pragma once

#include <string>
#include <vector>

#include <pinocchio/multibody/model.hpp>

namespace crisp_controllers {

/**
 * Controllers exchange per-joint values with ros2_control in the order of their
 * `joints` parameter but index q, v and tau by the Pinocchio model's order. The
 * two must match, or each joint receives another joint's torque. Requiring the
 * match (instead of remapping everywhere) keeps the control loops index-free and
 * makes a misordered config fail at configure rather than on the robot.
 *
 * @param model Reduced model containing only `joints`, all with one DoF.
 * @return Empty if `joints[i]` has velocity index i for all i, else an error
 *         message naming the required order.
 */
inline std::string check_joint_order(
  const pinocchio::Model & model, const std::vector<std::string> & joints) {
  bool ok = static_cast<size_t>(model.nv) == joints.size();
  for (size_t i = 0; ok && i < joints.size(); ++i) {
    ok = model.existJointName(joints[i]) &&
         model.joints[model.getJointId(joints[i])].idx_v() == static_cast<int>(i);
  }
  if (ok) {
    return {};
  }

  std::string required;
  for (pinocchio::JointIndex id = 1; id < model.joints.size(); ++id) {
    required += (id > 1 ? ", " : "") + model.names[id];
  }
  return "Parameter 'joints' must list the joints in the URDF tree order Pinocchio uses: [" +
         required +
         "]. Reorder per-joint parameter arrays (e.g. friction) and any publisher sending "
         "positional joint values the same way.";
}

}  // namespace crisp_controllers
