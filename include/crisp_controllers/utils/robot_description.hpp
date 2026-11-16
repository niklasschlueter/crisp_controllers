#pragma once

#include <chrono>
#include <future>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace crisp_controllers {

/**
 * URDF for a controller's on_configure. Prefers the description the controller
 * manager already holds (pass get_robot_description()), which needs no IPC.
 * Falls back to robot_state_publisher's parameter with timeouts: an unbounded
 * wait there hangs configure forever when the node is absent, and blocking on the
 * reply from inside the controller manager's executor could deadlock.
 *
 * @return The URDF, or an empty string if neither source provided one.
 */
inline std::string fetch_robot_description(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  const std::string & from_controller_manager) {
  if (!from_controller_manager.empty()) {
    return from_controller_manager;
  }
  auto client = std::make_shared<rclcpp::AsyncParametersClient>(node, "robot_state_publisher");
  if (!client->wait_for_service(std::chrono::seconds(1))) {
    return {};
  }
  auto future = client->get_parameters({"robot_description"});
  if (future.wait_for(std::chrono::seconds(2)) != std::future_status::ready) {
    return {};
  }
  const auto result = future.get();
  return result.empty() ? std::string{} : result[0].value_to_string();
}

}  // namespace crisp_controllers
