#pragma once

/**
 * @file pose_safety.hpp
 * @brief Utility functions for validating incoming target pose messages.
 *
 * These functions are shared between CartesianController and
 * CartesianAdmittanceController to avoid duplicating safety logic.
 */

#include <atomic>
#include <cmath>

#include <Eigen/Dense>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <pinocchio/spatial/se3.hpp>
#include <rclcpp/rclcpp.hpp>

/**
 * @brief Validate an incoming target pose message before applying it.
 *
 * Runs up to two checks (each independently enable/disable-able):
 *  1. NaN / Inf check    — rejects poses with non-finite position or orientation values.
 *  2. Max-distance check — rejects poses farther than max_distance_m from the current
 *                          end-effector position.
 *
 * @param msg                 Incoming PoseStamped message to validate.
 * @param end_effector_pose   Current end-effector pose (last-cycle FK result).
 * @param nan_check_enabled   Enable the NaN / Inf check.
 * @param max_distance_enabled Enable the maximum distance check.
 * @param max_distance_m      Maximum allowed distance (m) for the distance check.
 * @param logger              ROS logger for warning output.
 * @param clock               ROS clock for throttled warnings.
 * @return true  if the pose passed all enabled checks (safe to apply).
 * @return false if any check failed (pose should be discarded).
 */
inline bool validate_pose(
  const geometry_msgs::msg::PoseStamped & msg,
  const pinocchio::SE3 & end_effector_pose,
  bool nan_check_enabled,
  bool max_distance_enabled,
  double max_distance_m,
  rclcpp::Logger logger,
  rclcpp::Clock::SharedPtr clock)
{
  // ── Check 1: NaN / Inf ──────────────────────────────────────────────────
  // Malformed values (e.g. from a buggy publisher) would corrupt the
  // control computation and must be rejected before they reach the controller.
  if (nan_check_enabled) {
    const bool pos_ok = std::isfinite(msg.pose.position.x)
                     && std::isfinite(msg.pose.position.y)
                     && std::isfinite(msg.pose.position.z);
    const bool ori_ok = std::isfinite(msg.pose.orientation.x)
                     && std::isfinite(msg.pose.orientation.y)
                     && std::isfinite(msg.pose.orientation.z)
                     && std::isfinite(msg.pose.orientation.w);
    if (!pos_ok || !ori_ok) {
      RCLCPP_WARN_THROTTLE(logger, *clock, 1000,
        "Target pose contains NaN or Inf values — ignoring.");
      return false;
    }
  }

  // ── Check 2: Maximum distance from current end-effector ─────────────────
  // Prevents large jumps in the commanded target (e.g. from a mis-configured
  // or re-started publisher). end_effector_pose holds last-cycle's FK result,
  // which is accurate enough for this guard.
  if (max_distance_enabled) {
    const Eigen::Vector3d incoming_pos(
      msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
    const double dist = (incoming_pos - end_effector_pose.translation()).norm();
    if (dist > max_distance_m) {
      RCLCPP_WARN_THROTTLE(logger, *clock, 1000,
        "Target pose is %.3f m from end-effector (max=%.3f m) — ignoring.",
        dist, max_distance_m);
      return false;
    }
  }

  return true;
}

/**
 * @brief Timeout watchdog: freeze the target at the current end-effector
 *        position if no pose message has arrived within timeout_s seconds.
 *
 * Call this every update() cycle. When the timeout fires it overwrites
 * target_position and target_orientation with the current EE pose so that
 * the impedance controller commands near-zero torques instead of continuing
 * to drive toward a stale goal (e.g. when the sender crashes or the network
 * drops).
 *
 * @param time                  Current controller time (from update()).
 * @param last_pose_received_ns Atomic nanosecond timestamp set by the
 *                              subscriber callback; 0 means no pose received yet.
 * @param timeout_s             Maximum allowed age of the last pose (seconds).
 * @param end_effector_pose     Current EE pose (last-cycle FK result).
 * @param target_position       [out] Reset to EE translation when timeout fires.
 * @param target_orientation    [out] Reset to EE rotation when timeout fires.
 * @param logger                ROS logger for warning output.
 * @param clock                 ROS clock for throttled warnings.
 * @return true  if the watchdog fired (timeout exceeded).
 * @return false if still within the timeout window, or no pose received yet.
 */
inline bool apply_pose_timeout(
  const rclcpp::Time & time,
  const std::atomic<int64_t> & last_pose_received_ns,
  double timeout_s,
  const pinocchio::SE3 & end_effector_pose,
  Eigen::Vector3d & target_position,
  Eigen::Quaterniond & target_orientation,
  rclcpp::Logger logger,
  rclcpp::Clock::SharedPtr clock)
{
  const int64_t last_ns = last_pose_received_ns.load(std::memory_order_relaxed);

  // Skip if no pose has been received yet — no watchdog before first message.
  if (last_ns == 0) {
    return false;
  }

  const double age_s = (time - rclcpp::Time(last_ns, RCL_ROS_TIME)).seconds();
  if (age_s > timeout_s) {
    // NOTE: end_effector_pose is from the previous update() cycle.
    // A one-cycle lag on the freeze position is negligible.
    target_position    = end_effector_pose.translation();
    target_orientation = Eigen::Quaterniond(end_effector_pose.rotation());
    RCLCPP_WARN_THROTTLE(logger, *clock, 1000,
      "No target pose received for %.2f s (timeout=%.2f s). "
      "Holding current end-effector position.", age_s, timeout_s);
    return true;
  }

  return false;
}
