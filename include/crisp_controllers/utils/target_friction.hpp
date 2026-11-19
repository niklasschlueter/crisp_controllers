#pragma once

// Friction compensation driven by the target motion instead of the measured one.
//
// The measured-velocity model (friction_model.hpp) is zero while a joint sticks, so it gives no
// breakaway help, and at low stiffness its push in the direction of motion lets the arm drift.
// Here the direction comes from the target joint velocity dq_target (J^+ times the target twist),
// so the torque is zero while the target rests. Per drive i:
//   tau_i = scale * tanh(v_target_i / epsilon) * (Fk_i + stick_i)
//   stick_i = max(Fs_i - Fk_i, 0) * exp(-(v_i / stick_velocity)^2)   (only with breakaway given)
// Fk (kinetic) removes the lag and the plateau at target reversals. A constant push cannot remove
// stick-slip in the middle of a stroke (it only shifts the equilibrium by Fk/K), so the stick part
// adds the breakaway-minus-kinetic difference while the drive is (nearly) at rest.
// Friction sits in the drives, not the serial joints: v = A dq, and the drive torque maps back as
// A^T tau (power-consistent). A is the serial-to-drive velocity map (identity for direct drives).

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

// breakaway / viscous may be empty: no stick / viscous part.
inline Eigen::VectorXd target_friction_torque(
  const Eigen::VectorXd & dq_target, const Eigen::VectorXd & dq, const Eigen::MatrixXd & coupling,
  const Eigen::VectorXd & kinetic, const Eigen::VectorXd & breakaway, double epsilon,
  double stick_velocity, double scale, const Eigen::VectorXd & viscous = Eigen::VectorXd()) {
  const Eigen::VectorXd v_target = coupling * dq_target;
  const Eigen::VectorXd v = coupling * dq;
  Eigen::VectorXd tau_drive(v.size());
  for (Eigen::Index i = 0; i < v.size(); ++i) {
    double amplitude = kinetic[i];
    if (breakaway.size() == v.size()) {
      const double ratio = v[i] / stick_velocity;
      amplitude += std::max(breakaway[i] - kinetic[i], 0.0) * std::exp(-ratio * ratio);
    }
    tau_drive[i] = scale * amplitude * std::tanh(v_target[i] / epsilon);
    if (viscous.size() == v.size()) {
      tau_drive[i] += scale * viscous[i] * v[i];
    }
  }
  return coupling.transpose() * tau_drive;
}
