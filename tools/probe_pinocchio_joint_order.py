"""Probe: does Pinocchio's joint order follow the controller's `joints:` list?

crisp controllers build a reduced Pinocchio model holding only the configured
joints, then exchange per-joint values with ros2_control in list order. This
probe builds the same reduced model and prints, per listed joint, its list
position next to its Pinocchio velocity index (idx_v), plus the gravity torque
each joint would receive with and without mapping through idx_v.

Usage (pinocchio python bindings required):
    python tools/probe_pinocchio_joint_order.py ROBOT.urdf JOINT [JOINT ...]
Get ROBOT.urdf from xacro, or from a running robot via
    ros2 param get /robot_state_publisher robot_description --hide-type > robot.urdf

Expected: idx_v == list position only when the joints are listed in kinematic
(URDF tree) order. In any other order the "unmapped" column is what the
controllers used to command: each joint gets another joint's torque. This is
why on_configure now rejects such configs (utils/joint_ordering.hpp); the probe
shows the order a given URDF requires.
"""

import sys

import numpy as np
import pinocchio as pin

urdf_path, joints = sys.argv[1], sys.argv[2:]
raw = pin.buildModelFromXML(open(urdf_path).read())
missing = [j for j in joints if not raw.existJointName(j)]
if missing:
    sys.exit(f"joints not in URDF: {missing}")

# Same reduction as the controllers' on_configure: lock every unlisted joint at 0.
locked = [raw.getJointId(n) for n in raw.names if n not in joints and n != "universe"]
model = pin.buildReducedModel(raw, locked, pin.neutral(raw))
data = model.createData()

# Arbitrary non-singular pose, assigned per listed joint like the state interfaces.
q_list = np.linspace(0.3, -0.4, len(joints))
q = pin.neutral(model)
for i, name in enumerate(joints):
    q[model.joints[model.getJointId(name)].idx_q] = q_list[i]
tau = pin.computeGeneralizedGravity(model, data, q)

print(f"{'pos':>3}  {'joint':<20} {'idx_v':>5}  {'tau mapped':>10}  {'tau unmapped':>12}")
ok = True
for i, name in enumerate(joints):
    v = model.joints[model.getJointId(name)].idx_v
    ok &= v == i
    flag = "" if v == i else "   <-- MISMATCH"
    print(f"{i:>3}  {name:<20} {v:>5}  {tau[v]:>10.3f}  {tau[i]:>12.3f}{flag}")
print("list order == Pinocchio order" if ok else "list order != Pinocchio order: mapping required")
