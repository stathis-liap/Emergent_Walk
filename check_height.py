"""
Quick diagnostic to verify what data.qpos[2] actually measures on your Go2.

Spawns the robot in 3 poses:
  1. Standing (legs at Q_NOMINAL)
  2. Crouched (knees bent)
  3. Belly-down (forced lying on chest)

Prints qpos[2] and the actual lowest trunk-geom z for each. If qpos[2] is the
chassis centroid (not the lowest point), you'll see ~5-10cm offset.

Usage:
    python check_height.py
"""
import numpy as np
import mujoco

ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
data = mujoco.MjData(model)

# Identify the trunk body (it's the floating-base body, usually id=1)
# and which geoms belong to it
trunk_body_id = None
for b in range(1, model.nbody):
    # The trunk is the floating-base body — qpos starts here
    if model.body_jntnum[b] > 0 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
        trunk_body_id = b
        break
if trunk_body_id is None:
    trunk_body_id = 1   # fallback

trunk_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, trunk_body_id)
print(f"Trunk body: id={trunk_body_id}, name={trunk_name!r}")

# Find which geoms belong to the trunk body specifically (excluding limbs)
trunk_geom_ids = [g for g in range(model.ngeom) if model.geom_bodyid[g] == trunk_body_id]
print(f"Trunk geom ids: {trunk_geom_ids}")
for g in trunk_geom_ids:
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g)
    gtype = model.geom_type[g]
    size = model.geom_size[g]
    pos = model.geom_pos[g]  # local
    print(f"  geom {g}: name={name!r}, type={gtype}, size={size}, local_pos={pos}")


def lowest_trunk_z(data, model):
    """World-frame z of the bottommost point of any trunk geom."""
    mujoco.mj_kinematics(model, data)
    zs = []
    for g in trunk_geom_ids:
        gtype = model.geom_type[g]
        center_z = data.geom_xpos[g, 2]
        if gtype == mujoco.mjtGeom.mjGEOM_BOX:
            half_h = model.geom_size[g, 2]
            zs.append(center_z - half_h)
        elif gtype == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = model.geom_size[g, 0]
            zs.append(center_z - r)
        elif gtype == mujoco.mjtGeom.mjGEOM_CAPSULE or gtype == mujoco.mjtGeom.mjGEOM_CYLINDER:
            # Approximation: just use center − radius
            r = model.geom_size[g, 0]
            zs.append(center_z - r)
        else:
            zs.append(center_z)   # fallback
    return min(zs) if zs else float('nan')


# === Test 1: standing pose ===
print("\n=== Test 1: standing pose ===")
data.qpos[:] = 0
data.qpos[0:3] = [0, 0, 0.4]
data.qpos[3:7] = [1, 0, 0, 0]
Q_NOMINAL = np.array([0.0, 0.7, -1.4] * 4)
data.qpos[7:] = Q_NOMINAL
mujoco.mj_kinematics(model, data)
print(f"qpos[2] (trunk origin z) = {data.qpos[2]:.3f} m")
print(f"trunk lowest z          = {lowest_trunk_z(data, model):.3f} m")
print(f"offset                  = {data.qpos[2] - lowest_trunk_z(data, model):.3f} m")

# === Test 2: crouched pose ===
print("\n=== Test 2: crouched pose ===")
data.qpos[7:] = np.array([0.0, 1.5, -2.5] * 4)
data.qpos[2] = 0.20
mujoco.mj_kinematics(model, data)
print(f"qpos[2] = {data.qpos[2]:.3f} m")
print(f"trunk lowest z = {lowest_trunk_z(data, model):.3f} m")

# === Test 3: belly-down (trunk horizontal, on ground) ===
print("\n=== Test 3: belly-down (trunk on ground, all legs spread) ===")
data.qpos[7:] = np.array([0.0, 0.0, 0.0] * 4)   # legs flat
# Force trunk to lie flat on ground
data.qpos[0:3] = [0, 0, 0.05]   # roughly half trunk height
data.qpos[3:7] = [1, 0, 0, 0]
mujoco.mj_kinematics(model, data)
print(f"qpos[2] = {data.qpos[2]:.3f} m  <-- what the cost function sees")
print(f"trunk lowest z = {lowest_trunk_z(data, model):.3f} m  <-- physical reality")

# === Test 4: rolled on side ===
print("\n=== Test 4: rolled on side (90 deg roll) ===")
data.qpos[7:] = Q_NOMINAL
# 90 deg roll around x
data.qpos[3:7] = [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]
data.qpos[2] = 0.10
mujoco.mj_kinematics(model, data)
print(f"qpos[2] = {data.qpos[2]:.3f} m")
print(f"trunk lowest z = {lowest_trunk_z(data, model):.3f} m")

print("\nInterpretation:")
print("  If qpos[2] - lowest_z is roughly constant (~5cm), it's the chassis centroid")
print("  and the cost function W_H × |qpos[2] - z_des| is a PROXY, not a true fall detector.")
print("  When the robot falls flat, qpos[2] only drops by ~25cm (from 0.30 to 0.05),")
print("  giving W_H × 0.25 = 25 per step × 45 = 1125 cost — actually significant.")
print("  But during a LEAN (45 deg) without full collapse, qpos[2] might only drop 5cm,")
print("  giving 22 cost. Easy to miss.")