"""
Inspect what contacts the Go2 actually generates when standing.

Settles the robot in nominal pose, then enumerates all active contacts and
prints which geoms are involved, what bodies those belong to, and the normal
force. Helps debug per-foot contact attribution.
"""
import numpy as np
import mujoco

ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"

model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
model.opt.timestep = 0.005

# Mirror the actuator setup from the main script
for i in range(12):
    model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[i, 0] = 80.0
    model.actuator_biasprm[i, 1] = -80.0
    model.actuator_biasprm[i, 2] = -4.0

data = mujoco.MjData(model)

# === Spawn & settle ===
Q_NOMINAL = np.array([0.0, 0.7, -1.4] * 4)
initial_relaxed = np.array([0.0, 1.2, -2.5] * 4)
data.qpos[0:3] = [0.0, 0.0, 0.5]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[7:] = initial_relaxed

mujoco.mj_kinematics(model, data)
geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0]
bottoms = data.geom_xpos[geom_ids, 2] - model.geom_size[geom_ids, 0]
data.qpos[2] = (0.5 - np.min(bottoms)) + 0.001

print("Standing up...")
for i in range(400):
    alpha = (1 - np.cos(np.pi * i / 400)) / 2.0
    data.ctrl[:] = (1 - alpha) * initial_relaxed + alpha * Q_NOMINAL
    mujoco.mj_step(model, data)
for _ in range(200):
    data.ctrl[:] = Q_NOMINAL
    mujoco.mj_step(model, data)

print(f"\nSettled. qpos[2] = {data.qpos[2]:.3f}")
print(f"Number of active contacts: {data.ncon}\n")

# === Look up all body names ===
body_names = []
for b in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b)
    body_names.append(name if name else f"body_{b}")

# === Print every contact ===
force_buf = np.zeros(6)
print(f"{'idx':<4} {'g1':<5} {'g2':<5} {'body1':<25} {'body2':<25} {'normal_force':>12}")
print("-" * 90)
total_normal = 0.0
for c_idx in range(data.ncon):
    c = data.contact[c_idx]
    g1, g2 = c.geom1, c.geom2
    b1 = model.geom_bodyid[g1]
    b2 = model.geom_bodyid[g2]
    mujoco.mj_contactForce(model, data, c_idx, force_buf)
    f_normal = force_buf[0]
    total_normal += abs(f_normal)
    print(f"{c_idx:<4} {g1:<5} {g2:<5} {body_names[b1]:<25} {body_names[b2]:<25} {f_normal:>12.2f}")

print(f"\nTotal normal force sum: {total_normal:.2f} N")
print(f"Expected (body weight): ~147 N for a 15kg Go2")

# === Per-leg foot body lookup (replicating script logic) ===
print("\n=== Foot body lookup (replicating main script) ===")
foot_body_ids = []
foot_geom_ids_per_foot = []
for leg in ["FL", "FR", "RL", "RR"]:
    b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_foot")
    used = f"{leg}_foot"
    if b_id == -1:
        b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
        used = f"{leg}_calf"
    foot_body_ids.append(b_id)
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b_id]
    foot_geom_ids_per_foot.append(set(geoms))
    print(f"  {leg}: looked up '{used}' → body_id={b_id} ({body_names[b_id] if b_id>=0 else 'NONE'}); geoms={geoms}")

# === Did our matching work? ===
print("\n=== Force attribution check ===")
f_per_foot = [0.0, 0.0, 0.0, 0.0]
unmatched = 0
for c_idx in range(data.ncon):
    c = data.contact[c_idx]
    g1, g2 = c.geom1, c.geom2
    mujoco.mj_contactForce(model, data, c_idx, force_buf)
    f_normal = abs(force_buf[0])
    matched = False
    for foot_idx, geom_set in enumerate(foot_geom_ids_per_foot):
        if g1 in geom_set or g2 in geom_set:
            f_per_foot[foot_idx] += f_normal
            matched = True
            break
    if not matched:
        unmatched += 1

for leg, f in zip(["FL","FR","RL","RR"], f_per_foot):
    print(f"  {leg}: {f:.2f} N")
print(f"  Unmatched contacts: {unmatched}")
print(f"  Sum of per-foot: {sum(f_per_foot):.2f} N")
print(f"\n  If per-foot sums to 0 or very low, our matching is broken!")
print(f"  If unmatched > 0, contacts are happening between non-foot bodies.")

# === If matching fails, look at all geoms by body ===
if sum(f_per_foot) < 1.0 and total_normal > 10.0:
    print("\n=== Matching failed. Showing all leg-related body geoms: ===")
    for leg in ["FL", "FR", "RL", "RR"]:
        for suffix in ["hip", "thigh", "calf", "foot"]:
            b_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{suffix}")
            if b_id >= 0:
                geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == b_id]
                if geoms:
                    print(f"  {leg}_{suffix} (body {b_id}): geoms={geoms}")

# === CRITICAL: inspect each calf's geoms in detail ===
# We need to find WHICH geom is the foot sphere (the actual contact point)
# vs which geoms are the calf shaft (knee-walking would contact these).
GEOM_TYPE_NAMES = {
    0: "PLANE", 1: "HFIELD", 2: "SPHERE", 3: "CAPSULE",
    4: "ELLIPSOID", 5: "CYLINDER", 6: "BOX", 7: "MESH",
}
print("\n=== DETAILED CALF GEOM INSPECTION ===")
print("Looking for the foot sphere (contact point) vs the shaft geoms.")
for leg in ["FL", "FR", "RL", "RR"]:
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_calf")
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == cb]
    print(f"\n  {leg}_calf (body {cb}):")
    for g in geoms:
        gtype = model.geom_type[g]
        tname = GEOM_TYPE_NAMES.get(gtype, f"type{gtype}")
        size = model.geom_size[g]
        local_pos = model.geom_pos[g]
        contype = model.geom_contype[g]
        conaffinity = model.geom_conaffinity[g]
        # is this geom collidable with the world (contype/conaffinity)?
        collidable = "COLLIDES" if (contype != 0) else "visual-only"
        # mark the geom that actually contacted the floor in the stand test
        contacted = ""
        for c_idx in range(data.ncon):
            c = data.contact[c_idx]
            if c.geom1 == g or c.geom2 == g:
                contacted = "  <<< CONTACTED FLOOR"
        print(f"    geom {g}: {tname:9s} size={np.round(size,4)} "
              f"local_pos={np.round(local_pos,4)} {collidable}{contacted}")

print("\n=== INTERPRETATION ===")
print("The geom marked CONTACTED FLOOR during a normal stand IS the foot.")
print("Other COLLIDES geoms on the calf are the shaft — contact on those")
print("means the robot is knee-walking, which the cost function must penalize.")
print("visual-only geoms can be ignored entirely.")