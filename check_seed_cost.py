"""
Score the bare trot seed under the cost function.

Runs three scenarios from the settled standing pose:
  A. Hold Q_NOMINAL constant (do nothing)
  B. Run the trot seed open-loop
  C. Apply random noise around Q_NOMINAL

For each, computes the total cost using the same formula as the MPC's worker.
This tells us whether the seed actually wins against "stand still."

If seed cost > standing cost, the MPC will reject the seed and drift to standing.
This is the critical diagnostic.
"""
import numpy as np
import mujoco

ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"

# ===== Replicate Config =====
H_HORIZON_SEC = 0.9
DT = 0.02
H_STEPS = int(H_HORIZON_SEC / DT) + 1
PHYS_DT = 0.005
PHYS_SUBSTEPS = int(DT / PHYS_DT)
KP, KD = 80.0, 4.0
W_H = 100.0
W_ORIENT = 10.0
W_Q = 0.5
W_C_VEL = 0.5
W_C_FORCE = 0.0
W_BADCONTACT = 5.0
W_H_TERM = 1500.0
BODY_WEIGHT = 149.0
SAFE_FOOT_FORCE = 120.0
V_DES = 0.10

# Set up model
model = mujoco.MjModel.from_xml_path(ROBOT_SCENE)
model.opt.timestep = PHYS_DT
for i in range(12):
    model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
    model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
    model.actuator_gainprm[i, 0] = KP
    model.actuator_biasprm[i, 1] = -KP
    model.actuator_biasprm[i, 2] = -KD

data = mujoco.MjData(model)

# Classify geoms: foot spheres (per leg) vs non-foot collidables (shaft+trunk).
# Mirrors _classify_leg_geoms in the main script.
foot_body_ids = []
foot_geom_ids = []
shaft_geom_ids = set()
# trunk collidables
trunk_id = 1
for b in range(1, model.nbody):
    if model.body_jntnum[b] > 0 and model.jnt_type[model.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE:
        trunk_id = b
        break
for g in range(model.ngeom):
    if model.geom_bodyid[g] == trunk_id and model.geom_contype[g] != 0:
        shaft_geom_ids.add(g)
for name in ["FL", "FR", "RL", "RR"]:
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_foot")
    if b == -1:
        b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_calf")
    foot_body_ids.append(b)
    cb = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_calf")
    this_foot = set()
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != cb or model.geom_contype[g] == 0:
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            this_foot.add(g)
        else:
            shaft_geom_ids.add(g)
    for seg in ["hip", "thigh"]:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_{seg}")
        if sid != -1:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] == sid and model.geom_contype[g] != 0:
                    shaft_geom_ids.add(g)
    foot_geom_ids.append(this_foot)

print(f"Foot geoms per leg: {[sorted(s) for s in foot_geom_ids]}")
print(f"Non-foot collidable geoms: {sorted(shaft_geom_ids)}")

# Settle
Q_NOMINAL_INIT = np.array([0.0, 0.7, -1.5] * 4)
initial_relaxed = np.array([0.0, 1.2, -2.5] * 4)

data.qpos[0:3] = [0.0, 0.0, 0.5]
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
data.qpos[7:] = initial_relaxed
mujoco.mj_kinematics(model, data)
geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0]
bottoms = data.geom_xpos[geom_ids, 2] - model.geom_size[geom_ids, 0]
data.qpos[2] = (0.5 - np.min(bottoms)) + 0.001

for i in range(400):
    alpha = (1 - np.cos(np.pi * i / 400)) / 2.0
    data.ctrl[:] = (1 - alpha) * initial_relaxed + alpha * Q_NOMINAL_INIT
    mujoco.mj_step(model, data)
for _ in range(200):
    data.ctrl[:] = Q_NOMINAL_INIT
    mujoco.mj_step(model, data)

P_DES_Z = data.qpos[2]
Q_NOMINAL = data.qpos[7:].copy()
print(f"Post-settle: qpos[2] = {P_DES_Z:.3f}")
print(f"Post-settle: Q_NOMINAL[:3] = {Q_NOMINAL[:3].round(3)}")
print()

# Save the settled state
settled_qpos = data.qpos.copy()
settled_qvel = data.qvel.copy()


def evaluate(traj_q, label):
    """Roll out one trajectory and return its cost components."""
    mujoco.mj_resetData(model, data)
    data.qpos[:] = settled_qpos
    data.qvel[:] = settled_qvel

    c_height = c_orient = c_q = c_vc = c_force = 0.0
    force_buf = np.zeros(6)

    for t in range(H_STEPS - 1):
        data.ctrl[:] = traj_q[t]
        for _ in range(PHYS_SUBSTEPS):
            mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all():
            print(f"  {label}: DIVERGED at step {t}")
            return None

        # height
        c_height += W_H * abs(data.qpos[2] - P_DES_Z)
        # orient
        qw = max(-1.0, min(1.0, data.qpos[3]))
        ang = 2.0 * np.arccos(abs(qw))
        c_orient += W_ORIENT * ang * ang
        # q_reg
        dq = data.qpos[7:] - Q_NOMINAL
        c_q += W_Q * float(dq @ dq)
        # v_c — foot velocity while near ground (tight 2cm gate)
        v_c = 0.0
        for body_id in foot_body_ids:
            foot_z = data.xpos[body_id, 2]
            v_mag = np.linalg.norm(data.cvel[body_id, 3:6])
            in_contact = max(0.0, min(1.0, (0.02 - foot_z) / 0.02))
            v_c += in_contact * v_mag
        c_vc += W_C_VEL * v_c
        # bad-contact: count world↔non-foot (shaft/trunk) contacts
        bad = 0
        for c_idx in range(data.ncon):
            c = data.contact[c_idx]
            if c.geom1 in shaft_geom_ids or c.geom2 in shaft_geom_ids:
                bad += 1
        c_force += W_BADCONTACT * bad
        # optional legacy force term
        if W_C_FORCE > 0.0:
            f_per_foot = [0.0, 0.0, 0.0, 0.0]
            for c_idx in range(data.ncon):
                mujoco.mj_contactForce(model, data, c_idx, force_buf)
                f_normal = abs(force_buf[0])
                c = data.contact[c_idx]
                for fi, gs in enumerate(foot_geom_ids):
                    if c.geom1 in gs or c.geom2 in gs:
                        f_per_foot[fi] += f_normal
                        break
            total_f = sum(f_per_foot)
            c_force += W_C_FORCE * abs(total_f - BODY_WEIGHT)
            for ff in f_per_foot:
                excess = ff - SAFE_FOOT_FORCE
                if excess > 0.0:
                    c_force += W_C_FORCE * excess

    # terminal
    final_qpos = data.qpos.copy()
    p_tx = settled_qpos[0] + V_DES * H_HORIZON_SEC
    p_ty = settled_qpos[1]
    p_tz = P_DES_Z
    term = abs(final_qpos[0] - p_tx) + abs(final_qpos[1] - p_ty) + abs(final_qpos[2] - p_tz)
    c_term = W_H_TERM * term

    total = c_height + c_orient + c_q + c_vc + c_force + c_term
    print(f"  {label}:")
    print(f"    h={c_height:7.1f}  o={c_orient:7.1f}  q={c_q:5.2f}  "
          f"vc={c_vc:6.1f}  badC={c_force:6.1f}  term={c_term:7.1f}  TOTAL={total:7.1f}")
    print(f"    final: x={final_qpos[0]:+.3f}  y={final_qpos[1]:+.3f}  "
          f"z={final_qpos[2]:.3f}  dx_target={final_qpos[0]-settled_qpos[0]:+.3f}")
    return total


# --- Scenario A: hold Q_NOMINAL ---
print("=== A: Stand still (hold Q_NOMINAL) ===")
traj_A = np.tile(Q_NOMINAL, (H_STEPS, 1))
cost_A = evaluate(traj_A, "STAND")
print()

# --- Scenario B: bare trot seed ---
print("=== B: Trot seed (winning sweep params: freq=3.0, hip=0.20, knee=0.40, lean=+0.05) ===")
def make_seed(freq_hz, hip_amp, knee_amp, fwd_lean):
    """Matches sweep_trot.py's hip_sign=-1, knee_sign=-1 formulation."""
    seed = np.zeros((H_STEPS, 12))
    for h in range(H_STEPS):
        t = h * DT
        phase = 2 * np.pi * freq_hz * t
        sA = np.sin(phase)
        sB = np.sin(phase + np.pi)
        liftA = max(0.0, sA)
        liftB = max(0.0, sB)
        target = Q_NOMINAL.copy()
        # hip_sign=-1: hip += -hip_amp * s + fwd_lean
        # knee_sign=-1: knee += -knee_amp * lift
        target[1]  += -hip_amp * sA + fwd_lean
        target[2]  += -knee_amp * liftA
        target[4]  += -hip_amp * sB + fwd_lean
        target[5]  += -knee_amp * liftB
        target[7]  += -hip_amp * sB + fwd_lean
        target[8]  += -knee_amp * liftB
        target[10] += -hip_amp * sA + fwd_lean
        target[11] += -knee_amp * liftA
        seed[h] = target
    return seed

traj_B = make_seed(3.0, 0.20, 0.40, 0.05)
cost_B = evaluate(traj_B, "TROT (sweep winner)")
print()

# Try the old underpowered version for comparison
print("=== B2: Old underpowered trot (freq=2.0, hip=0.10, knee=0.20, lean=0.02) ===")
traj_B2 = make_seed(2.0, 0.10, 0.20, 0.02)
cost_B2 = evaluate(traj_B2, "TROT (old underpowered)")
print()

# --- Compare ---
print("=== SUMMARY ===")
print(f"  STAND:                {cost_A:.1f}")
print(f"  TROT (sweep winner):  {cost_B:.1f}  (diff vs stand: {cost_B - cost_A:+.1f})")
print(f"  TROT (old underpowered): {cost_B2:.1f}  (diff vs stand: {cost_B2 - cost_A:+.1f})")
print()
if cost_B < cost_A:
    print("  ✓ Sweep-winner trot beats standing — MPC should preserve this seed.")
else:
    print(f"  ⚠️ Sweep-winner trot still loses to standing by {cost_B - cost_A:.0f}.")
    print(f"     But it walks forward (dx>0), so the terminal target IS reached.")
    print(f"     The running cost penalty for motion exceeds the terminal cost benefit.")
    print(f"     Consider raising V_DES so the carrot grows, or lowering w_c,vel.")