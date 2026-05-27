"""
Actuator diagnostic. Spawns the robot, holds it in place, configures actuators
three different ways, and reports what torque MuJoCo actually applies.

Three configurations to compare:
  A. ORIGINAL affine-bias position actuator (KP=100, KD=5) — what worked
  B. MY DIRECT TORQUE setup — what now lies on the floor
  C. AFFINE BIAS with full Eq. 13 baked in (still no v_target capability,
     but I want to see if its torque output matches A's)

For each, I'll apply a known position target with current_q at a known value
and measure data.qfrc_actuator[joint_dof] to see what torque MuJoCo applied.
"""
import numpy as np
import mujoco

SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"

def build_model_position_actuator():
    """The OLD config that worked."""
    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = 0.005
    for i in range(12):
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_gainprm[i, 0] = 100.0   # KP
        m.actuator_biasprm[i, 0] = 0.0
        m.actuator_biasprm[i, 1] = -100.0  # -KP
        m.actuator_biasprm[i, 2] = -5.0    # -KD
    return m


def build_model_direct_torque():
    """My broken new config."""
    m = mujoco.MjModel.from_xml_path(SCENE)
    m.opt.timestep = 0.005
    for i in range(12):
        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_NONE
        m.actuator_gainprm[i, 0] = 1.0
        m.actuator_biasprm[i, 0] = 0.0
        m.actuator_biasprm[i, 1] = 0.0
        m.actuator_biasprm[i, 2] = 0.0
        m.actuator_ctrllimited[i] = 0
        m.actuator_ctrlrange[i, 0] = -1e9
        m.actuator_ctrlrange[i, 1] = 1e9
        m.actuator_forcelimited[i] = 1
        m.actuator_forcerange[i, 0] = -23.7
        m.actuator_forcerange[i, 1] = 23.7
    return m


def report(label, model, data, ctrl_value):
    """Apply a ctrl and report what torque comes out."""
    data.ctrl[:] = ctrl_value
    mujoco.mj_step1(model, data)   # forward kinematics + actuator
    # Report:
    #   - applied torque per actuator (data.actuator_force)
    #   - generalized force from actuators (data.qfrc_actuator)
    print(f"\n=== {label} ===")
    print(f"  ctrl[0] applied: {ctrl_value[0]:.4f}")
    print(f"  actuator_force[0..2]:    {data.actuator_force[:3]}")
    print(f"  qfrc_actuator[6..8]:     {data.qfrc_actuator[6:9]}")
    # also print the XML-defined limits as a sanity check
    print(f"  ctrlrange[0]: [{model.actuator_ctrlrange[0,0]:.4g}, {model.actuator_ctrlrange[0,1]:.4g}]  "
          f"limited={model.actuator_ctrllimited[0]}")
    print(f"  forcerange[0]: [{model.actuator_forcerange[0,0]:.4g}, {model.actuator_forcerange[0,1]:.4g}]  "
          f"limited={model.actuator_forcelimited[0]}")


def setup_known_state(model):
    """Reset to a fixed known qpos/qvel for fair comparison."""
    data = mujoco.MjData(model)
    # Floating base up in the air, joints at some known angle.
    data.qpos[0:3] = [0.0, 0.0, 0.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = [0.0, 1.2, -2.5] * 4   # initial relaxed
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return data


def main():
    print("=== ACTUATOR DIAGNOSTIC ===\n")
    print("Robot held at qpos[7] = 1.2 (hip in 'relaxed' = before stand-up)")
    print("Target ctrl = 0.7 (Q_NOMINAL for hip = standing config)")
    print()
    print("Position-error = (target - current) = 0.7 - 1.2 = -0.5 rad")
    print("Expected PD torque = KP * error + KD * (0 - 0) = 100 * -0.5 = -50 N·m")
    print("(saturated to actuator limit if applicable)")

    # --- Configuration A: original position actuator ---
    m1 = build_model_position_actuator()
    d1 = setup_known_state(m1)
    # OLD way: ctrl IS the position target
    ctrl_A = np.array([0.0, 0.7, -1.5] * 4)
    report("A: OLD affine-bias position actuator, ctrl=target_q", m1, d1, ctrl_A)

    # --- Configuration B: my new direct torque ---
    m2 = build_model_direct_torque()
    d2 = setup_known_state(m2)
    # NEW way: ctrl IS the torque, computed manually
    KP, KD = 100.0, 5.0
    target_q = np.array([0.0, 0.7, -1.5] * 4)
    target_v = np.zeros(12)
    q_now = d2.qpos[7:].copy()
    v_now = d2.qvel[6:].copy()
    ctrl_B = KP * (target_q - q_now) + KD * (target_v - v_now)
    print(f"\nComputed torque before clamp: ctrl_B = {ctrl_B[:3]}  (should be ~[0, -50, -100])")
    report("B: NEW direct-torque actuator, ctrl=Eq.13 PD torque", m2, d2, ctrl_B)

    # --- Print the original XML-defined ranges ---
    print("\n=== XML-DEFINED RANGES (from a freshly-loaded model) ===")
    m_fresh = mujoco.MjModel.from_xml_path(SCENE)
    for i in range(3):
        print(f"  act[{i}]: ctrlrange=[{m_fresh.actuator_ctrlrange[i,0]:.4f}, "
              f"{m_fresh.actuator_ctrlrange[i,1]:.4f}]  "
              f"ctrllimited={m_fresh.actuator_ctrllimited[i]}  "
              f"forcerange=[{m_fresh.actuator_forcerange[i,0]:.4f}, "
              f"{m_fresh.actuator_forcerange[i,1]:.4f}]  "
              f"forcelimited={m_fresh.actuator_forcelimited[i]}")


if __name__ == "__main__":
    main()