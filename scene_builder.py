import os

import numpy as np
import mujoco

from config import Config, FOOT_NAMES


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------

_SENSOR_BLOCK = "\n".join(
    [f'    <framepos name="mppi_{n}_pos" objtype="geom" objname="{n}"/>' for n in FOOT_NAMES] +
    [f'    <framelinvel name="mppi_{n}_vel" objtype="geom" objname="{n}"/>' for n in FOOT_NAMES]
)

_SCENE_WRAPPER = f"""<mujoco model="go2 scene mppi">
  <include file="scene.xml"/>
  <sensor>
{_SENSOR_BLOCK}
  </sensor>
</mujoco>
"""

_FLAT_WRAPPER = f"""<mujoco model="go2 flat mppi">
  <include file="go2.xml"/>
  <statistic center="0 0 0.1" extent="0.8"/>
  <asset>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4"
      rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
  <sensor>
{_SENSOR_BLOCK}
  </sensor>
</mujoco>
"""


def make_scene_with_sensors():
    """Write a wrapper XML (next to scene.xml, so includes resolve) that adds
    framepos/framelinvel sensors on the four foot spheres. Returns its path."""
    go2_dir = os.path.dirname(os.path.abspath(Config.ROBOT_SCENE))
    if Config.USE_FLAT_SCENE:
        path, content = os.path.join(go2_dir, "scene_mppi_flat.xml"), _FLAT_WRAPPER
    else:
        path, content = os.path.join(go2_dir, "scene_mppi.xml"), _SCENE_WRAPPER
    try:
        with open(path) as f:
            if f.read() == content:
                return path
    except FileNotFoundError:
        pass
    with open(path, "w") as f:
        f.write(content)
    return path


def prepare_model(xml_path, timestep):
    """Load the model, add per-foot touch sensors (for the paper's w_c,force
    term), and convert the torque motors into in-model PD servos:
    torque = Kp*(ctrl - q) - Kd*qvel, clipped to the motor's torque range.
    Planner and live sim both go through here -> identical dynamics."""
    spec = mujoco.MjSpec.from_file(xml_path)
    for n in FOOT_NAMES:
        body = spec.body(f"{n}_calf")
        site = body.add_site()
        site.name = f"mppi_{n}_touch"
        site.pos  = [-0.002, 0.0, -0.213]      # foot sphere center (go2.xml)
        site.type = mujoco.mjtGeom.mjGEOM_SPHERE
        site.size = [0.032, 0.0, 0.0]          # just larger than the 0.022 foot sphere
        sen = spec.add_sensor()
        sen.name = f"mppi_{n}_force"
        sen.type = mujoco.mjtSensor.mjSENS_TOUCH
        sen.objtype = mujoco.mjtObj.mjOBJ_SITE
        sen.objname = site.name
    model = spec.compile()
    model.opt.timestep = timestep
    for i in range(model.nu):
        torque_range = model.actuator_ctrlrange[i].copy()
        model.actuator_gaintype[i]      = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i]      = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[i, 0]    = Config.KP
        model.actuator_biasprm[i, 1]    = -Config.KP
        model.actuator_biasprm[i, 2]    = -Config.KD
        model.actuator_forcelimited[i]  = 1
        model.actuator_forcerange[i]    = torque_range      # clip torques to actuator limits
        model.actuator_ctrllimited[i]   = 0                 # ctrl is now a joint angle, not a torque
    return model


def actuator_joint_perm(model):
    """perm[i] = channel in the 12-dim joint-order vector driven by actuator i.
    go2.xml orders actuators FR,FL,RR,RL but qpos is FL,FR,RL,RR."""
    perm = np.empty(model.nu, dtype=int)
    for i in range(model.nu):
        jid     = model.actuator_trnid[i, 0]
        perm[i] = model.jnt_qposadr[jid] - 7
    return perm


def foot_sensor_columns(model):
    """Column indices into sensordata: z of each foot sphere, (4,3) linvel,
    and the touch-sensor normal force per foot."""
    z_cols = []
    v_cols = []
    f_cols = []
    for n in FOOT_NAMES:
        adr_p = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"mppi_{n}_pos")]
        adr_v = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"mppi_{n}_vel")]
        adr_f = model.sensor_adr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, f"mppi_{n}_force")]
        z_cols.append(adr_p + 2)
        v_cols.append([adr_v, adr_v + 1, adr_v + 2])
        f_cols.append(adr_f)
    return np.array(z_cols), np.array(v_cols), np.array(f_cols)
