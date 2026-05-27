import time
import os
import sys
import numpy as np
import mujoco
import cv2
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

from math_engine import CubicHermiteSpline, apply_derivative_clamp
from spline_recorder import SplineRecorder

class Config:
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"

    # === MPPI structural parameters ===
    H_HORIZON_SEC = 0.9
    CONTROL_FREQ  = 50.0
    DT            = 1.0 / CONTROL_FREQ
    H_STEPS       = int(H_HORIZON_SEC / DT) + 1

    PHYS_DT       = 0.005
    PHYS_SUBSTEPS = int(DT / PHYS_DT)

    K_NODES       = 10
    NUM_ITERATIONS = 3
    NUM_SAMPLES   = 30

    LAMBDA        = 0.1

    # Noise scaling
    SCALE_Q       = 0.1
    SCALE_V       = 0.01
    BETA_1        = 1.0
    BETA_2        = 1.0

    # === COST WEIGHTS ===
    W_H        = 100.0
    W_ORIENT   = 70.0
    W_Q        = 70.0
    W_C_VEL    = 0.5
    W_C_FORCE  = 0.05
    W_H_TERM   = 4500.0

    W_BADCONTACT = 0.0

    W_GAIT          = 800.0          # weight on the phase-clock cost
    GAIT_FREQ_HZ    = 2.0            # target stepping frequency (Go2 trots at ~2-3 Hz)
    GAIT_AMPLITUDE  = 0.08           # m. Target peak height difference between diagonals.

    # === PD controller ===
    KP = 40.0
    KD = 0.5

    # === Task targets ===
    P_DES_Z   = 0.28
    V_DES     = 0.3
    Q_NOMINAL = np.array([0.0, 0.7, -1.45] * 4)

    QUAT_DES = np.array([1.0, 0.0, 0.0, 0.0])

    JOINT_MIN = np.array([-0.8, -0.5, -2.6] * 4)
    JOINT_MAX = np.array([ 0.8,  2.5, -0.8] * 4)

    BODY_WEIGHT      = 149.0
    SAFE_FOOT_FORCE  = 80.0

    PARALLEL_MODE = 'process'
    NUM_WORKERS   = None

    # === Spline recording (debug) ===
    RECORD_SPLINES        = True
    RECORD_OUT_PATH       = "spline_log.npz"
    RECORD_EVERY_N_STEPS  = 10
    RECORD_MAX_STEPS      = 150
    RECORD_N_SAMPLES_KEPT = 5

_W_MODEL = None
_W_DATAS = None
_W_FOOT_IDS = None
_W_FOOT_GEOM_IDS = None
_W_SHAFT_GEOM_IDS = None
_W_HEAD_GEOM_ID = None
_W_CONFIG = None


def _classify_leg_geoms(model):
    foot_geom_ids = []
    nonfoot = set()

    trunk_id = 1
    for b in range(1, model.nbody):
        if model.body_jntnum[b] > 0:
            jadr = model.body_jntadr[b]
            if model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                trunk_id = b
                break
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == trunk_id and model.geom_contype[g] != 0:
            nonfoot.add(g)

    for name in ["FL", "FR", "RL", "RR"]:
        calf_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_calf")
        this_foot = set()
        if calf_id != -1:
            for g in range(model.ngeom):
                if model.geom_bodyid[g] != calf_id:
                    continue
                if model.geom_contype[g] == 0:
                    continue
                if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                    this_foot.add(g)
                else:
                    nonfoot.add(g)
        for seg in ["hip", "thigh"]:
            seg_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_{seg}")
            if seg_id != -1:
                for g in range(model.ngeom):
                    if model.geom_bodyid[g] == seg_id and model.geom_contype[g] != 0:
                        nonfoot.add(g)
        foot_geom_ids.append(this_foot)

    return foot_geom_ids, nonfoot


def _find_head_geom_id(model):
    trunk_id = 1
    for b in range(1, model.nbody):
        if model.body_jntnum[b] > 0:
            jadr = model.body_jntadr[b]
            if model.jnt_type[jadr] == mujoco.mjtJoint.mjJNT_FREE:
                trunk_id = b
                break
    candidates_sphere = []
    candidates_any = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] != trunk_id:
            continue
        local_x = model.geom_pos[g, 0]
        candidates_any.append((local_x, g))
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            candidates_sphere.append((local_x, g))
    if candidates_sphere:
        candidates_sphere.sort(reverse=True)
        return candidates_sphere[0][1]
    if candidates_any:
        candidates_any.sort(reverse=True)
        return candidates_any[0][1]
    return -1


def _worker_init(scene_path, kp, kd, phys_dt, lanes_per_worker, config_dict):
    global _W_MODEL, _W_DATAS, _W_FOOT_IDS, _W_FOOT_GEOM_IDS
    global _W_SHAFT_GEOM_IDS, _W_HEAD_GEOM_ID, _W_CONFIG

    _W_MODEL = mujoco.MjModel.from_xml_path(scene_path)
    _W_MODEL.opt.timestep = phys_dt
    for i in range(12):
        _W_MODEL.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        _W_MODEL.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        _W_MODEL.actuator_gainprm[i, 0] = kp
        _W_MODEL.actuator_biasprm[i, 1] = -kp
        _W_MODEL.actuator_biasprm[i, 2] = -kd

    _W_DATAS = [mujoco.MjData(_W_MODEL) for _ in range(lanes_per_worker)]

    _W_FOOT_IDS = []
    for name in ["FL", "FR", "RL", "RR"]:
        kin_body_id = mujoco.mj_name2id(_W_MODEL, mujoco.mjtObj.mjOBJ_BODY, f"{name}_foot")
        if kin_body_id == -1:
            kin_body_id = mujoco.mj_name2id(_W_MODEL, mujoco.mjtObj.mjOBJ_BODY, f"{name}_calf")
        _W_FOOT_IDS.append(kin_body_id)

    _W_FOOT_GEOM_IDS, _W_SHAFT_GEOM_IDS = _classify_leg_geoms(_W_MODEL)
    _W_HEAD_GEOM_ID = _find_head_geom_id(_W_MODEL)
    _W_CONFIG = config_dict


_SCRATCH_UP = np.zeros(3)
_SCRATCH_Z_AXIS = np.array([0.0, 0.0, 1.0])


def _evaluate_one(lane, start_qpos, start_qvel, traj_q, traj_v, gait_phase_start):
    """Rollout one trajectory under Eq. 13 PD with velocity feedforward.

    The ctrl-transform `ctrl = q_target + (KD/KP)*v_target` makes the
    affine-bias actuator produce K_p*(q_target - q) + K_d*(v_target - v),
    which is the paper's Eq. 13.

    gait_phase_start: the gait clock phase (radians) at the START of this rollout.
        The clock advances by 2π * GAIT_FREQ_HZ * DT per simulated control step.
        This is what makes the phase persist across control steps — each rollout
        starts where the live phase clock currently is, so consecutive control
        steps see consistent gait-clock targets.
    """
    cfg = _W_CONFIG
    data = _W_DATAS[lane]
    model = _W_MODEL
    foot_ids = _W_FOOT_IDS

    mujoco.mj_resetData(model, data)
    data.qpos[:] = start_qpos
    data.qvel[:] = start_qvel

    cost = 0.0
    H_STEPS = cfg['H_STEPS']
    PHYS_SUBSTEPS = cfg['PHYS_SUBSTEPS']
    DT = cfg['DT']
    P_DES_Z = cfg['P_DES_Z']
    Q_NOMINAL = cfg['Q_NOMINAL']
    W_H = cfg['W_H']
    W_ORIENT = cfg['W_ORIENT']
    W_Q = cfg['W_Q']
    W_C_VEL = cfg['W_C_VEL']
    W_C_FORCE = cfg['W_C_FORCE']
    W_H_TERM = cfg['W_H_TERM']
    V_DES = cfg['V_DES']
    H_HORIZON_SEC = cfg['H_HORIZON_SEC']
    KP = cfg['KP']
    KD = cfg['KD']
    kd_over_kp = KD / KP

    # Gait-clock parameters
    W_GAIT = cfg['W_GAIT']
    GAIT_FREQ = cfg['GAIT_FREQ_HZ']
    GAIT_AMP  = cfg['GAIT_AMPLITUDE']
    phase_step = 2.0 * np.pi * GAIT_FREQ * DT   # how much the clock advances per control step
    # foot id indices in foot_ids list: 0=FL, 1=FR, 2=RL, 3=RR
    # Diagonal A = {FL, RR} = indices (0, 3).  Diagonal B = {FR, RL} = indices (1, 2).

    _force_buf = np.zeros(6)

    for t in range(H_STEPS - 1):
        # ctrl = q_target + (KD/KP)*v_target.
        data.ctrl[:] = traj_q[t] + kd_over_kp * traj_v[t]
        for _ in range(PHYS_SUBSTEPS):
            mujoco.mj_step(model, data)

        if not np.isfinite(data.qpos).all():
            return 1e8

        cost += W_H * abs(data.qpos[2] - P_DES_Z)

        QUAT_DES = cfg['QUAT_DES']
        qw_err = np.dot(QUAT_DES, data.qpos[3:7])
        qw_err_clamped = max(-1.0, min(1.0, qw_err))
        ang = 2.0 * np.arccos(abs(qw_err_clamped))
        cost += W_ORIENT * ang * ang

        if W_Q > 0:
            dq = data.qpos[7:] - Q_NOMINAL
            cost += W_Q * float(dq @ dq)

        # =====================================================================
        # GAIT PHASE-CLOCK COST (Option B)
        # =====================================================================
        # Compute diagonal-pair mean foot heights, then penalize their
        # difference deviating from a sinusoidal target. The optimizer is
        # not told what each foot should do — only that the *difference*
        # between diagonal pairs should oscillate.
        if W_GAIT > 0.0:
            # foot world z heights (indices: 0=FL, 1=FR, 2=RL, 3=RR)
            z_FL = data.xpos[foot_ids[0], 2]
            z_FR = data.xpos[foot_ids[1], 2]
            z_RL = data.xpos[foot_ids[2], 2]
            z_RR = data.xpos[foot_ids[3], 2]
            d_A = 0.5 * (z_FL + z_RR)      # diagonal A average height
            d_B = 0.5 * (z_FR + z_RL)      # diagonal B average height
            # Clock phase at this rollout frame
            phase_t = gait_phase_start + (t + 1) * phase_step
            target_diff = GAIT_AMP * np.sin(phase_t)
            diff_err = (d_A - d_B) - target_diff
            cost += W_GAIT * diff_err * diff_err

        v_c = 0.0
        for body_id in foot_ids:
            if body_id == -1:
                continue
            foot_z = data.xpos[body_id, 2]
            foot_lin = data.cvel[body_id, 3:6]
            v_mag = (foot_lin[0]**2 + foot_lin[1]**2 + foot_lin[2]**2) ** 0.5
            in_contact = max(0.0, min(1.0, (0.03 - foot_z) / 0.03))
            v_c += in_contact * v_mag
        cost += W_C_VEL * v_c

        W_BADCONTACT = cfg['W_BADCONTACT']
        if W_BADCONTACT > 0.0:
            bad_contacts = 0
            for c_idx in range(data.ncon):
                contact = data.contact[c_idx]
                g1, g2 = contact.geom1, contact.geom2
                if g1 in _W_SHAFT_GEOM_IDS or g2 in _W_SHAFT_GEOM_IDS:
                    bad_contacts += 1
            cost += W_BADCONTACT * bad_contacts

        if W_C_FORCE > 0.0:
            BODY_WEIGHT = cfg['BODY_WEIGHT']
            SAFE_FOOT_FORCE = cfg['SAFE_FOOT_FORCE']
            f_per_foot = [0.0, 0.0, 0.0, 0.0]
            for c_idx in range(data.ncon):
                mujoco.mj_contactForce(model, data, c_idx, _force_buf)
                f_normal = abs(_force_buf[0])
                contact = data.contact[c_idx]
                g1, g2 = contact.geom1, contact.geom2
                for foot_idx, geom_set in enumerate(_W_FOOT_GEOM_IDS):
                    if g1 in geom_set or g2 in geom_set:
                        f_per_foot[foot_idx] += f_normal
                        break
            total_f = sum(f_per_foot)
            cost += W_C_FORCE * abs(total_f - BODY_WEIGHT)
            for f_foot in f_per_foot:
                excess = f_foot - SAFE_FOOT_FORCE
                if excess > 0.0:
                    cost += W_C_FORCE * excess

    p_target_x = start_qpos[0] + V_DES * H_HORIZON_SEC
    p_target_y = start_qpos[1]
    p_target_z = P_DES_Z
    terminal = (abs(data.qpos[0] - p_target_x)
              + abs(data.qpos[1] - p_target_y)
              + abs(data.qpos[2] - p_target_z))
    cost += W_H_TERM * terminal

    return cost


def _evaluate_chunk(start_qpos, start_qvel, traj_q_chunk, traj_v_chunk, gait_phase_start):
    n = len(traj_q_chunk)
    out = np.empty(n)
    for i in range(n):
        lane = i % len(_W_DATAS)
        out[i] = _evaluate_one(lane, start_qpos, start_qvel,
                                traj_q_chunk[i], traj_v_chunk[i],
                                gait_phase_start)
    return out


class GhostSimulator:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(Config.ROBOT_SCENE)
        self.model.opt.timestep = Config.PHYS_DT
        for i in range(12):
            self.model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_gainprm[i, 0] = Config.KP
            self.model.actuator_biasprm[i, 1] = -Config.KP
            self.model.actuator_biasprm[i, 2] = -Config.KD

        self.ghost_states = [mujoco.MjData(self.model) for _ in range(Config.NUM_SAMPLES)]

        self.foot_body_ids = []
        for name in ["FL", "FR", "RL", "RR"]:
            kin_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_foot")
            if kin_body_id == -1:
                kin_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_calf")
            self.foot_body_ids.append(kin_body_id)

        self.foot_geom_ids, self.shaft_geom_ids = _classify_leg_geoms(self.model)
        self.head_geom_id = _find_head_geom_id(self.model)
        print(f"  Head geom id: {self.head_geom_id}")
        for name, geoms in zip(["FL", "FR", "RL", "RR"], self.foot_geom_ids):
            print(f"  {name} foot geom: {sorted(geoms)}")

    def evaluate(self, sample_idx, start_qpos, start_qvel, traj_q, traj_v,
                 gait_phase_start=0.0, breakdown=False):
        """Rollout one trajectory under Eq. 13 PD with velocity feedforward."""
        data = self.ghost_states[sample_idx]
        mujoco.mj_resetData(self.model, data)
        data.qpos[:] = start_qpos
        data.qvel[:] = start_qvel

        cost = 0.0
        c_height = c_orient = c_q = c_vc = c_force = c_gait = 0.0
        kd_over_kp = Config.KD / Config.KP

        phase_step = 2.0 * np.pi * Config.GAIT_FREQ_HZ * Config.DT

        for t in range(Config.H_STEPS - 1):
            data.ctrl[:] = traj_q[t] + kd_over_kp * traj_v[t]
            for _ in range(Config.PHYS_SUBSTEPS):
                mujoco.mj_step(self.model, data)

            if not np.isfinite(data.qpos).all():
                if breakdown:
                    return {'total': 1e8, 'diverged': True}
                return 1e8

            term = Config.W_H * abs(data.qpos[2] - Config.P_DES_Z)
            c_height += term
            cost += term

            qw_err = np.dot(Config.QUAT_DES, data.qpos[3:7])
            qw_err_clamped = max(-1.0, min(1.0, qw_err))
            ang = 2.0 * np.arccos(abs(qw_err_clamped))
            term = Config.W_ORIENT * ang * ang
            c_orient += term
            cost += term

            if Config.W_Q > 0:
                dq = data.qpos[7:] - Config.Q_NOMINAL
                term = Config.W_Q * float(dq @ dq)
                c_q += term
                cost += term

            # --- gait phase-clock cost ---
            if Config.W_GAIT > 0.0:
                z_FL = data.xpos[self.foot_body_ids[0], 2]
                z_FR = data.xpos[self.foot_body_ids[1], 2]
                z_RL = data.xpos[self.foot_body_ids[2], 2]
                z_RR = data.xpos[self.foot_body_ids[3], 2]
                d_A = 0.5 * (z_FL + z_RR)
                d_B = 0.5 * (z_FR + z_RL)
                phase_t = gait_phase_start + (t + 1) * phase_step
                target_diff = Config.GAIT_AMPLITUDE * np.sin(phase_t)
                diff_err = (d_A - d_B) - target_diff
                term = Config.W_GAIT * diff_err * diff_err
                c_gait += term
                cost += term

            v_c = 0.0
            for body_id in self.foot_body_ids:
                if body_id == -1:
                    continue
                foot_z = data.xpos[body_id, 2]
                foot_lin = data.cvel[body_id, 3:6]
                v_mag = (foot_lin[0]**2 + foot_lin[1]**2 + foot_lin[2]**2) ** 0.5
                in_contact = max(0.0, min(1.0, (0.03 - foot_z) / 0.03))
                v_c += in_contact * v_mag
            term = Config.W_C_VEL * v_c
            c_vc += term
            cost += term

            if Config.W_BADCONTACT > 0.0:
                bad_contacts = 0
                for c_idx in range(data.ncon):
                    contact = data.contact[c_idx]
                    g1, g2 = contact.geom1, contact.geom2
                    if g1 in self.shaft_geom_ids or g2 in self.shaft_geom_ids:
                        bad_contacts += 1
                term = Config.W_BADCONTACT * bad_contacts
                c_force += term
                cost += term

            if Config.W_C_FORCE > 0:
                _force_buf = np.zeros(6)
                f_per_foot = [0.0, 0.0, 0.0, 0.0]
                for c_idx in range(data.ncon):
                    mujoco.mj_contactForce(self.model, data, c_idx, _force_buf)
                    f_normal = abs(_force_buf[0])
                    contact = data.contact[c_idx]
                    g1, g2 = contact.geom1, contact.geom2
                    for foot_idx, geom_set in enumerate(self.foot_geom_ids):
                        if g1 in geom_set or g2 in geom_set:
                            f_per_foot[foot_idx] += f_normal
                            break
                total_f = sum(f_per_foot)
                term = Config.W_C_FORCE * abs(total_f - Config.BODY_WEIGHT)
                for ff in f_per_foot:
                    excess = ff - Config.SAFE_FOOT_FORCE
                    if excess > 0.0:
                        term += Config.W_C_FORCE * excess
                c_force += term
                cost += term

        p_target_x = start_qpos[0] + Config.V_DES * Config.H_HORIZON_SEC
        p_target_y = start_qpos[1]
        p_target_z = Config.P_DES_Z
        term_x = abs(data.qpos[0] - p_target_x)
        term_y = abs(data.qpos[1] - p_target_y)
        term_z = abs(data.qpos[2] - p_target_z)
        c_terminal = Config.W_H_TERM * (term_x + term_y + term_z)
        cost += c_terminal

        if breakdown:
            return {
                'total': cost,
                'height': c_height,
                'orient': c_orient,
                'q_reg': c_q,
                'v_contact': c_vc,
                'f_contact': c_force,
                'gait': c_gait,
                'terminal': c_terminal,
                'terminal_dx': term_x,
                'terminal_dy': term_y,
                'terminal_dz': term_z,
                'final_z': data.qpos[2],
                'final_x': data.qpos[0],
            }
        return cost


class MPPIController:
    def __init__(self, initial_tau_q=None, seed_params=None):
        self.spline = CubicHermiteSpline(Config.H_HORIZON_SEC / (Config.K_NODES - 1))
        self.ghost = GhostSimulator()
        self.foot_body_ids = self.ghost.foot_body_ids
        self.seed_params = seed_params

        n_workers = Config.NUM_WORKERS or os.cpu_count() or 4
        n_workers = min(n_workers, Config.NUM_SAMPLES)
        chunk_size = max(1, (Config.NUM_SAMPLES + n_workers - 1) // n_workers)
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.mode = Config.PARALLEL_MODE
        print(f"--- Parallelism: mode={self.mode}, workers={n_workers}, chunk_size={chunk_size}")

        if self.mode == 'process':
            self.chunks = []
            for s in range(0, Config.NUM_SAMPLES, chunk_size):
                self.chunks.append((s, min(s + chunk_size, Config.NUM_SAMPLES)))
            cfg_dict = {k: getattr(Config, k) for k in [
                'H_STEPS', 'PHYS_SUBSTEPS', 'DT', 'P_DES_Z', 'Q_NOMINAL', 'QUAT_DES',
                'W_H', 'W_ORIENT', 'W_Q', 'W_C_VEL', 'W_C_FORCE',
                'W_BADCONTACT', 'W_H_TERM', 'V_DES', 'H_HORIZON_SEC',
                'BODY_WEIGHT', 'SAFE_FOOT_FORCE',
                'KP', 'KD',
                'W_GAIT', 'GAIT_FREQ_HZ', 'GAIT_AMPLITUDE',
            ]}
            ctx = mp.get_context('spawn')
            self.executor = ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=_worker_init,
                initargs=(Config.ROBOT_SCENE, Config.KP, Config.KD, Config.PHYS_DT,
                          chunk_size, cfg_dict),
            )
        elif self.mode == 'thread':
            self.executor = ThreadPoolExecutor(max_workers=n_workers)
        else:
            self.executor = None

        self.times = np.linspace(0, Config.H_HORIZON_SEC, Config.H_STEPS)

        # === Paper Sec. III-D: τ₀ and τ_best are SEPARATE state ===
        if initial_tau_q is not None:
            assert initial_tau_q.shape == (Config.H_STEPS, 12)
            self.tau_best_q = initial_tau_q.copy()
            node_idx = np.linspace(0, Config.H_STEPS - 1, Config.K_NODES).astype(int)
            self.theta_q = self.tau_best_q[node_idx].copy()
            dt_node = Config.H_HORIZON_SEC / (Config.K_NODES - 1)
            self.theta_v = np.zeros((Config.K_NODES, 12))
            for k in range(Config.K_NODES):
                if k == 0:
                    self.theta_v[k] = (self.theta_q[1] - self.theta_q[0]) / dt_node
                elif k == Config.K_NODES - 1:
                    self.theta_v[k] = (self.theta_q[k] - self.theta_q[k-1]) / dt_node
                else:
                    self.theta_v[k] = (self.theta_q[k+1] - self.theta_q[k-1]) / (2 * dt_node)
            sq0 = self.theta_q[None, :, :].copy()
            sv0 = self.theta_v[None, :, :].copy()
            _tq, tv = self.spline.interpolate(sq0, sv0, self.times)
            self.tau_best_v = np.ascontiguousarray(tv[0], dtype=np.float64)
            self.tau0_q = self.tau_best_q.copy()
            self.tau0_v = self.tau_best_v.copy()
        else:
            self.theta_q = np.tile(Config.Q_NOMINAL, (Config.K_NODES, 1))
            self.theta_v = np.zeros((Config.K_NODES, 12))
            self.tau_best_q = np.tile(Config.Q_NOMINAL, (Config.H_STEPS, 1))
            self.tau_best_v = np.zeros((Config.H_STEPS, 12))
            self.tau0_q = self.tau_best_q.copy()
            self.tau0_v = self.tau_best_v.copy()

        self.tau_best_cost = float('inf')
        self._original_seed = self.tau_best_q.copy()
        self.step_counter = 0
        self.tau_best_age = 0

        # === Gait phase clock — persists across control steps ===
        # Advances by 2π * GAIT_FREQ_HZ * DT each control step. This is the
        # absolute clock phase passed into every rollout so the gait target
        # is consistent across consecutive control steps.
        self.gait_phase = 0.0
        self._phase_step_per_control = 2.0 * np.pi * Config.GAIT_FREQ_HZ * Config.DT

        self._dbg_last_iter_sq = None
        self._dbg_last_iter_sv = None
        self._dbg_last_iter_costs = None

        print("--- Tracking foot bodies ---")
        for name, bid in zip(["FL", "FR", "RL", "RR"], self.foot_body_ids):
            print(f"  {name}: body_id={bid}")
        print(f"--- Gait clock: freq={Config.GAIT_FREQ_HZ}Hz, amp={Config.GAIT_AMPLITUDE}m, W_GAIT={Config.W_GAIT}")

    def shutdown(self):
        if self.executor is not None:
            self.executor.shutdown(wait=False)

    def _annealed_sigma(self, iter_i, node_k):
        I, K = Config.NUM_ITERATIONS, Config.K_NODES
        return float(np.exp(
            -(I - iter_i) / (Config.BETA_1 * I)
            -(K - node_k) / (Config.BETA_2 * K)
        ))

    def _compute_weights(self, costs):
        finite = costs < 1e7
        if not finite.any():
            return np.ones(len(costs)) / len(costs)
        cmin = costs[finite].min()
        cmax = costs[finite].max()
        if cmax == cmin:
            w = np.zeros(len(costs))
            w[finite] = 1.0 / finite.sum()
            return w
        norm = np.where(finite, (costs - cmin) / (cmax - cmin), 1.0)
        w = np.exp(-norm / Config.LAMBDA)
        s = w.sum()
        return w / s if s > 0 else np.ones(len(costs)) / len(costs)

    def control_step(self, current_qpos, current_qvel, compute_dt_seconds):
        self.step_counter += 1

        pred_qpos = current_qpos
        pred_qvel = current_qvel
        n_steps_ahead = 0

        if self.step_counter > 1:
            self.tau_best_q = np.concatenate([
                self.tau_best_q[1:],
                self.tau_best_q[-1:]
            ])
            self.tau_best_v = np.concatenate([
                self.tau_best_v[1:],
                self.tau_best_v[-1:]
            ])
            self.tau0_q = self.tau_best_q.copy()
            self.tau0_v = self.tau_best_v.copy()
            # Advance the gait clock by one control step
            self.gait_phase += self._phase_step_per_control
            # Wrap to keep numbers small (modulo 2π); doesn't change sin/cos
            self.gait_phase = self.gait_phase % (2.0 * np.pi)

        # Evaluate the shifted tau_best at the start of the step against the
        # current gait phase. This is the cost to beat for this control step.
        self.tau_best_cost = self.ghost.evaluate(
            0, pred_qpos, pred_qvel, self.tau_best_q, self.tau_best_v,
            gait_phase_start=self.gait_phase
        )

        node_idx = np.linspace(0, Config.H_STEPS - 1, Config.K_NODES).astype(int)
        self.theta_q = self.tau0_q[node_idx].copy()
        self.theta_v = self.tau0_v[node_idx].copy()

        dt_node = Config.H_HORIZON_SEC / (Config.K_NODES - 1)

        self._dbg_sample0_won = False
        self._dbg_winning_idx = -1
        self._dbg_sample0_cost = None
        self._dbg_cost_min = None
        self._dbg_cost_max = None
        self._dbg_cost_median = None

        improved = False

        for iter_i in range(Config.NUM_ITERATIONS, 0, -1):
            sq = np.zeros((Config.NUM_SAMPLES, Config.K_NODES, 12))
            sv = np.zeros((Config.NUM_SAMPLES, Config.K_NODES, 12))

            sigmas = np.array([self._annealed_sigma(iter_i, k) for k in range(Config.K_NODES)])

            for n in range(Config.NUM_SAMPLES):
                for k in range(Config.K_NODES):
                    sq_noise = np.random.normal(0, Config.SCALE_Q * sigmas[k], 12)
                    sv_noise = np.random.normal(0, Config.SCALE_V * sigmas[k], 12)
                    sq[n, k] = np.clip(
                        self.theta_q[k] + sq_noise,
                        Config.JOINT_MIN, Config.JOINT_MAX
                    )
                    v_raw = self.theta_v[k] + sv_noise
                    sv[n, k] = apply_derivative_clamp(
                        sq[n, k], v_raw, Config.JOINT_MIN, Config.JOINT_MAX, dt_node
                    )

            # Sample 0 is the unperturbed nominal
            sq[0] = self.theta_q.copy()
            sv[0] = self.theta_v.copy()

            tq_all, tv_all = self.spline.interpolate(sq, sv, self.times)
            tq_all = np.ascontiguousarray(tq_all, dtype=np.float64)
            tv_all = np.ascontiguousarray(tv_all, dtype=np.float64)

            if self.mode == 'process':
                futures = []
                for (s, e) in self.chunks:
                    chunk_q = tq_all[s:e]
                    chunk_v = tv_all[s:e]
                    futures.append(self.executor.submit(
                        _evaluate_chunk, pred_qpos, pred_qvel,
                        chunk_q, chunk_v, self.gait_phase))
                costs = np.empty(Config.NUM_SAMPLES)
                for (s, e), f in zip(self.chunks, futures):
                    costs[s:e] = f.result()
            elif self.mode == 'thread':
                futures = [
                    self.executor.submit(self.ghost.evaluate, n,
                                         pred_qpos, pred_qvel,
                                         tq_all[n], tv_all[n],
                                         gait_phase_start=self.gait_phase)
                    for n in range(Config.NUM_SAMPLES)
                ]
                costs = np.array([f.result() for f in futures])
            else:
                costs = np.array([
                    self.ghost.evaluate(n, pred_qpos, pred_qvel,
                                         tq_all[n], tv_all[n],
                                         gait_phase_start=self.gait_phase)
                    for n in range(Config.NUM_SAMPLES)
                ])

            weights = self._compute_weights(costs)

            self.theta_q = np.sum(sq * weights[:, None, None], axis=0)
            self.theta_v = np.sum(sv * weights[:, None, None], axis=0)

            iter_best_idx = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            self._dbg_sample0_cost = float(costs[0])
            self._dbg_cost_min = float(costs.min())
            self._dbg_cost_max = float(costs[costs < 1e7].max()) if (costs < 1e7).any() else 1e8
            self._dbg_cost_median = float(np.median(costs[costs < 1e7])) if (costs < 1e7).any() else 1e8
            if iter_best_idx == 0:
                self._dbg_sample0_won = True
            self._dbg_winning_idx = iter_best_idx

            if iter_i == 1:
                self._dbg_last_iter_sq = tq_all.copy()
                self._dbg_last_iter_sv = tv_all.copy()
                self._dbg_last_iter_costs = costs.copy()

            if iter_best_cost < self.tau_best_cost:
                self.tau_best_cost = iter_best_cost
                self.tau_best_q = tq_all[iter_best_idx].copy()
                self.tau_best_v = tv_all[iter_best_idx].copy()
                improved = True
                self.tau_best_age = 0

        if not improved:
            self.tau_best_age += 1

        breakdown = self.ghost.evaluate(0, current_qpos, current_qvel,
                                         self.tau_best_q, self.tau_best_v,
                                         gait_phase_start=self.gait_phase,
                                         breakdown=True)

        sq_nom = self.theta_q[None, :, :]
        sv_nom = self.theta_v[None, :, :]
        tq_nom, tv_nom = self.spline.interpolate(sq_nom, sv_nom, self.times)
        self.tau0_q = np.ascontiguousarray(tq_nom[0], dtype=np.float64)
        self.tau0_v = np.ascontiguousarray(tv_nom[0], dtype=np.float64)

        tau0_vs_best = float(np.abs(self.tau0_q - self.tau_best_q).mean())
        motion_content = float(self.tau_best_q[:10].std(axis=0).mean())

        seed_drift = None
        if self._original_seed is not None:
            seed_drift = float(np.abs(self.tau_best_q - self._original_seed).mean())

        return {
            'cost': self.tau_best_cost,
            'improved': improved,
            'n_steps_ahead': n_steps_ahead,
            'breakdown': breakdown,
            'sample0_won': self._dbg_sample0_won,
            'winning_idx': self._dbg_winning_idx,
            'sample0_cost': self._dbg_sample0_cost,
            'cost_min': self._dbg_cost_min,
            'cost_max': self._dbg_cost_max,
            'cost_median': self._dbg_cost_median,
            'seed_drift': seed_drift,
            'tau_best_age': self.tau_best_age,
            'tau0_vs_best': tau0_vs_best,
            'motion_content': motion_content,
            'gait_phase': self.gait_phase,
        }


def generate_trot_seed(base_pose, freq_hz=3.0, hip_amp=0.20, knee_amp=0.40,
                       fwd_lean=0.05, t_offset=0.0):
    seed = np.zeros((Config.H_STEPS, 12))
    for h in range(Config.H_STEPS):
        t = h * Config.DT + t_offset
        phase = 2 * np.pi * freq_hz * t
        sA = np.sin(phase)
        sB = np.sin(phase + np.pi)
        liftA = max(0.0, sA)
        liftB = max(0.0, sB)
        target = base_pose.copy()
        target[1]  += -hip_amp * sA + fwd_lean
        target[2]  += -knee_amp * liftA
        target[4]  += -hip_amp * sB + fwd_lean
        target[5]  += -knee_amp * liftB
        target[7]  += -hip_amp * sB + fwd_lean
        target[8]  += -knee_amp * liftB
        target[10] += -hip_amp * sA + fwd_lean
        target[11] += -knee_amp * liftA
        seed[h] = np.clip(target, Config.JOINT_MIN, Config.JOINT_MAX)
    return seed


if __name__ == "__main__":
    print("=== Reference-Free MPPI + Gait Phase Clock (Option B cost shaping) ===\n")

    live_model = mujoco.MjModel.from_xml_path(Config.ROBOT_SCENE)
    live_model.opt.timestep = Config.PHYS_DT
    for i in range(12):
        live_model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        live_model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        live_model.actuator_gainprm[i, 0] = Config.KP
        live_model.actuator_biasprm[i, 1] = -Config.KP
        live_model.actuator_biasprm[i, 2] = -Config.KD
    live_data = mujoco.MjData(live_model)

    width, height = 640, 480
    fps = int(1.0 / Config.DT)
    renderer = mujoco.Renderer(live_model, height, width)
    video_writer = cv2.VideoWriter(
        "emergent_walk.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 2.0
    camera.elevation = -20
    camera.azimuth = 90

    def capture_frame():
        camera.lookat[:] = live_data.qpos[:3]
        renderer.update_scene(live_data, camera=camera)
        frame = renderer.render()
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    live_data.qpos[0:3] = [0.0, 0.0, 0.5]
    live_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    initial_relaxed_pose = np.array([0.0, 1.2, -2.5] * 4)
    live_data.qpos[7:] = initial_relaxed_pose

    mujoco.mj_kinematics(live_model, live_data)
    geom_ids = [i for i in range(live_model.ngeom) if live_model.geom_bodyid[i] > 0]
    bottoms = live_data.geom_xpos[geom_ids, 2] - live_model.geom_size[geom_ids, 0]
    live_data.qpos[2] = (0.5 - np.min(bottoms)) + 0.001

    print("Stand-up routine...")
    stand_steps = int(2.0 / Config.PHYS_DT)
    for i in range(stand_steps):
        alpha = (1 - np.cos(np.pi * i / stand_steps)) / 2.0
        live_data.ctrl[:] = (1 - alpha) * initial_relaxed_pose + alpha * Config.Q_NOMINAL
        mujoco.mj_step(live_model, live_data)

    for _ in range(100):
        live_data.ctrl[:] = Config.Q_NOMINAL
        mujoco.mj_step(live_model, live_data)

    Config.P_DES_Z = float(live_data.qpos[2])
    Config.Q_NOMINAL = live_data.qpos[7:].copy()
    print(f"  Post-settle: qpos[2] = {Config.P_DES_Z:.3f} m  (used as P_DES_Z)")
    print(f"  Q_NOMINAL[:3] = {np.round(Config.Q_NOMINAL[:3], 3)}\n")

    print("Initializing optimizer from stable standing (no seed)...\n")

    mppi = MPPIController(initial_tau_q=None, seed_params=None)
    print()

    recorder = None
    if Config.RECORD_SPLINES:
        recorder = SplineRecorder(
            out_path=Config.RECORD_OUT_PATH,
            record_every=Config.RECORD_EVERY_N_STEPS,
            n_samples_to_keep=Config.RECORD_N_SAMPLES_KEPT,
            max_steps=Config.RECORD_MAX_STEPS,
        )
        print(f"SplineRecorder: logging every {Config.RECORD_EVERY_N_STEPS} steps "
              f"to {Config.RECORD_OUT_PATH} (max {Config.RECORD_MAX_STEPS} records)\n")

    SIM_SECONDS = 30.0
    total_steps = int(SIM_SECONDS / Config.DT)
    print(f"Running MPC for {SIM_SECONDS}s = {total_steps} control steps\n")

    kd_over_kp = Config.KD / Config.KP

    last_compute_dt = 0.0
    interrupted = False
    try:
      for step in range(total_steps):
        t0 = time.perf_counter()
        info = mppi.control_step(
            live_data.qpos.copy(),
            live_data.qvel.copy(),
            compute_dt_seconds=last_compute_dt,
        )
        last_compute_dt = time.perf_counter() - t0

        if recorder is not None:
            recorder.record(
                step=step,
                live_qpos=live_data.qpos,
                live_qvel=live_data.qvel,
                tau_best_q=mppi.tau_best_q,
                tau_best_v=mppi.tau_best_v,
                tau0_q=mppi.tau0_q,
                tau0_v=mppi.tau0_v,
                last_iter_samples_q=mppi._dbg_last_iter_sq,
                last_iter_samples_v=mppi._dbg_last_iter_sv,
                last_iter_costs=mppi._dbg_last_iter_costs,
            )

        live_data.ctrl[:] = mppi.tau_best_q[0] + kd_over_kp * mppi.tau_best_v[0]
        for _ in range(Config.PHYS_SUBSTEPS):
            mujoco.mj_step(live_model, live_data)

        capture_frame()

        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), live_data.qpos[3:7])
        tilt_deg = np.degrees(np.arccos(max(-1.0, min(1.0, up[2]))))
        body_x = np.zeros(3)
        mujoco.mju_rotVecQuat(body_x, np.array([1.0, 0.0, 0.0]), live_data.qpos[3:7])
        pitch_sign = "↓" if body_x[2] < -0.01 else ("↑" if body_x[2] > 0.01 else "·")
        flag = ('imp' if info['improved']
                else f'kpt({info["tau_best_age"]:3d})')
        # Show gait clock value for debugging
        clock_val = Config.GAIT_AMPLITUDE * np.sin(info['gait_phase'])
        sys.stdout.write(
            f"\r[{step:04d}/{total_steps}] "
            f"x={live_data.qpos[0]:+.2f}m  "
            f"z={live_data.qpos[2]:.2f}m  "
            f"vx={live_data.qvel[0]:+.2f}  "
            f"tilt={tilt_deg:.0f}°{pitch_sign}  "
            f"cost={info['cost']:7.1f}  "
            f"clk={clock_val:+.3f}  "
            f"{flag}  "
            f"{last_compute_dt*1000:.0f}ms"
        )
        sys.stdout.flush()

        if step % 25 == 0 and step > 0:
            b = info['breakdown']
            fl_id, fr_id, rl_id, rr_id = mppi.foot_body_ids
            fl_z = live_data.xpos[fl_id, 2]
            fr_z = live_data.xpos[fr_id, 2]
            rl_z = live_data.xpos[rl_id, 2]
            rr_z = live_data.xpos[rr_id, 2]
            # Measured diagonal-pair height difference (live, not in rollout)
            d_A_live = 0.5 * (fl_z + rr_z)
            d_B_live = 0.5 * (fr_z + rl_z)
            diff_live = d_A_live - d_B_live
            FOOT_DOWN = 0.03
            n_down = sum(z < FOOT_DOWN for z in (fl_z, fr_z, rl_z, rr_z))
            def fmark(z):
                if z < FOOT_DOWN: return "▼"
                if z < 0.08: return "·"
                return "▲"
            v_target_mag = float(np.abs(mppi.tau_best_v[0]).mean())
            print(f"\n  ↳ cost breakdown: "
                  f"height={b['height']:6.1f}  "
                  f"orient={b['orient']:6.1f}  "
                  f"q={b['q_reg']:5.2f}  "
                  f"vc={b['v_contact']:6.1f}  "
                  f"badC={b['f_contact']:6.1f}  "
                  f"gait={b.get('gait',0.0):6.1f}  "
                  f"term={b['terminal']:6.1f}")
            print(f"  ↳ feet ({n_down}/4 down): "
                  f"FL={fl_z:.3f}{fmark(fl_z)} "
                  f"FR={fr_z:.3f}{fmark(fr_z)} "
                  f"RL={rl_z:.3f}{fmark(rl_z)} "
                  f"RR={rr_z:.3f}{fmark(rr_z)}  "
                  f"| Δdiag={diff_live:+.3f}m  "
                  f"vs clk={clock_val:+.3f}m  "
                  f"| body_pitch_z={body_x[2]:+.3f}")

        if step % 10 == 0:
            won = info['sample0_won']
            widx = info['winning_idx']
            s0 = info['sample0_cost']
            cmin = info['cost_min']
            cmed = info['cost_median']
            cmax = info['cost_max']
            spread = (cmed - cmin) / max(cmed, 1.0)
            winner_str = "nominal(s0)" if won else f"noise#{widx:2d}"
            print(f"\n  ⌖ MPPI: {winner_str:12s}  "
                  f"s0={s0:.0f}  min={cmin:.0f}  med={cmed:.0f}  max={cmax:.0f}  "
                  f"spread={spread:.3f}  "
                  f"τ_best_age={info['tau_best_age']:3d}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[interrupted by Ctrl+C — flushing logs and exiting]")
    except Exception as e:
        interrupted = True
        print(f"\n\n[exception during loop: {type(e).__name__}: {e}]")
        import traceback; traceback.print_exc()
    finally:
        if not interrupted:
            print("\n\nSimulation done. Saving video...")
        else:
            print("Saving partial video and logs...")
        try:
            video_writer.release()
            renderer.close()
        except Exception as e:
            print(f"  (video writer cleanup error: {e})")

        if recorder is not None:
            try:
                recorder.save()
                print("To plot: python plot_splines.py", Config.RECORD_OUT_PATH)
            except Exception as e:
                print(f"  (recorder save error: {e})")

        try:
            mppi.shutdown()
        except Exception as e:
            print(f"  (mppi shutdown error: {e})")

        if not interrupted:
            print("Saved emergent_walk.mp4")