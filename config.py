"""
Configuration for the reference-free sampling-based MPC controller
(Schramm et al., arXiv:2511.19204) on the Go2.

All tunable parameters live in `Config`. Any field can be overridden from the
environment without editing this file: EW_CFG_<NAME>=value (e.g. EW_CFG_W_Q=0.3).
"""

import os

import numpy as np


class Config:
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml"
    USE_FLAT_SCENE = os.environ.get("EW_FLAT", "1") == "1"   # flat ground (no boxes/stairs)

    # === MPPI structural parameters ===
    H_HORIZON_SEC = 0.9
    CONTROL_FREQ  = 50.0
    DT            = 1.0 / CONTROL_FREQ
    H_STEPS       = int(H_HORIZON_SEC / DT) + 1          # 46

    LIVE_DT       = 0.005                                # live-sim physics step
    LIVE_SUBSTEPS = int(round(DT / LIVE_DT))             # 4
    PLAN_DT       = 0.02                                 # planner physics step (coarser = faster)
    PLAN_SUBSTEPS = int(round(DT / PLAN_DT))             # 2

    K_NODES        = 10
    NUM_ITERATIONS = 3
    NUM_SAMPLES    = 30
    NUM_THREADS    = min(NUM_SAMPLES, os.cpu_count() or 1)

    LAMBDA = 0.1

    # What to execute each step:
    #   0 = always the best raw sample (tau_best): decisive but noisy -> leg thrash
    #   1 = always the weighted-average nominal (tau0): smooth (jvel 2.6 -> 1.4,
    #       power 140 -> 65 W) but too timid to recover once tipped over
    #   2 = auto: nominal while upright, best sample when tilt > RECOVERY_TILT_DEG
    EXEC_NOMINAL      = 2
    RECOVERY_TILT_DEG = 25.0

    # Noise scaling (swept 2026-06: SCALE_Q 0.05 too timid->falls, 0.15 best; SCALE_V monotone, lower=stabler)
    SCALE_Q = 0.15        # position-node exploration std (rad)
    SCALE_V = 0.25        # velocity-node exploration std (rad/s); Eq.5 clamp keeps it safe
    BETA_1  = 1.0
    BETA_2  = 1.0

    # === COST WEIGHTS (paper Tab. II walking row — reference-free, no gait clock) ===
    # Swept: W_H 200 stops belly-grazing crouches, W_ORIENT 150 stops tip-overs;
    # W_Q>0 only hurt (0.3/1.0 stalled progress without calming joints), keep 0.
    W_H       = 100.0
    W_ORIENT  = 80.0
    W_Q       = 30.0
    W_C_VEL   = 1.0       # 1.5 made stepping itself too expensive -> falls
    W_C_FORCE = 0.1       # paper uses 5e-2; swept {0,.05,.15}: calms joints slightly but
                          # degrades walking and adds falls here — keep 0, term stays available
    W_H_TERM  = 3000.0    # paper value; 4500 caused speed overshoot and tilt spikes

    CONTACT_HEIGHT  = 0.05   # foot-sphere center below this counts as (partial) contact

    # === PD controller ===
    KP = 40.0
    KD = 1.0

    # === Task targets ===
    P_DES_Z   = 0.30
    V_DES     = 0.50
    Q_NOMINAL = np.array([0.0, 0.7, -1.45] * 4)
    QUAT_DES  = np.array([1.0, 0.0, 0.0, 0.0])

    JOINT_MIN = np.array([-0.8, -0.5, -2.6] * 4)
    JOINT_MAX = np.array([ 0.8,  2.5, -0.8] * 4)


# Tuning overrides without editing the file: EW_CFG_<NAME>=value (e.g. EW_CFG_W_Q=0.3)
for _k, _v in os.environ.items():
    if _k.startswith("EW_CFG_"):
        _name = _k[len("EW_CFG_"):]
        if hasattr(Config, _name):
            _old = getattr(Config, _name)
            setattr(Config, _name, type(_old)(float(_v)) if isinstance(_old, (int, float)) else _v)
            print(f"[config override] {_name} = {getattr(Config, _name)}")
# re-derive dependent values in case PLAN_DT / NUM_SAMPLES changed
Config.PLAN_SUBSTEPS = int(round(Config.DT / Config.PLAN_DT))
Config.LIVE_SUBSTEPS = int(round(Config.DT / Config.LIVE_DT))
Config.NUM_THREADS   = min(Config.NUM_SAMPLES, os.cpu_count() or 1)


FOOT_NAMES = ["FL", "FR", "RL", "RR"]   # named foot-sphere geoms in go2.xml
