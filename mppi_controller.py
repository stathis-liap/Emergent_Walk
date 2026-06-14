"""
Reference-free sampling-based MPC (Schramm et al., arXiv:2511.19204) on the Go2.

Planner backend: MuJoCo's native multithreaded `rollout` module (C++, releases
the GIL).  At 30-70 samples this is the architecture the paper validates for
real-time CPU control.  The previous MJX/GPU backend (kept in
mppi_core_mjx_backup.py) was dispatch-latency-bound: 3 iterations x 45 control
steps x 4 substeps = 540 *sequential* GPU kernel dispatches per control step,
which no batch size of 30 can amortize.

Planner and live simulation share the same XML, solver and actuator setup; the
only deliberate difference is the planner's coarser timestep (PLAN_DT).
"""

import numpy as np
import mujoco
from mujoco import rollout as mj_rollout

from config import Config
from scene_builder import (
    make_scene_with_sensors,
    prepare_model,
    actuator_joint_perm,
    foot_sensor_columns,
)
from math_engine import CubicHermiteSpline, apply_derivative_clamp_batch


# ---------------------------------------------------------------------------
# MPPI Controller
# ---------------------------------------------------------------------------

class MPPIController:
    def __init__(self, initial_tau_q=None, seed_params=None):
        scene_path = make_scene_with_sensors()
        self._model = prepare_model(scene_path, Config.PLAN_DT)
        self.act_perm = actuator_joint_perm(self._model)
        self._foot_z_cols, self._foot_v_cols, self._foot_f_cols = foot_sensor_columns(self._model)
        base_id  = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        self._f0 = self._model.body_subtreemass[base_id] * 9.81 / 4.0   # nominal per-foot stance force

        self._nq = self._model.nq
        self._nv = self._model.nv
        self._nstate = mujoco.mj_stateSize(self._model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        assert self._nstate == 1 + self._nq + self._nv, "unexpected state layout (act states?)"

        self._nstep_plan = (Config.H_STEPS - 1) * Config.PLAN_SUBSTEPS
        # cost is evaluated at the end of each 20 ms control step (as before)
        self._cost_idx = np.arange(Config.PLAN_SUBSTEPS - 1, self._nstep_plan, Config.PLAN_SUBSTEPS)

        # Persistent thread pool + per-thread MjData + preallocated outputs
        self._pool  = mj_rollout.Rollout(nthread=Config.NUM_THREADS)
        self._datas = [mujoco.MjData(self._model) for _ in range(Config.NUM_THREADS)]
        self._state_buf = np.empty((Config.NUM_SAMPLES, self._nstep_plan, self._nstate))
        self._sens_buf  = np.empty((Config.NUM_SAMPLES, self._nstep_plan, self._model.nsensordata))

        # ── Spline helper ──
        self._spline = CubicHermiteSpline(Config.H_HORIZON_SEC / (Config.K_NODES - 1))
        self._times  = np.linspace(0, Config.H_HORIZON_SEC, Config.H_STEPS)

        # ── Trajectory state ──
        if initial_tau_q is not None:
            assert initial_tau_q.shape == (Config.H_STEPS, 12)
            self.tau_best_q = initial_tau_q.astype(np.float64)
            node_idx = np.linspace(0, Config.H_STEPS - 1, Config.K_NODES).astype(int)
            self.theta_q    = self.tau_best_q[node_idx].copy()
            dt_node         = Config.H_HORIZON_SEC / (Config.K_NODES - 1)
            self.theta_v    = np.zeros((Config.K_NODES, 12))
            for k in range(Config.K_NODES):
                if k == 0:
                    self.theta_v[k] = (self.theta_q[1] - self.theta_q[0]) / dt_node
                elif k == Config.K_NODES - 1:
                    self.theta_v[k] = (self.theta_q[k] - self.theta_q[k-1]) / dt_node
                else:
                    self.theta_v[k] = (self.theta_q[k+1] - self.theta_q[k-1]) / (2 * dt_node)
            _, tv = self._spline.interpolate(self.theta_q[None], self.theta_v[None], self._times)
            self.tau_best_v = np.ascontiguousarray(tv[0])
        else:
            self.theta_q    = np.tile(Config.Q_NOMINAL, (Config.K_NODES, 1))
            self.theta_v    = np.zeros((Config.K_NODES, 12))
            self.tau_best_q = np.tile(Config.Q_NOMINAL, (Config.H_STEPS, 1))
            self.tau_best_v = np.zeros((Config.H_STEPS, 12))

        self.tau0_q = self.tau_best_q.copy()
        self.tau0_v = self.tau_best_v.copy()
        self.tau_best_cost = float('inf')

        self.step_counter = 0
        self.tau_best_age = 0
        self._dbg_sample0_won  = False
        self._dbg_winning_idx  = -1
        self._dbg_sample0_cost = None
        self._dbg_cost_min     = None
        self._dbg_cost_max     = None
        self._dbg_cost_median  = None
        self._last_breakdown   = None

        print(f"--- Planner: CPU threaded rollout  threads={Config.NUM_THREADS}  "
              f"samples={Config.NUM_SAMPLES}  iters={Config.NUM_ITERATIONS}  "
              f"plan_dt={Config.PLAN_DT*1000:.0f}ms ({self._nstep_plan} steps/rollout)\n")

    def shutdown(self):
        try:
            self._pool.close()
        except Exception:
            pass

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

    # ── Vectorised cost evaluation over the whole batch (numpy) ──
    def _evaluate(self, state, sens, start_qpos):
        nq  = self._nq
        idx = self._cost_idx
        qpos_t = state[:, idx, 1:1 + nq]              # (N, 45, 19)
        z      = qpos_t[:, :, 2]
        quat   = qpos_t[:, :, 3:7]
        qj     = qpos_t[:, :, 7:]

        c_h = Config.W_H * np.abs(z - Config.P_DES_Z).sum(axis=1)

        qw  = np.clip(np.abs(quat @ Config.QUAT_DES), 0.0, 1.0)
        ang = 2.0 * np.arccos(qw)
        c_o = Config.W_ORIENT * (ang * ang).sum(axis=1)

        dq  = qj - Config.Q_NOMINAL
        c_q = Config.W_Q * (dq * dq).sum(axis=(1, 2))

        sub    = sens[:, idx]                          # (N, 45, nsensordata)
        foot_z = sub[:, :, self._foot_z_cols]          # (N, 45, 4)  FL,FR,RL,RR
        foot_v = sub[:, :, self._foot_v_cols]          # (N, 45, 4, 3)

        # Contact velocity: penalise foot speed while (partially) in contact
        vnorm = np.linalg.norm(foot_v, axis=3)
        ic    = np.clip((Config.CONTACT_HEIGHT - foot_z) / Config.CONTACT_HEIGHT, 0.0, 1.0)
        c_vc  = Config.W_C_VEL * (ic * vnorm).sum(axis=(1, 2))

        # Contact force (paper Eq. 17): keep per-foot normal force near the
        # nominal stance share f0 = mg/4 — discourages slamming and flight
        foot_f = sub[:, :, self._foot_f_cols]          # (N, 45, 4)
        c_f    = Config.W_C_FORCE * np.abs(foot_f - self._f0).sum(axis=(1, 2))

        qpos_f = state[:, -1, 1:1 + nq]
        tx = np.abs(qpos_f[:, 0] - (start_qpos[0] + Config.V_DES * Config.H_HORIZON_SEC))
        ty = np.abs(qpos_f[:, 1] - start_qpos[1])
        tz = np.abs(qpos_f[:, 2] - Config.P_DES_Z)
        c_term = Config.W_H_TERM * (tx + ty + tz)

        total = c_h + c_o + c_q + c_vc + c_f + c_term
        total = np.where(np.isfinite(total), total, 1e8)

        terms = {
            'height': c_h, 'orient': c_o, 'q_reg': c_q,
            'v_contact': c_vc, 'f_contact': c_f, 'terminal': c_term,
            'terminal_dx': tx, 'terminal_dy': ty, 'terminal_dz': tz,
            'final_z': qpos_f[:, 2], 'final_x': qpos_f[:, 0],
        }
        return total, terms

    def control_step(self, current_qpos, current_qvel, compute_dt_seconds=0.0):
        self.step_counter += 1

        if self.step_counter > 1:
            self.tau_best_q = np.concatenate([self.tau_best_q[1:], self.tau_best_q[-1:]])
            self.tau_best_v = np.concatenate([self.tau_best_v[1:], self.tau_best_v[-1:]])
            self.tau0_q = self.tau_best_q.copy()
            self.tau0_v = self.tau_best_v.copy()

        node_idx = np.linspace(0, Config.H_STEPS - 1, Config.K_NODES).astype(int)
        self.theta_q = self.tau0_q[node_idx].copy()
        self.theta_v = self.tau0_v[node_idx].copy()
        dt_node = Config.H_HORIZON_SEC / (Config.K_NODES - 1)

        # Same start state for every rollout: [time, qpos, qvel]
        init_one = np.concatenate([[0.0], current_qpos, current_qvel])
        init_state = np.tile(init_one, (Config.NUM_SAMPLES, 1))

        kd_over_kp = Config.KD / Config.KP
        H = Config.H_STEPS

        # Re-baseline each step: the shifted tau_best was scored against the
        # previous state; force re-evaluation against the current one.
        self.tau_best_cost = float('inf')
        improved = False

        for iter_i in range(Config.NUM_ITERATIONS, 0, -1):
            sigmas = np.array([self._annealed_sigma(iter_i, k)
                               for k in range(Config.K_NODES)])  # (K,)

            rng_q = np.random.normal(0, 1, (Config.NUM_SAMPLES, Config.K_NODES, 12))
            rng_v = np.random.normal(0, 1, (Config.NUM_SAMPLES, Config.K_NODES, 12))
            rng_q *= Config.SCALE_Q * sigmas[None, :, None]
            rng_v *= Config.SCALE_V * sigmas[None, :, None]

            sq = np.clip(
                self.theta_q[None, :, :] + rng_q,
                Config.JOINT_MIN, Config.JOINT_MAX
            )
            sv = apply_derivative_clamp_batch(
                sq,
                self.theta_v[None, :, :] + rng_v,
                Config.JOINT_MIN, Config.JOINT_MAX,
                dt_node,
            )

            # Sample 0 is the unperturbed nominal
            sq[0] = self.theta_q.copy()
            sv[0] = self.theta_v.copy()

            # ── Interpolate splines (numpy, batched) ──
            tq_all, tv_all = self._spline.interpolate(sq, sv, self._times)

            # ── PD targets -> ctrl, joint order -> actuator order, control rate -> physics rate ──
            ctrl = (tq_all[:, :H - 1] + kd_over_kp * tv_all[:, :H - 1])[:, :, self.act_perm]
            ctrl = np.repeat(ctrl, Config.PLAN_SUBSTEPS, axis=1)
            ctrl = np.ascontiguousarray(ctrl)

            # ── Threaded CPU rollout ──
            self._pool.rollout(
                self._model, self._datas, init_state, control=ctrl,
                state=self._state_buf, sensordata=self._sens_buf,
            )
            costs, terms = self._evaluate(self._state_buf, self._sens_buf, current_qpos)

            weights = self._compute_weights(costs)
            self.theta_q = np.sum(sq * weights[:, None, None], axis=0)
            self.theta_v = np.sum(sv * weights[:, None, None], axis=0)

            iter_best_idx  = int(np.argmin(costs))
            iter_best_cost = float(costs[iter_best_idx])

            self._dbg_sample0_cost = float(costs[0])
            self._dbg_cost_min     = float(costs.min())
            self._dbg_cost_max     = float(costs[costs < 1e7].max()) if (costs < 1e7).any() else 1e8
            self._dbg_cost_median  = float(np.median(costs[costs < 1e7])) if (costs < 1e7).any() else 1e8
            self._dbg_sample0_won  = (iter_best_idx == 0)
            self._dbg_winning_idx  = iter_best_idx

            if iter_best_cost < self.tau_best_cost:
                self.tau_best_cost = iter_best_cost
                self.tau_best_q    = np.ascontiguousarray(tq_all[iter_best_idx])
                self.tau_best_v    = np.ascontiguousarray(tv_all[iter_best_idx])
                improved           = True
                self.tau_best_age  = 0
                self._last_breakdown = {k: float(v[iter_best_idx]) for k, v in terms.items()}
                self._last_breakdown['total'] = iter_best_cost

        if not improved:
            self.tau_best_age += 1

        # ── Update tau0 from weighted mean ──
        tq_nom, tv_nom = self._spline.interpolate(self.theta_q[None], self.theta_v[None], self._times)
        self.tau0_q = np.ascontiguousarray(tq_nom[0])
        self.tau0_v = np.ascontiguousarray(tv_nom[0])

        # Smooth execution (paper Tab. III "nominal only" ablation): act on the
        # weighted average rather than the noisy best sample — except in auto
        # mode when tilted, where the decisive raw sample is needed to recover.
        if Config.EXEC_NOMINAL == 1:
            use_nominal = True
        elif Config.EXEC_NOMINAL == 2:
            qw, qx, qy, _ = current_qpos[3:7]
            up_z = 1.0 - 2.0 * (qx * qx + qy * qy)
            tilt_deg = np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))
            use_nominal = tilt_deg < Config.RECOVERY_TILT_DEG
        else:
            use_nominal = False
        if use_nominal:
            self.tau_best_q = self.tau0_q.copy()
            self.tau_best_v = self.tau0_v.copy()

        return {
            'cost':         self.tau_best_cost,
            'improved':     improved,
            'breakdown':    self._last_breakdown,
            'sample0_won':  self._dbg_sample0_won,
            'winning_idx':  self._dbg_winning_idx,
            'sample0_cost': self._dbg_sample0_cost,
            'cost_min':     self._dbg_cost_min,
            'cost_max':     self._dbg_cost_max,
            'cost_median':  self._dbg_cost_median,
            'tau_best_age': self.tau_best_age,
        }


# ---------------------------------------------------------------------------
# Trot seed generator (optional warm start)
# ---------------------------------------------------------------------------

def generate_trot_seed(base_pose, freq_hz=3.0, hip_amp=0.20, knee_amp=0.40,
                       fwd_lean=0.05, t_offset=0.0):
    seed = np.zeros((Config.H_STEPS, 12))
    for h in range(Config.H_STEPS):
        t      = h * Config.DT + t_offset
        phase  = 2 * np.pi * freq_hz * t
        sA     = np.sin(phase)
        sB     = np.sin(phase + np.pi)
        liftA  = max(0.0, sA)
        liftB  = max(0.0, sB)
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
