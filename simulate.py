import os
import sys
import time

import numpy as np
import mujoco
import cv2

from config import Config, FOOT_NAMES
from scene_builder import make_scene_with_sensors, prepare_model, actuator_joint_perm
from mppi_controller import MPPIController


if __name__ == "__main__":
    print("=== Emergent Walk — MPPI with threaded CPU rollout ===\n")

    RECORD_VIDEO = os.environ.get("EW_RECORD", "1") != "0"
    SIM_SECONDS  = float(os.environ.get("EW_SIM_SECONDS", "30.0"))

    # ── Live (CPU) simulation: same XML + actuator setup as the planner ──
    scene_path = make_scene_with_sensors()
    live_model = prepare_model(scene_path, Config.LIVE_DT)
    live_data  = mujoco.MjData(live_model)
    act_perm   = actuator_joint_perm(live_model)
    foot_geom_ids = [mujoco.mj_name2id(live_model, mujoco.mjtObj.mjOBJ_GEOM, n)
                     for n in FOOT_NAMES]

    # ── Video ──
    if RECORD_VIDEO:
        width, height = 640, 480
        fps          = int(1.0 / Config.DT)
        renderer     = mujoco.Renderer(live_model, height, width)
        video_writer = cv2.VideoWriter(
            "emergent_walk.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height)
        )
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance  = 2.0
        camera.elevation = -20
        camera.azimuth   = 90

        def capture_frame():
            camera.lookat[:] = live_data.qpos[:3]
            renderer.update_scene(live_data, camera=camera)
            video_writer.write(cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR))

    # ── Drop robot to standing height ──
    live_data.qpos[0:3] = [0.0, 0.0, 0.5]
    live_data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    initial_relaxed = np.array([0.0, 1.2, -2.5] * 4)
    live_data.qpos[7:] = initial_relaxed
    mujoco.mj_kinematics(live_model, live_data)
    geom_ids = [i for i in range(live_model.ngeom) if live_model.geom_bodyid[i] > 0]
    bottoms  = live_data.geom_xpos[geom_ids, 2] - live_model.geom_size[geom_ids, 0]
    live_data.qpos[2] = (0.5 - np.min(bottoms)) + 0.001

    print("Stand-up routine...")
    stand_steps = int(2.0 / Config.LIVE_DT)
    for i in range(stand_steps):
        alpha = (1 - np.cos(np.pi * i / stand_steps)) / 2.0
        live_data.ctrl[:] = ((1 - alpha) * initial_relaxed + alpha * Config.Q_NOMINAL)[act_perm]
        mujoco.mj_step(live_model, live_data)
    for _ in range(100):
        live_data.ctrl[:] = Config.Q_NOMINAL[act_perm]
        mujoco.mj_step(live_model, live_data)

    Config.P_DES_Z   = float(live_data.qpos[2])
    Config.Q_NOMINAL = live_data.qpos[7:].copy()
    print(f"  Post-settle: z={Config.P_DES_Z:.3f} m  "
          f"Q[:3]={np.round(Config.Q_NOMINAL[:3], 3)}\n")

    print("Initializing MPPI from nominal standing (no warm-start seed)...\n")
    mppi = MPPIController(initial_tau_q=None)

    total_steps = int(SIM_SECONDS / Config.DT)
    print(f"Running {SIM_SECONDS}s = {total_steps} control steps\n")
    kd_over_kp  = Config.KD / Config.KP

    last_compute_dt = 0.0
    compute_dts     = []
    metrics         = {'vx': [], 'tilt': [], 'z': [], 'jvel': [], 'power': []}
    interrupted     = False
    try:
        for step in range(total_steps):
            t0   = time.perf_counter()
            info = mppi.control_step(
                live_data.qpos.copy(),
                live_data.qvel.copy(),
                compute_dt_seconds=last_compute_dt,
            )
            last_compute_dt = time.perf_counter() - t0
            compute_dts.append(last_compute_dt)

            live_data.ctrl[:] = (mppi.tau_best_q[0] + kd_over_kp * mppi.tau_best_v[0])[act_perm]
            for _ in range(Config.LIVE_SUBSTEPS):
                mujoco.mj_step(live_model, live_data)
            if RECORD_VIDEO:
                capture_frame()

            up = np.zeros(3)
            mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), live_data.qpos[3:7])
            tilt_deg = np.degrees(np.arccos(max(-1.0, min(1.0, up[2]))))
            body_x = np.zeros(3)
            mujoco.mju_rotVecQuat(body_x, np.array([1.0, 0.0, 0.0]), live_data.qpos[3:7])
            pitch_sign = "↓" if body_x[2] < -0.01 else ("↑" if body_x[2] > 0.01 else "·")
            metrics['vx'].append(live_data.qvel[0])
            metrics['tilt'].append(tilt_deg)
            metrics['z'].append(live_data.qpos[2])
            metrics['jvel'].append(np.abs(live_data.qvel[6:]).mean())
            metrics['power'].append(np.abs(live_data.actuator_force * live_data.qvel[6:][act_perm]).sum())
            flag = ('imp' if info['improved'] else f'kpt({info["tau_best_age"]:3d})')
            sys.stdout.write(
                f"\r[{step:04d}/{total_steps}] "
                f"x={live_data.qpos[0]:+.2f}m  "
                f"z={live_data.qpos[2]:.2f}m  "
                f"vx={live_data.qvel[0]:+.2f}  "
                f"tilt={tilt_deg:.0f}°{pitch_sign}  "
                f"cost={info['cost']:7.1f}  "
                f"{flag}  "
                f"{last_compute_dt*1000:.0f}ms"
            )
            sys.stdout.flush()

            if step % 25 == 0 and step > 0:
                b = info['breakdown']
                fl_z, fr_z, rl_z, rr_z = (live_data.geom_xpos[g, 2] for g in foot_geom_ids)
                d_A  = 0.5 * (fl_z + rr_z)
                d_B  = 0.5 * (fr_z + rl_z)
                FOOT_DOWN = 0.03
                n_down = sum(z < FOOT_DOWN for z in (fl_z, fr_z, rl_z, rr_z))
                def fmark(z):
                    if z < FOOT_DOWN: return "▼"
                    if z < 0.08:      return "·"
                    return "▲"
                print(f"\n  ↳ cost breakdown: "
                      f"height={b['height']:6.1f}  "
                      f"orient={b['orient']:6.1f}  "
                      f"q={b['q_reg']:5.2f}  "
                      f"vc={b['v_contact']:6.1f}  "
                      f"cf={b['f_contact']:6.1f}  "
                      f"term={b['terminal']:6.1f}")
                print(f"  ↳ feet ({n_down}/4 down): "
                      f"FL={fl_z:.3f}{fmark(fl_z)} "
                      f"FR={fr_z:.3f}{fmark(fr_z)} "
                      f"RL={rl_z:.3f}{fmark(rl_z)} "
                      f"RR={rr_z:.3f}{fmark(rr_z)}  "
                      f"| Δdiag={d_A - d_B:+.3f}m")

            if step % 10 == 0:
                won  = info['sample0_won']
                widx = info['winning_idx']
                s0   = info['sample0_cost']
                cmin = info['cost_min']
                cmed = info['cost_median']
                cmax = info['cost_max']
                spread = (cmed - cmin) / max(cmed, 1.0)
                winner_str = "nominal(s0)" if won else f"noise#{widx:2d}"
                print(f"\n  ⌖ MPPI: {winner_str:12s}  "
                      f"s0={s0:.0f}  min={cmin:.0f}  med={cmed:.0f}  max={cmax:.0f}  "
                      f"spread={spread:.3f}  τ_best_age={info['tau_best_age']:3d}")

    except KeyboardInterrupt:
        interrupted = True
        print("\n\n[Ctrl+C — flushing and exiting]")
    except Exception as e:
        interrupted = True
        print(f"\n\n[Exception: {type(e).__name__}: {e}]")
        import traceback; traceback.print_exc()
    finally:
        if compute_dts:
            arr = np.array(compute_dts) * 1000.0
            print(f"\n\nCompute time per control step: mean={arr.mean():.1f}ms  "
                  f"median={np.median(arr):.1f}ms  p95={np.percentile(arr, 95):.1f}ms  "
                  f"max={arr.max():.1f}ms")
        if metrics['vx']:
            vx, tl, zz, jv, pw = (np.array(metrics[k]) for k in ('vx', 'tilt', 'z', 'jvel', 'power'))
            fell = bool((zz < 0.12).any() or (tl > 70).any())
            print(f"METRICS dist={live_data.qpos[0]:.2f}m  vx_mean={vx.mean():.2f}  "
                  f"vx_err={np.abs(vx - Config.V_DES).mean():.2f}  "
                  f"tilt_mean={tl.mean():.1f}  tilt_p95={np.percentile(tl, 95):.1f}  "
                  f"tilt_max={tl.max():.1f}  z_min={zz.min():.2f}  "
                  f"jvel={jv.mean():.2f}  power={pw.mean():.0f}W  fell={fell}")
        if RECORD_VIDEO:
            print("Saving video..." if not interrupted else "Saving partial video...")
            try:
                video_writer.release()
                renderer.close()
            except Exception as e:
                print(f"  (video cleanup: {e})")
        mppi.shutdown()
        if RECORD_VIDEO and not interrupted:
            print("Saved emergent_walk.mp4")
