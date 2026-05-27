"""
Standalone live demo of the trot sinusoid.

Run this BEFORE the full MPPI pipeline to verify that the sinusoid alone produces
a real walk on the Go2. If this script can't walk, the MPPI seed can't either.

Usage:
    python demo_trot.py                  # default params
    python demo_trot.py --freq 1.5       # adjust frequency
    python demo_trot.py --freq 1.5 --hip 0.15 --knee 0.30   # tune
    python demo_trot.py --headless --duration 5  # no GUI, save video instead

Controls (live viewer):
    Press SPACE to pause/resume in the viewer window.
    Mouse: orbit/pan/zoom.

What this is:
    - Generates a continuous sinusoidal diagonal trot pattern.
    - Applies it to the Go2 via PD control (same gains as the MPPI script).
    - Lets you watch and tune amplitudes/frequencies until you find values
      that actually walk forward stably.
    - Once you find good values, plug them into Config.SEED_* in mppi_core.py.
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer


def make_trot_target(phase, base_pose, hip_amp, knee_amp, fwd_lean):
    """
    Returns a 12-vector of joint targets for the trot.

    Convention:
      Joint indices: [FL_yaw, FL_hip, FL_knee, FR_yaw, FR_hip, FR_knee,
                      RL_yaw, RL_hip, RL_knee, RR_yaw, RR_hip, RR_knee]

    Go2 hip-pitch convention: POSITIVE hip_pitch slopes the leg backward.
    Therefore, to walk FORWARD:
      - Swing phase: hip → NEGATIVE (leg swings forward, foot moves forward in air)
      - Stance phase: hip → POSITIVE (leg pushes backward on ground → body forward)
      - Forward lean: NEGATIVE bias (whole body tilts forward via hip angle reduction)

    Diagonal trot:
      Diagonal A = FL + RR move together (phase 0)
      Diagonal B = FR + RL move together (phase π)

    For each leg:
      - hip -= hip_amp * sin(phase)  (NEGATIVE sign — walks forward)
      - hip -= fwd_lean             (constant forward lean)
      - knee -= knee_amp * max(0, sin(phase))  (bend during swing)
    """
    sA = np.sin(phase)
    sB = np.sin(phase + np.pi)
    liftA = max(0.0, sA)
    liftB = max(0.0, sB)

    target = base_pose.copy()
    # FL — diagonal A
    target[1]  -= hip_amp * sA + fwd_lean   # NOTE: minus
    target[2]  -= knee_amp * liftA
    # FR — diagonal B
    target[4]  -= hip_amp * sB + fwd_lean
    target[5]  -= knee_amp * liftB
    # RL — diagonal B (SAME phase, SAME sign as FR)
    target[7]  -= hip_amp * sB + fwd_lean
    target[8]  -= knee_amp * liftB
    # RR — diagonal A (SAME phase, SAME sign as FL)
    target[10] -= hip_amp * sA + fwd_lean
    target[11] -= knee_amp * liftA

    return target


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scene", default="../unitree_mujoco/unitree_robots/go2/scene.xml",
                   help="Path to Go2 scene XML")
    p.add_argument("--freq", type=float, default=2.0, help="Trot frequency (Hz)")
    p.add_argument("--hip", type=float, default=0.10, help="Hip swing amplitude (rad)")
    p.add_argument("--knee", type=float, default=0.20, help="Knee bend amplitude (rad)")
    p.add_argument("--lean", type=float, default=0.02, help="Forward lean bias (rad)")
    p.add_argument("--kp", type=float, default=100.0, help="PD position gain")
    p.add_argument("--kd", type=float, default=5.0, help="PD velocity gain")
    p.add_argument("--duration", type=float, default=20.0, help="Run duration (s)")
    p.add_argument("--phys-dt", type=float, default=0.005, help="Physics timestep (s)")
    p.add_argument("--ramp", type=float, default=1.0,
                   help="Seconds to ramp the trot amplitude from 0 to full")
    p.add_argument("--headless", action="store_true",
                   help="No live viewer; save MP4 instead")
    p.add_argument("--out", default="demo_trot.mp4",
                   help="Output video path if --headless")
    args = p.parse_args()

    # --- Load model with PD-equipped actuators ---
    model = mujoco.MjModel.from_xml_path(args.scene)
    model.opt.timestep = args.phys_dt
    for i in range(12):
        model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        model.actuator_gainprm[i, 0] = args.kp
        model.actuator_biasprm[i, 1] = -args.kp
        model.actuator_biasprm[i, 2] = -args.kd

    data = mujoco.MjData(model)

    # --- Spawn & settle (same procedure as the main script) ---
    Q_NOMINAL = np.array([0.0, 0.7, -1.4] * 4)
    initial_relaxed = np.array([0.0, 1.2, -2.5] * 4)

    data.qpos[0:3] = [0.0, 0.0, 0.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = initial_relaxed

    mujoco.mj_kinematics(model, data)
    geom_ids = [i for i in range(model.ngeom) if model.geom_bodyid[i] > 0]
    bottoms = data.geom_xpos[geom_ids, 2] - model.geom_size[geom_ids, 0]
    data.qpos[2] = (0.5 - np.min(bottoms)) + 0.001

    print(f"Stand-up routine (2 s)...")
    stand_steps = int(2.0 / args.phys_dt)
    for i in range(stand_steps):
        alpha = (1 - np.cos(np.pi * i / stand_steps)) / 2.0
        data.ctrl[:] = (1 - alpha) * initial_relaxed + alpha * Q_NOMINAL
        mujoco.mj_step(model, data)
    for _ in range(100):
        data.ctrl[:] = Q_NOMINAL
        mujoco.mj_step(model, data)

    # Lock the post-settle pose as the new nominal so the trot oscillates
    # around the actual stable standing pose.
    base_pose = data.qpos[7:].copy()
    p_des_z = data.qpos[2]
    start_x = data.qpos[0]
    start_y = data.qpos[1]
    print(f"  Settled at z={p_des_z:.3f}m, base_pose[:3]={np.round(base_pose[:3], 3)}")
    print(f"\nTrot parameters:")
    print(f"  freq={args.freq} Hz   hip_amp={args.hip} rad ({np.degrees(args.hip):.1f}°)")
    print(f"  knee_amp={args.knee} rad ({np.degrees(args.knee):.1f}°)")
    print(f"  fwd_lean={args.lean} rad ({np.degrees(args.lean):.1f}°)")
    print(f"  ramp-in over {args.ramp}s, then run for {args.duration}s")
    print(f"  PD: Kp={args.kp}, Kd={args.kd}")
    print()

    # --- Stats during trot ---
    stats = {
        'max_tilt_deg': 0.0,
        'min_z': p_des_z,
        'max_z': p_des_z,
        'final_x': 0.0,
        'final_y': 0.0,
        'final_yaw_deg': 0.0,
    }
    trot_started_at_step = None

    def control_at_time(t_trot):
        """Compute control target at time t (seconds since trot started)."""
        ramp = min(1.0, t_trot / args.ramp) if args.ramp > 0 else 1.0
        phase = 2 * np.pi * args.freq * t_trot
        return make_trot_target(
            phase, base_pose,
            hip_amp=args.hip * ramp,
            knee_amp=args.knee * ramp,
            fwd_lean=args.lean * ramp,
        )

    def update_stats():
        up = np.zeros(3)
        mujoco.mju_rotVecQuat(up, np.array([0.0, 0.0, 1.0]), data.qpos[3:7])
        tilt = np.degrees(np.arccos(max(-1.0, min(1.0, up[2]))))
        stats['max_tilt_deg'] = max(stats['max_tilt_deg'], tilt)
        stats['min_z'] = min(stats['min_z'], data.qpos[2])
        stats['max_z'] = max(stats['max_z'], data.qpos[2])

    if args.headless:
        # --- Headless: write video ---
        import cv2
        width, height = 640, 480
        renderer = mujoco.Renderer(model, height, width)
        # Camera follows robot
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        camera.distance = 2.0
        camera.elevation = -20
        camera.azimuth = 90

        fps_render = 50
        phys_per_frame = int((1.0 / fps_render) / args.phys_dt)
        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'mp4v'),
                                 fps_render, (width, height))
        print(f"Recording to {args.out}...")

        t_start_trot = None
        total_steps = int(args.duration / args.phys_dt) + stand_steps
        for i in range(stand_steps, total_steps):
            t_trot = (i - stand_steps) * args.phys_dt
            data.ctrl[:] = control_at_time(t_trot)
            mujoco.mj_step(model, data)

            if i % phys_per_frame == 0:
                update_stats()
                camera.lookat[:] = data.qpos[:3]
                renderer.update_scene(data, camera=camera)
                frame = renderer.render()
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()
        renderer.close()
    else:
        # --- Interactive viewer ---
        print("Opening viewer. Press SPACE to pause/resume, ESC to quit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            t_start_wall = time.time()
            trot_start_step = None
            step_count = 0

            while viewer.is_running() and time.time() - t_start_wall < args.duration + 3.0:
                step_start = time.time()
                t_trot = max(0.0, time.time() - t_start_wall)
                data.ctrl[:] = control_at_time(t_trot)
                mujoco.mj_step(model, data)
                step_count += 1
                if step_count % 4 == 0:   # ~50 Hz stat update
                    update_stats()
                viewer.sync()

                # Pace to real-time
                elapsed = time.time() - step_start
                if elapsed < args.phys_dt:
                    time.sleep(args.phys_dt - elapsed)

    # Final stats
    stats['final_x'] = data.qpos[0] - start_x
    stats['final_y'] = data.qpos[1] - start_y
    # yaw from quaternion
    qw, qx, qy, qz = data.qpos[3], data.qpos[4], data.qpos[5], data.qpos[6]
    yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    stats['final_yaw_deg'] = np.degrees(yaw)
    avg_vx = stats['final_x'] / args.duration if args.duration > 0 else 0.0

    print(f"\n--- Results ---")
    print(f"  Forward distance:   {stats['final_x']:+.3f} m  (avg vx = {avg_vx:+.3f} m/s)")
    print(f"  Lateral drift:      {stats['final_y']:+.3f} m")
    print(f"  Final yaw:          {stats['final_yaw_deg']:+.1f}°")
    print(f"  Max body tilt:      {stats['max_tilt_deg']:.1f}°")
    print(f"  Body z range:       {stats['min_z']:.3f}–{stats['max_z']:.3f} m  (settled={p_des_z:.3f})")
    print()
    if stats['max_tilt_deg'] < 15 and abs(stats['final_y']) < 0.2 and abs(stats['final_yaw_deg']) < 15:
        print("  ✓ Trot looks stable. Good seed candidate.")
    else:
        print("  ✗ Trot is unstable. Try smaller amplitudes or different frequency.")
        if stats['max_tilt_deg'] > 30:
            print("    - High tilt: reduce hip_amp, increase freq.")
        if abs(stats['final_yaw_deg']) > 15:
            print("    - Yaw drift: pattern is asymmetric or step impacts are uneven.")
        if abs(stats['final_y']) > 0.3:
            print("    - Lateral drift: same as yaw — pattern is steering sideways.")


if __name__ == "__main__":
    main()