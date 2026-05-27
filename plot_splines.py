"""
Plot recorded MPPI spline data.

Usage: python plot_splines.py spline_log.npz

Produces 4 figures:
  1. tau_best continuity — frame 0 position & velocity across consecutive
     control steps for selected joints. Discontinuities here = visible jerks.
  2. tau_best vs tau_0 — divergence between the executed plan and the
     sampling center, for one representative step.
  3. Sample bundle — multiple samples + their winning one + previous
     tau_best, all on the same axes, for one step. Shows what alternatives
     the optimizer considered.
  4. Position vs velocity self-consistency — does τ^v match d/dt of τ^q?
     If the spline's velocity output doesn't agree with finite difference
     of its position output, the parameterization is inconsistent.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt


JOINT_NAMES = []
for leg in ["FL", "FR", "RL", "RR"]:
    for j in ["hip_ab", "thigh", "calf"]:
        JOINT_NAMES.append(f"{leg}_{j}")


def load(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def plot_tau_best_continuity(data, joint_indices=(1, 2, 7, 8), out="continuity.png"):
    """Frame 0 of tau_best across consecutive control steps.

    If the optimizer's executed command jumps between steps, you'll see
    sawtooth-like jumps in these traces.
    """
    steps = data['step']
    tau_best_q = data['tau_best_q']  # (n_records, H_STEPS, 12)
    tau_best_v = data['tau_best_v']

    fig, axes = plt.subplots(2, len(joint_indices), figsize=(4*len(joint_indices), 6), sharex=True)

    for col, j in enumerate(joint_indices):
        # Position at frame 0
        axes[0, col].plot(steps, tau_best_q[:, 0, j], 'b-', marker='o', markersize=3)
        axes[0, col].set_title(f"{JOINT_NAMES[j]} τ_best^q[0]")
        axes[0, col].set_ylabel("rad")
        axes[0, col].grid(True, alpha=0.3)

        # Velocity at frame 0
        axes[1, col].plot(steps, tau_best_v[:, 0, j], 'r-', marker='o', markersize=3)
        axes[1, col].set_title(f"{JOINT_NAMES[j]} τ_best^v[0]")
        axes[1, col].set_ylabel("rad/s")
        axes[1, col].set_xlabel("control step")
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle("τ_best frame 0 across control steps (jumps here = visible jerks)")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close()


def plot_tau_best_vs_tau0(data, step_idx=None, joint_indices=(1, 2, 7, 8), out="best_vs_nominal.png"):
    """Compare tau_best (executed) and tau_0 (sampling center) for one step."""
    if step_idx is None:
        step_idx = len(data['step']) // 2  # middle of recording

    tq_best = data['tau_best_q'][step_idx]  # (H_STEPS, 12)
    tv_best = data['tau_best_v'][step_idx]
    tq_0 = data['tau0_q'][step_idx]
    tv_0 = data['tau0_v'][step_idx]
    real_step = int(data['step'][step_idx])

    H = tq_best.shape[0]
    t = np.arange(H) * 0.02  # control DT = 0.02s

    fig, axes = plt.subplots(2, len(joint_indices), figsize=(4*len(joint_indices), 6), sharex=True)
    for col, j in enumerate(joint_indices):
        axes[0, col].plot(t, tq_best[:, j], 'b-', label='τ_best^q', linewidth=2)
        axes[0, col].plot(t, tq_0[:, j], 'g--', label='τ_0^q', linewidth=1)
        axes[0, col].set_title(f"{JOINT_NAMES[j]} position")
        axes[0, col].set_ylabel("rad")
        axes[0, col].grid(True, alpha=0.3)
        axes[0, col].legend(fontsize=8)

        axes[1, col].plot(t, tv_best[:, j], 'b-', label='τ_best^v', linewidth=2)
        axes[1, col].plot(t, tv_0[:, j], 'g--', label='τ_0^v', linewidth=1)
        axes[1, col].set_title(f"{JOINT_NAMES[j]} velocity")
        axes[1, col].set_ylabel("rad/s")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend(fontsize=8)

    fig.suptitle(f"τ_best vs τ_0 at control step {real_step}")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close()


def plot_sample_bundle(data, step_idx=None, joint_indices=(1, 2, 7, 8), out="samples.png"):
    """Show the sample bundle and which one won, for one step."""
    if 'sample_q' not in data:
        print("No sample data in log (last_iter_samples_q/v/costs not passed to recorder)")
        return
    if step_idx is None:
        step_idx = len(data['step']) // 2

    samples_q = data['sample_q'][step_idx]   # (n_samples_kept, H_STEPS, 12)
    samples_v = data['sample_v'][step_idx]
    costs = data['sample_cost'][step_idx]
    idx = data['sample_idx'][step_idx]
    tq_best = data['tau_best_q'][step_idx]
    real_step = int(data['step'][step_idx])

    H = tq_best.shape[0]
    t = np.arange(H) * 0.02

    fig, axes = plt.subplots(2, len(joint_indices), figsize=(4*len(joint_indices), 6), sharex=True)
    n = samples_q.shape[0]
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, n))

    for col, j in enumerate(joint_indices):
        for i in range(n):
            axes[0, col].plot(t, samples_q[i, :, j], color=colors[i], alpha=0.7,
                              label=f"s{idx[i]} c={costs[i]:.0f}", linewidth=1)
        axes[0, col].plot(t, tq_best[:, j], 'k-', label='τ_best^q (chosen)', linewidth=2)
        axes[0, col].set_title(f"{JOINT_NAMES[j]} position samples")
        axes[0, col].set_ylabel("rad")
        axes[0, col].grid(True, alpha=0.3)
        if col == 0:
            axes[0, col].legend(fontsize=7, loc='best')

        for i in range(n):
            axes[1, col].plot(t, samples_v[i, :, j], color=colors[i], alpha=0.7, linewidth=1)
        axes[1, col].set_title(f"{JOINT_NAMES[j]} velocity samples")
        axes[1, col].set_ylabel("rad/s")
        axes[1, col].set_xlabel("time (s)")
        axes[1, col].grid(True, alpha=0.3)

    fig.suptitle(f"Sample bundle at control step {real_step} "
                 f"(best cost={costs.min():.0f}, worst={costs.max():.0f})")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close()


def plot_spline_consistency(data, step_idx=None, joint_indices=(1, 2, 7, 8), out="consistency.png"):
    """Check if τ_best^v ≈ d/dt τ_best^q.

    If the spline parameterization is consistent, the velocity output of the
    spline should match the time-derivative of its position output. If they
    disagree significantly, the optimizer is being asked to track an
    inconsistent (q, v) pair, which can cause jerks in execution.
    """
    if step_idx is None:
        step_idx = len(data['step']) // 2

    tq = data['tau_best_q'][step_idx]
    tv = data['tau_best_v'][step_idx]
    real_step = int(data['step'][step_idx])

    H = tq.shape[0]
    dt = 0.02
    t = np.arange(H) * dt
    # Finite difference of position (central differences)
    tq_fd = np.zeros_like(tq)
    tq_fd[1:-1] = (tq[2:] - tq[:-2]) / (2 * dt)
    tq_fd[0] = (tq[1] - tq[0]) / dt
    tq_fd[-1] = (tq[-1] - tq[-2]) / dt

    fig, axes = plt.subplots(1, len(joint_indices), figsize=(4*len(joint_indices), 4), sharex=True)
    if len(joint_indices) == 1:
        axes = [axes]
    for col, j in enumerate(joint_indices):
        axes[col].plot(t, tv[:, j], 'r-', label="spline τ^v output", linewidth=2)
        axes[col].plot(t, tq_fd[:, j], 'b--', label="finite-diff(τ^q)", linewidth=1.5)
        axes[col].set_title(f"{JOINT_NAMES[j]}")
        axes[col].set_xlabel("time (s)")
        axes[col].set_ylabel("rad/s")
        axes[col].grid(True, alpha=0.3)
        axes[col].legend(fontsize=8)

    fig.suptitle(f"Spline τ^v vs d/dt τ^q at step {real_step} "
                 f"(divergence = inconsistent parameterization)")
    plt.tight_layout()
    plt.savefig(out, dpi=100)
    print(f"Saved {out}")
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plot_splines.py spline_log.npz [step_idx]")
        sys.exit(1)

    path = sys.argv[1]
    step_idx = int(sys.argv[2]) if len(sys.argv) > 2 else None

    data = load(path)
    print(f"Loaded {data['_n_records']} records from {path}")
    print(f"  (every {data['_record_every']} control steps)")

    # Plot 4 representative joints: FL hip, FL thigh, RL hip, RL thigh
    # (diagonal pair — should be in phase during trot)
    joints = (1, 2, 7, 8)

    plot_tau_best_continuity(data, joints, out="continuity.png")
    plot_tau_best_vs_tau0(data, step_idx, joints, out="best_vs_nominal.png")
    plot_sample_bundle(data, step_idx, joints, out="samples.png")
    plot_spline_consistency(data, step_idx, joints, out="consistency.png")
    print("\nDone. View continuity.png best_vs_nominal.png samples.png consistency.png")