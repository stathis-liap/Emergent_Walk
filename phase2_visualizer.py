import numpy as np
import mujoco
import matplotlib.pyplot as plt

# Import the simulator and config we just built
from ghost_sim import GhostSimulator, config

def evaluate_and_track(sim, start_qpos, start_qvel, traj_q, name):
    """
    Runs the ghost simulator but records the internal state at every step
    so we can plot exactly what the AI is 'imagining'.
    """
    mujoco.mj_resetData(sim.model, sim.data)
    sim.data.qpos[:] = start_qpos
    sim.data.qvel[:] = start_qvel
    
    p_x0 = sim.data.qpos[0]
    current_cost = 0.0
    
    # Tracking arrays
    times = [0.0]
    z_history = [sim.data.qpos[2]]
    tilt_history = [0.0]
    cost_history = [0.0]
    
    fatal = False
    
    for t in range(config.H_STEPS - 1):
        if not fatal:
            sim.data.ctrl[:] = traj_q[t]
            mujoco.mj_step(sim.model, sim.data, config.PHYS_SUBSTEPS)
            
            # Extract states
            z_height = sim.data.qpos[2]
            up_robot = np.zeros(3)
            mujoco.mju_rotVecQuat(up_robot, np.array([0.0, 0.0, 1.0]), sim.data.qpos[3:7])
            tilt = 1.0 - up_robot[2]
            
            # Death Penalty Checks
            if not np.isfinite(sim.data.qpos).all() or z_height < 0.18 or up_robot[2] < 0.5:
                current_cost += 1e6
                fatal = True
            else:
                # Running Costs
                current_cost += config.W_H * (z_height - config.P_DES_Z)**2
                current_cost += config.W_ORIENT * tilt
                current_cost += config.W_C_VEL * (sim.data.qvel[1]**2 + sim.data.qvel[2]**2)
                current_cost += config.W_Q * np.sum((sim.data.qpos[7:] - config.Q_NOMINAL)**2)
        
        # Record (If fatal, we just carry the last state forward for the plot)
        times.append((t + 1) * config.DT)
        z_history.append(sim.data.qpos[2] if not fatal else z_history[-1])
        tilt_history.append(tilt if not fatal else tilt_history[-1])
        cost_history.append(current_cost)

    # Terminal Cost
    if not fatal:
        target_x = p_x0 + (config.V_DES * config.HORIZON_SEC)
        current_cost += config.W_TERM * (sim.data.qpos[0] - target_x)**2
        cost_history[-1] = current_cost

    return times, z_history, tilt_history, cost_history

if __name__ == "__main__":
    print("Generating Phase 2 Visualizer Plots...")
    
    sim = GhostSimulator()
    
    # Safely anchor the robot
    start_qpos = np.zeros(19)
    start_qpos[0:3] = [0.0, 0.0, 0.5] 
    start_qpos[3:7] = [1.0, 0.0, 0.0, 0.0] 
    start_qpos[7:] = config.Q_NOMINAL 
    start_qvel = np.zeros(18)
    
    tmp_data = mujoco.MjData(sim.model)
    tmp_data.qpos[:] = start_qpos
    mujoco.mj_kinematics(sim.model, tmp_data)
    geom_ids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] > 0]
    bottoms = tmp_data.geom_xpos[geom_ids, 2] - sim.model.geom_size[geom_ids, 0]
    start_qpos[2] = (0.5 - np.min(bottoms)) + 0.001 
    
    # Generate Trajectories
    perfect_traj = np.tile(config.Q_NOMINAL, (config.H_STEPS, 1))
    terrible_traj = np.tile([0.0, 1.5, -2.7] * 4, (config.H_STEPS, 1)) # The true fold
    
    # Track States
    t_perf, z_perf, tilt_perf, cost_perf = evaluate_and_track(sim, start_qpos, start_qvel, perfect_traj, "Stand")
    t_terr, z_terr, tilt_terr, cost_terr = evaluate_and_track(sim, start_qpos, start_qvel, terrible_traj, "Collapse")
    
    # --- PLOTTING ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle('Phase 2: Ghost Simulator Trajectory Evaluation', fontsize=16)
    
    # 1. Z-Height Plot
    axes[0].plot(t_perf, z_perf, 'g-', linewidth=2, label='Perfect Stand')
    axes[0].plot(t_terr, z_terr, 'r-', linewidth=2, label='Collapse')
    axes[0].axhline(config.P_DES_Z, color='g', linestyle='--', alpha=0.5, label='Target Z (0.28m)')
    axes[0].axhline(0.18, color='r', linestyle='--', alpha=0.5, label='Death Penalty Threshold (0.18m)')
    axes[0].set_ylabel('Z-Height (m)')
    axes[0].set_title('Robot Torso Height over 0.3s Horizon')
    axes[0].legend()
    axes[0].grid(True)
    
    # 2. Tilt Plot
    axes[1].plot(t_perf, tilt_perf, 'g-', linewidth=2)
    axes[1].plot(t_terr, tilt_terr, 'r-', linewidth=2)
    axes[1].set_ylabel('Tilt Error (0=Flat)')
    axes[1].set_title('Robot Orientation Error (1.0 - Up_Z)')
    axes[1].grid(True)
    
    # 3. Accumulated Cost Plot
    axes[2].plot(t_perf, cost_perf, 'g-', linewidth=2)
    axes[2].plot(t_terr, cost_terr, 'r-', linewidth=2)
    axes[2].set_yscale('log') # Use Log scale because the death penalty is 1,000,000
    axes[2].set_ylabel('Accumulated Cost (Log Scale)')
    axes[2].set_xlabel('Time into Future Horizon (s)')
    axes[2].set_title('Trajectory Scoring (Lower is Better)')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig("phase2_validation.png", dpi=300)
    print("Plot saved as 'phase2_validation.png'. Open it to see the AI's imagination!")