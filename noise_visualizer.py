import numpy as np
import matplotlib.pyplot as plt
from math_engine import CubicHermiteSpline, calculate_noise_decay, apply_derivative_clamp

def visualize_noise_annealing():
    print("Generating Diffusion-Inspired Noise Visualization with Nodes...")
    
    # Parameters matching the paper's general structure
    K_nodes = 5
    dt_node = 0.1
    horizon = (K_nodes - 1) * dt_node
    num_samples = 30 # Number of noisy trajectories to plot
    num_iterations = 3
    
    # Base noise scales
    SCALE_Q = 0.5 # Exaggerated for visualization
    SCALE_V = 2.0
    
    # Nominal trajectory (a simple curve)
    times_nodes = np.linspace(0, horizon, K_nodes)
    nominal_q = np.array([[0.6], [0.55], [0.75], [0.95], [1.0]])
    nominal_v = np.array([[-0.8], [0.0], [2.0], [0.5], [-0.5]])
    
    # Setup Spline engine
    spline = CubicHermiteSpline(dt_node)
    dense_times = np.linspace(0, horizon, 100)
    
    # Create the plot
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, sharey='row')
    fig.suptitle('Reference-Free MPPI: Diffusion-Inspired Noise Annealing (with Nodes)', fontsize=16)
    
    for i in range(num_iterations):
        # Calculate iteration-specific noise
        current_iter = num_iterations - i # e.g., 3, 2, 1 (decreasing noise)
        
        # Prepare batched noisy nodes
        sq = np.zeros((num_samples, K_nodes, 1))
        sv = np.zeros((num_samples, K_nodes, 1))
        
        for n in range(num_samples):
            for k in range(K_nodes):
                if k == 0:
                    # Node 0 is anchored (Current measured state)
                    sq[n, k] = nominal_q[k]
                    sv[n, k] = nominal_v[k]
                else:
                    # Calculate decay for this specific node and iteration
                    decay = calculate_noise_decay(num_iterations, current_iter, K_nodes, k, beta_1=1.0, beta_2=1.0)
                    
                    # Apply noise to nominal trajectory
                    sq[n, k] = nominal_q[k] + np.random.normal(0, SCALE_Q * decay, 1)
                    v_raw = nominal_v[k] + np.random.normal(0, SCALE_V * decay, 1)
                    
                    # Optional: Clamp velocities to keep visual reasonable
                    sv[n, k] = np.clip(v_raw, -5.0, 5.0)
        
        # Generate dense splines for all samples
        tq_all, tv_all = spline.interpolate(sq, sv, dense_times)
        
        # --- Plot Position ---
        ax_q = axes[0, i]
        ax_q.set_title(f'Iteration {i+1}')
        for n in range(num_samples):
            color = plt.cm.winter(i / (num_iterations - 1)) 
            # Plot the spline curve
            ax_q.plot(dense_times, tq_all[n, :, 0], color=color, alpha=0.2)
            # Plot the noisy nodes (NEW)
            ax_q.scatter(times_nodes, sq[n, :, 0], color=color, alpha=0.5, s=15, zorder=4)
        
        # Plot nominal trajectory over the top
        nominal_dense_q, _ = spline.interpolate(nominal_q, nominal_v, dense_times)
        ax_q.plot(dense_times, nominal_dense_q[:, 0], 'k-', linewidth=2, zorder=5)
        ax_q.scatter(times_nodes, nominal_q[:, 0], color='k', s=50, zorder=6)
        ax_q.grid(True)
        
        if i == 0:
            ax_q.set_ylabel('Position (rad)', fontsize=12)
            
        # --- Plot Velocity ---
        ax_v = axes[1, i]
        for n in range(num_samples):
            color = plt.cm.winter(i / (num_iterations - 1))
            # Plot the spline curve
            ax_v.plot(dense_times, tv_all[n, :, 0], color=color, alpha=0.2)
            # Plot the noisy nodes (NEW)
            ax_v.scatter(times_nodes, sv[n, :, 0], color=color, alpha=0.5, s=15, zorder=4)
            
        _, nominal_dense_v = spline.interpolate(nominal_q, nominal_v, dense_times)
        ax_v.plot(dense_times, nominal_dense_v[:, 0], 'k-', linewidth=2, zorder=5)
        ax_v.scatter(times_nodes, nominal_v[:, 0], color='k', s=50, zorder=6)
        ax_v.grid(True)
        ax_v.set_xlabel('Time (s)', fontsize=12)
        
        if i == 0:
            ax_v.set_ylabel('Velocity (rad/s)', fontsize=12)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("noise_annealing_visualization.png", dpi=300)
    print("Plot saved as 'noise_annealing_visualization.png'")
    
if __name__ == "__main__":
    visualize_noise_annealing()