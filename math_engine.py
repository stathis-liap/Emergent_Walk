import numpy as np

# Optional: Import matplotlib for visual debugging
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

class CubicHermiteSpline:
    """
    Implements Equations 1-4 from Schramm et al. 2024.
    Upgraded to handle batched operations: (Num_Samples, Num_Nodes, Num_Joints)
    """
    def __init__(self, dt_node):
        self.dt_node = dt_node

    def _basis_functions(self, s):
        """Eq 3 & 4: Evaluates the cubic Hermite basis functions."""
        s2 = s**2
        s3 = s**3
        h00 = 2*s3 - 3*s2 + 1
        h10 = s3 - 2*s2 + s
        h01 = -2*s3 + 3*s2
        h11 = s3 - s2
        return h00, h10, h01, h11

    def _derivative_basis_functions(self, s):
        """Analytical derivatives of the basis functions to compute velocity v(t)."""
        s2 = s**2
        dh00 = 6*s2 - 6*s
        dh10 = 3*s2 - 4*s + 1
        dh01 = -6*s2 + 6*s
        dh11 = 3*s2 - 2*s
        return dh00, dh10, dh01, dh11

    def interpolate(self, theta_q, theta_v, times):
        """
        Batched Interpolation.
        theta_q: shape (K_nodes, num_joints) OR (batch_size, K_nodes, num_joints)
        theta_v: shape (K_nodes, num_joints) OR (batch_size, K_nodes, num_joints)
        """
        # Auto-detect if we are processing a single trajectory or a batch
        is_batched = (theta_q.ndim == 3)
        if not is_batched:
            theta_q = np.expand_dims(theta_q, axis=0)
            theta_v = np.expand_dims(theta_v, axis=0)

        batch_size, K_nodes, num_joints = theta_q.shape
        T_max = (K_nodes - 1) * self.dt_node
        
        dense_q = np.zeros((batch_size, len(times), num_joints))
        dense_v = np.zeros((batch_size, len(times), num_joints))

        for i, t in enumerate(times):
            t_clamped = min(max(t, 0.0), T_max)
            
            k = int(t_clamped // self.dt_node)
            k = min(k, K_nodes - 2) 
            
            t_k = k * self.dt_node
            s = (t_clamped - t_k) / self.dt_node

            h00, h10, h01, h11 = self._basis_functions(s)
            dh00, dh10, dh01, dh11 = self._derivative_basis_functions(s)

            # Extract batched nodes
            q_k, v_k = theta_q[:, k, :], theta_v[:, k, :]
            q_k1, v_k1 = theta_q[:, k+1, :], theta_v[:, k+1, :]

            # Position interpolation
            dense_q[:, i, :] = (h00 * q_k) + (h10 * self.dt_node * v_k) + \
                               (h01 * q_k1) + (h11 * self.dt_node * v_k1)

            # Velocity interpolation
            dense_v[:, i, :] = (dh00 * q_k / self.dt_node) + (dh10 * v_k) + \
                               (dh01 * q_k1 / self.dt_node) + (dh11 * v_k1)

        if not is_batched:
            return dense_q[0], dense_v[0]
        return dense_q, dense_v


def apply_derivative_clamp(theta_q, theta_v_raw, q_min, q_max, dt_node):
    """Eq 5: Mathematically prevents the spline from overshooting joint limits."""
    dist_to_max = q_max - theta_q
    dist_to_min = theta_q - q_min
    v_bound = np.minimum(dist_to_max, dist_to_min) / (dt_node / 2.0)
    return np.clip(theta_v_raw, -v_bound, v_bound)


def calculate_noise_decay(num_iterations, current_iteration, num_nodes, current_node, beta_1=1.0, beta_2=1.5):
    """Eq 8: Diffusion-inspired noise annealing schedule."""
    traj_decay = (num_iterations - current_iteration) / (beta_1 * num_iterations)
    action_decay = (num_nodes - current_node) / (beta_2 * num_nodes)
    return np.exp(-traj_decay - action_decay)


# ==========================================
# PHASE 1 VALIDATION & PLOTTING
# ==========================================
if __name__ == "__main__":
    print("--- Phase 1: Math Engine Validation ---")
    
    # Toggle this to True to see the visual graphs
    SHOW_PLOTS = False
    
    K_nodes = 4
    dt_node = 0.1 
    num_joints = 1
    q_min, q_max = -1.0, 1.0
    
    mock_theta_q = np.array([[0.0], [0.95], [0.0], [0.0]])
    mock_theta_v_raw = np.array([[0.0], [50.0], [0.0], [0.0]])
    
    mock_theta_v_clamped = apply_derivative_clamp(mock_theta_q, mock_theta_v_raw, q_min, q_max, dt_node)
    
    print(f"Sampled Position: {mock_theta_q[1,0]} rad (Limit: {q_max} rad)")
    print(f"AI Sampled Velocity: {mock_theta_v_raw[1,0]} rad/s")
    print(f"Clamped Safe Velocity: {mock_theta_v_clamped[1,0]:.2f} rad/s")
    
    spline = CubicHermiteSpline(dt_node)
    dense_times = np.linspace(0, 0.3, 100) 
    dense_q, dense_v = spline.interpolate(mock_theta_q, mock_theta_v_clamped, dense_times)
    
    print(f"\nMax Position reached in continuous spline: {np.max(dense_q):.3f} rad")
    if np.max(dense_q) > q_max:
        print("[FAIL] The spline overshot the physical joint limits!")
    else:
        print("[PASS] Eq. 5 mathematically guaranteed we stayed within limits!")

    # --- MATPLOTLIB VISUALIZATION ---
    if SHOW_PLOTS and MATPLOTLIB_AVAILABLE:
        print("\nOpening plots... Close the plot window to exit.")
        node_times = np.array([k * dt_node for k in range(K_nodes)])
        
        plt.figure(figsize=(10, 8))
        
        # 1. Position Plot
        plt.subplot(2, 1, 1)
        plt.plot(dense_times, dense_q[:, 0], 'b-', label='Interpolated Position')
        plt.scatter(node_times, mock_theta_q[:, 0], color='red', zorder=5, label='Sampled Nodes')
        plt.axhline(q_max, color='r', linestyle='--', alpha=0.5, label='Joint Max Limit')
        plt.axhline(q_min, color='r', linestyle='--', alpha=0.5, label='Joint Min Limit')
        plt.title('Hermite Spline Position (Eq. 4 & 5)')
        plt.ylabel('Position (rad)')
        plt.legend()
        plt.grid(True)

        # 2. Velocity Plot
        plt.subplot(2, 1, 2)
        plt.plot(dense_times, dense_v[:, 0], 'g-', label='Interpolated Velocity')
        plt.scatter(node_times, mock_theta_v_clamped[:, 0], color='orange', zorder=5, label='Clamped Node Velocity')
        plt.title('Hermite Spline Velocity')
        plt.xlabel('Time (s)')
        plt.ylabel('Velocity (rad/s)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig("spline_validation.png")
        print("\nPlot saved as 'spline_validation.png'. Check your folder!")
    elif SHOW_PLOTS and not MATPLOTLIB_AVAILABLE:
        print("\n[WARNING] matplotlib is not installed. Run 'pip install matplotlib' to view plots.")