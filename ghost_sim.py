import numpy as np
import mujoco

# ==========================================
# MOCK CONFIG (For Isolated Phase 2 Testing)
# ==========================================
class MockConfig:
    ROBOT_SCENE = "../unitree_mujoco/unitree_robots/go2/scene.xml" # Update to your path
    HORIZON_SEC = 0.3
    DT = 0.05
    H_STEPS = int(HORIZON_SEC / DT) + 1 
    
    PHYS_DT = 0.005
    PHYS_SUBSTEPS = int(DT / PHYS_DT) 
    
    # PD Gains 
    KP = 50.0
    KD = 1.5
    
    # Task Weights (Schramm Table II)
    W_H = 2500.0        
    W_ORIENT = 200.0    
    W_Q = 10.0          
    W_C_VEL = 10.0      
    W_TERM = 50.0       
    
    P_DES_Z = 0.28
    V_DES = 0.10 
    Q_NOMINAL = np.array([0.0, 0.8, -1.5] * 4)

config = MockConfig()

# ==========================================
# PHASE 2: THE GHOST SIMULATOR
# ==========================================
class GhostSimulator:
    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
        self.model.opt.timestep = config.PHYS_DT
        self.model.opt.o_solref = np.array([0.02, 1.0]) 
        
        for i in range(12):
            # [THE BUG FIX: Tell MuJoCo to actually use the PD parameters!]
            self.model.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
            self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
            
            self.model.actuator_gainprm[i, 0] = config.KP
            self.model.actuator_biasprm[i, 1] = -config.KP
            self.model.actuator_biasprm[i, 2] = -config.KD
            
        self.data = mujoco.MjData(self.model)

    def evaluate_trajectory(self, start_qpos, start_qvel, traj_q, debug_name=""):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = start_qpos
        self.data.qvel[:] = start_qvel
        
        cost = 0.0
        p_x0 = self.data.qpos[0] 
        
        for t in range(config.H_STEPS - 1):
            self.data.ctrl[:] = traj_q[t]
            mujoco.mj_step(self.model, self.data, config.PHYS_SUBSTEPS)
            
            # --- THE DEATH PENALTY (With Debugging) ---
            if not np.isfinite(self.data.qpos).all():
                if debug_name: print(f"  [{debug_name}] FATAL: Physics exploded (NaN)!")
                return 1e6 
                
            if self.data.qpos[2] < 0.18: # Increased from 0.15
                if debug_name: print(f"  [{debug_name}] FATAL: Belly-flop! Z-Height dropped to {self.data.qpos[2]:.3f}m")
                return 1e6 
                
            up_robot = np.zeros(3)
            mujoco.mju_rotVecQuat(up_robot, np.array([0.0, 0.0, 1.0]), self.data.qpos[3:7])
            if up_robot[2] < 0.5: # Changed from 0.0. (0.5 is a 60-degree tilt)
                if debug_name: print(f"  [{debug_name}] FATAL: Robot fell over! Tilt exceeded safe limits.")
                return 1e6
                
            # --- RUNNING COST ---
            cost += config.W_H * (self.data.qpos[2] - config.P_DES_Z)**2
            cost += config.W_ORIENT * (1.0 - up_robot[2])
            cost += config.W_C_VEL * (self.data.qvel[1]**2 + self.data.qvel[2]**2)
            cost += config.W_Q * np.sum((self.data.qpos[7:] - config.Q_NOMINAL)**2)

        # --- TERMINAL COST ---
        target_x = p_x0 + (config.V_DES * config.HORIZON_SEC)
        cost += config.W_TERM * (self.data.qpos[0] - target_x)**2
        
        return cost


# ==========================================
# PHASE 2 VALIDATION TEST
# ==========================================
if __name__ == "__main__":
    print("--- Phase 2: Ghost Simulator Validation ---")
    
    sim = GhostSimulator()
    
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
    
    print(f"Spawned robot with perfectly anchored Z-Height: {start_qpos[2]:.3f}m\n")
    
    # Test 1: The "Perfect Stand" Trajectory
    perfect_traj = np.tile(config.Q_NOMINAL, (config.H_STEPS, 1))
    cost_perfect = sim.evaluate_trajectory(start_qpos, start_qvel, perfect_traj, "Perfect Stand")
    print(f"Cost of standing still: {cost_perfect:.2f} (Should be ~100-300)")
    
    # Test 2: The "Collapse" Trajectory
    # [THE FIX: Command the legs to actually fold up so the body drops]
    terrible_traj = np.tile([0.0, 1.5, -2.7] * 4, (config.H_STEPS, 1))
    cost_terrible = sim.evaluate_trajectory(start_qpos, start_qvel, terrible_traj, "Collapse")
    print(f"Cost of belly-flopping: {cost_terrible:.2f} (Should be 1000000.00)")
    
    if cost_terrible >= 1e5 and cost_perfect < 1e5:
        print("\n[PASS] The physics are stable, the anchor works, and the scoring is accurate!")
    else:
        print("\n[FAIL] Something is still wrong with the scoring.")