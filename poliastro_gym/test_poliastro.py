import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Poliastro and Astropy for orbital mechanics
import astropy.units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

# Suppress a known warning from poliastro with newer astropy versions
# import warnings
# from astropy.utils.exceptions import AstropyWarning
# warnings.simplefilter('ignore', category=AstropyWarning)


class RendezvousEnv(gym.Env):
    """
    Custom Gymnasium Environment for a Satellite Rendezvous and Docking mission.

    The environment simulates a deputy satellite (the agent) attempting to
    rendezvous with a chief satellite (the target) in Low Earth Orbit (LEO).

    - Goal: Align the deputy at a fixed position relative to the chief
            (e.g., 10 meters ahead on the velocity vector) and match its velocity.
    - Dynamics: High-fidelity two-body dynamics using poliastro.
    - Randomization: Initial orbits for both satellites are randomized to
                     ensure the learned policy is robust.
    """
    metadata = {'render_modes': ['human'], 'render_fps': 30}

    def __init__(self):
        super().__init__()

        # --- CONFIGURATION PARAMETERS ---
        self.sim_step_dt = 10.0  # seconds per step
        self.max_steps = 500  # Max steps per episode
        self.max_fuel = 100.0  # Max fuel capacity (e.g., in m/s of delta-v)

        # Docking thresholds
        self.docking_pos_threshold = 10.0  # meters
        self.docking_vel_threshold = 0.1  # meters/second

        # Physical constants
        self.earth = Earth
        self.max_thrust = 1.0  # Max thrust capability in m/s per step

        # Target docking position in the chief's LVLH frame [V-bar, H-bar, R-bar]
        # This is 20m "ahead" of the chief satellite.
        self.target_docking_position = np.array([20.0, 0, 0])

        # Normalization constants (for observation space)
        self.max_distance = 10000  # 10 km
        self.max_velocity = 100  # 100 m/s

        # --- ACTION SPACE ---
        # Continuous 3D thrust vector, normalized between -1 and 1
        # [Thrust_V-bar, Thrust_H-bar, Thrust_R-bar]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # --- OBSERVATION SPACE ---
        # A combined vector of relevant information:
        # - Relative position (3) [x, y, z] in LVLH frame
        # - Relative velocity (3) [vx, vy, vz] in LVLH frame
        # - Target relative position (3) [tx, ty, tz] in LVLH frame
        # - Remaining fuel (1)
        # - Time remaining (1)
        # Total = 3 + 3 + 3 + 1 + 1 = 11 dimensions
        obs_low = np.array(
            [-1.0] * 3 +  # Rel Pos
            [-1.0] * 3 +  # Rel Vel
            [-1.0] * 3 +  # Target Pos
            [0.0] +       # Fuel
            [0.0]         # Time
        , dtype=np.float32)
        obs_high = np.array(
            [1.0] * 3 +   # Rel Pos
            [1.0] * 3 +   # Rel Vel
            [1.0] * 3 +   # Target Pos
            [1.0] +       # Fuel
            [1.0]         # Time
        , dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # --- SIMULATION STATE ---
        self.chief_orbit = None
        self.deputy_orbit = None
        self.current_step = 0
        self.fuel_remaining = self.max_fuel
        self.epoch = Time("2023-01-01 12:00:00", scale="tdb")


    def _get_lvlh_frame(self, orbit):
        """Computes the LVLH frame unit vectors for a given orbit."""
        r_vec = orbit.r.to_value(u.m)
        v_vec = orbit.v.to_value(u.m / u.s)
        
        r_norm = np.linalg.norm(r_vec)
        
        # R-bar (Radial, nadir pointing)
        r_bar = -r_vec / r_norm
        # H-bar (Angular momentum vector)
        h_vec = np.cross(r_vec, v_vec)
        h_bar = h_vec / np.linalg.norm(h_vec)
        # V-bar (Velocity direction)
        v_bar = np.cross(h_bar, r_bar)
        
        # Transformation matrix from Inertial to LVLH
        dcm = np.array([v_bar, h_bar, r_bar])
        return dcm

    def _get_observation(self):
        """
        Calculates the current observation state.
        This involves transforming inertial states into the chief's LVLH frame.
        """
        # Get chief's LVLH frame
        dcm_inertial_to_lvlh = self._get_lvlh_frame(self.chief_orbit)

        # Relative position and velocity in inertial frame
        rel_pos_inertial = (self.deputy_orbit.r - self.chief_orbit.r).to_value(u.m)
        rel_vel_inertial = (self.deputy_orbit.v - self.chief_orbit.v).to_value(u.m / u.s)

        # Transform to LVLH
        rel_pos_lvlh = dcm_inertial_to_lvlh @ rel_pos_inertial
        rel_vel_lvlh = dcm_inertial_to_lvlh @ rel_vel_inertial

        # Normalize observations
        norm_pos = rel_pos_lvlh / self.max_distance
        norm_vel = rel_vel_lvlh / self.max_velocity
        norm_target = self.target_docking_position / self.max_distance
        norm_fuel = self.fuel_remaining / self.max_fuel
        norm_time = (self.max_steps - self.current_step) / self.max_steps

        # Concatenate and clip to be safe
        obs = np.concatenate([
            norm_pos,
            norm_vel,
            norm_target,
            [norm_fuel],
            [norm_time]
        ]).astype(np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- RANDOMIZE CHIEF ORBIT ---
        # Start with a base circular LEO and add random perturbations
        base_alt = 700 + self.np_random.uniform(-100, 100)  # 600-800 km altitude
        base_inc = 51.6 + self.np_random.uniform(-1, 1)     # ISS-like inclination
        base_ecc = 0.001 * self.np_random.random()          # Near-circular
        
        self.chief_orbit = Orbit.from_classical(
            self.earth,
            base_alt * u.km,
            base_ecc * u.one,
            base_inc * u.deg,
            0 * u.deg, 0 * u.deg, 0 * u.deg, # RAAN, ArgP, Nu
            epoch=self.epoch
        )

        # --- RANDOMIZE DEPUTY ORBIT (NEARBY) ---
        # Add a random offset in position and velocity to the chief's state
        pos_offset = self.np_random.uniform(-1000, 1000, 3) * u.m  # +/- 1km offset
        vel_offset = self.np_random.uniform(-1, 1, 3) * u.m / u.s   # +/- 1 m/s offset

        self.deputy_orbit = Orbit.from_vectors(
            self.earth,
            r = self.chief_orbit.r + pos_offset,
            v = self.chief_orbit.v + vel_offset,
            epoch=self.epoch
        )

        # Reset simulation state
        self.current_step = 0
        self.fuel_remaining = self.max_fuel

        observation = self._get_observation()
        info = {} # You can add debug info here

        return observation, info

    def step(self, action):
        self.current_step += 1

        # --- 1. APPLY ACTION (THRUST) ---
        # Denormalize action to get thrust vector in m/s (delta-v)
        thrust_lvlh = action * self.max_thrust
        
        # Calculate fuel consumption (proportional to delta-v magnitude)
        fuel_consumed = np.linalg.norm(thrust_lvlh)
        
        if self.fuel_remaining > fuel_consumed:
            self.fuel_remaining -= fuel_consumed
            
            # Convert thrust from LVLH back to inertial frame to apply maneuver
            dcm_inertial_to_lvlh = self._get_lvlh_frame(self.deputy_orbit)
            dcm_lvlh_to_inertial = dcm_inertial_to_lvlh.T
            thrust_inertial = dcm_lvlh_to_inertial @ thrust_lvlh
            
            # Apply as an impulsive maneuver
            maneuver = Maneuver.impulse(thrust_inertial * (u.m / u.s))
            self.deputy_orbit = self.deputy_orbit.apply_maneuver(maneuver)
        else:
            # Not enough fuel, no thrust applied
            fuel_consumed = 0

        # --- 2. PROPAGATE DYNAMICS ---
        time_step = self.sim_step_dt * u.s
        self.chief_orbit = self.chief_orbit.propagate(time_step)
        self.deputy_orbit = self.deputy_orbit.propagate(time_step)

        # --- 3. CALCULATE STATE AND ERRORS ---
        # Get relative state in LVLH
        dcm = self._get_lvlh_frame(self.chief_orbit)
        rel_pos_inertial = (self.deputy_orbit.r - self.chief_orbit.r).to_value(u.m)
        rel_vel_inertial = (self.deputy_orbit.v - self.chief_orbit.v).to_value(u.m / u.s)
        rel_pos_lvlh = dcm @ rel_pos_inertial
        rel_vel_lvlh = dcm @ rel_vel_inertial
        
        # Calculate errors
        pos_error = np.linalg.norm(rel_pos_lvlh - self.target_docking_position)
        vel_error = np.linalg.norm(rel_vel_lvlh)

        # --- 4. DETERMINE REWARD ---
        # Reward is shaped to encourage getting closer and matching velocity
        # Penalties for distance, velocity error, and fuel usage
        reward = -0.1 * (pos_error / self.max_distance) \
                 -0.05 * (vel_error / self.max_velocity) \
                 -0.01 * (fuel_consumed / self.max_fuel)

        # --- 5. CHECK FOR TERMINATION / TRUNCATION ---
        terminated = False
        truncated = False
        
        is_docked = (pos_error < self.docking_pos_threshold) and \
                    (vel_error < self.docking_vel_threshold)
        
        if is_docked:
            reward += 100  # Large bonus for successful docking
            terminated = True
        
        if self.fuel_remaining <= 0:
            reward -= 10 # Penalty for running out of fuel
            terminated = True
            
        if pos_error > self.max_distance: # Drifted too far
            reward -= 10 # Penalty for drifting away
            terminated = True
            
        if self.current_step >= self.max_steps:
            truncated = True
            
        observation = self._get_observation()
        info = {
            'pos_error': pos_error,
            'vel_error': vel_error,
            'fuel_remaining': self.fuel_remaining,
            'rel_pos_lvlh': rel_pos_lvlh,
            'rel_vel_lvlh': rel_vel_lvlh
        }

        return observation, reward, terminated, truncated, info

    def render(self):
        """Rendering is disabled for this version."""
        pass

    def close(self):
        """No resources to close in non-rendering version."""
        pass


# Main block to test the environment and plot results
if __name__ == '__main__':
    from gymnasium.utils.env_checker import check_env
    import matplotlib.pyplot as plt

    # Create and check the environment
    env = RendezvousEnv()
    print("Checking environment compatibility with Gymnasium API...")
    check_env(env)
    print("Environment check passed!")

    # --- Setup for data collection ---
    history = {
        'steps': [], 'pos_error': [], 'vel_error': [], 'fuel': [],
        'reward': [], 'total_reward': [],
        'rel_pos_v': [], 'rel_pos_h': [], 'rel_pos_r': [],
    }
    
    # --- Run a full test episode ---
    print("\n--- Running Test Episode ---")
    obs, info = env.reset()
    done = False
    total_reward = 0
    step_count = 0

    while not done:
        action = env.action_space.sample()  # Use a random agent for this test
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step_count += 1
        
        # Record data
        history['steps'].append(step_count)
        history['pos_error'].append(info['pos_error'])
        history['vel_error'].append(info['vel_error'])
        history['fuel'].append(info['fuel_remaining'])
        history['reward'].append(reward)
        history['total_reward'].append(total_reward)
        history['rel_pos_v'].append(info['rel_pos_lvlh'][0])
        history['rel_pos_h'].append(info['rel_pos_lvlh'][1])
        history['rel_pos_r'].append(info['rel_pos_lvlh'][2])

    print("\n--- Episode Finished ---")
    print(f"Total Steps: {step_count}")
    print(f"Final Position Error: {info['pos_error']:.2f} m")
    print(f"Final Velocity Error: {info['vel_error']:.2f} m/s")
    print(f"Total Reward: {total_reward:.2f}")
    print("Backbone test complete.")

    # --- Plotting Results ---
    print("\n--- Generating Plots ---")
    fig, axs = plt.subplots(3, 2, figsize=(15, 18))
    fig.suptitle('Rendezvous Mission Analysis (Random Agent)', fontsize=16)

    # Plot 1: Position and Velocity Error
    ax = axs[0, 0]
    ax.plot(history['steps'], history['pos_error'], label='Position Error (m)', color='b')
    ax.set_ylabel('Position Error (m)', color='b')
    ax.tick_params(axis='y', labelcolor='b')
    ax2 = ax.twinx()
    ax2.plot(history['steps'], history['vel_error'], label='Velocity Error (m/s)', color='r')
    ax2.set_ylabel('Velocity Error (m/s)', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    ax.set_title('Errors Over Time')
    ax.set_xlabel('Step')
    ax.grid(True)

    # Plot 2: Fuel Remaining
    ax = axs[0, 1]
    ax.plot(history['steps'], history['fuel'], label='Fuel (delta-v m/s)', color='g')
    ax.set_title('Fuel Consumption')
    ax.set_xlabel('Step')
    ax.set_ylabel('Fuel Remaining (m/s)')
    ax.grid(True)

    # Plot 3: Cumulative Reward
    ax = axs[1, 0]
    ax.plot(history['steps'], history['total_reward'], label='Total Reward', color='purple')
    ax.set_title('Cumulative Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True)

    # Plot 4: Relative Trajectory (V-bar vs R-bar) - In-plane motion
    ax = axs[1, 1]
    ax.plot(history['rel_pos_r'], history['rel_pos_v'])
    ax.scatter(history['rel_pos_r'][0], history['rel_pos_v'][0], c='g', marker='o', label='Start')
    ax.scatter(history['rel_pos_r'][-1], history['rel_pos_v'][-1], c='r', marker='x', label='End')
    ax.scatter(0, 20, c='y', marker='*', s=100, label='Target')
    ax.set_title('Relative Trajectory (In-Plane)')
    ax.set_xlabel('R-bar (Radial) (m)')
    ax.set_ylabel('V-bar (Along-track) (m)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Plot 5: Relative Trajectory (V-bar vs H-bar) - Out-of-plane motion
    ax = axs[2, 0]
    ax.plot(history['rel_pos_h'], history['rel_pos_v'])
    ax.scatter(history['rel_pos_h'][0], history['rel_pos_v'][0], c='g', marker='o', label='Start')
    ax.scatter(history['rel_pos_h'][-1], history['rel_pos_v'][-1], c='r', marker='x', label='End')
    ax.scatter(0, 20, c='y', marker='*', s=100, label='Target')
    ax.set_title('Relative Trajectory (Out-of-Plane)')
    ax.set_xlabel('H-bar (Cross-track) (m)')
    ax.set_ylabel('V-bar (Along-track) (m)')
    ax.legend()
    ax.grid(True)
    ax.axis('equal')

    # Plot 6: Step-wise Reward
    ax = axs[2, 1]
    ax.plot(history['steps'], history['reward'], label='Step Reward', color='orange', alpha=0.7)
    ax.set_title('Step-wise Reward')
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    ax.grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('mission_analysis.png')
    print("\nPlots saved to mission_analysis.png")

