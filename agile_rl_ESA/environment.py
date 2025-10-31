"""
Create simulation Gym environment with Chief and Deputy satellites.
"""

import gymnasium as gym
import numpy as np

# import satellites
from satellites import ChiefSat, DeputySat

# import rewards
from rewards import DistanceReward, VelocityReward

# import bsk_rl libraries
from bsk_rl import scene, data # -> actions, data, observations
from bsk_rl import GeneralSatelliteTasking
# import os
# os.system("bskLargeData")
# define chief and deputy satellites
chief = ChiefSat(name="ChiefSat")
# define a target orbit for the deputy to reach (in ECI frame)
target_orbit = np.array([7000, 0, 0]) # km
target_velocity = np.array([0, 7.5, 0]) # km/s
deputy = DeputySat(name="DeputySat")

# create a simple env with just these two satellites
env = GeneralSatelliteTasking(
    satellites=[chief, deputy],
    scenario=scene.Scenario(),
    rewarder=[DistanceReward( weight=-3, target=target_orbit, position_fn=lambda sat: sat.dynamics.r_BN_N), VelocityReward(target=target_velocity, weight=-2, velocity_fn=lambda sat: sat.dynamics.v_BN_N), data.ResourceReward(0.1, resource_fn=lambda sat: sat.fsw.dv_available)]
)

# make class to make env into a gym interface
class GymEnv(gym.Env):
    """
    A Gym environment wrapper for the GeneralSatelliteTasking environment.
    """
    def __init__(self, env: GeneralSatelliteTasking):
        super().__init__()
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space

    def reset(self):
        """
        Resets the environment to an initial state and returns an initial observation.
        """
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        Executes one time step within the environment.
        :param action: An action provided by the agent.
        :return: A tuple (observation, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode='human'):
        """
        Renders the environment.
        :param mode: The mode to render with.
        """
        self.env.render(mode=mode)

# create gym environment
env = GymEnv(env)

# Add random action loop
episodes = 5
max_steps = 100

for episode in range(episodes):
    obs, info = env.reset()
    episode_reward = 0
    
    for step in range(max_steps):
        # Sample a random action
        action = env.action_space.sample()
        
        # Take a step in the environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        # Break if episode is done
        if terminated or truncated:
            break
            
    print(f"Episode {episode + 1} finished with reward: {episode_reward}")
