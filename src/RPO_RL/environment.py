# import required libraries
import gymnasium as gym
import numpy as np

# import bsk_rl libraries
from bsk_rl import act, data, obs, scene, sats # -> actions, data, observations, scenarios, satellites
from bsk_rl.sim import dyn, fsw # -> dynamics, flight software

# basilisk libraries
from Basilisk.architecture import bskLogging # for debug, info, warning, error logging etc.

# create a target satellite class -> not an agent, just a target
class TargetSat(sats.Satellite): # inherit from base satellite class#
    # implicitly call the parent class constructor
    observation_spec = [obs.Time()] # observation is time
    action_spec = [act.Drift()] # action is to do nothing
    dyn_type = dyn.BasicDynamicsModel 
    fsw_type = fsw.BasicFSWModel

# test environments
if __name__ == "__main__":
    # create a target satellite
    target_sat = TargetSat(name="TargetSat")

    # sat args
    target_sat.generate_sat_args() # randomize satellite arguments
    print("Target Satellite Arguments:", target_sat.sat_args)

    # create environment
    env = gym.make("SatelliteTasking-v1", satellite=target_sat, scenario=scene.Scenario(), rewarder = data.NoReward(), time_limit=5700, log_level="INFO")
    print("Environment created:", env)
    env.reset()  # reset the environment
    env.step(action=0)
    print("Step executed in environment.")