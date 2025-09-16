"""
Define the Deputy and Cheif Satellites - observation and action spaces.
"""
import gymnasium as gym
import numpy as np

# import bsk_rl libraries
from bsk_rl import act, obs, sats # -> actions, data, observations, scenarios, satellites
from bsk_rl.sim import dyn, fsw # -> dynamics, flight software

class ChiefSat(sats.Satellite): 
    """
    Chief satellite is passive.
    Observations: Time (1D)
    Actions: Drift (do nothing)
    """
    observation_spec = [
        obs.Time() # time in seconds since start of simulation, normalised by the total simulation time
        ]
    action_spec = [act.Drift()] # action is to do nothing
    dyn_type = dyn.BasicDynamicsModel
    fsw_type = fsw.BasicFSWModel

class DeputySat(sats.Satellite): 
    """
    Deputy satellite is an active agent.
    Observations: Fuel remaining (Continuous 1D); Hill frame position and velocity (Continuous 6D); Time (1D)
    Action: Impulsive Thrust in Hill frame (Continuous 4D - dir and time)
    """
    observation_spec = [obs.SatProperties( # self properties of the satellite
        # fuel remaining
        dict(prop="dv_available", module="fsw", norm=1, name="Fuel Remaining"),
    ),
    obs.RelativeProperties( # relative properties to the target satellite - using Hill frame
        dict(prop="r_DC_Hc", norm=1),
        dict(prop = "v_DC_Hc", norm=1),
        chief_name="ChiefSat" 
    ),
    obs.Time()
    ]
    action_spec = [act.ImpulsiveThrustHill("ChiefSat")] # continuous action is to apply an impulsive thrust in the Hill frame
    dyn_type = dyn.BasicDynamicsModel
    fsw_type = fsw.MagicOrbitalManeuverFSWModel # flight software model for the satellite -> has fuel remaining property for thruster control


# test environments
if __name__ == "__main__":
    # create a target satellite
    target_sat = ChiefSat(name="ChiefSat")

    # create a chaser satellite
    chaser_sat = DeputySat(name="DeputySat", sat_args=dict(dv_available_init=1))

     # Create lists to store data
    target_data = []
    chaser_data = []

    from bsk_rl import GeneralSatelliteTasking # -> general satellite tasking environment
    from bsk_rl import data, scene

    # create environment
    env = GeneralSatelliteTasking(satellites=[target_sat, chaser_sat], scenario=scene.Scenario(), rewarder = data.NoReward(), time_limit=5700, log_level="INFO", terminate_on_time_limit=True, vizard_dir="viz_output")
    obs = env.reset()
    print("Environment created:", env)
    # initialise orbits for both satellites
    

    
    # Simulation loop
    done = False
    i=0
    while not done:
        # Take a step with some action
        obs, reward, done, truncated, info = env.step([0, (2,1,0,0)])

    #     if i==0:
    #         target_kepler = obs[0][9:14]
    #         chaser_kepler = obs[1][9:14]

    #         target_data.append(target_kepler.tolist())
    #         chaser_data.append(chaser_kepler.tolist())

    #         i+=1
        
        time = obs[0][-1]*5700
        chaser_obs = obs[1][0:7]  # First 3 elements are position       
        # append to list
    
        chaser_data.append([time] + chaser_obs.tolist())
    

    # Convert to pandas DataFrames
    import pandas as pd
    chaser_df = pd.DataFrame(chaser_data)
    
    # Save to CSV files
    chaser_df.to_csv('chaser_trajectory.csv', index=False)
    
    print("Trajectory data saved to CSV files")