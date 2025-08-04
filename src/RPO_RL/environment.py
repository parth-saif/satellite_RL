# import required libraries
import gymnasium as gym
import numpy as np

# import bsk_rl libraries
from bsk_rl import act, data, obs, scene, sats # -> actions, data, observations, scenarios, satellites
from bsk_rl.sim import dyn, fsw # -> dynamics, flight software
from bsk_rl import GeneralSatelliteTasking # -> general satellite tasking environment

# create a target satellite class -> not an agent, just a target
class TargetSat(sats.Satellite): # inherit from base satellite class#
    # implicitly call the parent class constructor
    observation_spec = [
        obs.SatProperties( # self properties of the satellite
            # position in inertial frame
            dict(prop="r_BN_N", module="dynamics", norm=1, name="Position in Inertial Frame"),
            # velocity in inertial frame
            dict(prop="v_BN_N", module="dynamics", norm=1, name="Velocity in Inertial Frame"),
            # angular body rate relative to inertial frame in body
            dict(prop="omega_BN_B", module="dynamics", norm=1, name="Angular Body Rate in Body Frame"),

            # keplerian elements
            dict(prop="semi_major_axis", module="dynamics", name="Semi-major Axis"),
            dict(prop="eccentricity", module="dynamics", name="Eccentricity"),
            dict(prop="inclination", module="dynamics",  name="Inclination"),
            dict(prop="ascending_node", module="dynamics",  name="Ascending Node"),
            dict(prop="argument_of_periapsis", module="dynamics",  name="Argument of Periapsis"),
            dict(prop="true_anomaly", module="dynamics",  name="True Anomaly"),
            ),
        obs.Time() # time in seconds since start of simulation, normalised by the total simulation time
        ]
    action_spec = [act.Drift()] # action is to do nothing
    dyn_type = dyn.BasicDynamicsModel 
    fsw_type = fsw.BasicFSWModel

#create a chaser satellite class -> this is the agent
class ChaserSat(sats.Satellite): # inherit from base satellite class and target satellite class
    observation_spec = [obs.SatProperties( # self properties of the satellite
        # position in inertial frame
        dict(prop="r_BN_N", module="dynamics", norm=1, name="Position in Inertial Frame"),
        # velocity in inertial frame
        dict(prop="v_BN_N", module="dynamics", norm=1, name="Velocity in Inertial Frame"),
        # angular body rate relative to inertial frame in body frame
        dict(prop="omega_BN_B", module="dynamics", norm=1, name="Angular Body Rate in Body Frame"),
        

        # keplerian elements
        dict(prop="semi_major_axis", module="dynamics", name="Semi-major Axis"),
        dict(prop="eccentricity", module="dynamics", name="Eccentricity"),
        dict(prop="inclination", module="dynamics",  name="Inclination"),
        dict(prop="ascending_node", module="dynamics",name="Ascending Node"),
        dict(prop="argument_of_periapsis", module="dynamics", name="Argument of Periapsis"),
        dict(prop="true_anomaly", module="dynamics", name="True Anomaly"),

        # fuel remaining
        dict(prop="dv_available", module="fsw", norm=1, name="Fuel Remaining"),
    ),
    obs.RelativeProperties( # relative properties to the target satellite - using Hill frame as relative dynamics are more easily modeled -> might be easier for RL agent to learn
        # relative position of chaser to target in target's Hill frame
        dict(prop="r_DC_Hc", norm=1, name="Relative Position in Hill Frame"),
        # relative velocity of chaser to target in target's Hill frame
        dict(prop="v_DC_Hc", norm=1, name="Relative Velocity in Hill Frame"),
        ### possibly add more relative properties like relative angular rate, relative attitude etc. if needed (not currently implemented)
        chief_name="TargetSat" # name of the target satellite to get relative properties
    ),
    obs.Time() # time in seconds since start of simulation
    ]
    action_spec = [act.ImpulsiveThrust()] # continuous action is to apply an impulsive thrust in the Hill frame
    dyn_type = dyn.BasicDynamicsModel # dynamics model for the satellite
    fsw_type = fsw.MagicOrbitalManeuverFSWModel # flight software model for the satellite -> has fuel remaining property for thruster control


# test environments
if __name__ == "__main__":
    # create a target satellite
    target_sat = TargetSat(name="TargetSat")

    # create a chaser satellite
    chaser_sat = ChaserSat(name="ChaserSat")

     # Create lists to store data
    target_data = []
    chaser_data = []


    # create environment
    env = GeneralSatelliteTasking(satellites=[target_sat, chaser_sat], scenario=scene.Scenario(), rewarder = data.NoReward(), time_limit=5700, log_level="INFO", terminate_on_time_limit=True, vizard_dir="viz_output")
    obs = env.reset()
    print("Environment created:", env)
    target_sat.generate_sat_args()
    chaser_sat.generate_sat_args()
    print(target_sat.sat_args)

    import pandas as pd
    
    # Simulation loop
    done = False
    i=0
    while not done:
        # Take a step with some action
        obs, reward, done, truncated, info = env.step([0, (0,0,0,0)])

        if i==0:
            target_kepler = obs[0][9:14]
            chaser_kepler = obs[1][9:14]

            target_data.append(target_kepler.tolist())
            chaser_data.append(chaser_kepler.tolist())

            i+=1
        
        # time = obs[0][-1]*5700
        # target_pos = obs[0][0:3]  # First 3 elements are position
        # chaser_pos = obs[1][0:3]  # First 3 elements are position       

        # # append to list
        # target_data.append([time] + target_pos.tolist())
        # chaser_data.append([time] + chaser_pos.tolist())
    

    # # Convert to pandas DataFrames
    # target_df = pd.DataFrame(target_data, columns=['time', 'x', 'y', 'z'])
    # chaser_df = pd.DataFrame(chaser_data, columns=['time', 'x', 'y', 'z'])#

    target_df = pd.DataFrame(target_data, columns=['semi_major_axis', 'eccentricity', 'inclination', 'ascending_node', 'argument_of_periapsis'])
    chaser_df = pd.DataFrame(chaser_data, columns=['semi_major_axis', 'eccentricity', 'inclination', 'ascending_node', 'argument_of_periapsis'])
    
    # Save to CSV files
    target_df.to_csv('target_trajectory.csv', index=False)
    chaser_df.to_csv('chaser_trajectory.csv', index=False)
    
    print("Trajectory data saved to CSV files")