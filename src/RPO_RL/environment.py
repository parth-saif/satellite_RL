# import required libraries
import gymnasium as gym
import numpy as np

# import bsk_rl libraries
from bsk_rl import act, data, obs, scene, sats # -> actions, data, observations, scenarios, satellites
from bsk_rl.sim import dyn, fsw # -> dynamics, flight software
from bsk_rl import GeneralSatelliteTasking # -> general satellite tasking environment
# basilisk libraries
from Basilisk.architecture import bskLogging # for debug, info, warning, error logging etc.

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
            ),
        obs.Time() # time in seconds since start of simulation
        ]
    action_spec = [act.Drift()] # action is to do nothing
    dyn_type = dyn.BasicDynamicsModel 
    fsw_type = fsw.BasicFSWModel

#create a chaser satellite class -> this is the agent
class ChaserSat(sats.Satellite): 
    observation_spec = [obs.SatProperties( # self properties of the satellite
        # position in inertial frame
        dict(prop="r_BN_N", module="dynamics", norm=1, name="Position in Inertial Frame"),
        # velocity in inertial frame
        dict(prop="v_BN_N", module="dynamics", norm=1, name="Velocity in Inertial Frame"),
        # angular body rate relative to inertial frame in body frame
        dict(prop="omega_BN_B", module="dynamics", norm=1, name="Angular Body Rate in Body Frame"),
        
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

    # sat args
    target_sat.generate_sat_args() # randomize satellite arguments
    chaser_sat.generate_sat_args() # randomize satellite arguments
    print("Target Satellite Arguments:", target_sat.sat_args)
    print("Chaser Satellite Arguments:", chaser_sat.sat_args)

    # create environment
    env = GeneralSatelliteTasking(satellites=[target_sat, chaser_sat], scenario=scene.Scenario(), rewarder = data.NoReward(), time_limit=5700, log_level="INFO", vizard_dir="viz_output")
    print("Environment created:", env)
    env.reset()  # reset the environment
    env.step([0,(100,1,1,5700)])
    print("Step executed in environment.")