"""
Define the Deputy and Cheif Satellites.
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
