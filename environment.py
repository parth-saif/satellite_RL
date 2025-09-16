"""
Define an Environment class that handles configuration of simulation/satellites and returns Gym-wrapped env.
Configuration parameters are passed as dict (could be from parsed json)

The returned gym environment use bsk_rl Gym API.
"""
# import bsk_rl libararies
from bsk_rl import GeneralSatelliteTasking
from bsk_rl import scene
from bsk_rl.data import ResourceReward

# import satellite types
from satellites import ChiefSat, DeputySat

# import reward types
from rewards import DistanceReward, VelocityReward
