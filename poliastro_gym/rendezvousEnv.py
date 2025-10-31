import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Poliastro and Astropy for orbital mechanics
import astropy.units as u
from astropy.time import Time
from poliastro.bodies import Earth
from poliastro.twobody import Orbit
from poliastro.maneuver import Maneuver

class RendezvousEnv(gym.Env):
    