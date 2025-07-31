""""
fly.py - Module for controlling Crazyflie drones in a testbed environment.
1. Connect to Crazyflie drones.
2. Log data from drones.
3. Edit parameters of drones.
4. Generate drone trajectories from satellite trajectories.
5. Upload trajectories to drones.
6. Control drones using motion commands.
7. Handle asynchronous logging and parameter editing.

"""


# basic imports
import time
import logging

# crazyflie api imports
import cflib.crtp # -> scanning for available Crazyflies
from cflib.crazyflie import Crazyflie # -> Crazyflie class tp connect/send/recieve data
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie # -> synchronous interface to Crazyflie (turns into blocking calls)

from cflib.crazyflie.log import LogConfig # -> configuration for logging data from Crazyflie
from cflib.crazyflie.syncLogger import SyncLogger # -> synchronous logger to log data from Crazyflie
from cflib.utils import uri_helper # -> parse URIs for Crazyflie connections

from cflib.positioning.motion_commander import MotionCommander # -> high-level motion commands for Crazyflie

from cflib.crazyflie.swarm import CachedCfFactory # -> cache Crazyflie instances for swarm operations to reduce connection overhead
from cflib.crazyflie.swarm import Swarm # -> manage a swarm of Crazyflies

