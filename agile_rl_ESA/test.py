import gymnasium as gym
import numpy as np
from bsk_rl import act, data, obs, scene, sats
from bsk_rl.sim import dyn, fsw

class MyScanningSatellite(sats.AccessSatellite):
    observation_spec = [
        obs.SatProperties(
            dict(prop="storage_level_fraction"),
            dict(prop="battery_charge_fraction")
        ),
        obs.Eclipse(),
    ]
    action_spec = [
        act.Scan(duration=60.0),  # Scan for 1 minute
        act.Charge(duration=600.0),  # Charge for 10 minutes
    ]
    dyn_type = dyn.ContinuousImagingDynModel
    fsw_type = fsw.ContinuousImagingFSWModel

MyScanningSatellite.default_sat_args()

sat_args = {}

# Set some parameters as constants
sat_args["imageAttErrorRequirement"] = 0.05
sat_args["dataStorageCapacity"] = 1e10
sat_args["instrumentBaudRate"] = 1e7
sat_args["storedCharge_Init"] = 50000.0

# Randomize the initial storage level on every reset
sat_args["storageInit"] = lambda: np.random.uniform(0.25, 0.75) * 1e10

# Make the satellite
sat = MyScanningSatellite(name="EO1", sat_args=sat_args)

env = gym.make(
    "SatelliteTasking-v1",
    satellite=sat,
    scenario=scene.UniformNadirScanning(),
    rewarder=data.ScanningTimeReward(),
    time_limit=5700.0,  # approximately 1 orbit
    log_level="INFO",
)
observation, info = env.reset(seed=1)

print("Initial data level:", observation[0], "(randomized by sat_args)")
for _ in range(3):
    observation, reward, terminated, truncated, info = env.step(action=0)
print("  Final data level:", observation[0])

while not truncated:
    observation, reward, terminated, truncated, info = env.step(action=1)
    print(f"Charge level: {observation[1]:.3f} ({env.unwrapped.simulator.sim_time:.1f} seconds)\n\tEclipse: start: {observation[2]:.1f} end: {observation[3]:.1f}")