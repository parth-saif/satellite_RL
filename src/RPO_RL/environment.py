"""
Multi-agent environment definitions.
Define the Deputy and Cheif Satellites' observation and action spaces.
"""
import gymnasium as gym
import numpy as np

# import bsk_rl libraries
from bsk_rl import act, data, obs, scene, sats # -> actions, data, observations, scenarios, satellites
from bsk_rl.sim import dyn, fsw # -> dynamics, flight software
from bsk_rl import GeneralSatelliteTasking # -> general satellite tasking environment

import rel_obs_elems as roe # relative orbital elements

class OEDynamics(dyn.BasicDynamicsModel):
    """
    Modified dynamics model to calculate further orbital elements:
    - Eccentric Anomaly, E
    - Mean Anomaly, M
    - Argument of Latitude, u
    - Eccentricy vector, e_vec
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def eccentric_anomaly(self):
        """
        Calculate the Eccentric Anomaly E from True Anomaly nu and Eccentricity e.
        """
        nu = self.true_anomaly
        e = self.eccentricity

        E_cos = (e + np.cos(nu)) / (1 + e * np.cos(nu))
        E_sin = (np.sqrt(1 - e**2) * np.sin(nu)) / (1 + e * np.cos(nu))
        E = np.arctan2(E_sin, E_cos)
        return E
    
    @property
    def mean_anomaly(self):
        """
        Calculate Mean Anomaly M.
        """
        e = self.eccentricity
        E = self.eccentric_anomaly
        M = E - e*np.sin(E)
        return M
    
    @property
    def argument_of_latitude(self):
        """
        Calculate the Argument of Latitude u from Argument of Periapsis omega.
        """
        omega = self.argument_of_periapsis
        u = omega + self.mean_anomaly
        return u
    
    @property
    def eccentricity_vec(self):
        """
        Eccentricity Vector
        """
        e =  self.eccentricity
        omega = self.argument_of_periapsis
        e_vec =  e*np.array([np.cos(omega), np.sin(omega)])
        return e_vec

class ChiefSat(sats.Satellite): 
    """
    Chief satellite defined as agent
    Observations: Time (1D)
    Actions: Drift (do nothing)
    """
    observation_spec = [
        obs.Time() # time in seconds since start of simulation, normalised by the total simulation time
        ]
    action_spec = [act.Drift()] # action is to do nothing
    dyn_type = OEDynamics 
    fsw_type = fsw.BasicFSWModel

class DeputySat(sats.Satellite): 
    """
    Deputy satellite defined as agent.
    Observations: Fuel remaining (Continuous 1D); Relative Orbital Elements (Continuous 6D); Time (1D)
    Action: Impulsive Thrust in Hill frame (Continuous 4D)
    """
    observation_spec = [obs.SatProperties( # self properties of the satellite
        # fuel remaining
        dict(prop="dv_available", module="fsw", norm=1, name="Fuel Remaining"),
    ),
    obs.RelativeProperties( # relative properties to the target satellite - using Hill frame
        dict(fn=roe.delta_a, name="delta_a"), # relative semi-major axis
        dict(fn=roe.delta_lambda, name="delta_lambda"), # relative mean longitude
        dict(fn=roe.delta_e_vec, name="delta_e_vec"), # relative eccentricity vector
        dict(fn=roe.delta_i_vec, name="delta_i_vec"), # relative inclination vector
        chief_name="ChiefSat" 
    ),
    obs.Time()
    ]
    action_spec = [act.ImpulsiveThrustHill("ChiefSat")] # continuous action is to apply an impulsive thrust in the Hill frame
    dyn_type = OEDynamics
    fsw_type = fsw.MagicOrbitalManeuverFSWModel # flight software model for the satellite -> has fuel remaining property for thruster control


# test environments
if __name__ == "__main__":
    # create a target satellite
    target_sat = ChiefSat(name="ChiefSat")

    # create a chaser satellite
    chaser_sat = DeputySat(name="DeputySat")

     # Create lists to store data
    target_data = []
    chaser_data = []


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