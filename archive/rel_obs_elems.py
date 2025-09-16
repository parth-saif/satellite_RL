"""
Define non-linear functions for calculating quasi-nonsingular Relative Orbital Elements (ROE) between Deputy and Chief.
"""
import numpy as np
from bsk_rl import sats
from bsk_rl.sim import dyn

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
    def eccentric_anomaly(self) -> float:
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
    def mean_anomaly(self) -> float:
        """
        Calculate Mean Anomaly M.
        """
        e = self.eccentricity
        E = self.eccentric_anomaly
        M = E - e*np.sin(E)
        return M
    
    @property
    def argument_of_latitude(self) -> float:
        """
        Calculate the Argument of Latitude u from Argument of Periapsis omega.
        """
        omega = self.argument_of_periapsis
        u = omega + self.mean_anomaly
        return u
    
    @property
    def eccentricity_vec(self) -> np.ndarray:
        """
        Eccentricity Vector
        """
        e =  self.eccentricity
        omega = self.argument_of_periapsis
        e_vec =  e*np.array([np.cos(omega), np.sin(omega)])
        return e_vec

def delta_a(deputy: sats.Satellite, chief: sats.Satellite) -> float:
    """
    Calculate normalised relative difference between semi-major axes.
    """
    a_d = deputy.dynamics.semi_major_axis
    a_c = chief.dynamics.semi_major_axis
    return (a_d-a_c)/a_c

def delta_lambda(deputy: sats.Satellite, chief: sats.Satellite) -> float:
    """
    Calculate the relative mean longitude as non-linear combination.
    """
    u_d = deputy.dynamics.argument_of_latitude
    u_c = chief.dynamics.argument_of_latitude
    
    i_c = chief.dynamics.inclination

    Omega_d = deputy.dynamics.ascending_node
    Omega_c = chief.dynamics.ascending_node

    return (u_d-u_c) + (Omega_d - Omega_c)*np.cos(i_c)

def delta_e_vec(deputy: sats.Satellite, chief: sats.Satellite) -> np.ndarray:
    """
    Calculate the difference in eccentricity vectors.
    """
    e_d_vec = deputy.dynamics.eccentricity_vec
    e_c_vec = chief.dynamics.eccentricity_vec
    return e_d_vec - e_c_vec

def delta_i_vec(deputy: sats.Satellite, chief: sats.Satellite) -> np.ndarray:
    """
    Calculate the change in inclination vector as a non-linear combination.
    """

    i_d = deputy.dynamics.inclination
    i_c = chief.dynamics.inclination

    Omega_d = deputy.dynamics.ascending_node
    Omega_c = chief.dynamics.ascending_node

    i_vec = np.array([i_d-i_c, (Omega_d - Omega_c)*np.sin(i_c)])
    return i_vec