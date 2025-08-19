"""
Define non-linear functions for calculating quasi-nonsingular Relative Orbital Elements (ROE) between Deputy and Chief.
"""
import numpy as np

def delta_a(deputy, chief):
    """
    Calculate normalised relative difference between semi-major axes.
    """
    a_d = deputy.dynamics.semi_major_axis
    a_c = chief.dynamics.semi_major_axis
    return (a_d-a_c)/a_c

def delta_lambda(deputy, chief):
    """
    Calculate the relative mean longitude as non-linear combination.
    """
    u_d = deputy.dynamics.argument_of_latitude
    u_c = chief.dynamics.argument_of_latitude
    
    i_c = chief.dynamics.inclination

    Omega_d = deputy.dynamics.ascending_node
    Omega_c = chief.dynamics.ascending_node

    return (u_d-u_c) + (Omega_d - Omega_c)*np.cos(i_c)

def delta_e_vec(deputy, chief):
    """
    Calculate the difference in eccentricity vectors.
    """
    e_d_vec = deputy.dynamics.eccentricity_vec
    e_c_vec = chief.dynamics.eccentricity_vec
    return e_d_vec - e_c_vec

def delta_i_vec(deputy, chief):
    """
    Calculate the change in inclination vector as a non-linear combination.
    """

    i_d = deputy.dynamics.inclination
    i_c = chief.dynamics.inclination

    Omega_d = deputy.dynamics.ascending_node
    Omega_c = chief.dynamics.ascending_node

    i_vec = np.array([i_d-i_c, (Omega_d - Omega_c)*np.sin(i_c)])
    return i_vec