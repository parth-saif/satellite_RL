"""
Define Datastores for computing position and velocity error Data.
"""

import numpy as np
from bsk_rl.data.base import DataStore
from typing import Callable

# import data type classes
from data import DistanceData, VelocityData

"""
Datastores hold Data from previous step. At the end of one step, it compares the new Data to the previous.
We update the Data value through this comparison/computation.
"""

class DistanceDataStore(DataStore):
    """
    Data store for computing/tracking position error.
    """
    data_type  = DistanceData # added for clarity

    def __init__(self, *args, position_fn: Callable, target: np.ndarray, **kwargs):
        """
        :param position_fn: A function that, given a satellite object, returns
                            the position vector to be used for the calculation.
        :param target: The target position vector to compare against.
        """
        super().__init__(*args, **kwargs)
        self.position_fn = position_fn
        self.target = np.array(target)

    def get_log_state(self) -> np.ndarray:
        """
        Gets the current position vector of the satellite from the provided function.
        """
        return self.position_fn(self.satellite)
    
    def compare_log_states(self, prev_state: np.ndarray, new_state: np.ndarray) -> DistanceData:
        """
        Calculates the distance error based on the new position state. This method must be defined for DataStore.

        For distance error, we only care about the most recent state,
        so the 'prev_state' is ignored.
        """
        # Distance norm is Euclidean distance
        error = np.linalg.norm(new_state - self.target)
        
        # Return the error packaged in our custom Data class
        return DistanceData(distance_err=error)
    
class VelocityDataStore(DataStore):
    """DataStore for computing/tracking velocity error from a target velocity."""
    data_type = VelocityData

    def __init__(self, *args, velocity_fn: Callable, target: np.ndarray, **kwargs):
        """
        :param velocity_fn: A function that, given a satellite object, returns
                         the velocity vector for the calculation.
        :param target: The target velocity vector to compare against.
        """
        super().__init__(*args, **kwargs)
        self.velocity_fn = velocity_fn
        self.target = np.array(target)

    def get_log_state(self) -> np.ndarray:
        """
        Gets the current velocity vector of the satellite.
        """
        return self.velocity_fn(self.satellite)

    def compare_log_states(self, prev_state: np.ndarray, new_state: np.ndarray) -> VelocityData:
        """
        Calculates the velocity error based on the new state.
        The prev_state is ignored for this type of state-based error again.
        """
        # Calculate the Euclidean norm of the velocity error vector
        error = np.linalg.norm(new_state - self.target)
        
        # Return the error packaged in our custom Data class
        return VelocityData(velocity_err=error)
