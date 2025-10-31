"""
Define data containers for position and velocity error data -> used by Datastores.
We use the architecture of Data, DataStores and Rewarder in bsk_rl
"""

from bsk_rl.data.base import Data

"""
Data is data container that holds the result of the calculations after each step.
A Data class is stored and updated by a DataStore
"""
class DistanceData(Data):
    """
    Data for tracking position error from a target. This data is updated at the end of a step.
    """
    def __init__(self, distance_err: float =0.0) -> None:
        """
        Construct distance error data
        :param distance_err: Euclidean distance from the target
        """
        self.distance_err = distance_err
    def __add__(self, other:"DistanceData") -> "DistanceData":
        """
        Define how data is added after each step.
        """
        total_err = self.distance_err + other.distance_err
        return DistanceData(total_err)
    def __repr__(self) -> str:
        """
        String representation of the data
        """
        return f"DistanceData(distance_error={self.distance_err})"

class VelocityData(Data):
    """
    Data for tracking the velocity error from a target.
    """
    def __init__(self, velocity_err: float = 0.0) -> None:
        """
        Construct velocity data.
        :param velocity_error: The calculated Euclidean norm of the velocity error vector.
        """
        self.velocity_error = velocity_err

    def __add__(self, other: "VelocityData") -> "VelocityData":
        """
        Combine two units of VelocityData.
        """
        total_error = self.velocity_error + other.velocity_error
        return VelocityData(total_error)

    def __repr__(self) -> str:
        """
        String representation of the VelocityData.
        """
        return f"VelocityData(velocity_error={self.velocity_error})"