"""
Define Rewards for distance and velocity error from target.
"""
import numpy as np
from typing import Callable
# import data and data stores for position and velocity error
from datastores import DistanceDataStore, VelocityDataStore
from data import DistanceData, VelocityData

# import base reward class
from bsk_rl.data.base import GlobalReward

# --- distance reward ---
class DistanceReward(GlobalReward):
    """A reward that penalizes the agent for being far from a target position."""
    data_store_type = DistanceDataStore

    def __init__(
        self,
        weight: float,
        target: list,
        position_fn: Callable,
    ) -> None:
        """
        :param weight (float): The scaling factor for the reward. Should be negative
                        to create a penalty for distance. -> error is positive, so weight should be negative.
        :param target (list): The target position vector (e.g., [-100.0, 0.0, 0.0]).
        :param position_fn (Callable): A function that extracts the relevant position
                                vector from the satellite object. -> this can be any frame you want
        """
        super().__init__()
        self.weight = weight
        # arguments to pass to data store
        self.data_store_kwargs = dict(position_fn=position_fn, target=target)
        
    def calculate_reward(
        self, new_data_dict: dict[str, DistanceData]
    ) -> dict[str, float]:
        """
        Calculates the reward for each satellite based on its distance error.
        :param new_data_dict: A dictionary mapping satellite names to their
                         newly computed DistanceData objects.
        """
        rewards = {
            sat_name: self.weight * data.distance_err
            for sat_name, data in new_data_dict.items()
        } # compute dictionary of computed rewards based on distance to target

        return rewards

# --- velocity reward ---
class VelocityReward(GlobalReward):
    """A reward that penalizes an agent for having a non-zero velocity error."""
    data_store_type = VelocityDataStore

    def __init__(
        self,
        weight: float,
        target: list,
        velocity_fn: Callable,
    ) -> None:
        super().__init__()
        self.weight = weight
        self.data_store_kwargs = dict(velocity_fn=velocity_fn, target=target)
        
    def calculate_reward(
        self, new_data_dict: dict[str, VelocityData]
    ) -> dict[str, float]:
        """
        Calculates the reward for each satellite based on its velocity error.
        """
        rewards = {
            sat_name: self.weight * data.velocity_error
            for sat_name, data in new_data_dict.items()
        }
        return rewards