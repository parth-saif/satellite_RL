"""
Define the rewards.
"""
import numpy as np
from bsk_rl.data.base import GlobalReward, Data
from bsk_rl.data import ResourceReward

class DistanceData(Data):
    """
    Data for tracking 
    """

class DistanceReward(GlobalReward):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def compute_reward(self, ) -> np.ndarray:
        """
        Compute a reward for the chaser satellite based on its state.        
        """
        pass

    def __potential(self) -> float:
        """
        Calculate a potential function that computes a weighted norm of ROEs.
        """


