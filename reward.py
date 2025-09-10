# Define reward function for the RPO_RL environment

# import GlobalReward from bsk_rl.data.base -> base reward class has access to global states
from bsk_rl.data.base import GlobalReward
import numpy as np
# create reward class for rendezvous and proximity operations (RPO) tasks
# this class will compute the reward based on the state of the chaser satellite
class RendezvousReward(GlobalReward):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def compute_reward(self, observation: np.ndarray) -> np.ndarray:
        """
        Compute a reward for the chaser satellite based on its state.
        State: Fuel remaining (Continuous 1D); Relative Orbital Elements (Continuous 6D); Time (1D)
        """
        pass

    def __potential(self) -> float:
        """
        Calculate a potential function that computes a weighted norm of ROEs.
        """


