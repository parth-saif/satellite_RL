# Define reward function for the RPO_RL environment

# import GlobalReward from bsk_rl.data.base -> base reward class has access to global states
from bsk_rl.data.base import GlobalReward

# create reward class for rendezvous and proximity operations (RPO) tasks
# this class will compute the reward based on the state and action of the chaser satellite
class RendezvousReward(GlobalReward):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)

    def compute_reward(self, state, action):
        pass

