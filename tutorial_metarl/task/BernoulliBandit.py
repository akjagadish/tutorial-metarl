import torch
import numpy as np
import  math 
from tutorial_metarl.trainers.utils import set_random_seed

class BernoulliBandit():
    def __init__(self, seed, probs):
        """Bernoulli Bandit Task w/ Dependent Arms

        Args:
            seed ([int]): seed for the dataloader
            probs ([list]): list of probs of rewards for arm 1
        """
        self.seed = seed
        self.probs  = probs
        self.num_arms = 2
        set_random_seed(seed)
        
    def sample(self, num_episodes, prob=None):
        """sample reward per arm

        Args:
            num_episodes ([int]): number of episodes to sample rewards for
            prob ([float], optional): probability of reward for arm 1. Defaults to None.

        Returns:
            Y [list]: rewards per arm for num_episodes
        """
        X, Y = [], []
        for rnd in np.arange(num_episodes):
            if prob is None:
                prob = np.random.choice(self.probs)
            y = self._sample_one_episode(prob)
            Y.append(y)
        
        Y = torch.stack(Y)
        return Y
            
    def _sample_one_episode(self, prob): 
        r1 = torch.bernoulli(torch.as_tensor([prob]))
        r2 = torch.bernoulli(torch.as_tensor([1-prob]))
        y = torch.cat((r1, r2))
        return y