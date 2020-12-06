import torch
import numpy as np
import  math 


class BernoulliBandit():

    def __init__(self, probs):
        
        self.probs  = probs
        self.num_arms = 2
        
    def sample(self, num_episodes, prob=None):
       
        X, Y = [], []
        for rnd in np.arange(num_episodes):
            if prob is None:
                prob = np.random.choice(self.probs)
            y = self._sample_one_episode(prob)
            #X.append(x) 
            Y.append(y)
        #if num_episodes>1:
        Y = torch.stack(Y)
        #X = torch.stack(X)
        return Y
            
    def _sample_one_episode(self, prob): 
        r1 = torch.bernoulli(torch.as_tensor([prob]))
        r2 = 1. - r1
        y = torch.cat((r1, r2))
        return y
    
'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', context='talk')

    # build a sampler

    task = BernoulliBandit()

    # sample
    X, Y = task.sample()
    print(f'X shape = {np.shape(X)}, n_example x time x x-dim')
    print(f'Y shape = {np.shape(Y)},  n_example x time x y-dim')

    # show one trial
    i = 0
    input = X[i]
    target = int(Y[i][0])
    vmin = np.min(X)
    vmax = np.max(X)

    f, ax = plt.subplots(1, 1, figsize=(3, 5))
    sns.heatmap(
        input.T,
        vmin=vmin, vmax=vmax,
        cmap='RdBu_r', yticklabels=10, center=0,
        ax=ax
    )
    ax.axvline(t_noise_off, color='grey', linestyle='--')
    ax.axhline(obs_dim, color='black', linestyle='--')
    ax.set_title(f'Stimulus for a trial, y = {target}')
    ax.set_xlabel('Time')
    ax.set_ylabel('x-dim: context | input')
    f.savefig(f'../figs/eg-{target}.png', dpi=100, bbox_inches='tight')
