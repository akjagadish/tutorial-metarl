import torch
import numpy as np
import  math 


class ContextualBandit():

    def __init__(self, probs, ctx_dim, num_arms):
        
        self.probs  = probs
        self.num_arms = num_arms
        self.ctx_dim = ctx_dim
        self.context = np.random.randint(2, size=(num_arms, ctx_dim)) 
        while len(np.unique(self.context, axis=0))<num_arms: # such that we sample unique contexts
            self.context = np.random.randint(2, size=(num_arms, ctx_dim))  
        self.arm_per_context = np.random.choice(np.arange(num_arms), size=num_arms, replace=False)
        
    def sample(self, num_episodes, num_trials, prob=None, cxt_per_epoch=False):
       
        X, Y = [], []
        if cxt_per_epoch:
            self.context = np.random.randint(2, size=(self.num_arms, self.ctx_dim)) 
            while len(np.unique(self.context, axis=0))<self.num_arms: # such that we sample unique contexts
                self.context = np.random.randint(2, size=(self.num_arms, self.ctx_dim))  
            self.arm_per_context = np.random.choice(np.arange(self.num_arms), size=self.num_arms, replace=False)
        contexts = self.make_bag_of_tasks(num_episodes)
        for cxt in contexts:
            if prob is None:
                prob = np.random.choice(self.probs)
            x, b = self.context[cxt], self.arm_per_context[cxt]
            x, y = self._sample_one_episode(x, b, prob, num_trials)
            X.append(x)
            Y.append(y)
        Y = torch.stack(Y)
        X = torch.stack(X)
        return X, Y
            
    def _sample_one_episode(self, x, b, prob, num_trials):
        
        X, Y = [], []
        for _ in range(num_trials):
            y = np.zeros(self.num_arms)
            r = np.random.choice([0, 1], size=(self.num_arms-1), p=[prob, 1-prob])
            y[np.arange(self.num_arms)!=b] = r
            y[b] = torch.bernoulli(torch.as_tensor([prob]))
            Y.append(torch.as_tensor(y))
            X.append(torch.as_tensor(x).type(torch.FloatTensor))
        return torch.stack(X), torch.stack(Y)
    
    def make_bag_of_tasks(self, num_episodes):
        samples = np.repeat(np.arange(self.num_arms), int(num_episodes/len(self.context)))
        np.random.shuffle(samples)
        return samples


    
'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', context='talk')

    # build a sampler

    task = ContextualBandit()

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
