import torch
import numpy as np
import  math 


class CompositionalTwoArmedBandit():

    def __init__(self, probs, ctx_dim, num_arms, num_ctx=400, max_ctx=1000):
        
        self.probs = np.asarray(probs)
        self.num_arms = num_arms
        self.ctx_dim = ctx_dim
        self.num_ctx = num_ctx
        self.max_ctx = max_ctx
        self.context = self.make_contexts(ctx_dim, num_ctx, max_ctx)
        
    def sample(self, num_episodes=1000, num_trials=100, prob=None, cxt_per_epoch=False, repeats=None):
       
        if cxt_per_epoch:
            # generate unique contexts
            self.context = self.make_contexts(self.ctx_dim, self.num_ctx, self.max_ctx)
        # group them into high and rewarding
        highrwd_context = self.context[:int(self.num_ctx/2)]
        lowrwd_context  = self.context[int(self.num_ctx/2):]
        # make copies and generate samples for both contexts
        highsamples = self.make_bag_of_tasks(num_episodes, repeats=repeats)
        lowsamples = highsamples.copy()
        np.random.shuffle(lowsamples)
        # set low and high probs
        low_prob, high_prob  = self.probs
        probs = self.probs.copy()

        X, Y = [], []
        ctx = torch.zeros(self.num_arms, self.ctx_dim)
        for hsample, lsample in zip(highsamples, lowsamples):
            # change high and low rewarding arm
            np.random.shuffle(probs)
            # sample contexts and assign to respective arms
            ctx[probs == low_prob] = lowrwd_context[lsample]
            ctx[probs == high_prob] = highrwd_context[hsample]
            x, y = self._sample_one_episode(ctx.reshape(-1), probs, num_trials)
            X.append(x)
            Y.append(y)
        Y = torch.stack(Y)
        X = torch.stack(X)
        return X, Y
            
    def _sample_one_episode(self, x, probs, num_trials):
        
        X, Y = [], []
        low_prob, high_prob  = self.probs
        for _ in range(num_trials):
            y = np.zeros(self.num_arms)
            y[probs == low_prob] =  np.random.choice([0, 1], size=(1,), p=self.probs[::-1])
            y[probs == high_prob] = np.random.choice([0, 1], size=(1,), p=self.probs)
            Y.append(torch.as_tensor(y))
            X.append(torch.as_tensor(x).type(torch.FloatTensor))

        return torch.stack(X), torch.stack(Y)
    
    def make_bag_of_tasks(self, num_episodes, repeats=None):
        num_contexts_per_group = int(self.num_ctx/2)
        if repeats is None:
            repeats = int(num_episodes/num_contexts_per_group)
        samples = np.repeat(np.arange(num_contexts_per_group), repeats)
        samples = samples[:num_episodes]
        np.random.shuffle(samples)
        return samples

    def make_contexts(self, ctx_dim, num_ctx, max_ctx):
        sample_contexts = np.random.randint(2, size=(max_ctx, ctx_dim)) 
        while len(np.unique(sample_contexts, axis=0))<num_ctx: # such that we sample unique contexts
            sample_contexts = np.random.randint(2, size=(max_ctx, ctx_dim))  
        sample_contexts = np.unique(sample_contexts, axis=0)[:num_ctx]
        np.random.shuffle(sample_contexts)
        return torch.tensor(sample_contexts).type(torch.FloatTensor)


    
'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', context='talk')

    # build a sampler

    task = CompositionalTwoArmedBandit()

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
