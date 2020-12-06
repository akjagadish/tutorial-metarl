import torch
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns

class CompositionalBandit():

    def __init__(self, bandit, rules, num_rounds=10, normalize=True):
        
        # default_cues = {'linear': torch.tensor([1.0, 0.0]), 'periodic': torch.tensor([0.0, 1.0]), 'linperiodic': torch.tensor([1.0, 1.0])}
        # default_arms = {'linear': end_arm, 'periodic': end_arm-1, 'linperiodic': end_arm-1} 
        
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.cues = bandit.cues
        self.num_cues = bandit.num_cues
        self.num_rounds = num_rounds
        self.rules = rules
        self.num_rules = len(rules)
        self.aoi = bandit.aoi
        self.cue_per_epoch = self.bandit.cue_per_epoch
        # 2nd level params
        self.normalize = normalize
        
    def sample(self, start_rnd=0, end_rnd=None, cue='linpos_pereven_add'):

        #cue1, cue2, rule = cue.split(sep='_')
        split_cues = cue.split(sep='_')
        rule = split_cues[-1]
        cues = split_cues[:-1]

        if end_rnd is None:
            end_rnd = start_rnd + self.num_rounds

        X, Y = [], []
        for rnd in np.arange(start_rnd, end_rnd):
            # generate per round sample 
            # x, y = self._sample_one_round(cue1, cue2, rule)
            # loop over cues to compose
            for idx, cue in enumerate(cues):
                if idx == 0:
                    x, y = self.bandit._sample_one_round(cue=cue)
                else:
                    x, y = self._sample_one_round(x, y, cue, rule)
                        #if self.normalize:
            if self.normalize:
                y = y-y.min()
                y = y/y.max()
            X.append(x)
            Y.append(y)

        Y = torch.stack(Y)#/torch.stack(Y).max(1).values.mean()

        X = torch.stack(X)

        return X, Y

    def _sample_one_round(self, x1, y1, cue, rule='add'):

        #x1, y1 = self.bandit._sample_one_round(cue=cue1)
        x2, y2 = self.bandit._sample_one_round(cue=cue)
        x, y =   self.do_compose(x1, y1, x2, y2, rule=rule)

        return x, y

    def do_compose(self, x1, y1, x2, y2, rule='add'):

        x = torch.cat((x1, x2, torch.as_tensor(self.rules[rule]))) #torch.tensor([0.])))
        if rule == 'add' or rule == 'sum':
                y = y1 + y2

        elif rule == 'chngpnt' or rule == 'change_point':
                y = torch.cat((y1[:int(self.num_arms/2)].type(torch.FloatTensor), y2[int(self.num_arms/2):].type(torch.FloatTensor)))

        return x, y


'''how to use'''

if __name__ == "__main__":

    sns.set(style='white', context='talk')

    # build a sampler
    start_arm = 0
    end_arm = 7
    ctx_dim = 2
    num_rounds = 10
    task = CompositionalBandit(start_arm=start_arm, end_arm=end_arm, ctx_dim=ctx_dim, num_rounds=num_rounds)

    # sample
    end_rnd = 2
    X, Y = task.sample(end_rnd=end_rnd, cue= 'linear', to_torch=False)
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
