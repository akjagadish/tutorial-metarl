import torch
import numpy as np
import math 
import matplotlib.pyplot as plt
import seaborn as sns
import pdb

class BlockMABs():

    def __init__(self, bandit, num_blocks,  num_rounds=10, composition_block=True, cues=None, rule='add', normalize=True):
          
        self.bandit = bandit
        self.num_arms = bandit.num_arms
        self.composition_block = composition_block
        self.cues = cues if cues is not None else bandit.cues
        self.num_cues = len(self.cues)
        self.num_rounds = num_rounds
        self.num_blocks = num_blocks
        self.aoi = bandit.aoi
        self.cue_per_epoch = self.bandit.cue_per_epoch
        self.normalize = normalize
        self.eval_control = False
        self.return_composition_only = False
        self.rule = rule
        
    def sample(self, start_rnd=0, end_rnd=None, block=['linear'], rule=None):

        end_rnd = end_rnd if end_rnd else start_rnd+self.num_rounds 
        rule = rule if rule else self.rule

        X, Y, S = [], [], []
        for kernel_indx, kernel in enumerate(block):
            
            for _ in np.arange(start_rnd, end_rnd):

                if self.composition_block or (self.eval_control and len(block)>1):

                    if kernel == 'linear':
                        kernel = np.random.choice(['linpos', 'linneg'])
                    elif kernel == 'periodic':
                        kernel = np.random.choice(['perodd', 'pereven'])
                    
                    if kernel == 'linperiodic':
                        x, y = self.composed_x, self.composed_y
                    else:
                        x, y = self.bandit._sample_one_round(cue=kernel)    

                    if kernel_indx<(len(block)-1):
                        _ = self.do_composition(x, y, kernel_indx, rule=rule)

                else:

                    if kernel == 'linperiodic' and len(block)==1:
                        xlin, ylin = self.bandit._sample_one_round(cue=np.random.choice(['linpos', 'linneg']))
                        _ = self.do_composition(xlin, ylin, 0, rule=rule)
                        xper, yper = self.bandit._sample_one_round(cue=np.random.choice(['perodd', 'pereven']))
                        _ = self.do_composition(xper, yper, 1, rule=rule)
                        x, y = self.composed_x, self.composed_y

                # if kernel_indx == 0:
                #     xcomp = x
                #     ycomp = y
                # else:
                #     xcomp = xcomp + x
                #     ycomp = ycomp + y
                # if self.normalize:
                #     y = y-y.min()
                #     y = y/y.max()
                y = self.norm(y) if self.normalize else y

                X.append(x)
                Y.append(y)
                S.append(torch.tensor(kernel_indx).reshape(-1))

        if self.return_composition_only:
            X, Y, S = [], [], []
            X.append(self.composed_x)
            Y.append(self.norm(self.composed_y))
            S.append(torch.tensor(0).reshape(-1))
                
        Y = torch.stack(Y)
        X = torch.stack(X)
        S = torch.stack(S)

        return X, Y, S

    def evaluate_control(self):
        self.eval_control = True
        self.return_composition_only = True

    def evaluate_composition(self):
        self.eval_control = False
        self.return_composition_only = False
                # if cue == 'linear':
                #     cue = np.random.choice(['linpos', 'linneg'])
                #     x, y = self.bandit._sample_one_round(cue=cue)
                #     xlin = x
                #     ylin = y
                # elif cue == 'periodic':
                #     cue = np.random.choice(['perodd', 'pereven'])
                #     x, y = self.bandit._sample_one_round(cue=cue)
                #     xper = x
                #     yper = y   
                # if cue == 'linperiodic':  
                #     if len(block)<2:
                #         xlin, ylin = self.bandit._sample_one_round(cue='linear')
                #         xper, yper = self.bandit._sample_one_round(cue='periodic')
                #     x, y = xlin + xper, ylin + yper 
                #     y = y-y.min()
                #     y = y/y.max()

            
                #if self.normalize:
                #    Y = torch.stack(Y)/torch.stack(Y).max(1).values.mean()
                
    def do_composition(self, x, y, kernel_idx, rule='add'):

        if kernel_idx == 0:
            self.composed_x = x
            self.composed_y = y
        else:
            if rule == 'add' or rule == 'sum':
                self.composed_y =  self.composed_y + y
                self.composed_y = self.composed_y/2
                self.composed_x = self.composed_x + x

        return self.composed_x, self.composed_y

    @staticmethod
    def norm(y):    
        y = y-y.min()
        y = y/y.max()
        return y

    @staticmethod
    def prepare_data(X, Y, S, block, n_trials):
        X = np.repeat(X, n_trials, axis=0)
        Y = np.repeat(Y, n_trials, axis=0)
        S = np.repeat(S, n_trials, axis=0)
        cues = np.repeat(block, n_trials)
        return X, Y, S, cues

        
'''how to use'''

if __name__ == "__main__":

    sns.set(style='white', context='talk')

    # build a sampler
    start_arm = 0
    end_arm = 7
    ctx_dim = 2
    num_rounds = 10
    task = BlockMABs(start_arm=start_arm, end_arm=end_arm, ctx_dim=ctx_dim, num_rounds=num_rounds)

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
