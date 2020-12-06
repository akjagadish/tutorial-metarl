import torch
import numpy as np
import  math 
from sklearn.metrics.pairwise import linear_kernel


class MultiArmedBandit():

    def __init__(self, cues=None, start_arm=0, end_arm=7, ctx_dim=2, num_rounds=10, normalize=True, best_arms=None, noise_per_arm=False, cue_per_epoch=False):
        
        default_cues = {'linear': [1., 0.], 'periodic': [0., 1.], 'linperiodic': [1., 1.]}
        default_arms = {'linear': end_arm, 'periodic': end_arm-1, 'linperiodic': end_arm-1} 
        self.cues =  default_cues if cues is None else cues
        self.aoi = default_arms if best_arms is None else best_arms
        self.num_cues  = len(self.cues)
        self.ctx_dim = ctx_dim
        self.num_rounds = num_rounds
        self.start_arm = start_arm
        self.end_arm = end_arm 
        self.num_arms = (end_arm - start_arm) + 1
        self.noise_per_arm = noise_per_arm
        self.cue_per_epoch = cue_per_epoch

        # 2nd level params
        self.x_dim = self.ctx_dim + 1
        self.y_dim = self.num_arms
        self.normalize=normalize
        
    def sample(self, start_rnd=0, end_rnd=None, cue='linear', data=None, noise_var=2.0):
       
        if end_rnd is None:
            end_rnd = start_rnd + self.num_rounds

        X, Y = [], []
        for rnd in np.arange(start_rnd, end_rnd):
            # perform additive composition based on past data
            if bool(data) and (cue == 'linperiodic' or cue =='combined'):
                pass_data = []
                pass_data.append(data['linear', rnd][1])
                pass_data.append(data['periodic', rnd][1])           
            else:         
                pass_data = None

            # generate per round sample 
            x, y = self._sample_one_round(cue=cue, data=pass_data, noise_var=noise_var)
            X.append(x)
            Y.append(y)

        Y = torch.stack(Y)/torch.stack(Y).max(axis=1, keepdims=True).values #.mean()
        X = torch.stack(X)
        return X, Y
            
    def _sample_one_round(self, cue='linear', data=None, slope=5.0, mean=4., offset=25., #mean=5.0, offset=15.0
                          freq=0.25, amp=10., phase=1.0, noise_var=2., noise_slope=10.0): #amp=5.0

        # arms to generate rewards 
        x =  torch.linspace(self.start_arm, self.end_arm, self.num_arms)
        if cue in self.cues:
            ctx = self.cues[cue]
        elif cue == 'linpos' or cue == 'linneg':
            ctx = self.cues['linear']
        elif cue == 'perodd' or cue == 'pereven':
            ctx = self.cues['periodic']
           

        if self.noise_per_arm:
            noise = torch.randn((self.num_arms,)) * np.sqrt(noise_var)
        else:
            noise = torch.randn(1) * np.sqrt(noise_var)

        # generate rewards
        if cue == 'periodic' or cue == 'pereven':
            phases =  [1.] #### [1.] 
            phase = np.random.choice(phases)
            y =  amp*torch.abs(torch.sin((x-phase) * (2*np.pi*freq))) + noise

        
        elif cue == 'perodd':
            phases = [0.] 
            phase = np.random.choice(phases)
            y =  amp*torch.abs(torch.sin((x-phase) * (2*np.pi*freq))) + noise
            
        elif cue == 'linear' or cue == 'linpos':
        
            mean = self.end_arm
            K1 = linear_kernel(x.reshape(-1,1)-mean, x.reshape(-1,1)-mean)
            mu = x*2.5  # torch.linspace(0, self.end_arm, self.num_arms) 
            y = torch.as_tensor(np.random.multivariate_normal(mu, K1, size=1).reshape(1,-1)).squeeze() + noise + offset

        elif cue == 'linneg':

            mean = self.end_arm
            K1 = linear_kernel(x.reshape(-1,1)-mean, x.reshape(-1,1)-mean)
            mu = -x*2.5  #torch.linspace(0, self.end_arm, self.num_arms) 
            y = torch.as_tensor(np.random.multivariate_normal(mu, K1, size=1).reshape(1,-1)).squeeze() + noise + offset

        elif cue == 'linperiodic' or cue == 'combined':
            if data is None:
                y =  (torch.abs(amp*torch.sin((x-phase)) * (2*np.pi*freq)) + offset + (x-mean)*torch.abs(slope + np.sqrt(noise_slope)*torch.rand((1))) + noise)/2 
            else:
                y = data[0] + data[1]
                
        x = torch.as_tensor(ctx)
        if self.normalize:
            y = y-y.min()
            y = y/y.max()

        return x.type(torch.FloatTensor), y.type(torch.FloatTensor)
    
'''how to use'''
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style='white', context='talk')

    # build a sampler
    start_arm = 0
    end_arm = 7
    ctx_dim = 2
    num_rounds = 10
    task = MultiArmedBandit(start_arm=start_arm, end_arm=end_arm, ctx_dim=ctx_dim, num_rounds=num_rounds)

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
