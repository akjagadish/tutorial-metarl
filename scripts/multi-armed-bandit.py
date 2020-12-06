# TODO
# [] save model per seeds

import time
import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from compositional_metarl.task import MultiArmedBandit
from compositional_metarl.model import QDNDLSTM as Agent
from compositional_metarl.utils import compute_stats, to_sqnp
from compositional_metarl.model.DND import compute_similarities
from compositional_metarl.model.utils import get_reward, compute_returns, compute_a2c_loss, get_reward_mab, run_agent_inference
import matplotlib as mpl
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import torch.nn.functional as F

sns.set(style='white', context='talk', palette='colorblind')


#seed_val = 0 # or 100
seeds = [1344, 3536, 2489]#, 7648]
FACTORS = [32, 64, 128] #, 128, 256]
n_seeds = len(seeds)

'''init task'''

start_arm = 0
end_arm = 7
ctx_dim = 2
n_arms = (end_arm - start_arm) + 1
n_rounds = 30
n_trials = 10

'''define cues '''

CUES =  {'linperiodic': torch.tensor([1.0, 1.0])} #'periodic': torch.tensor([1.0, 0.0])} #'linear': torch.tensor([0.0, 1.0])} #, 'periodic': torch.tensor([1.0, 0.0])} #, 'linperiodic': torch.tensor([1.0, 1.0])} 
aoi = {'linear': 7, 'periodic': 6, 'linperiodic': 6} #np.random.choice([0, 2, 4, 6])
n_cues = len(CUES)
mode = 'training'


'''init model and trainer'''

# DNDLSTM params
dim_hidden = 16 
inp_dim = 2 # 4 
dim_output = n_arms
estimate_Qvals = True
dict_len = 100
kernel = 'cosine' # '1NN'
dnd_policy = 'norm' # 'softmax'
unique_keys = True
exclude_key = True
gamma = 0. #0.8
normalize_return = True

# training parameters
learning_rate = 5e-4
n_epochs = 20

# log
log_return = np.zeros((n_seeds, n_epochs, n_cues, n_rounds))
log_loss_value = np.zeros((n_seeds, n_epochs, n_cues, n_rounds))
log_loss_policy = np.zeros((n_seeds, n_epochs, n_cues, n_rounds))
log_Y = np.zeros((n_seeds, n_epochs, n_cues, n_rounds, n_arms))
log_Y_hat = np.zeros((n_seeds, n_epochs, n_cues, n_rounds, n_trials))
log_regret = np.zeros((n_seeds, n_epochs, n_cues, n_rounds, n_trials))
log_loss_entropy = np.zeros((n_seeds, n_epochs, n_cues, n_rounds))

'''train'''
for dim_hidden in FACTORS:

    for seed_indx, seed_val in enumerate(seeds):
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

        # init task, agent and optimizer
        task = MultiArmedBandit(start_arm=start_arm, end_arm=end_arm, ctx_dim=ctx_dim, num_rounds=n_rounds)
        agent = Agent(inp_dim, dim_hidden, dim_output, dict_len, kernel=kernel, dnd_policy=dnd_policy, unique_keys=unique_keys, q_est=estimate_Qvals)
        optimizer = torch.optim.Adam(agent.parameters(), lr=learning_rate)

        # loop over epoch
        for i in range(n_epochs):
            time_start = time.time()
            # flush hippocampus which is context and representation
            agent.reset_memory()
            agent.turn_on_retrieval()

            for c_indx, cue in enumerate(CUES):
                # get data for this task
                X, Y = task.sample(end_rnd=n_rounds, cue=cue)
                
                # try different rounds of task
                for m in range(n_rounds): 
                    # prealloc
                    cumulative_reward = 0
                    probs, rewards, values, entropys = [], [], [], []
                    h_t, c_t = agent.get_init_states()
                    a_t = torch.randint(high=dim_output,size=(1,))
                    r_t = Y[m][a_t].type(torch.FloatTensor).data.reshape(-1)
                    
                    if cue == 'linperiodic' and mode == 'inference':
                        agent.eval()
                        actions, cumulative_reward = run_agent_inference(agent, h_t, c_t, X[m].view(1, 1, -1)[0][0], Y[m], n_trials)
                        log_Y_hat[i, c_indx, m, :] = actions
                        log_return[i, c_indx, m] = cumulative_reward / n_trials
                        agent.train()
                        continue
                    
                    # loop over time, for one training example
                    for t in range(n_trials):
                        
                        # only save memory at the end of the last trial
                        agent.turn_off_encoding()
                        if t == n_trials-1: # and cue == 'linear':
                            agent.turn_on_encoding()

                        # recurrent computation at time t
                        x_t = X[m].view(1, 1, -1)[0][0]
                        if inp_dim==4:
                            x_t = torch.cat((x_t, a_t.reshape(-1).type(torch.FloatTensor), r_t.reshape(-1).type(torch.FloatTensor)), dim=0)
                        output_t, cache_t = agent(x_t, h_t, c_t)
                        a_t, prob_a_t, v_t, h_t, c_t = output_t
                        f_t, i_t, o_t, rg_t, m_t, q_t = cache_t
                        
                        # compute immediate reward
                        r_t = get_reward_mab(a_t, Y[m])

                        #compute entropy
                        prob_t = torch.nn.functional.softmax(q_t.squeeze())
                        entropy = -torch.sum(torch.log(prob_t)*prob_t)

                        # log
                        probs.append(prob_a_t)
                        rewards.append(r_t)
                        values.append(v_t)
                        entropys.append(entropy)
                        cumulative_reward += r_t
                        log_Y_hat[seed_indx, i, c_indx, m, t] = a_t.item()
                        log_regret[seed_indx, i, c_indx, m, t] = get_reward_mab(aoi[cue], Y[m]) - r_t

                    returns = compute_returns(rewards, gamma=gamma, normalize=normalize_return)
                    loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
                    loss_entropy = torch.stack(entropys).sum()
                    loss = loss_policy + loss_value - 1.*loss_entropy
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # log
                    log_Y[seed_indx, i, c_indx] = np.squeeze(Y[m].numpy())
                    log_return[seed_indx, i, c_indx, m] = cumulative_reward / n_trials
                    log_loss_value[seed_indx, i, c_indx, m] += loss_value.item() / n_trials
                    log_loss_policy[seed_indx, i, c_indx, m] += loss_policy.item() / n_trials
                    log_loss_entropy[seed_indx, i, c_indx, m] += loss_entropy.item() / n_trials

                # print out some stuff
                time_end = time.time()
                run_time = time_end - time_start
                print(
                    'Seed %2d | Epoch %3d | return = %.2f | loss: val = %.2f, pol = %.2f, entropy = %.2f | time = %.2f' %
                    (seed_indx, i, log_return[seed_indx, i, c_indx, :].mean(), log_loss_value[seed_indx, i, c_indx, m], log_loss_policy[seed_indx, i, c_indx, m], log_loss_entropy[seed_indx, i, c_indx, m], run_time)
                )
    
    '''' save result '''
    loss =  'pol_val_entrp'
    file_name = 'lp_nseeds{}_inpdim{}_hiddim{}kernel_{}policy_{}_mode{}_loss{}'.format(n_seeds, inp_dim, dim_hidden, kernel, dnd_policy, mode, loss)
    np.savez(file_name, n_rounds=n_rounds, n_epochs=n_epochs, n_trials=n_trials, log_Y=log_Y, log_Y_hat=log_Y_hat, log_return=log_return,log_loss_value=log_loss_value, log_loss_policy=log_loss_policy)