from copy import deepcopy
import time
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import pdb 
from tutorial_metarl.trainers.utils import EarlyStopping, set_random_seed
from tutorial_metarl.models.utils import get_reward, compute_returns, compute_a2c_loss, compute_a2c_loss_multidimMAB
from tutorial_metarl.models.utils import get_reward_mab, get_reward_multidimMAB, one_hot_embedding, run_agent_inference

class Trainer:

    def __init__(self, model, dataloaders, seed, lr= 5e-4, beta=1.0, gamma=0.7, n_trials=10,
                 n_epochs=100, loss='policy_value_entropy', normalize_return=True, early_stopping=True, shuffle_task=True, slack_reporter=False,
                 label=None, device=torch.device('cuda'), infer_composition=False, simulated_annealing=False, tensorboard_writer=None, patience=100, **kwargs):
        
        self.model = model
        self.dataloaders = dataloaders
        self.loss = loss
        self.seed = seed 
        self.lr = lr
        self.epochs = int(n_epochs)
        self.device = device
        self.gamma = gamma
        self.beta = beta
        self.normalize_return = normalize_return
        self.infer_composition = infer_composition
        self.simulated_annealing = simulated_annealing
        self.shuffle_task = shuffle_task
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)    
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max', factor=.3, patience=1000, min_lr=1e-8, verbose=True, threshold_mode='abs')
        self.early_stopping = EarlyStopping(mode='max', patience=20, verbose=True) if early_stopping else None
        self.tensorboard_writer = tensorboard_writer
        self.train_losses = []

        # from dataloaders
        n_cues = dataloaders.num_cues
        n_rounds = dataloaders.num_rounds
        n_arms = dataloaders.num_arms
        aoi = dataloaders.aoi

        #log
        self.log_cum_return = np.zeros((n_epochs, n_cues, n_rounds))
        self.log_regret = np.zeros((n_epochs, n_cues, n_rounds, n_trials))
        self.log_return = np.zeros((n_epochs, n_cues, n_rounds, n_trials))
        self.log_loss_value = np.zeros((n_epochs, n_cues, n_rounds))
        self.log_loss_policy = np.zeros((n_epochs, n_cues, n_rounds))
        self.log_loss_entropy = np.zeros((n_epochs, n_cues, n_rounds))
        self.log_Y = np.zeros((n_epochs, n_cues, n_rounds, n_arms))
        self.log_Y_hat = np.zeros((n_epochs, n_cues, n_rounds, n_trials))
        self.n_trials = n_trials
        self.n_rounds = n_rounds
        self.n_cues = n_cues
        self.n_arms = n_arms
        self.aoi = aoi
        self.lrs = []
        self.label = "({})".format(label) if label else ""
        self.current_epoch = 0

    def train_one_episode(self, X, Y, AOI, c_indx, rnd, n_trials=10):
    
        # init
        cumulative_reward = 0
        probs, rewards, values, entropys = [], [], [], []
        h_t, c_t = self.model.get_init_states()

        # sample random action for zeroth trial
        a_t = None # torch.randint(high=self.n_arms, size=(1,))
        r_t = Y[0].mean() #Y[0][a_t].type(torch.FloatTensor).data.reshape(-1)
    
        for trial in range(n_trials):
            # only save memory at the end of the last trial
            self.model.turn_off_encoding()
            if trial == n_trials-1: 
                self.model.turn_on_encoding()

            x_t =  self._input(X, a_t, r_t, trial)
            output_t, cache_t = self.model(x_t, h_t, c_t)
            a_t, prob_a_t, v_t, h_t, c_t = output_t
            f_t, i_t, o_t, rg_t, m_t, q_t, p_at = cache_t
            
            prob_at = torch.nn.functional.softmax(q_t.squeeze(), dim=0)
            entropy = -torch.sum(torch.log(prob_at)*prob_at)

            # compute immediate reward
            r_t = get_reward_mab(a_t, Y[0])
            
            # log
            probs.append(prob_a_t)
            rewards.append(r_t)
            entropys.append(entropy)
            values.append(v_t) 
            cumulative_reward += r_t
            self.log_Y_hat[self.current_epoch, c_indx, rnd, trial] = a_t.item()
            self.log_regret[self.current_epoch, c_indx, rnd, trial] = Y[0].max() - r_t #get_reward_mab(AOI, Y[0]) - r_t
            self.log_return[self.current_epoch, c_indx, rnd, trial] = r_t

        return rewards, probs, values, entropys, cumulative_reward
    
    def train_one_epoch(self, bag_of_tasks):
        """Training function for a single epoch.

        Returns:
            float: Loss (of the first iteration).

        """
        self.model.train()
        losses = []
        losses_ = 0.
        self.rnd_per_cue = torch.zeros(self.n_cues,dtype=torch.int32) 
        self.optimizer.zero_grad()

        for task_indx in range(self.n_rounds*self.n_cues):
            cue, c_indx = self.sample_task(bag_of_tasks, task_indx)
            rnd = self.rnd_per_cue[c_indx]
            # get data for this task
            X, Y = self.dataloaders.sample(end_rnd=1, cue=cue)
            AOI = self.aoi[cue]
            rewards, probs, values, entropys, cumulative_reward = self.train_one_episode(X, Y, AOI, c_indx, rnd, self.n_trials)
            loss, cache = self.compute_loss(rewards, probs, values, entropys)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            losses_ += loss.item()
            losses.append(loss.item())
            loss_policy, loss_value, loss_entropy = cache
            self.log_cum_return[self.current_epoch, c_indx, rnd] = cumulative_reward / self.n_trials
            self.log_loss_value[self.current_epoch, c_indx, rnd] += loss_value.item() / self.n_trials
            self.log_loss_policy[self.current_epoch, c_indx, rnd] += loss_policy.item() / self.n_trials
            self.log_loss_entropy[self.current_epoch, c_indx, rnd] += loss_entropy.item() / self.n_trials
            self.log_Y[self.current_epoch, c_indx] = np.squeeze(Y[0].numpy())
            self.rnd_per_cue[c_indx] +=1

        return losses_ # return the cumulative loss across tasks in the epoch
    
    def train(self):
        # set the manual seed before running the epochs
        set_random_seed(self.seed)

        # establish a baseline regret
        with torch.no_grad():
            mean_cum_reward  = 0.
        if self.early_stopping:
            self.early_stopping(mean_cum_reward, self.model)
        
        old_lr = self.lr
        lrs_counter = 0
        self.mean_cum_rewards = []
        for self.current_epoch in range(self.epochs):
            
            current_lr = self.optimizer.param_groups[0]['lr']
            if current_lr - old_lr:
                self.retrive_best_model()
                old_lr = current_lr
            
                if hasattr(self.model, 'freeze_training') and (lrs_counter == 0):
                    print("Freezing training for params specified in the model..", flush=True)
                    self.model.freeze_training()
                lrs_counter += 1

            time_start = time.time()
            self.model.reset_memory()
            self.model.turn_on_retrieval()
            # shuffle cues every epoch
            if self.dataloaders.cue_per_epoch:
                self.shuffle_cues()
            bag_of_tasks = self.make_bag_of_tasks(self.dataloaders.cues, self.n_rounds, self.n_cues, shuffle=self.shuffle_task)
            loss = self.train_one_epoch(bag_of_tasks)
            self.train_losses.append(loss)
            self.mean_cum_reward = self.log_cum_return[self.current_epoch,:,:].mean()
            self.mean_cum_rewards.append(self.mean_cum_reward)
            self.lrs.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            
            time_end = time.time()
            run_time = time_end - time_start
            print_string = "Epoch {} / {}| cum. return = {:.2f} | loss: val = {:.2f}, pol = {:.2f}, entropy = {:.2f} | time = {:.2f}"
                    
            msg = print_string.format(self.current_epoch+1, self.epochs, self.log_cum_return[self.current_epoch, :, :].mean(),
                                     self.log_loss_value[self.current_epoch, :, :].mean(), self.log_loss_policy[self.current_epoch, :, :].mean(), 
                                     self.log_loss_entropy[self.current_epoch, :, :].mean(), run_time)
            print(msg, flush=True)

            if self.early_stopping:
                self.early_stopping(self.mean_cum_reward, self.model)
                if self.early_stopping.early_stop:
                    msg = "{} Early stopping at epoch {}. Best mean cum. reward : {:.3f}".format(self.label,
                                                                                    self.current_epoch, 
                                                                                    np.array(self.mean_cum_rewards).max())
                    print(msg, flush=True)
                    break

            self.scheduler.step(self.mean_cum_reward)

            # Tensoboard logging
            if self.tensorboard_writer:
                # self.tensorboard_writer.add_scalar('Train_loss', train_loss, self.current_epoch+1)
                # self.tensorboard_writer.flush()
                pass
                
        if self.early_stopping:
            self.retrive_best_model()

        return (self.model,
                np.array(self.train_losses),
                np.array(self.mean_cum_rewards),
                self.lrs)

    def retrive_best_model(self):
        if self.early_stopping:
            if self.early_stopping.best_model_state_dict is not None:
                print("Retrieve best model..", flush=True)
                self.model.load_state_dict(deepcopy(self.early_stopping.best_model_state_dict))
            else:
                print("Keep existing model..", flush=True)

    def compute_loss(self, rewards, probs, values, entropys):

        returns = compute_returns(rewards, gamma=self.gamma, normalize=self.normalize_return)
        loss_policy, loss_value = compute_a2c_loss(probs, values, returns)
        loss_entropy = torch.stack(entropys).sum()
        cache = [loss_policy, loss_value, loss_entropy]
        beta = self.beta
        if self.simulated_annealing:
            beta = torch.exp(-self.beta*self.current_epoch/self.epochs)
        if self.loss == 'policy_value_entropy':
            return loss_policy + 0.5*loss_value - beta*loss_entropy, cache
        elif self.loss == 'policy_value':
            return loss_policy + loss_value, cache
        elif self.loss == 'policy_only':
            return loss_policy, cache
        elif self.loss == 'value_only':
            return loss_value, cache
        elif self.loss == 'policy_entropy':
            return loss_policy - beta*loss_entropy, cache
    
    def make_bag_of_tasks(self, cues, n_rounds, n_cues, shuffle=True):
        bag_of_tasks = np.repeat(list(cues.keys()),n_rounds)
        if shuffle:
            bag_of_tasks = np.random.choice(bag_of_tasks, size=(n_rounds*n_cues), replace=False)
        return bag_of_tasks


    def sample_task(self, bag_of_tasks, rnd):
        cue = bag_of_tasks[rnd]
        # if number of unique rounds = 1 then set to 0 otherwise assign based on cues
        if len(np.unique(np.array(list(bag_of_tasks))))==1:
            return cue, 0

        if cue == 'linear' or cue == 'linpos':
            cue_indx = 0
        elif cue == 'periodic' or cue == 'pereven':
            cue_indx = 1
        elif cue == 'linneg':
            cue_indx = 2
        elif cue == 'perodd':
            cue_indx = 3

        return cue, cue_indx

    def _input(self, X, a_t, r_t, t):
        x_t = X[0].view(1, 1, -1)[0][0]
        if self.model.inputs == 'context_action_reward':
            one_hot_a_t = one_hot_embedding(a_t, self.n_arms)
            x_t = torch.cat((x_t.type(torch.FloatTensor), one_hot_a_t.reshape(-1).type(torch.FloatTensor),
                             r_t.reshape(-1).type(torch.FloatTensor), torch.tensor(t).reshape(-1).type(torch.FloatTensor)), dim=0)
        return x_t

    def run_agent_inference(self, Y):
    
        # run model for n_runs
        self.model.eval()
        self.model.turn_off_encoding()
        h_t, c_t = self.model.get_init_states()

        # sample random action for zeroth trial
        a_t = torch.randint(high=self.n_arms, size=(1,))
        r_t = Y[0][a_t].type(torch.FloatTensor).data.reshape(-1)

        h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)  
        actions, cum_reward, rewards= [], 0., []
        for t in range(self.dataloaders.n_trials):
            output_t, _ = self.model(x_t, h_t, c_t) 
            a_t, _, _, _, _ = output_t
            r_t = get_reward_mab(a_t, Y)
            cum_reward += r_t
            actions.append(a_t)
            rewards.append(r_t)
        self.model.train()

        return actions, rewards

    def shuffle_cues(self):
        values = np.asarray(list(self.dataloaders.cues.values())) #[::-1]
        np.random.shuffle(values)
        for indx, key in enumerate(self.dataloaders.cues.keys()):
            self.dataloaders.cues[key] = values[indx]
