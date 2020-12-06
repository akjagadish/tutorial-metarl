from copy import deepcopy
import time
import numpy as np
import torch
from torch import optim, nn
from torch.nn import functional as F
import pdb 

#from tqdm import tqdm
#from torch.utils.data import TensorDataset, DataLoader
#from torch.utils.tensorboard import SummaryWriter
from compositional_metarl.trainers.utils import EarlyStopping, set_random_seed
from compositional_metarl.model.utils import get_reward, compute_returns, compute_a2c_loss, compute_a2c_loss_multidimMAB
from compositional_metarl.model.utils import get_reward_mab, get_reward_multidimMAB, one_hot_embedding, run_agent_inference

# TODO
# 1. beta coefficient for value function is now hard coded
# 2. 

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

    # def send_info(self):
    #     self.slack_reporter.send_text("====================== Model Params {} ======================".format(self.label))
    #     for attr, val in self.model.__dict__.items():
    #         if all([char not in attr for char in ["train", "val", "test", "slack"]]) and attr[0] != "_":
    #             msg = "{} = {}".format(attr, val)
    #             self.slack_reporter.send_text(msg)

    #     self.slack_reporter.send_text("====================== trainer Params {} ======================".format(self.label))
    #     for attr, val in self.__dict__.items():
    #         if all([char not in attr for char in ["train", "val", "test", "slack"]]) and attr[0] != "_":
    #             msg = "{} = {}".format(attr, val)
    #             self.slack_reporter.send_text(msg)

class CompositionTrainer(Trainer):

    def __init__(self, *args, validation='LOSO', **kwargs):
        super().__init__(*args, **kwargs)
        
        # from dataloaders
        n_cues = self.dataloaders.num_cues
        n_rules = self.dataloaders.num_rules
        n_rounds = self.dataloaders.num_rounds
        n_arms = self.dataloaders.num_arms
        n_epochs = self.epochs
        n_trials = self.n_trials
        n_combinations = len(self.dataloaders.aoi)
        self.aoi = self.dataloaders.aoi
        self.n_rules = self.dataloaders.num_rules

        # validation model
        self.validation = validation

        #log
        self.log_cum_return = np.zeros((n_epochs, n_rules, n_rounds))
        self.log_loss_value = np.zeros((n_epochs, n_rules, n_rounds))
        self.log_loss_policy = np.zeros((n_epochs, n_rules, n_rounds))
        self.log_loss_entropy = np.zeros((n_epochs, n_rules, n_rounds))
        self.log_Y = np.zeros((n_epochs, n_rules, n_rounds, n_arms))
        self.log_Y_hat = np.zeros((n_epochs,  n_combinations, n_rounds, n_trials))
        self.log_regret = np.zeros((n_epochs, n_combinations, n_rounds, n_trials))
        self.log_return = np.zeros((n_epochs, n_combinations, n_rounds, n_trials))

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
            self.log_regret[self.current_epoch, c_indx, rnd, trial] = get_reward_mab(AOI, Y[0]) - r_t
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
        self.rnd_per_cue = torch.zeros(self.n_rules, dtype=torch.int32) 
        self.optimizer.zero_grad()
        for task_indx in range(len(bag_of_tasks)): 

            # get cue, index for cue and function 
            cue, rule_indx, fun_indx = self.sample_task(bag_of_tasks, task_indx)
            rnd = self.rnd_per_cue[rule_indx]
            # get data for this task
            X, Y = self.dataloaders.sample(end_rnd=1, cue=cue)
            AOI = self.aoi[cue]
            # ru nmodel and update parameters
            rewards, probs, values, entropys, cumulative_reward = self.train_one_episode(X, Y, AOI, fun_indx, rnd, self.n_trials)
            loss, cache = self.compute_loss(rewards, probs, values, entropys)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            losses_ += loss.item()
            losses.append(loss.item())
            loss_policy, loss_value, loss_entropy = cache
            self.log_cum_return[self.current_epoch, rule_indx, rnd] = cumulative_reward / self.n_trials
            self.log_loss_value[self.current_epoch, rule_indx, rnd] += loss_value.item() / self.n_trials
            self.log_loss_policy[self.current_epoch, rule_indx, rnd] += loss_policy.item() / self.n_trials
            self.log_loss_entropy[self.current_epoch, rule_indx, rnd] += loss_entropy.item() / self.n_trials
            self.log_Y[self.current_epoch, rule_indx] = np.squeeze(Y[0].numpy())
            self.rnd_per_cue[rule_indx] +=1

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

            # shuffle cues within epochs

            if self.dataloaders.cue_per_epoch and self.current_epoch>1:
                self.shuffle_cues_and_rules()
            bag_of_tasks = self.make_bag_of_tasks(self.dataloaders.cues, self.dataloaders.rules, self.n_rounds, shuffle=self.shuffle_task)

            loss = self.train_one_epoch(bag_of_tasks)
            self.train_losses.append(loss)
            self.mean_cum_reward = self.log_cum_return[self.current_epoch].mean()
            self.mean_cum_rewards.append(self.mean_cum_reward)
            self.lrs.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            
            time_end = time.time()
            run_time = time_end - time_start
            print_string = "Epoch {} / {}| cum. return = {:.2f} | loss: val = {:.2f}, pol = {:.2f}, entropy = {:.2f} | time = {:.2f}"
                    
            msg = print_string.format(self.current_epoch+1, self.epochs, self.log_cum_return[self.current_epoch].mean(), 
                                      self.log_loss_value[self.current_epoch].mean(), self.log_loss_policy[self.current_epoch].mean(), 
                                      self.log_loss_entropy[self.current_epoch].mean(), run_time)
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

    def make_bag_of_tasks(self, cues, rules, n_rounds, shuffle=True):

        rules = list(rules.keys())
        n_rules = len(rules)
        cues = list(cues.keys())

        if self.validation == 'LOSO':

            cue_list = []
            # 0: linpos, 1: linneg, 2: perodd, 3:pereven
            cue_list.append(list(cues[idx] for idx in [0, 1, 2])) # no pereven
            cue_list.append(list(cues[idx] for idx in [0, 2, 3])) # no linneg
            # here 2 is num_rules
            kernel, rule = np.random.choice(cue_list[0], size=(int(n_rounds/n_rules), 2)), np.repeat(rules[0], int(n_rounds/n_rules)).reshape(-1,1)
            bag_of_tasks = np.hstack((kernel, rule))
            kernel, rule = np.random.choice(cue_list[1], size=(int(n_rounds/n_rules), 2)), np.repeat(rules[1], int(n_rounds/n_rules)).reshape(-1,1)
            bag_of_tasks = np.vstack((bag_of_tasks, np.hstack((kernel, rule)))) # np.repeat(list(cues.keys()),n_rounds)

        elif self.validation == 'LOCO':
            
            ## addition
            kernel = np.random.choice(cues, size=(int(n_rounds*2), 2))
            LOKernel = [['linneg', 'perodd']] # for addition
            while len(kernel)>n_rounds:
                # find kernels not containing combinationss
                kernel = kernel[np.product(kernel == LOKernel, 1) == 0]
                LOKernel[0].reverse()
                kernel = kernel[np.product(kernel == LOKernel, 1) == 0]
                kernel = kernel[:n_rounds]
            rule = np.repeat(rules[0], int(n_rounds/n_rules)).reshape(-1,1)
            bag_of_tasks = np.hstack((kernel[:int(n_rounds/n_rules)], rule))

            ## changepoint
            kernel = np.random.choice(cues, size=(int(n_rounds*2), 2))
            LOKernel = [['linpos', 'pereven']] # for change point
            while len(kernel)>n_rounds:
                # keep kernels not containing above combination
                kernel = kernel[np.product(kernel == LOKernel, 1) == 0]
                LOKernel[0].reverse()
                # keep kernels not containing reverse combination
                kernel = kernel[np.product(kernel == LOKernel, 1) == 0]
                kernel = kernel[:n_rounds]
            rule = np.repeat(rules[1], int(n_rounds/n_rules)).reshape(-1, 1)
            bag_of_tasks = np.vstack((bag_of_tasks, np.hstack((kernel[int(n_rounds/n_rules):], rule))))

        if shuffle:
            np.random.shuffle(bag_of_tasks)
        return bag_of_tasks

    def sample_task(self, bag_of_tasks, rnd):

        cue = bag_of_tasks[rnd]
        # rule = cue[2]
        if len(self.dataloaders.rules)>1:
            if cue[2] == 'add':
                cue_indx = 0
            elif cue[2] == 'chngpnt':
                cue_indx = 1
        else:
            cue_indx = 0

        cue = '_'.join(cue)
        fun_indx = list(self.aoi.keys()).index(cue)

        return  cue, cue_indx, fun_indx
    
    def shuffle_cues_and_rules(self):
        # cues
        values = np.asarray(list(self.dataloaders.cues.values())) 
        np.random.shuffle(values)
        for indx, key in enumerate(self.dataloaders.cues.keys()):
            self.dataloaders.cues[key] = values[indx]
        
        # rules
        values = np.asarray(list(self.dataloaders.rules.values())) 
        np.random.shuffle(values)
        for indx, key in enumerate(self.dataloaders.rules.keys()):
            self.dataloaders.rules[key] = values[indx]

class MultiStateTrainer(Trainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        #self.log_Y_hat = np.zeros((n_epochs, n_cues, n_rounds, n_trials, 2))
    
    def train_one_episode(self, X, Y, AOI, c_indx, rnd, n_trials=10):
        # init
        cumulative_reward = 0
        probs, rewards, values, entropys = [], [], [], []
        h_t, c_t = self.model.get_init_states()

        # sample random action for zeroth trial
        a_t1 = torch.randint(high=self.n_arms, size=(1,))
        a_t2 = torch.randint(high=self.n_arms, size=(1,))
        r_t = Y[0][a_t1, a_t2].type(torch.FloatTensor).data.reshape(-1)
        
        for trial in range(n_trials):
            # only save memory at the end of the last trial
            self.model.turn_off_encoding()
            if trial == n_trials-1: 
                self.model.turn_on_encoding()

            # recurrent computation at time t
            x_t = X[0].view(1, 1, -1)[0][0]
            if self.model.inputs == 'context_action_reward':
                one_hot_a_t = one_hot_embedding(a_t, self.n_arms)
                x_t = torch.cat((x_t, one_hot_a_t.reshape(-1).type(torch.FloatTensor), r_t.reshape(-1).type(torch.FloatTensor)), dim=0)

            output_t, cache_t = self.model(x_t, h_t, c_t)
            a_t1, prob_a_t1, v_t1, a_t2, prob_a_t2, v_t2, h_t, c_t = output_t
            f_t, i_t, o_t, rg_t, m_t, q_t1, p_at1, q_t2, p_at2 = cache_t
            
            prob_at1 = torch.nn.functional.softmax(q_t1.squeeze())
            prob_at2 = torch.nn.functional.softmax(q_t2.squeeze())
            entropy = -(torch.sum(torch.log(prob_at1)*prob_at1)+torch.sum(torch.log(prob_at2)*prob_at2))

            # compute immediate reward
            r_t = get_reward_multidimMAB([a_t1, a_t2], Y[0])
            
            # log
            prob_a_t = [prob_a_t1, prob_a_t2]
            v_t = v_t1 + v_t2
            probs.append(prob_a_t)
            rewards.append(r_t)
            entropys.append(entropy)
            values.append(v_t) 
            cumulative_reward += r_t
            #self.log_Y_hat[self.current_epoch, c_indx, rnd, trial, 0] = a_t1.item()
            #self.log_Y_hat[self.current_epoch, c_indx, rnd, trial, 1] = a_t2.item()
            #self.log_regret[self.current_epoch, c_indx, rnd, trial] = get_reward_multidimMAB(AOI, Y[0]) - r_t
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
            #self.log_Y[self.current_epoch, c_indx] = np.squeeze(Y[0].numpy())
            self.rnd_per_cue[c_indx] +=1

        return losses_ # return the cumulative loss across tasks in the epoch

    def compute_loss(self, rewards, probs, values, entropys):

        returns = compute_returns(rewards, gamma=self.gamma, normalize=self.normalize_return)
        loss_policy, loss_value = compute_a2c_loss_multidimMAB(probs, values, returns)
        loss_entropy = torch.stack(entropys).sum()
        cache = [loss_policy, loss_value, loss_entropy]

        if self.loss == 'policy_value_entropy':
             return loss_policy + loss_value - self.beta*loss_entropy, cache
        elif self.loss == 'policy_value':
             return loss_policy + loss_value, cache
        elif self.loss == 'policy_only':
             return loss_policy, cache
        elif self.loss == 'value_only':
             return loss_value, cache
    
    def sample_task(self, bag_of_tasks, rnd):
        cue = bag_of_tasks[rnd]
        if cue == 'linear-random':
            cue_indx = 0
        elif cue == 'random-periodic':
            cue_indx = 1
        else:
            cue_indx = 2

        return cue, cue_indx

class BlockTrainer(Trainer):

    def __init__(self, *args, block_structure=True, **kwargs):
        super().__init__(*args, **kwargs)
        
        # from dataloaders
        n_cues = self.dataloaders.num_cues
        n_blocks = self.dataloaders.num_blocks
        n_rounds = self.dataloaders.num_rounds
        n_arms = self.dataloaders.num_arms
        n_epochs = self.epochs
        n_trials = self.n_trials
        self.aoi = self.dataloaders.aoi
        self.n_blocks = self.dataloaders.num_blocks
        self.block_structure = block_structure

        #log
        self.log_cum_return = np.zeros((n_epochs, n_cues, n_blocks))
        self.log_loss_value = np.zeros((n_epochs, n_blocks))
        self.log_loss_policy = np.zeros((n_epochs,  n_blocks))
        self.log_loss_entropy = np.zeros((n_epochs, n_blocks))
        self.log_Y = np.zeros((n_epochs, n_cues, n_blocks, n_arms))
        self.log_Y_hat = np.zeros((n_epochs,  n_cues, n_blocks, n_trials))
        self.log_regret = np.zeros((n_epochs, n_cues, n_blocks, n_trials))
        self.log_return = np.zeros((n_epochs, n_cues, n_blocks, n_trials))

    def train_one_episode(self, X, Y, S, cues, n_trials=10):
    
        # init
        cumulative_reward = 0.
        probs, rewards, values, entropys = [], [], [], []
        h_t, c_t = self.model.get_init_states()

        # sample random action for zeroth trial
        a_t = None # torch.randint(high=self.n_arms, size=(1,))
        r_t = Y[0].mean() #Y[0][a_t].type(torch.FloatTensor).data.reshape(-1)
        
        for trial in range(n_trials):

            # only save memory at the end of the last trial
            self.model.turn_off_encoding()
            if trial%self.n_trials == self.n_trials-1: 
                self.model.turn_on_encoding()

            # cue indx
            c_indx, AOI = self.get_info(cues[trial])    
            rnd = int(self.round_counter[c_indx])

            x_t =  self._input(X[trial], S[trial], a_t, r_t, trial%self.n_trials)
            output_t, cache_t = self.model(x_t, h_t, c_t)
            a_t, prob_a_t, v_t, h_t, c_t = output_t
            f_t, i_t, o_t, rg_t, m_t, q_t, p_at = cache_t
            
            prob_arms = torch.nn.functional.softmax(q_t.squeeze(), dim=0)
            entropy = -torch.sum(torch.log(prob_arms)*prob_arms)

            # compute immediate reward
            r_t = get_reward_mab(a_t, Y[trial])
            
            # log
            probs.append(prob_a_t)
            rewards.append(r_t)
            entropys.append(entropy)
            values.append(v_t) 
            cumulative_reward += r_t

            self.log_Y_hat[self.current_epoch,  c_indx, rnd, trial%self.n_trials] = a_t.item()
            self.log_regret[self.current_epoch, c_indx, rnd, trial%self.n_trials] = Y[trial].max() - r_t
            self.log_return[self.current_epoch, c_indx, rnd, trial%self.n_trials] = r_t

            if (trial+1)%self.n_trials==0:
                self.log_cum_return[self.current_epoch, c_indx, rnd] = cumulative_reward / self.n_trials
                self.log_Y[self.current_epoch, c_indx, rnd] = np.squeeze(Y[trial].numpy())
                cumulative_reward = 0.
                # h_t, c_t = self.model.get_init_states()
            

        return rewards, probs, values, entropys, cumulative_reward

    def train_one_block(self, X, Y, S, block, block_indx):

        losses = []
        losses_ = 0.
        self.block_size = len(block)

        # create block structure
        X, Y, S, cues = self.dataloaders.prepare_data(X, Y, S, block, self.n_trials)
        num_trials = self.n_trials*self.block_size
        # X = np.repeat(X, self.n_trials, axis=0)
        # Y = np.repeat(Y, self.n_trials, axis=0)
        # S = np.repeat(S, self.n_trials, axis=0)
        # cues = np.repeat(block, self.n_trials)

        # train one episode
        rewards, probs, values, entropys, cumulative_reward = self.train_one_episode(X, Y, S, cues, num_trials)
        loss, cache = self.compute_loss(rewards, probs, values, entropys)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        losses_ += loss.item()
        losses.append(loss.item())
        loss_policy, loss_value, loss_entropy = cache
        self.log_loss_value[self.current_epoch, block_indx] += loss_value.item() / self.n_trials
        self.log_loss_policy[self.current_epoch, block_indx] += loss_policy.item() / self.n_trials
        self.log_loss_entropy[self.current_epoch, block_indx] += loss_entropy.item() / self.n_trials
        self.round_counter += 1

        return losses_
        # for block_indx, cue in enumerate(block):

        #     # cue indx
        #     cue_indx, aoi = self.get_info(cue, block)    
        #     rnd = self.round_counter[cue_indx]
        #     # train one episode
        #     rewards, probs, values, entropys, cumulative_reward = self.train_one_episode(X[block_indx], Y[block_indx], S[block_indx], cue_indx, rnd, aoi, self.n_trials)
        #     loss, cache = self.compute_loss(rewards, probs, values, entropys)

        #     loss.backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()
            
        #     losses_ += loss.item()
        #     losses.append(loss.item())
        #     loss_policy, loss_value, loss_entropy = cache
        #     self.log_cum_return[self.current_epoch, cue_indx, rnd] = cumulative_reward / self.n_trials
        #     self.log_loss_value[self.current_epoch, cue_indx, rnd] += loss_value.item() / self.n_trials
        #     self.log_loss_policy[self.current_epoch, cue_indx, rnd] += loss_policy.item() / self.n_trials
        #     self.log_loss_entropy[self.current_epoch, cue_indx, rnd] += loss_entropy.item() / self.n_trials
        #     self.log_Y[self.current_epoch, cue_indx] = np.squeeze(Y[block_indx].numpy())
        #     self.round_counter[cue_indx] += 1
        
    def train_one_epoch(self, bag_of_tasks):
        """Training function for a single epoch.

        Returns:
            float: Loss (of the first iteration).

        """

        self.model.train()
        self.round_counter = torch.zeros(self.n_cues, dtype=torch.int32) 
        
        self.optimizer.zero_grad()
        for block_indx in range(len(bag_of_tasks)): 

            # get block
            block = bag_of_tasks[block_indx]
            # get data for this task
            X, Y, S = self.dataloaders.sample(end_rnd=1, block=block)
            # run nmodel and update parameters
            losses_ = self.train_one_block(X, Y, S, block, block_indx)
            

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

            # shuffle cues within epochs
            if self.dataloaders.cue_per_epoch and self.current_epoch>1 and len(self.dataloaders.cues.keys())>1:
                self.shuffle_cues()
                #print('cue is shuffled')
                
            # make bag of tasks
            bag_of_tasks = self.make_bag_of_tasks(self.dataloaders.cues, self.dataloaders.num_blocks, self.n_rounds, block_structure=self.block_structure)

            loss = self.train_one_epoch(bag_of_tasks)
            self.train_losses.append(loss)
            self.mean_cum_reward = self.log_cum_return[self.current_epoch].mean()
            self.mean_cum_rewards.append(self.mean_cum_reward)
            self.lrs.append(self.optimizer.state_dict()['param_groups'][0]['lr'])
            
            time_end = time.time()
            run_time = time_end - time_start
            print_string = "Epoch {} / {}| cum. return = {:.2f} | loss: val = {:.2f}, pol = {:.2f}, entropy = {:.2f} | time = {:.2f}"
                    
            msg = print_string.format(self.current_epoch+1, self.epochs, self.log_cum_return[self.current_epoch].mean(), 
                                      self.log_loss_value[self.current_epoch].mean(), self.log_loss_policy[self.current_epoch].mean(), 
                                      self.log_loss_entropy[self.current_epoch].mean(), run_time)
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

    def make_bag_of_tasks(self, cues, n_blocks, n_rounds, block_structure=True, shuffle=True):

        if block_structure: 
            bag_of_tasks = np.repeat(np.asarray(list(cues.keys())).reshape(1,-1), n_blocks, axis=0)
        else:
            bag_of_tasks = np.repeat(list(cues.keys()), n_rounds) 

        if shuffle:
            np.random.shuffle(bag_of_tasks)
        return bag_of_tasks

    def get_info(self, cue):

        aoi = self.aoi[cue]
        if self.block_size == 1:
            cue_indx = 0
        else:
            if cue == 'linear' or cue == 'linpos' or cue == 'linneg':
                cue_indx = 0
            elif cue == 'periodic' or cue == 'perodd' or cue == 'pereven':
                cue_indx = 1
            else:
                cue_indx = 2

        return cue_indx, aoi
    
        
    def shuffle_cues(self):
        # cues
        values = np.asarray(list(self.dataloaders.bandit.cues.values())[:2]) 
        np.random.shuffle(values)
        for indx, key in enumerate(list(self.dataloaders.bandit.cues.keys())[:2]):
            self.dataloaders.cues[key] = values[indx]

    def _input(self, X, S, a_t, r_t, t):
        x_t = X
        one_hot_a_t = one_hot_embedding(a_t, self.n_arms)
        s_t = S
        if self.model.inputs == 'context_action_reward':
            x_t = torch.cat((x_t.type(torch.FloatTensor), one_hot_a_t.reshape(-1).type(torch.FloatTensor),
                             r_t.reshape(-1).type(torch.FloatTensor), torch.tensor(t).reshape(-1).type(torch.FloatTensor)), dim=0)
        elif self.model.inputs  == 'context_block_action_reward':
            x_t = torch.cat((x_t.type(torch.FloatTensor), s_t.reshape(-1).type(torch.FloatTensor), one_hot_a_t.reshape(-1).type(torch.FloatTensor),
                             r_t.reshape(-1).type(torch.FloatTensor), torch.tensor(t).reshape(-1).type(torch.FloatTensor)), dim=0)

        return x_t
