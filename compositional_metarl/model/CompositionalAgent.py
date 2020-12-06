"""
A DND-based LSTM based on ...
Ritter, et al. (2018).
Been There, Done That: Meta-Learning with Episodic Recall.
Proceedings of the International Conference on Machine Learning (ICML).
"""
import torch
import torch.nn as nn
from compositional_metarl.model.DND import DND
from compositional_metarl.model.A2C import A2C_linear, A2C
from compositional_metarl.trainers.utils import set_random_seed
#import ipdb 

# constants
N_GATES = 5

class CompositionalAgent(nn.Module):

    def __init__(self, hidden_dim, output_dim,
            dict_len, input_dim=None, function_dim=6, rule_dim=1, inputs='context_action_reward', bias=True, 
            function_kernel='cosine', rule_kernel='consine', 
            function_dnd_policy='softmax', rule_dnd_policy='softmax', 
            unique_keys=False, q_est=False, seed=0):
        super(CompositionalAgent, self).__init__() 
        
        if input_dim is None:
            if inputs == 'context_action_reward':
                input_dim = function_dim + rule_dim + output_dim + 1 # context + num_arms + reward
            # elif inputs == 'context': 
            #     input_dim 
        self.input_dim = input_dim
        self.inputs = inputs 
        self.function_dim = function_dim
        self.rule_dim = rule_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        # set the manual seed 
        set_random_seed(seed)
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # kernel dnd
        self.function_dnd = DND(dict_len, hidden_dim, function_kernel, unique_keys=unique_keys)
        self.function_dnd_policy = function_dnd_policy
        # rule dnd
        self.rule_dnd = DND(dict_len, hidden_dim, rule_kernel, unique_keys=unique_keys)
        self.rule_dnd_policy = rule_dnd_policy
        #
        self.a2c = A2C(hidden_dim, hidden_dim, output_dim, q_est=q_est) # A2C_linear(hidden_dim, output_dim)
        # init
        self.reset_parameter()

    def reset_parameter(self):
        for name, wts in self.named_parameters():
            if 'weight' in name:
                torch.nn.init.orthogonal_(wts)
            elif 'bias' in name:
                torch.nn.init.constant_(wts, 0)

    def forward(self, x_t, h, c):
        # unpack activity
        h = h.view(h.size(1), -1)
        c = c.view(c.size(1), -1)
        x_t = x_t #.view(x_t.size(1), -1)
        # transform the input info
        Wx = self.i2h(x_t)
        Wh = self.h2h(h)
        preact = Wx + Wh
        # get all gate values
        gates = preact[:, : N_GATES * self.hidden_dim].sigmoid()
        # split input(write) gate, forget gate, output(read) gate
        f_t = gates[:, :self.hidden_dim]
        i_t = gates[:, self.hidden_dim: 2*self.hidden_dim]
        o_t = gates[:, 2*self.hidden_dim: 3*self.hidden_dim]
        fun_t = gates[:, 3*self.hidden_dim: 4*self.hidden_dim]
        rul_t = gates[:, -self.hidden_dim:]
        
        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        # retrieve memory
        memfun_t = self.function_dnd.get_memory(x_t[:self.function_dim], self.function_dnd_policy).tanh()
        memrule_t = self.rule_dnd.get_memory(x_t[self.function_dim:self.function_dim+self.rule_dim], self.rule_dnd_policy).tanh()
        # gate the memory; in general, can be any transformation of it
        c_t = c_t + torch.mul(fun_t, memfun_t) + torch.mul(rul_t, memrule_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())
    
        # take a episodic snapshot (split up into rule and function component)
        with torch.no_grad():
            c_ft = c_t.clone()
            c_rt = c_t.clone()
            c_ft[0][int(self.hidden_dim/2):] = 0.
            c_rt[0][:int(self.hidden_dim/2)] = 0.
        #ipdb.set_trace()
        self.function_dnd.save_memory(x_t[:self.function_dim], c_ft)
        self.rule_dnd.save_memory(x_t[self.function_dim:self.function_dim+self.rule_dim], c_rt)

        # policy
        pi_a_t, v_t, q_t = self.a2c.forward(h_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # fetch activity
        output = [a_t, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, fun_t, rul_t, q_t, pi_a_t]
        return output, cache

    def pick_action(self, action_distribution):
        """action selection by sampling from a multinomial.

        Parameters
        ----------
        action_distribution : 1d torch.tensor
            action distribution, pi(a|s)

        Returns
        -------
        torch.tensor(int), torch.tensor(float)
            sampled action, log_prob(sampled action)

        """
        m = torch.distributions.Categorical(action_distribution)
        a_t = m.sample()
        log_prob_a_t = m.log_prob(a_t)
        return a_t, log_prob_a_t

    def get_init_states(self, scale=.1):
        h_0 = torch.randn(1, 1, self.hidden_dim) * scale
        c_0 = torch.randn(1, 1, self.hidden_dim) * scale
        return h_0, c_0

    def turn_off_encoding(self):
        self.function_dnd.encoding_off = True
        self.rule_dnd.encoding_off = True

    def turn_on_encoding(self):
        self.function_dnd.encoding_off = False
        self.rule_dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.function_dnd.retrieval_off = True
        self.rule_dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.function_dnd.retrieval_off = False
        self.rule_dnd.retrieval_off = False

    def reset_memory(self):
        self.function_dnd.reset_memory()
        self.rule_dnd.reset_memory()

    def get_all_mems(self):
        n_mems = len(self.function_dnd.keys)
        K = [self.function_dnd.keys[i] for i in range(n_mems)]
        V = [self.function_dnd.vals[i] for i in range(n_mems)]

        n_mems = len(self.rule_dnd.keys)
        K = [self.rule_dnd.keys[i] for i in range(n_mems)]
        V = [self.rule_dnd.vals[i] for i in range(n_mems)]
        return K, V


