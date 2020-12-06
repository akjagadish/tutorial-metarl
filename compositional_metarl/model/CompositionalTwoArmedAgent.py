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
import numpy as np

# constants
N_GATES = 4

class CompositionalTwoArmedAgent(nn.Module):

    def __init__(self, hidden_dim, output_dim,
            dict_len, input_dim=None, ctx_dim=10, retrieval_dim=10, inputs='context_action_reward', 
            kernel='cosine', bias=True, dnd_policy='softmax', 
            unique_keys=False, q_est=False, seed=0):
        super(CompositionalTwoArmedAgent, self).__init__() 
        
        if input_dim is None:
            if inputs == 'context_action_reward':
                input_dim = ctx_dim + output_dim + 1 + 1 # context + num_arms + reward + trial_idx
        self.input_dim = input_dim
        self.inputs = inputs
        self.ctx_dim = ctx_dim
        self.retrieval_dim = retrieval_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        # set the manual seed 
        set_random_seed(seed)
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # dnd
        self.dnd = DND(dict_len, hidden_dim, kernel, unique_keys=unique_keys)
        self.dnd_policy = dnd_policy
        #
        self.a2c = A2C(hidden_dim, hidden_dim, output_dim, q_est=q_est) #A2C_linear(hidden_dim, output_dim)
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
        r_t = gates[:, -self.hidden_dim:]
        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()
        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)
        # retrieve memory
        pick_arm = np.random.choice([0, 1])
        m_t = self.dnd.get_memory(x_t[pick_arm*self.retrieval_dim:(pick_arm+1)*self.retrieval_dim], self.dnd_policy).tanh()
        # gate the memory; in general, can be any transformation of it
        c_t = c_t + torch.mul(r_t, m_t)
        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())
        # take a episodic snapshot
        self.dnd.save_memory(x_t[pick_arm*self.retrieval_dim:(pick_arm+1)*self.retrieval_dim], c_t)
        # policy
        pi_a_t, v_t, q_t = self.a2c.forward(h_t)
        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)
        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)
        # fetch activity
        output = [a_t, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, r_t, m_t, q_t, pi_a_t]
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
        self.dnd.encoding_off = True

    def turn_on_encoding(self):
        self.dnd.encoding_off = False

    def turn_off_retrieval(self):
        self.dnd.retrieval_off = True

    def turn_on_retrieval(self):
        self.dnd.retrieval_off = False

    def reset_memory(self):
        self.dnd.reset_memory()

    def get_all_mems(self):
        n_mems = len(self.dnd.keys)
        K = [self.dnd.keys[i] for i in range(n_mems)]
        V = [self.dnd.vals[i] for i in range(n_mems)]
        return K, V


