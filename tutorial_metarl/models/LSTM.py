import torch
import torch.nn as nn
from tutorial_metarl.models.DND import DND
from tutorial_metarl.models.A2C import A2C_linear, A2C
from tutorial_metarl.trainers.utils import set_random_seed

# constants
N_GATES = 4

class LSTM(nn.Module):

    def __init__(self, hidden_dim, output_dim, bias=True, q_est=False, seed=0):
        super(LSTM, self).__init__() 
        
        # input, hidden and output units
        input_dim =  output_dim + 1 + 1 # arms + reward + trial index
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.bias = bias
        # set the manual seed 
        set_random_seed(seed)
        # input-hidden weights
        self.i2h = nn.Linear(input_dim, (N_GATES+1) * hidden_dim, bias=bias)
        # hidden-hidden weights
        self.h2h = nn.Linear(hidden_dim, (N_GATES+1) * hidden_dim, bias=bias)
        self.a2c = A2C(hidden_dim, hidden_dim, output_dim, q_est=q_est) 
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

        # stuff to be written to cell state
        c_t_new = preact[:, N_GATES * self.hidden_dim:].tanh()

        # new cell state = gated(prev_c) + gated(new_stuff)
        c_t = torch.mul(f_t, c) + torch.mul(i_t, c_t_new)

        # get gated hidden state from the cell state
        h_t = torch.mul(o_t, c_t.tanh())
        
        # policy
        pi_a_t, v_t, q_t = self.a2c.forward(h_t)

        # pick an action
        a_t, prob_a_t = self.pick_action(pi_a_t)

        # reshape data
        h_t = h_t.view(1, h_t.size(0), -1)
        c_t = c_t.view(1, c_t.size(0), -1)

        # fetch activity
        output = [a_t, prob_a_t, v_t, h_t, c_t]
        cache = [f_t, i_t, o_t, q_t, pi_a_t] 
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