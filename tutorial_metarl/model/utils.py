import torch
import numpy as np
from torch.nn.functional import smooth_l1_loss


'''helpers'''

eps = np.finfo(np.float32).eps.item()


def compute_returns(rewards, gamma=0, normalize=False):
    """compute return in the standard policy gradient setting.

    Parameters
    ----------
    rewards : list, 1d array
        immediate reward at time t, for all t
    gamma : float, [0,1]
        temporal discount factor
    normalize : bool
        whether to normalize the return
        - default to false, because we care about absolute scales

    Returns
    -------
    1d torch.tensor
        the sequence of cumulative return

    """
    R = 0
    returns = []
    for r in rewards[::-1]: # reverse the rewards
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    if normalize:
        returns = (returns - returns.mean()) / (returns.std() + eps)
    return returns


def get_reward(a_t, a_t_targ):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    a_t_targ : int
        target action

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    if a_t == a_t_targ:
        r_t = 1
    else:
        r_t = 0
    return torch.tensor(r_t).type(torch.FloatTensor).data


def compute_a2c_loss(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-prob_t * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value

def get_reward_mab(a_t, rewards):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    rewards : list
        rewards across arms

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    r_t = rewards[a_t]
    return r_t.type(torch.FloatTensor).data #torch.tensor(r_t).type(torch.FloatTensor).data

def get_reward_multidimMAB(a_t, rewards):
    """define the reward function at time t

    Parameters
    ----------
    a_t : int
        action
    rewards : list
        rewards across arms

    Returns
    -------
    torch.FloatTensor, scalar
        immediate reward at time t

    """
    r_t = rewards[a_t[0], a_t[1]]
    return torch.tensor(r_t).type(torch.FloatTensor).data

def compute_a2c_loss_mab(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-prob_t * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value

def compute_a2c_loss_multidimMAB(probs, values, returns):
    """compute the objective node for policy/value networks

    Parameters
    ----------
    probs : list
        action prob at time t
    values : list
        state value at time t
    returns : list
        return at time t

    Returns
    -------
    torch.tensor, torch.tensor
        Description of returned object.

    """
    policy_grads, value_losses = [], []
    for prob_t, v_t, R_t in zip(probs, values, returns):
        A_t = R_t - v_t.item()
        policy_grads.append(-(prob_t[0]+prob_t[1]) * A_t)
        value_losses.append(
            smooth_l1_loss(torch.squeeze(v_t), torch.squeeze(R_t))
        )
    loss_policy = torch.stack(policy_grads).sum()
    loss_value = torch.stack(value_losses).sum()
    return loss_policy, loss_value

def run_agent_inference(agent, h_t, c_t, x_t, Y, n_runs):
    """run inference on the agent

    Args:
        agent ([PyTorchModel]): model
        h_t ([FloatTensor]): hidden states
        c_t ([FloatTensor]): context states
        x_t ([FloatTensor]): input states
        Y ([FloatTensor]): reward structure over arms
        n_runs ([int]): number of runs of the model
    """
    # run model for n_runs
    agent.eval()
    agent.turn_off_encoding()
    h_t, c_t = agent.get_init_states()
    # sample random action for zeroth trial
    a_t = torch.randint(high=agent.n_arms, size=(1,))
    r_t = Y[0][a_t].type(torch.FloatTensor).data.reshape(-1)
    h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t)  
    actions, cum_reward, rewards = [], 0., []
    for t in range(n_runs):
        output_t, _ = agent(x_t, h_t, c_t) 
        a_t, _, _, _, _ = output_t
        r_t = get_reward_mab(a_t, Y)
        cum_reward += r_t
        actions.append(a_t)
        rewards.append(r_t)
    agent.train()
    return actions, rewards

def one_hot_embedding(arm, num_arms):
    """Embedding labels to one-hot form.

    Args:
      arms: (LongTensor) arms for which embedding is required, sized [N,].
      num_arms: (int) number of arms.

    Returns:
      (tensor) encoded arms, sized [N, #classes].
    """
    y = torch.eye(num_arms) 
    
    return torch.zeros(num_arms) if arm is None else y[arm]
    
def run_model_eval(agent, X, Y, S, n_trials, total_trials=None, init_at=None):
    """ runs the model for one episode

    Args:
        agent ([pyTorch.model]): Meta-RL agent
        X ([torch.tensor]): cues shown to agent per trial stacked one below the other
        Y ([torch.tensor]): Rewards across arms per trial stacked one below the other
        S ([torch.tensor]): (Optional) Block the given base function belongs to
        n_trials ([int]): Number of trials per episode of a base function.
        total_trials ([int]): (Optional) Total number of trials in the episode. Defaults to None
        init_at ([type], optional): [description]. Defaults to None.

    Returns:
        actions: actions taken by the agent in all trials
        rewards: rewards obtained by the agent in all trials
        values: Q-values predicted for all trials
        regrets: regret per trial for all trials
    """
    
    # set model to eval mode
    agent.eval()
    agent.turn_off_encoding()

    # set init states
    actions, values, regrets, rewards =  [], [], [], []
    total_trials = total_trials if total_trials else n_trials
    h_t, c_t = agent.get_init_states()
    h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t) 
    a_t = init_at
    r_t = Y[0].mean() 
    n_arms = Y[0].shape[0]
    
    
    # run model
    for t in range(total_trials):

        if agent.inputs == 'context_block_action_reward':
            x_t = X[t] 
            s_t = S[t]
            trial = t%n_trials
            one_hot_a_t = one_hot_embedding(a_t, n_arms)
            x_t = torch.cat((x_t.type(torch.FloatTensor), s_t.reshape(-1).type(torch.FloatTensor), one_hot_a_t.reshape(-1).type(torch.FloatTensor), 
                             r_t.reshape(-1).type(torch.FloatTensor), torch.tensor(trial).reshape(-1).type(torch.FloatTensor)), dim=0)
        elif agent.inputs == 'context_action_reward':
            x_t = X[t] 
            one_hot_a_t = one_hot_embedding(a_t, n_arms)
            x_t = torch.cat((x_t.type(torch.FloatTensor), one_hot_a_t.reshape(-1).type(torch.FloatTensor), 
                             r_t.reshape(-1).type(torch.FloatTensor), torch.tensor(t).reshape(-1).type(torch.FloatTensor)), dim=0)

        output_t, c = agent(x_t, h_t, c_t)
        a_t, prob_a_t, v_t, h_t, c_t = output_t
        f_t, i_t, o_t, rg_t, m_t, q_t, pa_t = c

        # bootstrap reward from q-values
        r_t = Y[t][a_t]
        

        # predicted q-value normalized
        Q = q_t.detach().numpy().T
        Q = Q - Q.min()
        Q = Q/Q.max()

        # store results
        rewards.append(r_t)
        actions.append(a_t) 
        regrets.append(Y[t].max() - r_t)
        values.append(Q)
    
    return torch.stack(actions), torch.stack(rewards), torch.stack(regrets), np.hstack(values).T

def _input(one_hot_a_t, r_t, t):
    """ prepares input to the model

    Args:
        one_hot_a_t ([torch.tensor]): one hot coded actions
        r_t ([torch.tensor]): reward
        t [torch.tensor]: trial index

    Returns:
        input to the model
    
    """
    
    return torch.cat((one_hot_a_t.reshape(-1).type(torch.FloatTensor), 
                  r_t.reshape(-1).type(torch.FloatTensor), 
                  torch.tensor(t).reshape(-1).type(torch.FloatTensor)), dim=0)