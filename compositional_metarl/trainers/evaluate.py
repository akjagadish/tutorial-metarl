import torch 


def evaluate(model, dataloaders, n_runs=100):
    """ model evaluation

    Args:
        model ([PyTorchModel object]): model of the agent
        dataloaders ([Datagenerator class]): pass data generator class
        n_runs (int, optional): pass number of runs as input. Defaults to 100.

    Returns:
        mean_performance: percentage of best actions
    """
    n_cues = dataloaders.num_cues
    aoi = dataloaders.aoi
    n_best_actions = torch.zeros(n_cues)
    AOI = {'linear': [7], 'periodic': [0, 2, 4, 6], 'linperiodic': [6]}

    for c_indx, cue in enumerate(dataloaders.cues):

        X, Y = dataloaders.sample(end_rnd=1, cue=cue)

        # set model to eval mode
        model.eval()
        model.turn_off_encoding()

        # set init states 
        h_t, c_t = model.get_init_states()
        h_t, c_t = torch.zeros_like(h_t), torch.zeros_like(c_t) 
        a_t = torch.randint(high=dataloaders.num_arms,size=(1,)) #model.output_dim
        r_t = Y[0][a_t] 
        
        with torch.no_grad():
            # run model
            for t in range(n_runs):
                x_t = dataloaders.cues[cue] 
                if model.inputs == 'action-reward':
                    x_t = torch.cat((x_t, a_t.reshape(-1).type(torch.FloatTensor), r_t.reshape(-1).type(torch.FloatTensor)), dim=0)
                _, cache = model(x_t, h_t, c_t) 
                _, _, _, _, _, q_t, _ = cache
                a_t = torch.argmax(q_t)
                
                # bootstrap reward from q-values
                n_best_actions[c_indx] += a_t in AOI[cue]

    percent_best_actions = n_best_actions/n_runs
    return percent_best_actions.mean().item()

            
        
