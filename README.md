<!-- ### Task description 

`src/contextual-choice.ipynb` tests the model on an 
<a href="https://en.wikipedia.org/wiki/Two-alternative_forced_choice#Behavioural_experiments">evidence accumulation task </a>
with "context". 

More concretely, in the i-th trial ... 

- At time t, the model receives noisy observation, x_t (e.g. random dots moving around, slightly drifting to left/right)
and a "context" for this trial, call it context_i (e.g. an image of an apple)
- The task is to press button 0 if x_t is, on average, negative and press 1 otherwise (like press left/right button according to the average direction of the moving dots). Let's denote the response target by y_i, so y_i \in {0, 1}.  
- If the model never saw trial i before, it has to base its decision in x_t. However, if it this is the 2nd encounter of trial i, assuming the model cached the association between context_i and y_i in its episodic memory, then the model can just output y_i. 


Since context is always presented within a trial, making decisions based on recalling the context-target association allows the model to respond faster, which leads to greater cumulative reward.  -->

<!-- ### Results -->
<!-- 
### Dir structure 
```
.
├── LICENSE
├── README.md
├── figs
├── requirements.txt
└── src
    ├── contextual-choice.ipynb         # train the model on a contextual choice task, in .ipynb
    ├── contextual-choice.py            # train the model on a contextual choice task, in .py
    ├── model   
    │   ├── A2C.py                      # an advantage actor critic agent
    │   ├── DND.py                      # the memory module 
    │   ├── DNDLSTM.py                  # a LSTM-based A2C agent with DND memory 
    │   ├── utils.py
    └── └── __init__.py
    ├── task
    │   ├── ContextualChoice.py         # the definition of the contextual choice task
    │   └── __init__.py
    └── utils.py
``` -->
<!-- 
### Extra note 

1. A variant of the DND part is implemented in 
<a href="https://princetonuniversity.github.io/PsyNeuLink/">psyneulink</a> 
as <a href="https://princetonuniversity.github.io/PsyNeuLink/MemoryFunctions.html?highlight=dnd#psyneulink.core.components.functions.statefulfunctions.memoryfunctions.ContentAddressableMemory">    pnl.ContentAddressableMemory</a>. 

2. The original paper uses A3C. I'm doing A2C instead - no asynchronous parallel rollouts. If you are not familiar with these ideas, here's a <a href="https://github.com/qihongl/demo-advantage-actor-critic">a standalone demo of A2C</a>. 

3. The memory module is called a "differentiable neural dictionary", but note that it is not fully differentiable, unlike end-to-end models (e.g. <a href="https://arxiv.org/abs/1410.5401">NTM</a>, <a href="https://www.nature.com/articles/nature20101/">DNC</a>). 
By giving up end-to-end differentiability, one can impose some explicit structure of what the memory module suppose to do, such as one-nearest neighbor search or kernel-weighted averaging. 
 -->

### References

- Ritter, S., Wang, J. X., Kurth-Nelson, Z., Jayakumar, S. M., Blundell, C., Pascanu, R., & Botvinick, M. (2018). Been There, Done That: Meta-Learning with Episodic Recall. arXiv [stat.ML]. Retrieved from http://arxiv.org/abs/1805.09692

    - also see Blundell et al. 2016, Pritzel et al. 2017 and Kaiser et al 2017... 

- Mnih, V., Badia, A. P., Mirza, M., Graves, A., Lillicrap, T. P., Harley, T., … Kavukcuoglu, K. (2016). Asynchronous Methods for Deep Reinforcement Learning. Retrieved from http://arxiv.org/abs/1602.01783

- Forked from base repo: https://github.com/qihongl/dnd-lstm/

