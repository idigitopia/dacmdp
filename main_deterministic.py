#!/usr/bin/env python
# coding: utf-8

# ## Imports

# In[9]:


# !jupyter nbconvert --to script main_deterministic.ipynb


# In[1]:


# %load_ext autoreload
# %autoreload 2


# In[5]:


import numpy as np
import gym
from lmdp.data.buffer import StandardBuffer,ReplayBuffer, gather_data_in_buffer, get_iter_indexes
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.utils.utils_eval import evaluate_on_env
from core.args import get_args, print_args, wandbify_args
import torch
from beta_gym.cont_cartpole import ContinuousCartPoleEnv
from collections import defaultdict
import time
from d4rl.infos import DATASET_URLS as d4rl_envs
from exp_track.experiments import ExpPool
import wandb as wandb_logger
import core.utils.server_helper as sh
from datetime import datetime


# ## Parse Args

# In[4]:


# options = "--env_name CartPole-cont-v1 --MAX_S_COUNT 110000 --buffer_size 100000 --tran_type_count 40 --MAX_NS_COUNT 1 --mdp_build_k 1 --plcy_k 1 \
# --normalize_by_distance --penalty_beta 10 --no_wandb_logging --load_buffer --buffer_name Robust_cartpole-cont-v1_0 --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/"

# options = "--no_cuda --env_name maze2d-medium-v1 --MAX_S_COUNT 1100000 --buffer_size 1000000 --tran_type_count 5 --MAX_NS_COUNT 1 --mdp_build_k 1 --plcy_k 1 \
# --normalize_by_distance --penalty_beta 10 --no_wandb_logging  --load_buffer"
if sh.in_notebook():
    options = ExpPool.get_by_id("Det-S0-DRobust-tt10-p100").expSuffix + " --no_wandb_logging"
    args = get_args(options)
else:
    args = get_args()


# In[5]:


print_args(args)


# ## Setup Environment Variables 

# In[21]:


import os 


# ## Setup Wandb Logging

# In[22]:


if args.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    wandb_logger.init( id = args.logArgs.exp_id + "-" +datetime.now().strftime('%b%d_%H-%M-%S'),
            entity="xanga",
            project=args.logArgs.wandb_project,
            config = wandbify_args(args),
            resume = "allow")


# ## Define Environment

# In[23]:


from beta_gym.cont_cartpole import ContinuousCartPoleEnv
env = ContinuousCartPoleEnv() if args.envArgs.env_name == "CartPole-cont-v1" else gym.make(args.envArgs.env_name)


# In[ ]:





# ## Define Buffer

# In[24]:


train_buffer = StandardBuffer(state_shape = env.observation_space.shape,
                           action_shape = env.action_space.shape, 
                           batch_size=64, 
                           buffer_size=args.dataArgs.buffer_size,
                           device=args.dataArgs.buffer_device)


# In[25]:


args.dataArgs.buffer_name


# In[26]:


if args.dataArgs.load_buffer:
    print('Loading buffer!')
    if args.envArgs.env_name in d4rl_envs:
        dataset = env.get_dataset()
        for i in range(min(args.dataArgs.buffer_size,len(dataset['observations']))):
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i+1]
            action = dataset['actions'][i]
            reward = dataset['rewards'][i]
            done_bool = bool(dataset['terminals'][i])
            train_buffer.add(obs, action, new_obs, reward, done_bool)
    else: 
        train_buffer.load(f"{args.dataArgs.data_dir}{args.dataArgs.buffer_name}")
    print('Loaded buffer!')
else: 
    print('Collecting buffer!')
    train_buffer, info = gather_data_in_buffer(train_buffer, env,policy = lambda s:env.action_space.sample(), episode_count=99999, frame_count=args.dataArgs.buffer_size)
    print('Collected buffer!')


# In[ ]:





# ## Define Representation

# In[27]:


class DummyNet():
    def __init__(self, sim, add_noise=False):
        self.simulator = sim

    def encode_state_single(self, o):
        return tuple(np.array(o).astype(np.float32))

    def encode_state_batch(self, o_batch):
        return [self.encode_state_single(o) for o in o_batch]
    
    def encode_action_single(self, a):
        return tuple(np.array(a).astype(np.float32))
    
    def encode_action_batch(self,a_batch):
        return [self.encode_action_single(a) for a in a_batch]

    def predict_single_transition(self, o, a):
        assert False, "Not Implemented Error"

    def predict_batch_transition(self, o_batch, a_batch):
        assert False, "Not Implemented Error"


# ## Define and Solve MDP

# In[28]:


from core.mdp_agents.cont_agent_deterministic import SimpleAgent, get_tran_types


# In[29]:


tt_action_space = get_tran_types(args.mdpBuildArgs.tran_type_count)

empty_MDP = FullMDP(A= tt_action_space, 
                    build_args=args.mdpBuildArgs, 
                    solve_args=args.mdpSolveArgs)

myAgent = SimpleAgent(seed_mdp= empty_MDP, 
                      repr_model= DummyNet(None), 
                      build_args=args.mdpBuildArgs, 
                      solve_args=args.mdpSolveArgs, 
                      eval_args=args.evalArgs,
                      action_space = env.action_space,).verbose()

myAgent.process(train_buffer)


# In[30]:


# a


# ## Sanity Check

# In[10]:


# go.Figure(data=go.Histogram(x=[val for qs, val_d in myAgent.mdp_T.s_qvalDict.items() for a,val in val_d.items()]))
# import matplotlib.pyplot as plt
# plt.hist([val for qs, val_d in myAgent.mdp_T.s_qvalDict.items() for a,val in val_d.items() if val > args.mdpBuildArgs.ur])


# ## Evaluate policies

# In[32]:


eval_rewards = {}
for pi_name, pi in myAgent.policies.items(): 
    avg_rewards, info = evaluate_on_env(env, pi, eps_count=args.evalArgs.eval_episode_count,progress_bar=True)
    print(f"Policy Name: {pi_name} Avg Rewards:{avg_rewards}")
    eval_rewards[pi_name] = avg_rewards
    time.sleep(1)


# In[ ]:


if args.logArgs.no_wandb_logging:
    pass
else:
    wandb_logger.log(eval_rewards)

