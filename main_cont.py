#!/usr/bin/env python
# coding: utf-8

# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 4#### Imports

# In[1]:


# !jupyter nbconvert --to script main_cont.ipynb


# In[2]:


# %load_ext autoreload
# %autoreload 2


# In[549]:


import numpy as np
import gym
import torch
from collections import defaultdict
import time
import wandb as wandb_logger
from datetime import datetime
import plotly.graph_objects as go

from beta_gym.cont_cartpole import ContinuousCartPoleEnv

from lmdp.data.buffer import StandardBuffer,ReplayBuffer, gather_data_in_buffer, get_iter_indexes
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.utils.utils_eval import evaluate_on_env

from dacmdp.exp_track.experiments import ExpPool
from dacmdp.core.args import BaseConfig
import dacmdp.core.utils.server_helper as sh

from d4rl.infos import DATASET_URLS as d4rl_envs


# ## Parse Args

# In[547]:


# options = "--env_name CartPole-cont-v1 --MAX_S_COUNT 11000 --buffer_size 10000 --tran_type_count 40 --MAX_NS_COUNT 1 --mdp_build_k 1 --plcy_k 1 \
# --normalize_by_distance --penalty_beta 10 --no_wandb_logging --load_buffer --buffer_name Robust_cartpole-cont-v1_0 \
# --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
# --results_folder results/CartPole-cont-v1/test_run-Apr07"

# options = "--env_name halfcheetah-random-v0 --tran_type_count 10 --wandb_project DACMDPCONT-V0 \
# --load_buffer --buffer_size 1000000 \
# --MAX_S_COUNT 1100000 --MAX_NS_COUNT 1 --mdp_build_k 1 --normalize_by_distance --penalty_beta 0.1 \
# --gamma 0.99 --slip_prob 0.1 --default_mode GPU \
# --eval_episode_count 100 --plcy_k 1"

if sh.in_notebook():
    options = ExpPool.get_by_id("S0-DRobust-tt5-p100-ns5").expSuffix
    config = BaseConfig(options)
    config.dataArgs.buffer_size = 10000
    config.mdpBuildArgs.MAX_S_COUNT = int(1.1 * config.dataArgs.buffer_size) 
    config.mdpBuildArgs.MAX_NS_COUNT = 2
    config.mdpBuildArgs.mdp_build_k = 2
    config.mdpBuildArgs.tran_type_count = 2
    config.evalArgs.eval_episode_count = 10
    config.logArgs.no_wandb_logging = True
    config.mdpBuildArgs.knn_delta = 0.001
else:
    config = BaseConfig()


# In[548]:


print(config)


# ## Setup Environment Variables 

# In[ ]:





# ## Setup Wandb Logging

# In[524]:


if config.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    wandb_logger.init( id = config.logArgs.wandb_id,
            entity=config.logArgs.wandb_entity,
            project=config.logArgs.wandb_project,
            config = config.flat_args,
            resume = "allow")


# ## Define Environment

# In[525]:


from beta_gym.cont_cartpole import ContinuousCartPoleEnv
env = ContinuousCartPoleEnv() if config.envArgs.env_name == "CartPole-cont-v1" else gym.make(config.envArgs.env_name)


# ## Define Buffer

# In[526]:


train_buffer = StandardBuffer(state_shape = env.observation_space.shape,
                           action_shape = env.action_space.shape, 
                           batch_size=64, 
                           buffer_size = config.dataArgs.buffer_size,
                           device = config.dataArgs.buffer_device)


# In[528]:


if config.dataArgs.load_buffer:
    print('Loading buffer!')
    if config.envArgs.env_name in d4rl_envs:
        dataset = env.get_dataset()
        for i in range(min(config.dataArgs.buffer_size,len(dataset['observations']))-1):
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i+1]
            action = dataset['actions'][i]
            reward = dataset['rewards'][i]
            done_bool = bool(dataset['terminals'][i])
            train_buffer.add(obs, action, new_obs, reward, done_bool)
    else: 
        train_buffer.load(f"{config.dataArgs.data_dir}{config.dataArgs.buffer_name}")
    print('Loaded buffer!')
else: 
    print('Collecting buffer!')
    train_buffer, info = gather_data_in_buffer(train_buffer, env, policy = lambda s:env.action_space.sample(), episode_count=99999, frame_count=args.dataArgs.buffer_size)
    print('Collected buffer!')


# In[529]:





# ## Define Representation

# In[ ]:





# ## Define and Solve MDP

# In[530]:


from dacmdp.core.repr_nets import DummyNet
from dacmdp.core.mdp_agents.cont_agent import StochasticAgent,StochasticAgentWithDelta, get_tran_types


# In[532]:


AgentClass = StochasticAgent
# AgentClass = StochasticAgentWithDelta


# In[533]:


tt_action_space = get_tran_types(config.mdpBuildArgs.tran_type_count)

empty_MDP = FullMDP(A= tt_action_space, 
                    build_args=config.mdpBuildArgs, 
                    solve_args=config.mdpSolveArgs)

myAgent = AgentClass(seed_mdp= empty_MDP, 
                      repr_model= DummyNet(None), 
                      build_args=config.mdpBuildArgs, 
                      solve_args=config.mdpSolveArgs, 
                      eval_args=config.evalArgs,
                      action_space = env.action_space,).verbose()

myAgent.process(train_buffer)


# ## Evaluate policies

# In[546]:


eval_rewards = {}
for pi_name, pi in myAgent.policies.items(): 
    avg_rewards, info = evaluate_on_env(env, pi, eps_count=config.evalArgs.eval_episode_count,progress_bar=True)
    print(f"Policy Name: {pi_name} Avg Rewards:{avg_rewards}")
    eval_rewards[pi_name] = avg_rewards
    time.sleep(1)


# In[36]:


if config.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    wandb_logger.log(eval_rewards)    
    for name, distr in myAgent.mdp_distributions.items():
        wandb_logger.log({"MDP Histogram" + "/" + name: go.Figure(data=[go.Histogram(x=distr)]), 
                         "mdp_frame_count": len(myAgent.mdp_T.s2i)})


# In[ ]:





# In[ ]:





# In[ ]:




