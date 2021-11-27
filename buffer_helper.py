# Default Python Packages.
from collections import defaultdict
import logging
import sys
import os
import time
import numpy as np

# Standard Python Packages.
import torch
import numpy as np
import gym
from sklearn.cluster import KMeans
from tqdm import tqdm

# Project Specific Dependencies 
from lmdp.data.buffer import StandardBuffer, gather_data_in_buffer
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.mdp.MDP_GPU_FACTORED import FullMDPFactored
from lmdp.utils.utils_eval import evaluate_on_env
from dacmdp.core.args import BaseConfig
from beta_gym.cont_cartpole import ContinuousCartPoleEnv
from dacmdp.exp_track.experiments import ExpPool
import dacmdp.core.utils.server_helper as sh

# Data Dependencies
from d4rl.infos import DATASET_URLS as d4rl_envs
from d4rl.offline_env import OfflineEnv


def get_d4rl_dataset(env, d4rl_path):
    o_env = OfflineEnv(env)
    o_env.observation_space = env.observation_space
    o_env.action_space = env.action_space
    d4rl_dataset = o_env.get_dataset(d4rl_path)

    return d4rl_dataset

def convert_from_d4rl_dataset(config, buffer, d4rl_dataset):
    d_size = min(config.dataArgs.buffer_size,len(d4rl_dataset['observations']))
    
    for i in range(d_size):
        obs = d4rl_dataset['observations'][i]
        new_obs = d4rl_dataset['observations'][i+1] if i < d_size -1 else d4rl_dataset['observations'][i]
        action = d4rl_dataset['actions'][i]
        reward = d4rl_dataset['rewards'][i]
        done_bool = bool(d4rl_dataset['terminals'][i])
        if i < (d_size - 1) or bool(d4rl_dataset['terminals'][d_size - 1]):
            buffer.add(obs, action, new_obs, reward, done_bool)
        
    return buffer

def load_buffer(config,env):
    """
    defines a new buffer, and loads it from the pre-configured load path. 
    follows d4rl data definition for storing datasets
    """
    print('Loading buffer!')
    
    action_shape = [1] if config.envArgs.env_name == "CartPole-cont-v1" else env.action_space.shape

    buffer = StandardBuffer(state_shape = env.observation_space.shape,
                           action_shape = action_shape, 
                           batch_size=32, 
                           buffer_size=config.dataArgs.buffer_size,
                           device="cpu")
    
    
    if config.envArgs.env_name == "CartPole-cont-v1":
        # replay_buffer.load(f"{args.output_dir}/buffers/{buffer_name}")
        fname = '%s-%s.hdf5' % (str(config.envArgs.env_name).lower(), config.dataArgs.buffer_name)
        fpath = os.path.join(config.dataArgs.data_dir,fname)
        d4rl_dataset = get_d4rl_dataset(env, fpath)
        train_buffer = convert_from_d4rl_dataset(config, buffer, d4rl_dataset)
        
    elif config.envArgs.env_name in d4rl_envs:
        d4rl_dataset = env.get_dataset()
        train_buffer = convert_from_d4rl_dataset(config, buffer, d4rl_dataset)    
    else:
        assert False, f"Data load logic for given env {config.envArgs.buffer_size} is not defined."
    
    print('Loaded buffer!')
    
    return train_buffer




def collect_buffer(config, env):
    """
    Collect buffer. using a random policy. 
    returns a data buffer. 
    """

    print('Collecting buffer!')


    train_buffer = StandardBuffer(state_shape = env.observation_space.shape,
                               action_shape = [1], # for discrete settings. 
                               batch_size=32, 
                               buffer_size=config.dataArgs.buffer_size,
                               device="cpu")
    
    train_buffer, info = gather_data_in_buffer(train_buffer, env, 
                                           policy = lambda s:env.action_space.sample(),
                                           episode_count=99999, 
                                           frame_count=config.dataArgs.buffer_size)

    print('Collected buffer!')    
    
    return train_buffer


def pred_miss_idx_count(buffer):
    """
    predicts if all next states are seen in the dataset. 
    outputs the number of missing indexes
    """
    i=0
    all_states, _, all_next_states, _ ,_ = buffer.sample_indices(range(len(buffer)))
    for s in tqdm(all_next_states):
        if s not in all_states:
            i +=1
            print(s)
    return i 