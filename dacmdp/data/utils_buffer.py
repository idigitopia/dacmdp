from collections import namedtuple
from typing import Tuple
import numpy as np
import torch
import time
import copy
from types import SimpleNamespace
import os 
import d4rl



# Generic replay buffer for standard gym tasks
# most commonly used.
class StandardBuffer(object):
    """
    Initializes an array for elements of transitions as per the maximum buffer size. 
    Keeps track of the crt_size. 
    Saves the buffer element-wise as numpy array. Fast save and retreival compared to pickle dumps. 
    """
    def __init__(self, state_shape, action_shape,  buffer_size, device, batch_size = 64):
        
        self.state_shape = state_shape
        self.action_shape = action_shape
        
        self.batch_size = batch_size
        self.max_size = int(buffer_size)
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, *state_shape))
        self.action = np.zeros((self.max_size, *action_shape))
        self.next_state = np.zeros_like(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))

        # Normalization parameters. 
        self.norm_params = SimpleNamespace(is_state_normalized = False,  is_action_normalized = False, 
                                        state_mean = None, state_std = None,
                                        action_mean = None, action_std = None)
    
    def __len__(self):
        return self.crt_size

    def __repr__(self):
        return f"Standard Buffer: \n \
                Total number of transitions: {len(self)}/{self.max_size} \n \
                State Store Shape: {self.state.shape} \n \
                Action Store Shape: {self.action.shape} \n"

    @property
    def all_states(self):
        return self.state[:self.crt_size]

    @property
    def all_next_states(self):
        return self.next_state[:self.crt_size]
    
    @property
    def all_actions(self):
        return self.action[:self.crt_size]

    @property
    def all_not_ep_ends(self):
        return self.not_done[:self.crt_size]

    @property
    def all_ep_ends(self):
        return 1- self.not_done[:self.crt_size]

    @property
    def all_rewards(self):
        return self.reward[:self.crt_size]


    def add(self, state, action, next_state, reward, done, episode_done=None, episode_start=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)


    def sample_indices(self, batch_size = None):
        indxs = np.random.randint(0, self.crt_size, size = batch_size or self.batch_size)
        return indxs

    def sample_using_indices(self, indxs, device = None):
        device = device or self.device
        indxs = np.array(indxs)
        return (
            torch.FloatTensor(self.state[indxs]).to(device),
            torch.FloatTensor(self.action[indxs]).to(device),
            torch.FloatTensor(self.next_state[indxs]).to(device),
            torch.FloatTensor(self.reward[indxs]).to(device),
            torch.FloatTensor(self.not_done[indxs]).to(device)
        )
    
    def sample(self, batch_size= None):
        indxs = self.sample_indices(batch_size or self.batch_size)
        return self.sample_using_indices(indxs)
    

    def sample_indices_for_seq(self, seq_len, batch_size = None):
        batch_size =batch_size or self.batch_size
        indxs = self.sample_indices(2*batch_size)
        # if the eipsode ends before seq_len, ignore
        eligible_indxs = indxs[[np.sum(self.not_done[i:i+seq_len]) == seq_len for i in indxs]][:batch_size]
        return eligible_indxs

    def sample_seq_using_indices(self, seq_len, indxs, device = None):
        device = device or self.device
        stacked_indxs = np.stack([np.arange(i,i+seq_len) for i in indxs])

        return (
            torch.FloatTensor(self.state[stacked_indxs]).to(device),
            torch.FloatTensor(self.action[stacked_indxs]).to(device),
            torch.FloatTensor(self.next_state[stacked_indxs]).to(device),
            torch.FloatTensor(self.reward[stacked_indxs]).to(device),
            torch.FloatTensor(self.not_done[stacked_indxs]).to(device)
        )
        
    def sample_seq(self, seq_len , batch_size = None, device = None):
        indxs = self.sample_indices_for_seq(seq_len, batch_size)
        return self.sample_seq_using_indices(seq_len, indxs, device)



    def save(self, save_folder):
        np.save(f"{save_folder}_state.npy", self.state[:self.crt_size])
        np.save(f"{save_folder}_action.npy", self.action[:self.crt_size])
        np.save(f"{save_folder}_next_state.npy", self.next_state[:self.crt_size])
        np.save(f"{save_folder}_reward.npy", self.reward[:self.crt_size])
        np.save(f"{save_folder}_not_done.npy", self.not_done[:self.crt_size])
        np.save(f"{save_folder}_ptr.npy", self.ptr)

    def load(self, save_folder, size=-1):
        reward_buffer = np.load(f"{save_folder}_reward.npy")

        # Adjust crt_size if we're using a custom size
        size = min(int(size), self.max_size) if size > 0 else self.max_size
        self.crt_size = min(reward_buffer.shape[0], size)

        self.state[:self.crt_size] = np.load(f"{save_folder}_state.npy")[:self.crt_size]
        self.action[:self.crt_size] = np.load(f"{save_folder}_action.npy")[:self.crt_size]
        self.next_state[:self.crt_size] = np.load(f"{save_folder}_next_state.npy")[:self.crt_size]
        self.reward[:self.crt_size] = reward_buffer[:self.crt_size]
        self.not_done[:self.crt_size] = np.load(f"{save_folder}_not_done.npy")[:self.crt_size]

        print(f"Replay Buffer loaded with {self.crt_size} elements.")


    def get_tran_tuples(self):
        batch = self.sample_using_indices(list(range(0, len(self))))
        batch_ob, batch_a, batch_ob_prime, batch_r, batch_nd = batch
        batch_d = 1 - batch_nd
        tran_tuples = [(tuple(s), tuple(a), tuple(ns), r, d) for s, a, ns, r, d in zip(batch_ob.numpy(),
                                                                                       batch_a.numpy(),
                                                                                       batch_ob_prime.numpy(),
                                                                                       batch_r.view((-1,)).numpy(),
                                                                                       batch_d.view((-1,)).numpy())]
        return tran_tuples


    def fetch_normalization_statistics(self, eps):
        state_mean = torch.FloatTensor(self.state[:self.crt_size].mean(0,keepdims=True))
        state_std = torch.FloatTensor(self.state[:self.crt_size].std(0,keepdims=True) + eps)
        state_std[state_std<1e-5] = 1

        action_mean = torch.FloatTensor(self.action[:self.crt_size].mean(0,keepdims=True))
        action_std = torch.FloatTensor(self.action[:self.crt_size].std(0,keepdims=True) + eps)
        action_std[action_std<1e-5] = 1

        return (state_mean, state_std), (action_mean, action_std)

    # Normalization Logic
    def normalize(self, normalize_state = True, normalize_action = False, eps = 1e-3):
        if normalize_state:
            self.state[:self.crt_size] = (self.state[:self.crt_size] - mean)/std
            self.next_state[:self.crt_size] = (self.next_state[:self.crt_size] - mean)/std

            self.norm_params.is_state_normalized = True  
            self.norm_params.state_mean, self.norm_params.state_std = mean, std
        
        if normalize_action:
            mean = self.action[:self.crt_size].mean(0,keepdims=True)
            std = self.action[:self.crt_size].std(0,keepdims=True) + eps
            self.action[:self.crt_size] = (self.action[:self.crt_size] - mean)/std

            self.norm_params.is_action_normalized = True  
            self.norm_params.action_mean, self.norm_params.action_std = mean, std
        
        return self.norm_params

    def query_normalized_state(self, state):
        if self.norm_params.is_state_normalized:
            return (state - self.norm_params.state_mean.reshape(-1))/self.norm_params.state_std.reshape(-1)
        else:
            return  state
    
    def query_denormalized_state(self, state):
        if self.norm_params.is_state_normalized:
            return (state * self.norm_params.state_std.reshape(-1)) + self.norm_params.state_mean.reshape(-1)
        else:
            return  state

    def query_normalized_action(self, action):
        if self.norm_params.is_action_normalized:
            return (action - self.norm_params.action_mean.reshape(-1))/self.norm_params.action_std.reshape(-1)
        else:
            return action

    def query_denormalized_action(self, action):
        if self.norm_params.is_action_normalized:
            return (action * self.norm_params.action_std.reshape(-1)) + self.norm_params.action_mean.reshape(-1)
        else:
            return action



    @staticmethod
    def populate_buffer(buffer, env, policy, episode_count=1,frame_count=None, render= False,  pad_attribute_fxn=None,
                                verbose=False):
        """

        :param exp_buffer:
        :param env:
        :param episodes:
        :param render:
        :param policy:
        :param frame_count:
        :param pad_attribute_fxn:
        :param verbose: Can be None,  True or 2 for maximum verboseness.
        :param policy_on_states:  if set to true , the policy provided is assumed to be on the state variable of unwrapped env
        :return:
        """
        # experience = obs, action, next_obs, reward, terminal_flag
        start_time = time.time()

        cum_rewards = 0
        frame_counter = 0
        eps_count = 0
        all_rewards = []
        for _ in range(episode_count):
            eps_count += 1
            episode_timesteps = 0
            done = False
            state = env.reset()
            ep_reward = 0
            episode_start = True
            while not done:
                episode_timesteps += 1
                if render:
                    env.render()

                action =policy(state)

                # Perform action and log results
                next_state, reward, done, info = env.step(action)
                ep_reward += reward

                # Only consider "done" if episode terminates due to failure condition
                done_float = float(done) if episode_timesteps < env._max_episode_steps else 0

                # Store data in replay buffer
                buffer.add(state, action, next_state, reward, done_float, done, episode_start)
                state = copy.copy(next_state)
                episode_start = False


            frame_counter += episode_timesteps
            all_rewards.append(ep_reward)

            if verbose:
                print(ep_reward, frame_count)

            if frame_count and frame_counter > frame_count:
                break

        print('Average Reward of collected trajectories:{}'.format(round(np.mean(all_rewards), 3)))

        info = {"all_rewards":all_rewards}
        if verbose:
            print("Data Collection Complete in {} Seconds".format(time.time()-start_time))
        return buffer, info



class StandardElasticBuffer(StandardBuffer):
    """
    Initializes an array for elements of transitions as per the maximum buffer size. 
    Keeps track of the crt_size. 
    Saves the buffer element-wise as numpy array. Fast save and retreival compared to pickle dumps. 
    """
    def __init__(self, state_shape, action_shape, buffer_size, device, batch_size = 64):
        
        super().__init__(state_shape, action_shape,  buffer_size, device, batch_size)
    

    def add(self, state, action, next_state, reward, done, episode_done=None, episode_start=None):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done


        self.ptr = self.ptr + 1
        self.crt_size = self.crt_size + 1

        if self.ptr >= self.max_size:
            self.increase_buffer_size(50000)

    def increase_buffer_size(self, increment=50000):
        self.max_size = self.max_size + increment
        self.state = np.concatenate((self.state, np.zeros((increment, *self.state_shape))))
        self.action = np.concatenate((self.action, np.zeros((increment, *self.action_shape))))
        self.next_state = np.concatenate((self.next_state, np.zeros((increment, *self.state_shape))))
        self.reward = np.concatenate((self.reward, np.zeros((increment, 1))))
        self.not_done =np.concatenate((self.not_done, np.zeros((increment, 1))))

        assert self.state.shape[0] == self.max_size

    def __repr__(self):
        return f"Standard Elastic Buffer: \n \
                Total number of transitions: {len(self)}/{self.max_size} \n \
                State Store Shape: {self.state.shape} \n \
                Action Store Shape: {self.action.shape} \n"

    def append_buffer(self, buffer):
        if (self.max_size - self.crt_size) < buffer.crt_size:
            self.increase_buffer_size(buffer.crt_size)
        
        self.state[self.ptr:self.ptr+buffer.crt_size] = buffer.state[:buffer.crt_size]
        self.action[self.ptr:self.ptr+buffer.crt_size] = buffer.action[:buffer.crt_size]
        self.next_state[self.ptr:self.ptr+buffer.crt_size] = buffer.next_state[:buffer.crt_size]
        self.reward[self.ptr:self.ptr+buffer.crt_size] =  buffer.reward[:buffer.crt_size]
        self.not_done[self.ptr:self.ptr+buffer.crt_size] = buffer.not_done[:buffer.crt_size]
        
        self.crt_size += buffer.crt_size
        self.ptr += buffer.ptr





# Data Dependencies
from d4rl.infos import DATASET_URLS as d4rl_envs
from d4rl.offline_env import OfflineEnv

def get_d4rl_dataset(env, d4rl_path):
    o_env = OfflineEnv(env)
    o_env.observation_space = env.observation_space
    o_env.action_space = env.action_space
    d4rl_dataset = d4rl.qlearning_dataset(o_env)
    assert "next_observations" in list(d4rl_dataset.keys())
    return d4rl_dataset

def convert_from_d4rl_dataset(config, buffer, d4rl_dataset):
    d_size = min(config.dataArgs.buffer_size,len(d4rl_dataset['observations']))
    
    for i in range(d_size):
        obs = d4rl_dataset['observations'][i]
        new_obs = d4rl_dataset['next_observations'][i]
        action = d4rl_dataset['actions'][i]
        reward = d4rl_dataset['rewards'][i]
        done_bool = bool(d4rl_dataset['terminals'][i])
        if i < (d_size - 1) or bool(d4rl_dataset['terminals'][d_size - 1]):
            buffer.add(obs, action, new_obs, reward, done_bool)
    return buffer


def load_from_d4rl(config,env):
    """
    defines a new buffer, and loads it from the pre-configured load path. 
    follows d4rl data definition for storing datasets
    """
    print('Loading buffer!')
    if config.envArgs.env_name in ["CartPole-cont-v1", "CartPole-cont-noisy-v1"]:
        action_shape = [1]  
    else:
        action_shape = env.action_space.shape

    buffer = StandardBuffer(state_shape = env.observation_space.shape,
                           action_shape = action_shape, 
                           batch_size=32, 
                           buffer_size=config.dataArgs.buffer_size,
                           device="cpu")
    
    
    if config.envArgs.env_name in ["CartPole-cont-v1", "CartPole-cont-noisy-v1"]:
        # replay_buffer.load(f"{args.output_dir}/buffers/{buffer_name}")
        fname = '%s-%s.hdf5' % (config.envArgs.env_name, config.dataArgs.buffer_name)
        fpath = os.path.join(config.dataArgs.data_dir,fname)
        d4rl_dataset = get_d4rl_dataset(env, fpath)
        train_buffer = convert_from_d4rl_dataset(config, buffer, d4rl_dataset)
        
    elif config.envArgs.env_name in d4rl_envs:
        d4rl_dataset = d4rl.qlearning_dataset(env)
        train_buffer = convert_from_d4rl_dataset(config, buffer, d4rl_dataset)    
    else:
        assert False, f"Data load logic for given env {config.dataArgs.buffer_size} is not defined."
    
    print('Loaded buffer!')
    
    return train_buffer


def generate_or_load_buffer(config, env):
    assert hasattr(config, "dataArgs")
    assert hasattr(config.dataArgs, "load_buffer")

    if config.dataArgs.load_buffer:
        return load_from_d4rl(config, env)
    else:
        return collect_buffer(config, env)

def collect_buffer(config, env):
    """
    Collect buffer. using a random policy. 
    returns a data buffer. 
    """

    print('Collecting buffer!')


    train_buffer = StandardBuffer(state_shape = env.observation_space.shape,
                               action_shape = [len(env.action_space.sample())], # for discrete settings. 
                               batch_size=32, 
                               buffer_size=config.dataArgs.buffer_size,
                               device="cpu")
    
    train_buffer, info = StandardBuffer.populate_buffer(train_buffer, env, 
                                           policy = lambda s:env.action_space.sample(),
                                           episode_count=99999, 
                                           frame_count=config.dataArgs.buffer_size)

    print('Collected buffer!')    
    
    return train_buffer
