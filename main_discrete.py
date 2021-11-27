# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

# %% [markdown]
# ## Imports

# %%
# Default Python Packages.
from collections import defaultdict
import logging
import sys

# Standard Python Packages.
import torch
import numpy as np
import gym

# Project Specific Dependencies 
from lmdp.data.buffer import StandardBuffer,ReplayBuffer, gather_data_in_buffer, get_iter_indexes
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.mdp.MDP_GPU_FACTORED import FullMDPFactored
from lmdp.utils.utils_eval import evaluate_on_env
from dacmdp.core.args import BaseConfig
import dacmdp.core.utils.server_helper as sh


# %%
# Logging Basics
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

# %% [markdown]
# ## Parse Args

# %%

if sh.in_notebook():
    options = "--no_cuda --env_name CartPole-v1 --buffer_size 50000 --MAX_S_COUNT 51000 --MAX_NS_COUNT 5 --mdp_build_k 5 --plcy_k 1 --normalize_by_distance --penalty_beta 1 --gamma 0.99"
    config = BaseConfig(options)
else:
    config = BaseConfig()
print(config)

# %% [markdown]
# ## Define Environment

# %%
env = gym.make(config.envArgs.env_name)
# Over Ride tran type count if discrete actions 
if isinstance(env.action_space, gym.spaces.discrete.Discrete):
    config.mdpBuildArgs.tran_type_count = env.action_space.n
config.mdpBuildArgs.tran_type_count

# %% [markdown]
# ## Define Buffer

# %%
train_buffer = StandardBuffer(state_shape = env.observation_space.shape,
                           action_shape = [1], 
                           batch_size=32, 
                           buffer_size=config.dataArgs.buffer_size,
                           device="cpu")


# %%
train_buffer, info = gather_data_in_buffer(train_buffer, env,policy = lambda s:np.random.randint(2), 
                                           episode_count=99999, frame_count=config.dataArgs.buffer_size)

# %% [markdown]
# ## Define Representation

# %%
from dacmdp.core.repr_nets import DummyNet
from dacmdp.core.mdp_agents.disc_agent import DACAgentBase, get_action_list_from_space
from dacmdp.mdp_wrappers import get_agent_model_class, get_repr_model


# %%



# %%
# config.mdpBuildArgs.dac_build


# %%
AgentClass = get_agent_model_class(config, config.mdpBuildArgs.dac_build)
repr_model = get_repr_model(config, config.reprArgs.repr_build)


logger.info('DAC Agent Class :'.ljust(25) + config.mdpBuildArgs.dac_build)
logger.info('Representation Build : '.ljust(25) + config.reprArgs.repr_build)


# %%



# %%
import time

# %% [markdown]
# ## Define and Solve MDP

# %%
# the action space is later replaced. the number of actions is relevant here. 
empty_MDP = FullMDPFactored(A= get_action_list_from_space(env.action_space), 
                    build_args=config.mdpBuildArgs, 
                    solve_args=config.mdpSolveArgs)

myAgent = AgentClass(action_space = env.action_space, 
                     seed_mdp= empty_MDP, 
                      repr_model= DummyNet(None), 
                      build_args=config.mdpBuildArgs, 
                      solve_args=config.mdpSolveArgs, 
                      eval_args=config.evalArgs).verbose()


myAgent.cache_buffer =  train_buffer
myAgent.process(train_buffer)


# %%



# %%
myAgent.parsed_unique_actions, myAgent.build_args.tran_type_count


# %%


# %% [markdown]
# ## Get MDP Policies

# %%
policies = {pi_name:lambda obs: int(pol(obs)[0]) for pi_name, pol in myAgent.policies.items()}


# %%


# %% [markdown]
# ## Evaluate policies

# %%
eval_rewards = {}
myAgent.eval_args.plcy_k = 1

for pi_name, pol in myAgent.policies.items(): 
    pol_wrapper = lambda obs: int(pol(obs)[0])
    print(f"Policy Name: {pi_name}")
    avg_rewards, info = evaluate_on_env(env, pol_wrapper, eps_count=50) #config.evalArgs.eval_episode_count,progress_bar=True)
    eval_rewards[pi_name] = avg_rewards
    time.sleep(1)


# %%
eval_rewards = {}
myAgent.eval_args.plcy_k = 11

for pi_name, pol in myAgent.policies.items(): 
    pol_wrapper = lambda obs: int(pol(obs)[0])
    print(f"Policy Name: {pi_name}")
    avg_rewards, info = evaluate_on_env(env, pol_wrapper, eps_count=50) #config.evalArgs.eval_episode_count,progress_bar=True)
    eval_rewards[pi_name] = avg_rewards
    time.sleep(1)


# %%
myAgent.


# %%



