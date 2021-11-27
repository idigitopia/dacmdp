# %%
# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# ## Imports
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
import plotly.graph_objects as go
import wandb as wandb_logger

# Project Specific Dependencies 
from beta_gym.cont_cartpole import ContinuousCartPoleEnv
from lmdp.data.buffer import StandardBuffer,ReplayBuffer, gather_data_in_buffer, get_iter_indexes
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.mdp.MDP_GPU_FACTORED import FullMDPFactored
from lmdp.utils.utils_eval import evaluate_on_env
from dacmdp.core.args import BaseConfig
from dacmdp.core.utils import server_helper as sh
from dacmdp.exp_track.experiments import ExpPool
from dacmdp.core.repr_nets import DummyNet
from dacmdp.core.mdp_agents.disc_agent import CustomActionSpace, get_action_list_from_space, get_one_hot_list
from dacmdp.core.mdp_agents.cont_agent import DACAgentContNNBaseline
from dacmdp.mdp_wrappers import get_agent_model_class, get_repr_model
from buffer_helper import collect_buffer, load_buffer

# Data Dependencies
import d4rl 
from d4rl.infos import DATASET_URLS as d4rl_envs
d4rl.infos.REF_MIN_SCORE.update({"CartPole-cont-v1":15})
d4rl.infos.REF_MAX_SCORE.update({"CartPole-cont-v1":500})


# %%
# Logging Basics
logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
                     level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)



# %%
# ## Parse Args

options = "--seed 0 \
--env_name CartPole-cont-v1 --buffer_name random --load_buffer --buffer_size 100000 \
--data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
--tran_type_count 10 --MAX_S_COUNT 110000 --MAX_NS_COUNT 5 --mdp_build_k 5 --normalize_by_distance --penalty_beta 10 \
--gamma 0.99 --plcy_k 1 --no_wandb_logging \
--repr_build identity --dac_build DACAgentContNNBaseline \
--wandb_project cartpoleCont --wandb_entity dacmdp --wandb_id TEST-MCNB-TT-10-P10-K5"
config = BaseConfig(options if sh.in_notebook() else None)
if config.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    wandb_logger.init( id = config.logArgs.wandb_id,
            entity=config.logArgs.wandb_entity,
            project=config.logArgs.wandb_project,
            config = config.flat_args,
            resume = "allow")
print(config)



# %%
# ## Define Environment
from beta_gym.cont_cartpole import ContinuousCartPoleEnv
env = ContinuousCartPoleEnv() if config.envArgs.env_name == "CartPole-cont-v1" else gym.make(config.envArgs.env_name)

# %% [markdown]
# ## Define Buffer

if config.dataArgs.load_buffer:
    train_buffer = load_buffer(config, env)
else: 
    train_buffer = collect_buffer(config, env)


# %%
# Define the dac agent and representation model. 
AgentClass = get_agent_model_class(config, config.mdpBuildArgs.dac_build)
repr_model = get_repr_model(config, config.reprArgs.repr_build)

logger.info('DAC Agent Class :'.ljust(25) + config.mdpBuildArgs.dac_build)
logger.info('Representation Build : '.ljust(25) + config.reprArgs.repr_build)



# In[532]:

tt_action_space = get_one_hot_list(config.mdpBuildArgs.tran_type_count)

empty_MDP = FullMDPFactored(A= tt_action_space, 
                    build_args=config.mdpBuildArgs, 
                    solve_args=config.mdpSolveArgs)

myAgent = AgentClass(action_space = env.action_space, 
                    seed_mdp= empty_MDP, 
                   repr_model= DummyNet(None), 
                      build_args=config.mdpBuildArgs, 
                      solve_args=config.mdpSolveArgs, 
                      eval_args=config.evalArgs,).verbose()

myAgent.process(train_buffer)


# ## Get MDP Policies
eval_rewards = {}
for pi_name, pi in myAgent.policies.items(): 
    avg_rewards, info = evaluate_on_env(env, pi, eps_count=config.evalArgs.eval_episode_count,progress_bar=True)
    print(f"Policy Name: {pi_name} Avg Rewards:{avg_rewards}")
    eval_rewards[pi_name] = d4rl.get_normalized_score(config.envArgs.env_name, avg_rewards) * 100
    time.sleep(1)

# ## Wandb Logging
if config.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    wandb_logger.log(eval_rewards)    
    for name, distr in myAgent.mdp_distributions.items():
        wandb_logger.log({"MDP Histogram" + "/" + name: go.Figure(data=[go.Histogram(x=distr)]), 
                         "mdp_frame_count": len(myAgent.mdp_T.s2i)})