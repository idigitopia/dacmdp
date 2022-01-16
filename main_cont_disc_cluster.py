
# from IPython import get_ipython
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
from dacmdp.core.mdp_agents.disc_agent import DACAgentBase, CustomActionSpace, get_action_list_from_space
from dacmdp.mdp_wrappers import get_agent_model_class, get_repr_model
from utils_buffer import collect_buffer, load_buffer

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
options = "--no_cuda --env_name CartPole-cont-v1 --buffer_size 5000 --MAX_S_COUNT 51000 --MAX_NS_COUNT 5 --mdp_build_k 5 --plcy_k 1 --normalize_by_distance --penalty_beta 1 --gamma 0.99 --tran_type_count 20 --buffer_name random --load_buffer --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/"
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
# ## Instantiate the clustering model
all_actions = train_buffer.action[:train_buffer.crt_size]
KMeans_model = KMeans(n_clusters = config.mdpBuildArgs.tran_type_count).fit(all_actions)
new_disc_actions = KMeans_model.predict(all_actions)
disc_action_space = CustomActionSpace([tuple(np.array(a).astype(np.float32))
                                            for a in KMeans_model.cluster_centers_])

# ## Convert the actions to discrete action space.
for i, cluster_id in enumerate(new_disc_actions):     
    train_buffer.action[i] = KMeans_model.cluster_centers_[cluster_id]

# %%
# Define the dac agent and representation model. 
AgentClass = get_agent_model_class(config, config.mdpBuildArgs.dac_build)
repr_model = get_repr_model(config, config.reprArgs.repr_build)

logger.info('DAC Agent Class :'.ljust(25) + config.mdpBuildArgs.dac_build)
logger.info('Representation Build : '.ljust(25) + config.reprArgs.repr_build)


# ## Define and Solve MDP

# the action space is later replaced. the number of actions is relevant here. 
empty_MDP = FullMDPFactored(A= get_action_list_from_space(disc_action_space), 
                    build_args=config.mdpBuildArgs, 
                    solve_args=config.mdpSolveArgs)

# Here action space refers to the list of actions. 
myAgent = AgentClass(action_space = disc_action_space, 
                     seed_mdp= empty_MDP, 
                      repr_model= DummyNet(None), 
                      build_args=config.mdpBuildArgs, 
                      solve_args=config.mdpSolveArgs, 
                      eval_args=config.evalArgs).verbose()

myAgent.cache_buffer =  train_buffer
myAgent.process(train_buffer)


# ## Get MDP Policies
eval_rewards = {}
for pi_name, pol in myAgent.policies.items(): 
    print(f"Policy Name: {pi_name}")
    pol_wrapper = lambda obs: np.array(pol(obs))
    avg_rewards, info = evaluate_on_env(env, pol_wrapper, eps_count = config.evalArgs.eval_episode_count, progress_bar=True)
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