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
from utils_mdp_repr_wrappers import get_agent_model_class, get_repr_model
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

options = "--seed 0 \
--env_name CartPole-cont-v1 --buffer_name random --load_buffer --buffer_size 100000 \
--data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
--tran_type_count 10 --MAX_S_COUNT 110000 --MAX_NS_COUNT 5 --mdp_build_k 5 --normalize_by_distance --penalty_beta 10 \
--gamma 0.99 --plcy_k 1 --no_wandb_logging \
--repr_build identity --dac_build DACAgentContNNBaseline \
--wandb_project cartpoleCont --wandb_entity dacmdp --wandb_id TEST-MCNB-TT-10-P10-K5 \
--results_folder /nfs/hpc/share/shrestaa/storage/dac_ws/ \
--repr_save_dir /nfs/guille/afern/users/shrestaa/dac_workspace/src/orepr/orepr/third_party_srcs/TD3_BC/models"
config = BaseConfig(options if sh.in_notebook() else None)
if config.logArgs.no_wandb_logging:
    print("Skipped Logging at Wandb")
else:
    config.logArgs.wandb_id  += hex(int(time.time()/1000))
    wandb_logger.init( id = config.logArgs.wandb_id ,
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
                   repr_model= repr_model, 
                      build_args=config.mdpBuildArgs, 
                      solve_args=config.mdpSolveArgs, 
                      eval_args=config.evalArgs,).verbose()

myAgent.process(train_buffer)


# ## Get MDP Policies
eval_rewards = {}
for pi_name, pi in myAgent.policies.items(): 
    avg_rewards, info = evaluate_on_env(env, pi, eps_count=config.evalArgs.eval_episode_count,progress_bar=True)
    print(f"Policy Name: {pi_name} Avg Rewards:{avg_rewards}")
    # eval_rewards[pi_name] = d4rl.get_normalized_score(config.envArgs.env_name, avg_rewards) * 100
    all_rewards = [v['sum_reward'] for k,v in info['run_info'].items()]
    eval_rewards[pi_name+"_mean"] = d4rl.get_normalized_score(config.envArgs.env_name, np.mean(all_rewards)) * 100
    eval_rewards[pi_name+"_median"] = d4rl.get_normalized_score(config.envArgs.env_name, np.median(all_rewards)) * 100
    time.sleep(1)

## Get repr evaluations
if config.reprArgs.repr_build == "td3_bc":
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    myAgent.repr_model.td3_net.actor = myAgent.repr_model.td3_net.actor.to(device)
    myAgent.repr_model.td3_net.actor = myAgent.repr_model.td3_net.actor.to(device)

    def TD3BC_policy(state):
        state = (np.array(state).reshape(1,-1) - myAgent.repr_model.mean)/myAgent.repr_model.std
        action = myAgent.repr_model.td3_net.select_action(state)
        return action

    td3_rewards, td3_info = evaluate_on_env(env, TD3BC_policy, 
                                            eps_count=config.evalArgs.eval_episode_count,progress_bar=True)
    all_rewards = [v['sum_reward'] for k,v in td3_info['run_info'].items()]
    eval_rewards["TD3"+"_mean"] = d4rl.get_normalized_score(config.envArgs.env_name, np.mean(all_rewards)) * 100
    eval_rewards["TD3"+"_median"] = d4rl.get_normalized_score(config.envArgs.env_name, np.median(all_rewards)) * 100
    

    myAgent.repr_model.td3_net.actor = myAgent.repr_model.td3_net.actor.to("cpu")
    myAgent.repr_model.td3_net.actor = myAgent.repr_model.td3_net.actor.to("cpu")

    
# ## Wandb Logging
if config.logArgs.no_wandb_logging:
    print("Skipped Logging Metrics at Wandb")
else:
    wandb_logger.log(eval_rewards)    
    for name, distr in myAgent.mdp_distributions.items():
        wandb_logger.log({"MDP Histogram" + "/" + name: go.Figure(data=[go.Histogram(x=distr)]), 
                         "mdp_frame_count": len(myAgent.mdp_T.s2i)})
        
        
        
# log utils #  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########
from utils_log import write_gif 

    
def get_value_and_cost_from_obs(env, obs):
    state = myAgent.repr_model.encode_obs_single(obs)
    nn_s_id, dist = list(myAgent.s_kdTree.get_knn_idxs(state, 1).items())[0]
    expected_cost = dist * myAgent.build_args.penalty_beta 
    expected_penalty= myAgent.mdp_T.cD_cpu[nn_s_id] + expected_cost
    expected_value= myAgent.mdp_T.vD_cpu[nn_s_id] - expected_cost

    return {"val":expected_value[0], "penalty":expected_penalty[0], "cost":expected_cost}


if config.logArgs.log_video:
    # Get Eval Renders.
    eval_rewards = {}
    pi_name, pi = "optimal",  myAgent.policies["optimal"] 
    avg_rewards, info = evaluate_on_env(env, pi, eps_count=config.logArgs.log_video_count,
                                        progress_bar=True, render_mode = "rgb_array", render = True, 
                                       every_step_hook=get_value_and_cost_from_obs)
    print(f"Policy Name: {pi_name} Avg Rewards:{avg_rewards}")
    eval_rewards[pi_name] = d4rl.get_normalized_score(config.envArgs.env_name, avg_rewards) * 100

    # Get Eval Dashboard Renders. 
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    eval_render_path = {}

    for run_id in info['run_info']:
        video_array = np.array(info['run_info'][run_id]['renders']).astype( np.uint8)
        gif_path = write_gif(video_array, 
                  [     info['run_info'][run_id]['rewards'], 
                        [out['cost'] for out in info['run_info'][run_id]['hook_outs']],
                        [out['penalty'] for out in info['run_info'][run_id]['hook_outs']], 
                        [out['val'] for out in info['run_info'][run_id]['hook_outs']],], 
                  [    {"title": "Episode Rewards (True)", "xaxis_title":"Time Step", "yaxis_title":"Step Reward"},
                       {"title": "Episode Cost (DAC-Pred)", "xaxis_title":"Time Step", "yaxis_title":"Step Cost"},
                       {"title": "Episode Penalties (DAC-Pred)", "xaxis_title":"Time Step", "yaxis_title":"Step Penalty"},
                       {"title": "Episode Values (DAC-Pred)", "xaxis_title":"Time Step", "yaxis_title":"Step Value"}, ], 
                  os.path.join(config.logArgs.results_folder, f"eval_render_{run_id}.gif"), scaling = 0.5, save_mp4 = True)
        eval_render_path[f"Eval Render - {run_id}"] = gif_path
        
    
    # Log to Wandb. 
    if config.logArgs.no_wandb_logging:
        print("Skipped Logging Videos at Wandb")
    else:
        wandb_logger.log({**{"Optimal Policy"+run_id:wandb_logger.Video(gif_path, fps=24, format="gif") for run_id,gif_path in eval_render_path.items()}
                             })
    ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########  ##########