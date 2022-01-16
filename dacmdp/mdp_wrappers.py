import os
import gym
import numpy as np
import policybazaar
import torch
import wandb
from .core.args import BaseConfig as MDPBaseConfig
from .core.mdp_agents.disc_agent import DACAgentBase 
from .core.mdp_agents.cont_agent import DACAgentContNNBaseline, DACAgentContNNBDeltaPred,  DACAgentThetaDynamics, DACAgentContNNBSARepr
from .core.mdp_agents.pi_agent import DACAgentThetaDynamicsBiasPi, DACAgentThetaDynamicsPlusPi,  DACAgentThetaDynamicsPlusPiWithOODEval
from .core.repr_nets import DummyNet, LatentDynamicsNet, LatentPolicyNetObs, LatentPolicyNetState

from lmdp.data.buffer import StandardBuffer, gather_data_in_buffer
from lmdp.mdp.MDP_GPU import FullMDP
from lmdp.mdp.MDP_GPU_FACTORED import FullMDPFactored
from tqdm import tqdm 
import sys

def add_wargs_ldynamics(parser):
    # Wrapper Arguments
    wrapperArgs = parser.add_argument_group(title="wrapperArgs", description='Environment Specification')
    wrapperArgs.add_argument("--cbplcy_size", help="Number of transtiions to collect for base policy given.", type=int, default=0)
    wrapperArgs.add_argument("--pretrained_label", help="Number of Tran Types to consider", type=int, default= 1)
    wrapperArgs.add_argument("--dynamics_id", help="run id of dynamics", type=str, default= "1gjbhk2j")
    wrapperArgs.add_argument("--save_path", help="Base folder for saving mdp results", type=str, default= "./")
    wrapperArgs.add_argument("--wdb_save_run_id", help="wandb id of offline dynamics", type=str, default= "20fgtxat")
    wrapperArgs.add_argument("--wdb_save_run_id_full", help="full run ide of offline dynamics", type=str, default = f"offline_drl_team/offline-dynamics/20fgtxat")
    wrapperArgs.add_argument("--upload_to_wandb", help="Upload MDP cache to wandb", action="store_true")
    wrapperArgs.add_argument("--download_from_wandb", help="Download MDP cache from wandb", action="store_true")
    return parser


def add_wargs_ope(parser):
    # Wrapper Arguments
    wrapperArgs = parser.add_argument_group(title="wrapperArgs", description='Environment Specification')
    wrapperArgs.add_argument("--cbplcy_size", help="Number of transtiions to collect for base policy given.", type=int, default=0)
    wrapperArgs.add_argument("--dynamics_id", help="run id of dynamics", type=str, default= "dummy")
    wrapperArgs.add_argument("--pretrained_label", help="Number of Tran Types to consider", type=int, default= 1)
    wrapperArgs.add_argument("--ope_base_dir", help="base dir of ope", type=str, default= "/nfs/guille/afern/users/shrestaa/new_projects/deep_ope")
    wrapperArgs.add_argument("--ope_policy_dir", help="policy dir of ope", type=str, default= "/nfs/hpc/share/frg-students/Policies/d4rl")
    wrapperArgs.add_argument("--save_path", help="Base folder for saving mdp results", type=str, default= "./")
    wrapperArgs.add_argument("--wdb_save_run_id", help="wandb id of offline dynamics", type=str, default= "20fgtxat")
    wrapperArgs.add_argument("--wdb_save_run_id_full", help="full run ide of offline dynamics", type=str, default = f"offline_drl_team/offline-dynamics/20fgtxat")
    wrapperArgs.add_argument("--upload_to_wandb", help="Upload MDP cache to wandb", action="store_true")
    wrapperArgs.add_argument("--download_from_wandb", help="Download MDP cache from wandb", action="store_true")
    return parser

DEFAULT_MDP_ARG_STR = "--wandb_project offline-dynamics --load_buffer --buffer_device cuda \
--MAX_S_COUNT 11000 --normalize_by_distance --save_mdp2cache --save_folder to_be_changed --ur 0 \
--gamma 0.99 --slip_prob 0.1 --default_mode GPU --eval_episode_count 100 --plcy_k 1 "


def header_hash(mdp_config):
    return os.path.join( f"mdp_results",
           f"D-{mdp_config.wrapperArgs.dynamics_id}" +
           f"_P-{mdp_config.wrapperArgs.pretrained_label}" +
           f"_A-{mdp_config.mdpBuildArgs.agent_class}" +
           f"_B-{mdp_config.dataArgs.buffer_size}",
           f"DET-TT{mdp_config.mdpBuildArgs.tran_type_count}" +
           f"_K-{mdp_config.mdpBuildArgs.mdp_build_k}" +
           f"_P-{mdp_config.mdpBuildArgs.penalty_beta}",
           f"_Dev-{mdp_config.dataArgs.buffer_device}/")

def get_mdp_config(arg_str = None, modify_parser_fxn= lambda x:x):
    mdp_arg_str = (DEFAULT_MDP_ARG_STR + " ".join(sys.argv[1:])) if arg_str is None else (DEFAULT_MDP_ARG_STR + arg_str)

    mdp_config = MDPBaseConfig("--env_name dummy")
    mdp_config.modify_parser = modify_parser_fxn
    mdp_config._initialize(mdp_arg_str)
    
    mdp_config.mdpBuildArgs.pretrained_label = mdp_config.wrapperArgs.pretrained_label
    mdp_config.mdpBuildArgs.save_folder = os.path.join(mdp_config.wrapperArgs.save_path,header_hash(mdp_config))
    mdp_config.mdpBuildArgs.base_folder = mdp_config.wrapperArgs.save_path
    mdp_config.logArgs.wdb_save_run_id = mdp_config.wrapperArgs.wdb_save_run_id
    mdp_config.logArgs.wdb_save_run_id_full = mdp_config.wrapperArgs.wdb_save_run_id_full
    os.makedirs(mdp_config.mdpBuildArgs.save_folder, exist_ok=True)
    return mdp_config
    
    
def get_buffer(env, config, dataset, policy):
    """ Convert Datset to Buffer of LMDP """
    train_buffer = StandardBuffer(state_shape=env.observation_space.shape,
                                  action_shape=env.action_space.shape,
                                  batch_size=64,
                                  buffer_size=config.dataArgs.buffer_size,
                                  device=config.dataArgs.buffer_device)

    if config.dataArgs.load_buffer:
        print('Loading buffer!')
        for i in range(min(config.dataArgs.buffer_size, len(dataset['observations'])) - 1):
            obs = dataset['observations'][i]
            new_obs = dataset['observations'][i + 1]
            action = dataset['actions'][i]
            reward = dataset['rewards'][i]
            done_bool = bool(dataset['terminals'][i])
            train_buffer.add(obs, action, new_obs, reward, done_bool)
        print('Loaded buffer!')
        
    if config.wrapperArgs.cbplcy_size > 0:
        train_buffer = gather_data_in_buffer(train_buffer, env, policy, episode_count = 999, frame_count = config.wrapperArgs.cbplcy_size )
        
    return train_buffer



# def get_mdp_agent_wrapper(env, agent_model_class, repr_model, mdp_config, build=True, dataset=None, dataset_size = None, factored_mdp = True):    
    
#     # Prep Logic
#     if dataset is not None:
#         assert dataset_size is not None
#         mdp_config.dataArgs.buffer_size = dataset_size
#         mdp_config.mdpBuildArgs.MAX_S_COUNT = int(1.1*dataset_size)

#     if mdp_config.wrapperArgs.download_from_wandb:
#             wandb_restore_mdp_params(mdp_config)
            
#     if build:
#         mdp_config.mdpBuildArgs.rebuild_mdpfcache = False
#         mdp_config.mdpBuildArgs.save_mdp2cache = True
#     else:
#         mdp_config.mdpBuildArgs.rebuild_mdpfcache = True
#         mdp_config.mdpBuildArgs.save_mdp2cache = False
    

#     # Main Logic
#     train_buffer = get_buffer(env, mdp_config, dataset, repr_model.predict_action_single)
    
#     tt_action_space = get_tran_types(mdp_config.mdpBuildArgs.tran_type_count)

#     if factored_mdp:
#         empty_MDP = FullMDPFactored(A=tt_action_space,
#                         build_args=mdp_config.mdpBuildArgs,
#                         solve_args=mdp_config.mdpSolveArgs)
#     else:
#         empty_MDP = FullMDP(A=tt_action_space,
#                         build_args=mdp_config.mdpBuildArgs,
#                         solve_args=mdp_config.mdpSolveArgs)
    

#     myAgent = agent_model_class(seed_mdp=empty_MDP,
#                          repr_model=repr_model,
#                         build_args=mdp_config.mdpBuildArgs,
#                             solve_args=mdp_config.mdpSolveArgs,
#                              eval_args=mdp_config.evalArgs,
#                             action_space=env.action_space,).verbose()
        
#     myAgent.process(train_buffer, match_hash=False)

    
#     return myAgent


def wandb_restore_mdp_params(config):
    wandb.init(id=config.logArgs.wdb_save_run_id,
               resume='allow',
               project='offline-dynamics',
               entity='offline_drl_team')

    file_names = ["_qD_cpu.npy", "_vD_cpu.npy", "_s_qD_cpu.npy", "_s_vD_cpu.npy", "_stt2a_idx_matrix.npy", "_hmaps.pk"]
    for fn in file_names:
        f = os.path.join(config.mdpBuildArgs.save_folder, fn)
        f = "/".join([d for d in f.split("/") if d != '.'])
        wandb.restore(name=f,
                      run_path=config.logArgs.wdb_save_run_id_full,
                      replace=True, root=config.mdpBuildArgs.base_folder)
    wandb.finish()


def wandb_save_mdp_params(config):
    wandb.init(id=config.logArgs.wdb_save_run_id,
               resume='allow',
               project='offline-dynamics',
               entity='offline_drl_team')

    file_names = ["_qD_cpu.npy", "_vD_cpu.npy", "_s_qD_cpu.npy", "_s_vD_cpu.npy", "_stt2a_idx_matrix.npy", "_hmaps.pk"]
    for fn in file_names:
        f = os.path.join(config.mdpBuildArgs.save_folder, fn)
        wandb.save(f, policy='now', base_path=config.mdpBuildArgs.base_folder)
    wandb.finish()

    
    
def get_agent_model_class(config, dac_build):
    """
    Holds the map of dac_build and dac agent class. 
    Returns the correct DACMDP Agent Class 
    input: config - current context and arguments. 
    input: dac_build - name of the dac build for the dac agent class.  
    """
    if dac_build == "DACAgentBase": return DACAgentBase;
    elif dac_build == "DACAgentContNNBaseline": return DACAgentContNNBaseline;
    elif dac_build == "DACAgentContNNBDeltaPred": return DACAgentContNNBDeltaPred;
#     elif dac_build == "StochasticAgent": return DACAgent;
#     elif dac_build == "StochasticAgent_o": return DACAgent;
#     elif dac_build == "StochasticAgent_s": return DACAgent;
#     elif dac_build == "StochasticAgent_sa": return DACAgentSARepr;
#     elif dac_build == "StochasticAgentWithDelta_o": return DACAgentDelta;
#     elif dac_build == "StochasticAgentWithDelta_s": return DACAgentDelta;
#     elif dac_build == "StochasticAgentWithParametricPredFxn_o": return DACAgentThetaDynamics;
#     elif dac_build == "StochasticAgentWithParametricPredFxn_s": return DACAgentThetaDynamics;
#     elif dac_build == "StchExtendedAgent_o": return DACAgentThetaDynamicsPlusPi;
#     elif dac_build == "StchExtendedAgent_s": return DACAgentThetaDynamicsPlusPi;
#     elif dac_build == "StchExtendedAgentSafeBase_o":return DACAgentThetaDynamicsPlusPiWithOODEval;
#     elif dac_build == "StchExtendedAgentSafeBase_s":return DACAgentThetaDynamicsPlusPiWithOODEval;
#     elif dac_build == "PIAgent_o" : return DACAgentThetaDynamicsBiasPi;
#     elif dac_build == "PIAgent_s" : return DACAgentThetaDynamicsBiasPi;
    else: assert False, "Agent Model Not Found"
    
def get_repr_model(config, repr_build):
    if repr_build == "identity": return DummyNet();
    elif repr_build == "td3_bc": return DummyNet();
#     elif repr_build == "DeterministicAgent_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
#     elif repr_build == "StochasticAgent": return DummyNet();
#     elif repr_build == "StochasticAgent_o": return DummyNet();
#     elif repr_build == "StochasticAgent_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
#     elif repr_build == "StochasticAgent_sa": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
#     elif repr_build == "StochasticAgentWithDelta_o": return DummyNet();
#     elif repr_build == "StochasticAgentWithDelta_s": return LatentDynamicsNet(config.reprArgs.dynamics_model, config.dataArgs.buffer_device);
#     elif repr_build == "StochasticAgentWithParametricPredFxn_o": return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "StochasticAgentWithParametricPredFxn_s": return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "StchExtendedAgent_o": return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "StchExtendedAgent_s": return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "StchExtendedAgentSafeBase_o":return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "StchExtendedAgentSafeBase_s":return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "PIAgent_o" : return LatentPolicyNetObs(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
#     elif repr_build == "PIAgent_s" : return LatentPolicyNetState(config.reprArgs.dynamics_model, config.reprArgs.policy_model, config.dataArgs.buffer_device);
    else: 
        print("Agent Class Not Found","returning dummy representation")
        return DummyNet();






























