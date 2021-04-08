import argparse
import torch
from munch import munchify
import itertools 
import hashlib
from datetime import datetime
from os import path
import os

def get_args(s = None):
    parser = argparse.ArgumentParser()
    
    # Env Arguments
    envArgs = parser.add_argument_group(title="envArgs", description='Environment Specification')
    envArgs.add_argument("--env_name", help="environment name", type=str, default="CartPole-v1")
    envArgs.add_argument("--seed", help="choice of seed to use for single start state.", type=int, default=4444)

    # Log Arguments
    logArgs = parser.add_argument_group(title="logArgs", description='Logger / Save Specification')
    logArgs.add_argument("--exp_id", help="Id of the experiment", type=str, default="test_run")
    logArgs.add_argument("--exp_meta", help="meta data of the Experiment", type=str, default="test experiment")
    
    logArgs.add_argument("--wandb_project", help="Wandb Project", type=str, default="DACMDP_Cont_V0")
    logArgs.add_argument("--no_wandb_logging", help="set to log a video of the evaluation run", action="store_true")
    
    logArgs.add_argument("--log_mdp_sol", help="Set to log the mdp value vector to the wnadb server.", action="store_true")
    logArgs.add_argument("--log_mdp_attributes", help="Set to log different charactersitic distributiosn of the mdp.", action="store_true")
    logArgs.add_argument("--log_video", help="set to log a video of the evaluation run", action="store_true")
    logArgs.add_argument("--log_video_count", help="Number of episodes to evaluate and log the video of.", type=int, default=2)


    # System Arguments 
    sysArgs = parser.add_argument_group(title="sysArgs", description='System Specification')
    sysArgs.add_argument("--no_cuda", help="environment name", action="store_true")
#     sysArgs.add_argument("--device", help="environment name", type=str, default="cpu")

    # Buffer / Dataset Arguments
    dataArgs = parser.add_argument_group(title="dataArgs", description="dataset / buffer arguments")
    dataArgs.add_argument("--data_dir", help="Directory where the data is stored", type=str, default= "./")
    dataArgs.add_argument("--buffer_name", help="Name Identifier of the buffer", type=str, default= "default")
    dataArgs.add_argument("--buffer_size", help="Size of the buffer", type=int, default= 100000)
    dataArgs.add_argument("--load_buffer", help="Do a bellman backups every __k frames", action="store_true")
    dataArgs.add_argument("--buffer_device", help="Default device to use for the sampled tensors", type=str, default= "cpu")

    # MDP Build parameters
    mdpBuildArgs = parser.add_argument_group(title="mdpBuildArgs", description="MDP build arguments")
    mdpBuildArgs.add_argument("--rmax_reward", help="Default reward for RMAX reward", type=int, default= 10000)
    mdpBuildArgs.add_argument("--balanced_exploration", help="Try to go to all states equally often", type=int, default= 0)
    mdpBuildArgs.add_argument("--rmax_threshold", help="Number of travesal before annealing rmax reward", type=int, default= 2)
    mdpBuildArgs.add_argument("--MAX_S_COUNT", help="maximum state count  for gpu rewource allocation", type=int, default= 250000)
    mdpBuildArgs.add_argument("--MAX_NS_COUNT", help="maximum nest state count  for gpu rewource allocation", type=int, default=20)
    mdpBuildArgs.add_argument("--fill_with", help="Define how to fill missing state actions", type=str, default="0Q_src-KNN", choices=["0Q_src-KNN", "1Q_dst-KNN","kkQ_dst-1NN", "none"])
    mdpBuildArgs.add_argument("--mdp_build_k", help="Number of Nearest neighbor to consider k", type=int, default= 1)
    mdpBuildArgs.add_argument("--knn_delta", help="Define the bias parmeter for nearest neighbor distance", type=float, default=1e-8)
    mdpBuildArgs.add_argument("--penalty_type", help="penalized predicted rewards based on the distance to the state", type=str, default="linear", choices=["none", "linear", "exponential"])
    mdpBuildArgs.add_argument("--penalty_beta", help="beta multiplyer for penalizing rewards based on distance", type=float, default= 1)
    mdpBuildArgs.add_argument("--filter_with_abstraction", help="Set to true, to filter the states to be added based on the radius.", type=int, default= 0)
    mdpBuildArgs.add_argument("--normalize_by_distance", help="set it on if the transition probabilities should be normalized by distance.", action = "store_true")
    mdpBuildArgs.add_argument("--tran_type_count", help="Number of Tran Types to consider", type=int, default= 10)
    mdpBuildArgs.add_argument("--ur", help="Reward for unknown transition, default = -1000.", type=float, default= -1000)
    
    mdpBuildArgs.add_argument("--rebuild_mdpfcache", help="Set to rebuild the mdp from cache solution.", action="store_true")
    mdpBuildArgs.add_argument("--save_mdp2cache", help="Set to cache th esolution vectors", action="store_true")
    mdpBuildArgs.add_argument("--save_folder", help="Folder where the cached vectors will be saved.", type=str, default= "./def_run")


    # MDP solve and lift up parameters
    mdpSolveArgs = parser.add_argument_group(title="mdpSolveArgs", description="MDP build arguments")
    mdpSolveArgs.add_argument("--default_mode", help="Default device to use for Solving the MDP", type=str, default= "GPU")
    mdpSolveArgs.add_argument("--gamma", help="Discount Factor for Value iteration", type=float, default= 0.99)
    mdpSolveArgs.add_argument("--slip_prob", help="Slip probability for safe policy", type=float, default= 0.1)
    mdpSolveArgs.add_argument("--target_vi_error", help="target belllman backup error for considering solved", type=float, default= 0.001)
    mdpSolveArgs.add_argument("--bellman_backup_every", help="Do a bellman backups every __k frames", type=int, default= 100)
    mdpSolveArgs.add_argument("--n_backups", help="The number of backups for every backup step", type=int, default= 10)

    # Evaluation Parameters
    evalArgs = parser.add_argument_group(title="evalArgs", description="Evaluation Arguments")
    evalArgs.add_argument("--eval_episode_count", help="Number of episodes to evaluate the policy", type=int, default=50)
    evalArgs.add_argument("--soft_at_plcy", help="Sample according to Q values rather than max action", action="store_true")
    evalArgs.add_argument("--plcy_k", help="Set the lift up parameter policy_k you want to test with",  type= int, default=1)
    evalArgs.add_argument("--plcy_sweep", help="Set if evaluate the policy under different nearst neighbor numbers k", action="store_true")
    evalArgs.add_argument("--plcy_sweep_k", help="List the sweep lift up parameter policy_k you want to test with", nargs="+", type= int, default=[1,5,11]) 

    # parser.add_argument("--all_gammas", help="Name of the Environment to guild", type=int, default= "[0.1 ,0.9 ,0.99 ,0.999]")
    
    
    # Parse Arguments
    parsedArgs = parser.parse_args(s.split(" ") if s is not None else None)
    parsedArgs.device = 'cuda' if (not parsedArgs.no_cuda) and torch.cuda.is_available() else 'cpu'

    # Process Parsed arguments
    argGroups = {}
    for group in parser._action_groups:
        title_map = {"positional arguments": "posArgs", "optional arguments":"optArgs"}
        title = title_map[group.title] if group.title in title_map else group.title
        argGroups[title]={a.dest:getattr(parsedArgs,a.dest,None) for a in group._group_actions}

    return munchify(argGroups)

class BaseConfig(object):
    def __init__(self, arg_str=None):
        
        # Initialize Base Arguments
        arg_groups = self._get_arg_groups(arg_str)
        self.arg_gnames = list(arg_groups)
        for group_name, args in arg_groups.items():
            setattr(self,group_name,args)

        self.seed_additional_arguments()
    
    def seed_additional_arguments(self):
        self.logArgs.wandb_id = self.pad_datetime(self.logArgs.exp_id) \
                                if self.logArgs.wandb_id == "default" else self.logArgs.wandb_id
        self.logArgs.results_folder = path.join("results",self.envArgs.env_name,self.logArgs.wandb_id) \
                                        if self.logArgs.results_folder == "default" else self.logArgs.results_folder
        self.mdpBuildArgs.save_folder = path.join(self.logArgs.results_folder, "mdp_sol") \
                                        if self.mdpBuildArgs.save_folder == "default" else self.mdpBuildArgs.save_folder
        
        if self.logArgs.cache_mdp2wandb and not self.mdpBuildArgs.save_mdp2cache:
            print("Setting save 2 cache as True, cannot upload to Wandb without Saving")
            self.mdpBuildArgs.save_mdp2cache = True
        
        os.makedirs(self.logArgs.results_folder, exist_ok = True)
        os.makedirs(self.mdpBuildArgs.save_folder, exist_ok = True)
        
    def pad_datetime(self,s):
        return s + "-" + datetime.now().strftime('%b%d_%H-%M-%S')
    
    @property
    def flat_args(self):
        args = {}
        for grp_name in self.arg_gnames:
            group = getattr(self,grp_name)
            for arg_name, arg_value in group.items():
                args[f"{grp_name}:{arg_name}"] = arg_value
        return args
    
    def __str__(self):
        get_header = lambda title: "#" * 45 + " " * 4 + title + " " * 4 + "#" * 45
        out_str = get_header("All Arguments") + "\n"
        for grp_name in self.arg_gnames: 
            group = getattr(self,grp_name)
            out_str+="\n" + get_header(grp_name) + "\n"
            for arg1, arg2 in itertools.zip_longest(*[iter(group.items())]*2):
                out_str+=f"{str(arg1[0]).ljust(30)}:{str(arg1[1]).ljust(30)}" + \
                      (f"{str(arg2[0]).ljust(30)}:{str(arg2[1])}" if arg2 else "") + "\n"

        out_str+="\n" + "#" * len(get_header("All Arguments"))
        return out_str


    def _get_arg_groups(self,s = None):
        parser = argparse.ArgumentParser()

        # Env Arguments
        envArgs = parser.add_argument_group(title="envArgs", description='Environment Specification')
        envArgs.add_argument("--env_name", help="environment name", type=str, default="CartPole-v1")
        envArgs.add_argument("--seed", help="choice of seed to use for single start state.", type=int, default=4444)

        # Log Arguments
        logArgs = parser.add_argument_group(title="logArgs", description='Logger / Save Specification')
        logArgs.add_argument("--exp_id", help="Id of the experiment", type=str, default="test_run")
        logArgs.add_argument("--exp_meta", help="meta data of the Experiment", type=str, default="test experiment")

        logArgs.add_argument("--no_wandb_logging", help="set to log a video of the evaluation run", action="store_true")
        logArgs.add_argument("--wandb_project", help="Wandb Project", type=str, default="DACMDP_Cont_V0")
        logArgs.add_argument("--wandb_entity", help="Wandb Entity", type=str, default="xanga")
        logArgs.add_argument("--wandb_id", help="Wandb Id", type=str, default="default")
        logArgs.add_argument("--cache_mdp2wandb", help="Set to upload the mdp Solution vectors to Wandb", action = "store_true")
        logArgs.add_argument("--log_mdp_attributes", help="Set to log different charactersitic distributiosn of the mdp.", action="store_true")
        logArgs.add_argument("--log_video", help="set to log a video of the evaluation run", action="store_true")
        logArgs.add_argument("--log_video_count", help="Number of episodes to evaluate and log the video of.", type=int, default=2)
        
        logArgs.add_argument("--results_folder", help="base folder for results", type=str, default = "default")
        
        # System Arguments 
        sysArgs = parser.add_argument_group(title="sysArgs", description='System Specification')
        sysArgs.add_argument("--no_cuda", help="environment name", action="store_true")
    #     sysArgs.add_argument("--device", help="environment name", type=str, default="cpu")

        # Buffer / Dataset Arguments
        dataArgs = parser.add_argument_group(title="dataArgs", description="dataset / buffer arguments")
        dataArgs.add_argument("--data_dir", help="Directory where the data is stored", type=str, default= "./")
        dataArgs.add_argument("--buffer_name", help="Name Identifier of the buffer", type=str, default= "default")
        dataArgs.add_argument("--buffer_size", help="Size of the buffer", type=int, default= 100000)
        dataArgs.add_argument("--load_buffer", help="Do a bellman backups every __k frames", action="store_true")
        dataArgs.add_argument("--buffer_device", help="Default device to use for the sampled tensors", type=str, default= "cpu")

        # MDP Build parameters
        mdpBuildArgs = parser.add_argument_group(title="mdpBuildArgs", description="MDP build arguments")
        mdpBuildArgs.add_argument("--rmax_reward", help="Default reward for RMAX reward", type=int, default= 10000)
        mdpBuildArgs.add_argument("--balanced_exploration", help="Try to go to all states equally often", type=int, default= 0)
        mdpBuildArgs.add_argument("--rmax_threshold", help="Number of travesal before annealing rmax reward", type=int, default= 2)
        mdpBuildArgs.add_argument("--MAX_S_COUNT", help="maximum state count  for gpu rewource allocation", type=int, default= 250000)
        mdpBuildArgs.add_argument("--MAX_NS_COUNT", help="maximum nest state count  for gpu rewource allocation", type=int, default=20)
        mdpBuildArgs.add_argument("--fill_with", help="Define how to fill missing state actions", type=str, default="0Q_src-KNN", choices=["0Q_src-KNN", "1Q_dst-KNN","kkQ_dst-1NN", "none"])
        mdpBuildArgs.add_argument("--mdp_build_k", help="Number of Nearest neighbor to consider k", type=int, default= 1)
        mdpBuildArgs.add_argument("--knn_delta", help="Define the bias parmeter for nearest neighbor distance", type=float, default=1e-8)
        mdpBuildArgs.add_argument("--penalty_type", help="penalized predicted rewards based on the distance to the state", type=str, default="linear", choices=["none", "linear", "exponential"])
        mdpBuildArgs.add_argument("--penalty_beta", help="beta multiplyer for penalizing rewards based on distance", type=float, default= 1)
        mdpBuildArgs.add_argument("--filter_with_abstraction", help="Set to true, to filter the states to be added based on the radius.", type=int, default= 0)
        mdpBuildArgs.add_argument("--normalize_by_distance", help="set it on if the transition probabilities should be normalized by distance.", action = "store_true")
        mdpBuildArgs.add_argument("--tran_type_count", help="Number of Tran Types to consider", type=int, default= 10)
        mdpBuildArgs.add_argument("--ur", help="Reward for unknown transition, default = -1000.", type=float, default= -1000)

        mdpBuildArgs.add_argument("--rebuild_mdpfcache", help="Set to rebuild the mdp from cache solution.", action="store_true")
        mdpBuildArgs.add_argument("--save_mdp2cache", help="Set to cache th esolution vectors", action="store_true")
        mdpBuildArgs.add_argument("--save_folder", help="Folder where the cached vectors will be saved.", type=str, default= "default")


        # MDP solve and lift up parameters
        mdpSolveArgs = parser.add_argument_group(title="mdpSolveArgs", description="MDP build arguments")
        mdpSolveArgs.add_argument("--default_mode", help="Default device to use for Solving the MDP", type=str, default= "GPU")
        mdpSolveArgs.add_argument("--gamma", help="Discount Factor for Value iteration", type=float, default= 0.99)
        mdpSolveArgs.add_argument("--slip_prob", help="Slip probability for safe policy", type=float, default= 0.1)
        mdpSolveArgs.add_argument("--target_vi_error", help="target belllman backup error for considering solved", type=float, default= 0.001)
        mdpSolveArgs.add_argument("--bellman_backup_every", help="Do a bellman backups every __k frames", type=int, default= 100)
        mdpSolveArgs.add_argument("--n_backups", help="The number of backups for every backup step", type=int, default= 10)

        # Evaluation Parameters
        evalArgs = parser.add_argument_group(title="evalArgs", description="Evaluation Arguments")
        evalArgs.add_argument("--eval_episode_count", help="Number of episodes to evaluate the policy", type=int, default=50)
        evalArgs.add_argument("--soft_at_plcy", help="Sample according to Q values rather than max action", action="store_true")
        evalArgs.add_argument("--plcy_k", help="Set the lift up parameter policy_k you want to test with",  type= int, default=1)
        evalArgs.add_argument("--plcy_sweep", help="Set if evaluate the policy under different nearst neighbor numbers k", action="store_true")
        evalArgs.add_argument("--plcy_sweep_k", help="List the sweep lift up parameter policy_k you want to test with", nargs="+", type= int, default=[1,5,11]) 

        # parser.add_argument("--all_gammas", help="Name of the Environment to guild", type=int, default= "[0.1 ,0.9 ,0.99 ,0.999]")


        # Parse Arguments
        parsedArgs = parser.parse_args(s.split(" ") if s is not None else None)
        parsedArgs.device = 'cuda' if (not parsedArgs.no_cuda) and torch.cuda.is_available() else 'cpu'

        # Process Parsed arguments
        argGroups = {}
        for group in parser._action_groups:
            title_map = {"positional arguments": "posArgs", "optional arguments":"optArgs"}
            title = title_map[group.title] if group.title in title_map else group.title
            argGroups[title]={a.dest:getattr(parsedArgs,a.dest,None) for a in group._group_actions}

        return munchify(argGroups)

def print_args(argGroups, to_show_groups=None):
    get_header = lambda title: "#" * 45 + " " * 4 + title + " " * 4 + "#" * 45
    all_args_flag = not to_show_groups
    
    print(get_header("All Arguments") if all_args_flag else "")
    for group_name, group in argGroups.items(): 
        if to_show_groups and group_name not in to_show_groups:
            continue
        print("\n",get_header(group_name))
        for arg1, arg2 in itertools.zip_longest(*[iter(group.items())]*2):
            print(f"{str(arg1[0]).ljust(30)}:{str(arg1[1]).ljust(30)}", 
                  f"{str(arg2[0]).ljust(30)}:{str(arg2[1])}" if arg2 else "")
        
    print("\n","#" * len(get_header("All Arguments")) if all_args_flag else "")

def wandbify_args(argGroups):
    args = {}
    for grp_name, group in argGroups.items():
        for arg_name, arg_value in group.items():
            args[f"{grp_name}:{arg_name}"] = arg_value
    return args