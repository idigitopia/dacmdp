from munch import munchify 
import torch
import time
import gym
import dacmdp
import dacmdp.envs as ce

from dacmdp.core.models_action import NNActionModel, GlobalClusterActionModel, EnsembleActionModel
from dacmdp.core.models_sa_repr import DeltaPredictonRepr
from dacmdp.core.utils_knn import THelper

from dacmdp.eval.utils_eval import evaluate_on_env
from dacmdp.core.dac_core import DACTransitionBatch
from dacmdp.core.dac_build import DACBuildWithActionNames
from dacmdp.core.utils_knn import THelper


config = munchify({
"envArgs":{'env_name': 'CartPole-cont-v1', 'seed': 0},
"dataArgs": {'buffer_name': 'random', 'buffer_size': 50000, 'load_buffer': False, 'buffer_device': 'gpu', "data_dir":""},
"reprModelArgs": {'repr_model_name': 'OracleDynamicsRepr', 's_multiplyer': 1, 'a_multiplyer': 10, 'repr_dim': 4},
"actionModelArgs": {'action_model_name': 'NNActionModelCuda', 'nn_engine': "torch_pykeops"},
"mdpBuildArgs": {'n_tran_types': 10, 'n_tran_targets': 5, 'penalty_beta': 1.0, 'penalty_type': 'linear', 'rebuild_mdpfcache': False,
                 'save_mdp2cache': False, 'save_folder': '/nfs/hpc/share/shrestaa/storage/dac_storage_22_Q4/mdp_dumps/random_hash'},
"mdpSolveArgs": {'device': 'cuda', 'max_n_backups': 10000, "gamma": 0.999, 'epsilon': 0.0001, 'penalty_beta': 1, "operator": "simple_backup"},
"evalArgs": {'eval_episode_count': 50, "skip_eval":True, "skip_dist_log":True},
})

env = gym.make(config.envArgs.env_name)
seed_buffer = dacmdp.utils_buffer.generate_or_load_buffer(config, env)



######### Get Action and Repr Models ####################################
cluster_action_count = 10
cluster_action_model = GlobalClusterActionModel(action_space=env.action_space,
                                   n_actions= cluster_action_count,
                                   data_buffer=seed_buffer)
nn_action_model = NNActionModel(action_space = env.action_space,
                               n_actions = 5,
                               data_buffer = seed_buffer,
                               projection_fxn=lambda s: s, 
                                batch_knn_engine=THelper.batch_calc_knn_pykeops
                               )
action_model = EnsembleActionModel(env.action_space,[nn_action_model, cluster_action_model])    
    
sa_repr_model = DeltaPredictonRepr(s_multiplyer=2, 
                                   a_multiplyer=1,
                                   buffer=seed_buffer)
######################################################################################################




# Instantiate Elastic Agent
config.mdpBuildArgs.n_tran_types = action_model.n_actions
config.mdpBuildArgs.repr_dim = 4
data_buffer = seed_buffer 

elasticAgent = DACBuildWithActionNames( config = config, 
                                    action_space = env.action_space, 
                                    action_model = action_model, # Update this later.
                                    repr_model = sa_repr_model, 
                                    effective_batch_size= 1000, 
                                    batch_knn_engine=THelper.batch_calc_knn_pykeops
                                    )

######### TT 3: DACMDP Elastic Build   ###########################################################################
transitions = DACTransitionBatch(states =torch.FloatTensor(data_buffer.all_states).clone().detach(),
                                actions = torch.FloatTensor(data_buffer.all_actions).clone().detach(),
                                next_states = torch.FloatTensor(data_buffer.all_next_states).clone().detach(),
                                rewards = torch.FloatTensor(data_buffer.all_rewards.reshape(-1)).clone().detach(), 
                                terminals = torch.LongTensor(data_buffer.all_ep_ends.reshape(-1)).clone().detach(),
                                )

st = time.time()
elasticAgent.consume_transitions(transitions, verbose = True, batch_size = 1000)
elasticAgent.dacmdp_core.solve(max_n_backups = config.mdpSolveArgs.max_n_backups, 
                               penalty_beta = config.mdpSolveArgs.penalty_beta, 
                               epsilon = config.mdpSolveArgs.epsilon, 
                               gamma = config.mdpSolveArgs.gamma, 
                               operator="dac_backup", 
                               bellman_backup_batch_size=500, 
                               reset_values=True)

print(f"Graph built and solved in {time.time()-st:.2f} Seconds")
######################################################################################################

config.evalArgs.eval_episode_count = 20
elasticAgent.batch_knn_engine = THelper.batch_calc_knn_jit
avg_rewards, info = evaluate_on_env(env,elasticAgent.dac_lifted_policy, eps_count=config.evalArgs.eval_episode_count)
print(avg_rewards)