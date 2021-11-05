from .exp_track_helper import Experiment, ExperimentPool
import sys


### Story 1 ##############################################################################################################################################################
### Determinsitic Continuous MDP ### 
### Run experiments for 3 different datasets of CartPole under different number of transition types ### 
### Also run som experiments on the different values of penalty_beta ### 

cartPoleExps =  ExperimentPool()

for seed in [0,2,4]:
    for dataset in ["Robust", "Random", "Optimal"]:
        for tran_type_count in [1, 5, 10, 20, 40]:
            for penalty_beta in [0,0.1,1,10,100,1000,10000,100000]:
                for MAX_NS_COUNT in [1, 5, 10, 20, 40]:
                    env_name = "cartpole-cont-v1"
                    dataset_map= {d:f"{d}_{env_name}_{seed}" for d in ["Robust", "Random", "Optimal"]}
                    cartPoleExps.add_experiment(Experiment(id=f"S{seed}-D{dataset}-tt{tran_type_count}-p{penalty_beta}-ns{MAX_NS_COUNT}",
                                           meta="Stochastic MDP Build For Continuous Action Spaces. Story 1.",
                                           expPrefix="python main_cont.py ",
                                           expSuffix=f"--env_name CartPole-cont-v1 --tran_type_count {tran_type_count} --wandb_project DACMDPCONT-V0 \
                                            --load_buffer --buffer_size 100000  --buffer_name {dataset_map[dataset]} \
                                            --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
                                            --MAX_S_COUNT 110000 --MAX_NS_COUNT {MAX_NS_COUNT} --mdp_build_k {MAX_NS_COUNT} \
                                            --normalize_by_distance --penalty_beta {penalty_beta} --ur 0 --knn_delta 1e-5\
                                            --gamma 0.99 --slip_prob 0.1 --save_mdp2cache \
                                            --save_folder /nfs/hpc/share/shrestaa/projects/dacmdp_cont/results \
                                            --eval_episode_count 100 --plcy_k 1"))

    
### Story 2 ##############################################################################################################################################################
### Determinsitic Continuous MDP ### 

d4rlExps =  ExperimentPool()
d4rlGym_envs = ["halfcheetah-random-v0","halfcheetah-medium-v0","halfcheetah-expert-v0","halfcheetah-medium-replay-v0","halfcheetah-medium-expert-v0","walker2d-random-v0","walker2d-medium-v0","walker2d-expert-v0","walker2d-medium-replay-v0","walker2d-medium-expert-v0","hopper-random-v0","hopper-medium-v0","hopper-expert-v0","hopper-medium-replay-v0","hopper-medium-expert-v0"]
d4rlMaze_envs = ["maze2d-open-v0","maze2d-umaze-v1","maze2d-medium-v1","maze2d-large-v1","maze2d-open-dense-v0","maze2d-umaze-dense-v1","maze2d-medium-dense-v1","maze2d-large-dense-v1"]
d4rlAntmaze_envs = ["antmaze-umaze-v0","antmaze-umaze-diverse-v0","antmaze-medium-diverse-v0","antmaze-medium-play-v0","antmaze-large-diverse-v0","antmaze-large-play-v0"]
d4rlAirdroit_envs = ["pen-human-v0","pen-cloned-v0","pen-expert-v0","hammer-human-v0","hammer-cloned-v0","hammer-expert-v0","door-human-v0","door-cloned-v0","door-expert-v0","relocate-human-v0","relocate-cloned-v0","relocate-expert-v0"]


agent_classes = ["DeterministicAgent", "DeterministicAgent_o", "DeterministicAgent_s", "StochasticAgent", "StochasticAgent_o","StochasticAgent_s","StochasticAgentWithDelta_o","StochasticAgentWithDelta_s","StochasticAgentWithParametricPredFxn_o","StochasticAgentWithParametricPredFxn_s","StchExtendedAgent_o","StchExtendedAgent_s","PIAgent_o" ,"PIAgent_s" ]

def d4rl_template(agent_class, env_name, tran_type_count, penalty_beta, MAX_NS_COUNT):
    return Experiment(id=f"{agent_class}-{env_name}-tt{tran_type_count}-p{penalty_beta}-ns{MAX_NS_COUNT}",
                                                   meta="Stochastic MDP Build For Continuous Action Spaces. Story 1.",
                                                   expPrefix="python main_cont.py ",
                                                   expSuffix=f"--env_name {env_name} --tran_type_count {tran_type_count} --wandb_project DACMDPCONT-V0 \
                                                    --load_buffer --buffer_size 1000000 \
                                                    --MAX_S_COUNT 1100000 --MAX_NS_COUNT {MAX_NS_COUNT} --mdp_build_k {MAX_NS_COUNT} --normalize_by_distance --penalty_beta {penalty_beta} --ur 0 \
                                                    --gamma 0.99 --slip_prob 0.1 --default_mode GPU --save_mdp2cache \
                                                    --save_folder /nfs/hpc/share/shrestaa/projects/dacmdp_cont/results \
                                                    --eval_episode_count 100 --plcy_k 1")


# for env_name in d4rlGym_envs + d4rlMaze_envs + d4rlAntmaze_envs + d4rlAirdroit_envs:
#     for agent_class in agent_classes:
#         for tran_type_count in [1, 5, 10, 20, 40]:
#             for penalty_beta in [0,0.1,1,10,100,1000,10000,100000]:
#                 for MAX_NS_COUNT in [1, 5, 10, 20, 40]:
#                     d4rlExps.add_experiment()


                                  
############################################################################################################################################################################
                                                                              
                                                          
### Main Variables #########################################################################################################################################################                                            
pools = [cartPoleExps, d4rlExps]
                                                          
# make sure experiment ids across the pools are unique                                                    
all_exp_keys = [exp for pool in pools for exp in pool.expPool.keys()]
assert len(all_exp_keys) == len(set(all_exp_keys))
                                                          
# Main ExpPool that holds all experiment parameters                                     
ExpPool = ExperimentPool.joinPools(*pools)

                                                          ############################################################################################################################################################################
                                                          
                                                          
if __name__ == '__main__':
    query_exp_id = sys.argv[1]
    for exp_id in ExpPool.expPool:
        if query_exp_id in exp_id:
            print(exp_id)