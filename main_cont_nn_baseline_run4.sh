#!/bin/bash
# Define a string variable with a value\
EnvNames=("halfcheetah-random-v0"
	"hopper-random-v0"
	"walker2d-random-v0"
	"halfcheetah-medium-v0"
	"hopper-medium-v0"
	"walker2d-medium-v0"
	"halfcheetah-expert-v0"
	"hopper-expert-v0"
	"walker2d-expert-v0"
	"halfcheetah-medium-expert-v0"
	"hopper-medium-expert-v0"
	"walker2d-medium-expert-v0"
	"halfcheetah-medium-replay-v0"
	"hopper-medium-replay-v0"
	"walker2d-medium-replay-v0")
    
TranTypeCounts="10"
MDPBuildKs="5"
PenaltyBetas="1"
BufferNames="None"
Seeds="0"

# Iterate the string variable using for loop
for env_name in ${EnvNames[*]}; do
{
    for tran_type_count in $TranTypeCounts; do
    {
        for buffer_name in $BufferNames; do
        {
            for mdp_build_k in $MDPBuildKs; do 
            {
                for penalty_beta in $PenaltyBetas; do 
                {
                    for seed in $Seeds; do
                    {
                    
                    python main_cont_nn_baseline.py --seed ${seed} \
        --env_name ${env_name} --buffer_name ${buffer_name} --load_buffer --buffer_size 1000000 \
        --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
        --tran_type_count ${tran_type_count} --MAX_S_COUNT 1100000 --MAX_NS_COUNT ${mdp_build_k} --mdp_build_k ${mdp_build_k} --normalize_by_distance --penalty_beta ${penalty_beta} \
        --gamma 0.99 --plcy_k 1 \
        --repr_build identity --dac_build DACAgentContNNBaseline \
        --wandb_project d4rl_dac --wandb_entity dacmdp --wandb_id MCNB-DD${env_name}-TT${tran_type_count}-P${penalty_beta}-K${mdp_build_k}-S${seed} \
        --results_folder /nfs/hpc/share/shrestaa/storage/dac_ws/
                    }
                    done
                }
                done
            }
            done
        }
        done
    }
    done
}
done