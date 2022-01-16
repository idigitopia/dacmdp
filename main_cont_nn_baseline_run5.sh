#!/bin/bash
# Define a string variable with a value\
EnvNames=("maze2d-open-v0"
    "maze2d-umaze-v1"
    "maze2d-medium-v1"
    "maze2d-large-v1")
    
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
        
        
                    python main_cont_nn_baseline.py --seed ${seed} \
                    --env_name ${env_name} --buffer_name ${buffer_name} --load_buffer --buffer_size 1000000 \
                    --data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
                    --tran_type_count ${tran_type_count} --MAX_S_COUNT 1100000 --MAX_NS_COUNT ${mdp_build_k} --mdp_build_k ${mdp_build_k} --normalize_by_distance --penalty_beta ${penalty_beta} \
                    --gamma 0.99 --plcy_k 1 \
                    --repr_build identity --dac_build DACAgentContNNBDeltaPred \
                    --wandb_project d4rl_dac --wandb_entity dacmdp --wandb_id MCNBDP-DD${env_name}-TT${tran_type_count}-P${penalty_beta}-K${mdp_build_k}-S${seed} \
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
