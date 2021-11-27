#!/bin/bash
# Define a string variable with a value
EnvNames="CartPole-cont-v1"
TranTypeCounts="10 20 40"
MDPBuildKs="5 10 20"
PenaltyBetas="1 10 100"

# Iterate the string variable using for loop
for env_name in $EnvNames; do
{
    for tran_type_count in $TranTypeCounts; do
    {
        for mdp_build_k in $MDPBuildKs; do 
        {
            for penalty_beta in $PenaltyBetas; do 
            {
            python main_cont_disc_cluster.py --seed 0 \
--env_name ${env_name} --buffer_name random --load_buffer --buffer_size 100000 \
--data_dir /nfs/hpc/share/shrestaa/projects/dacmdp_cont/buffers/ \
--tran_type_count ${tran_type_count} --MAX_S_COUNT 110000 --MAX_NS_COUNT ${mdp_build_k} --mdp_build_k ${mdp_build_k} --normalize_by_distance --penalty_beta ${penalty_beta} \
--gamma 0.99 --plcy_k 1 \
--wandb_project cartpoleCont --wandb_entity dacmdp --wandb_id MCDC-TT-${tran_type_count}-P${penalty_beta}-K${mdp_build_k}
            }
            done
        }
        done
    }
    done
}
done