#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
## SBATCH --job-name=main_eval_mp_Greedy_NAVMESH_GT_Potential
## SBATCH --job-name=main_eval_mp_DP_NAVMESH_GT_Potential_SqrtD
## SBATCH --job-name=main_eval_mp_DP_NAVMESH_GT_Potential_Skeleton_Dall
## BATCH --job-name=build_map
#SBATCH --job-name=large_A3_500steps
## SBATCH --job-name=B1_1000steps
## SBATCH --job-name=C1_500steps
## SBATCH --job-name=C4_1000steps

## General partitions: all-HiPri, bigmem-HiPri   --   (12 hour limit)
##                     all-LoPri, bigmem-LoPri, gpuq  (5 days limit)
## Restricted: CDS_q, CS_q, STATS_q, HH_q, GA_q, ES_q, COS_q  (10 day limit)
#SBATCH --partition=gpuq

## Separate output and error messages into 2 files.
## NOTE: %u=userID, %x=jobName, %N=nodeID, %j=jobID, %A=arrayID, %a=arrayTaskID
#SBATCH --output=/scratch/yli44/logs/%x-%N-%j.out  # Output file
#SBATCH --error=/scratch/yli44/logs/%x-%N-%j.err   # Error file

## Slurm can send you updates via email
#SBATCH --mail-type=BEGIN,END,FAIL         # ALL,NONE,BEGIN,END,FAIL,REQUEUE,..
#SBATCH --mail-user=yli44@gmu.edu     # Put your GMU email address here

## Specify how much memory your job needs. (2G is the default)
#SBATCH --mem=300G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=5-00:00   # Total time needed for job: Days-Hours:Minutes

#SBATCH --gres=gpu:3
#SBATCH --nodelist=NODE078

#SBATCH --cpus-per-task 18

## Load the relevant modules needed for the job
module load cuda/11.2
module load python/3.7.4
module load gcc/7.5.0

source /scratch/yli44/habitat_env_argo/bin/activate

## Run your program or script
#python main_eval_multiprocess.py --config='exp_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_SqrtD_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS_whole_skeleton_graph.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS_whole_skeleton_graph_pruned.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_SqrtD_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#python build_map_multiprocess.py --config='build_map.yaml'
#python main_eval_multiprocess.py --config='large_exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_FME_NAVMESH_MAP.yaml'
#python main_eval_multiprocess.py --j=1 --config='large_exp_360degree_FME_NAVMESH_MAP.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS_whole_skeleton_graph_ratio1dot7.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_500STEPS_whole_skeleton_graph_pruned_ratio1dot7.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_DP_NAVMESH_MAP_GT_Potential_D_Skeleton_Dall_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --config='exp_360degree_FME_NAVMESH_MAP_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=6 --config='exp_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=6 --config='exp_360degree_Greedy_NAVMESH_MAP_GT_Potential_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='exp_90degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='exp_90degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=1 --config='exp_90degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=1 --config='exp_90degree_FME_NAVMESH_MAP_1STEP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='exp_360degree_ANS_NAVMESH_MAP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=1 --config='exp_90degree_ANS_NAVMESH_MAP_1000STEPS.yaml'
#python main_eval_multiprocess.py --j=4 --config='large_exp_360degree_Greedy_NAVMESH_MAP_UNet_OCCandSEM_Potential_1STEP_1000STEPS.yaml'
python main_eval_multiprocess.py --j=6 --config='large_exp_360degree_DP_NAVMESH_MAP_UNet_OCCandSEM_Potential_D_Skeleton_Dall_1STEP_500STEPS.yaml'