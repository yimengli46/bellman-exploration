#!/bin/sh

## Give your job a name to distinguish it from other jobs you run.
## SBATCH --job-name=train_UNet_occ_only
#SBATCH --job-name=train_ResNet_sseg_and_depth

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
#SBATCH --mem=50G        # Total memory needed per task (units: K,M,G,T)

## Specify how much time your job needs. (default: see partition above)
#SBATCH --time=5-00:00   # Total time needed for job: Days-Hours:Minutes

#SBATCH --gres=gpu:2
#SBATCH --nodelist=NODE040
#SBATCH --cpus-per-task 15

## Load the relevant modules needed for the job
module load cuda/11.2
module load python/3.7.4
module load gcc/7.5.0

source /scratch/yli44/habitat_env_argo/bin/activate

## Run your program or script
#python train_UNet_input_partial_map.py
python train_ResNet_input_view.py