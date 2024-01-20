#!/bin/bash

# Parameters
#SBATCH --array=0-1%2
#SBATCH --cpus-per-task=4
#SBATCH --error=/mnt/qb/work/macke/mgloeckler90/probjax/results/bm_hh/.submitit/%A_%a/%A_%a_0_log.err
#SBATCH --gres=gpu
#SBATCH --job-name=hydra_script
#SBATCH --mem=32GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --open-mode=append
#SBATCH --output=/mnt/qb/work/macke/mgloeckler90/probjax/results/bm_hh/.submitit/%A_%a/%A_%a_0_log.out
#SBATCH --signal=USR2@60
#SBATCH --time=700
#SBATCH --wckey=submitit

# command
export SUBMITIT_EXECUTOR=slurm
srun --unbuffered --output /mnt/qb/work/macke/mgloeckler90/probjax/results/bm_hh/.submitit/%A_%a/%A_%a_%t_log.out --error /mnt/qb/work/macke/mgloeckler90/probjax/results/bm_hh/.submitit/%A_%a/%A_%a_%t_log.err /mnt/qb/work/macke/mgloeckler90/miniconda3/envs/probjax/bin/python -u -m submitit.core._submit /mnt/qb/work/macke/mgloeckler90/probjax/results/bm_hh/.submitit/%j
