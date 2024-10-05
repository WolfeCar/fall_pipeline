#!/bin/bash
#SBATCH --partition=all
#SBATCH --mem=1400G
##SBATCH --gpus-per-node=tesla_v100s:1
#SBATCH --time=120:00:00
#SBATCH --job-name=pipeline_fall
#SBATCH --output=/mnt/projects/debruinz_project/carly_wolfe/fall_pipeline/out/ml-job.%j.out
#SBATCH --error=/mnt/projects/debruinz_project/carly_wolfe/fall_pipeline/error/ml-job%j.err
#SBATCH --mail-user=wolfeca@mail.gvsu.edu
#SBATCH --mail-type=begin,end,fail

module purge
module spider miniforge3
module load miniforge3/24.3.0-0
conda env list

source activate /mnt/home/wolfeca/.conda/envs/scib-pipeline-R4.0  
# source activate /mnt/home/wolfeca/.conda/envs/snakie
# conda env list
module load miniforge3/24.3.0-0

snakemake  --configfile configs/first.yaml --cores 'all'

conda activate base
conda deactivate
module unload miniforge3

