#! /bin/bash
#SBATCH --job-name=SDgeneration
#################RESSOURCES#################
#SBATCH --partition=24-2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
############################################
#SBATCH --output=SDgeneration.out
#SBATCH --error=SDgeneration.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm-yagi30
###########################################


srun python SD1.5_generation.py