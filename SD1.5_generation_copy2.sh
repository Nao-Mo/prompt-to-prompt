#! /bin/bash
#SBATCH --job-name=SDgen3
#################RESSOURCES#################
#SBATCH --partition=48-2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
############################################
#SBATCH --output=SDgen3.out
#SBATCH --error=SDgen3.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm-yagi30
###########################################


srun python SD1.5_generation_copy2.py