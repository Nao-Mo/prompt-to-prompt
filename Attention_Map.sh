#! /bin/bash
#SBATCH --job-name=Attention_Map1
#################RESSOURCES#################
#SBATCH --partition=48-2
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
############################################
#SBATCH --output=Attn1.out
#SBATCH --error=Attn1.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ldm-yagi30
###########################################


srun python prompt-to-prompt_stable.py