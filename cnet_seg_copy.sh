#! /bin/bash
#SBATCH --job-name=cnet_s2
#################RESSOURCES#################
#SBATCH --partition=48-4
#SBATCH --nodelist="yagi35"
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
############################################
#SBATCH --output=cnet_seg1.out
#SBATCH --error=cnet_seg1.err
#SBATCH -v
source ~/anaconda3/etc/profile.d/conda.sh
conda activate controlnet
###########################################


srun python cnet_seg.py