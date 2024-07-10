#!/bin/bash
#SBATCH --job-name=compare_tcm_job
#SBATCH --output=compare_tcm_output.txt
#SBATCH --error=compare_tcm_error.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=defq

# Load necessary modules
module load python3

# Activate your Python environment if needed
source ~/miniconda3/bin/activate scimap

# Run the compare_tcm.py script
srun python compare_tcm.py