#!/bin/bash
#SBATCH --job-name=compare_tcm_job
#SBATCH --output=compare_tcm_output_%j.txt
#SBATCH --error=compare_tcm_error_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --partition=defq
#SBATCH --nodelist=node09

# Load necessary modules
module load python/3.8

# Activate your Python environment if needed
source /path/to/your/venv/bin/activate

# Run the compare_tcm.py script
srun python compare_tcm.py