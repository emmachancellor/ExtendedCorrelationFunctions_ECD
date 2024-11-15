import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import sys
import os

# Custom modules
from simulate_tumors import TumorCellSimulator, plot_cell_distribution
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ecd_helperFunctions import *
from helperFunctions import *

simulations_dir = '/home/ecdyer/PROJECTS/mIF_stats/simulations'
simulation_sample_list = os.listdir(simulations_dir)

markers = ['tumor_cell', 'immune_cell']
labels = {1: 'Tumor', 2: 'Immune'}
keep_cols = ['x', 'y']
num_perturbations = 10
delta_range = (-0.01, 0.01)
all_max_sens = []
results = {'sample': []}
save_dir = '/home/ecdyer/PROJECTS/mIF_stats/simulations/tcm_max_sens'

for sample in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sample)
    simulation_list = os.listdir(sim_path)
    results['sample'].append(sample)
    for i, sim_type in enumerate(simulation_list):
        baseline_tcm = None
        sim_type_path = os.path.join(sim_path, sim_type)
        print(f'Processing {sim_type} ({i+1} of {len(simulation_list)})')
        for j in range(num_perturbations):
            if j != 0:
                perturb_matrix = True
            else:
                perturb_matrix = False
            tcm = calculate_tcm_from_df(sim_type_path,
                                   markers,
                                   labels,
                                   keep_cols,
                                   typea='Tumor',
                                   typeb='Immune',
                                   pointcloud_name='tumor_immune',
                                   perturb_matrix=perturb_matrix)
            if j == 0:
                baseline_tcm = tcm
                continue
            else:
                max_sens = calculate_max_sens(baseline_tcm, tcm)
                all_max_sens.append(max_sens)
                print(f'Max sensitivity: {max_sens}')
        if sim_type not in results:
            results[sim_type] = [max_sens]
        results[sim_type].append(max_sens)
        # Create line plot of max sensitivity values
        plt.figure(figsize=(10,6))
        sns.lineplot(data=all_max_sens, markers='o')
        plt.xlabel('Perturbation Index')
        plt.ylabel('Maximum Sensitivity')
        plt.title(f'Maximum Sensitivity Over {num_perturbations} Perturbations\n{sim_type}')
        plt.tight_layout()
        plt.show()
        plt.savefig(os.path.join(save_dir, f'{sample}_{sim_type}_max_sens_over_perturbations.png'))


        # Create heatmap of results
        plt.figure(figsize=(10,8))
        results_df = pd.DataFrame(results).set_index('sample')
        sns.heatmap(results_df, cmap='YlOrRd', annot=True, fmt='.3f')
        plt.xlabel('Simulation Type')
        plt.ylabel('Sample')
        plt.title('Maximum Sensitivity Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sample}_max_sens_heatmap.png'))
        plt.close()


        
        