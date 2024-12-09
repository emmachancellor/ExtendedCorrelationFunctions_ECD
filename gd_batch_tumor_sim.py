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

simulations_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/simulations'
simulation_sample_list = os.listdir(simulations_dir)

all_explainer_stats = []
markers = ['tumor_cell', 'immune_cell']
labels = {1: 'Tumor', 2: 'Immune'}
keep_cols = ['x', 'y']
num_perturbations = 2
delta_range = (-0.01, 0.01)
results = {
        'sample_name': [],
        'raw_simulation': [],
        'high_immune_infiltration': [], 
        'immune_exclusion': [],
        'immune_ring_formation': [],
        'scattered_immune_surveillance': [],
        'dense_tumor_clustering': [],
        'diffuse_mixed_distribution': []
    }
save_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/gd_batch_tumor_sim'

for sample in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sample)
    simulation_list = os.listdir(sim_path)
    results['sample_name'].append(sample)

    for i, sim_type in enumerate(simulation_list):
        all_explainer = []
        sim_name = sim_type.replace('.csv', '')
        baseline_gd = None
        sim_type_path = os.path.join(sim_path, sim_type)
        print(f'Processing {sim_name} ({i+1} of {len(simulation_list)})')
        
        for j in range(num_perturbations):
            if j != 0:
                perturb_matrix = True
            else:
                perturb_matrix = False
        
            gd = calculate_gd(sim_type_path, 
                            perturb_matrix=True)
            
            if j == 0:
                baseline_gd = gd
                explainer = 0
                continue
        else:
            #TODO: make this iterative for ROIs, so that the key between the baseline and perturbed gd is the same
            #TODO: and the explainer is calculated for each ROI
            explainer = calculate_max_sens(baseline_gd, gd, explainer)
            all_explainer_stats.append(explainer)
            # Add max sensitivity values to appropriate key in results dict
        if 'raw' in sim_name:
            results['raw_simulation'].append(explainer)
        elif 'high_infiltration' in sim_name:
            results['high_immune_infiltration'].append(explainer)
        elif 'immune_exclusion' in sim_name:
            results['immune_exclusion'].append(explainer)
        elif 'immune_ring' in sim_name:
            results['immune_ring_formation'].append(explainer)
        elif 'immune_surveillance' in sim_name:
            results['scattered_immune_surveillance'].append(explainer)
        elif 'dense_cluster' in sim_name:
            results['dense_tumor_clustering'].append(explainer)
        elif 'diffuse_mixed' in sim_name:
            results['diffuse_mixed_distribution'].append(explainer)

        # Create line plot of max sensitivity values
        plt.figure(figsize=(10,6))
        sns.lineplot(data=all_explainer, markers='o')
        plt.xlabel('Perturbation Index')
        plt.ylabel('Maximum Sensitivity')
        plt.title(f'Maximum Sensitivity Over {num_perturbations} Perturbations\n{sim_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sample}_{sim_name}_explainer_over_perturbations.png'))
        plt.close()
print(results)

# Convert results dictionary to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, f'gd_max_sens_results.csv'))
