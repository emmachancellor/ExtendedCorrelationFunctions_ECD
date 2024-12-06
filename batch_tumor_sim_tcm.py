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

markers = ['tumor_cell', 'immune_cell']
labels = {1: 'Tumor', 2: 'Immune'}
keep_cols = ['x', 'y']
num_perturbations = 25
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
save_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/tcm_max_sens'

for sample in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sample)
    simulation_list = os.listdir(sim_path)
    results['sample_name'].append(sample)

    for i, sim_type in enumerate(simulation_list):
        all_max_sens = []
        sim_name = sim_type.replace('.csv', '')
        baseline_tcm = None
        sim_type_path = os.path.join(sim_path, sim_type)
        print(f'Processing {sim_name} ({i+1} of {len(simulation_list)})')
        
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
                baseline_shape = baseline_tcm.shape
                max_sens = 0
                continue
            else:
                # Check if shapes match and pad/trim if necessary
                if tcm.shape != baseline_shape:
                    # Pad with zeros or trim to match baseline shape
                    padded_tcm = np.zeros(baseline_shape)
                    min_rows = min(baseline_shape[0], tcm.shape[0])
                    min_cols = min(baseline_shape[1], tcm.shape[1])
                    padded_tcm[:min_rows, :min_cols] = tcm[:min_rows, :min_cols]
                    tcm = padded_tcm
                #max_sens = calculate_max_sens(baseline_tcm, tcm, max_sens)
                # note, for simplicity I'm not changign the max_sens variable name, but this is calculating infidelity
                max_sens = calculate_infidelity(baseline_tcm, tcm, max_sens)
                all_max_sens.append(max_sens)

        # Add max sensitivity values to appropriate key in results dict
        if 'raw' in sim_name:
            results['raw_simulation'].append(max_sens)
        elif 'high_infiltration' in sim_name:
            results['high_immune_infiltration'].append(max_sens)
        elif 'immune_exclusion' in sim_name:
            results['immune_exclusion'].append(max_sens)
        elif 'immune_ring' in sim_name:
            results['immune_ring_formation'].append(max_sens)
        elif 'immune_surveillance' in sim_name:
            results['scattered_immune_surveillance'].append(max_sens)
        elif 'dense_cluster' in sim_name:
            results['dense_tumor_clustering'].append(max_sens)
        elif 'diffuse_mixed' in sim_name:
            results['diffuse_mixed_distribution'].append(max_sens)

        # Create line plot of max sensitivity values
        plt.figure(figsize=(10,6))
        sns.lineplot(data=all_max_sens, markers='o')
        plt.xlabel('Perturbation Index')
        plt.ylabel('Maximum Sensitivity')
        plt.title(f'Maximum Sensitivity Over {num_perturbations} Perturbations\n{sim_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sample}_{sim_name}_max_sens_over_perturbations.png'))
        plt.close()
print(results)

# Convert results dictionary to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, f'max_sens_results.csv'))

#Create heatmap of results
# plt.figure(figsize=(10,8))
# results_df = pd.DataFrame(results).set_index('sample')
# sns.heatmap(results_df, cmap='YlOrRd', annot=True, fmt='.3f')
# plt.xlabel('Simulation Type')
# plt.ylabel('Sample')
# plt.title('Maximum Sensitivity Heatmap')
# plt.tight_layout()
# plt.savefig(os.path.join(save_dir, f'{sample}_max_sens_heatmap.png'))
# plt.close()



        
        