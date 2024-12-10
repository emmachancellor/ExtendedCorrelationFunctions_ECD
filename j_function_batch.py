import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import sys
import os

# Custom modules
from simulate_tumors import TumorCellSimulator, plot_cell_distribution
from helperFunctions import *
from ecd_helperFunctions import *
from ecd_j_function import calculate_j_function

# Define directories
simulations_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/simulations'
simulation_sample_list = os.listdir(simulations_dir)

# Define perturbation parameters
num_perturbations = 25
delta_range = (-0.01, 0.01)

# Define results dictionary and save directory
max_sens_results = {
    'sample_name': [],
    'raw_simulation': [],
    'high_immune_infiltration': [], 
    'immune_exclusion': [],
    'immune_ring_formation': [],
    'scattered_immune_surveillance': [],
    'dense_tumor_clustering': [],
    'diffuse_mixed_distribution': []
}
infidelity_results = {
    'sample_name': [],
    'raw_simulation': [],
    'high_immune_infiltration': [], 
    'immune_exclusion': [],
    'immune_ring_formation': [],
    'scattered_immune_surveillance': [],
    'dense_tumor_clustering': [],
    'diffuse_mixed_distribution': []
}

save_dir = '/home/ecdyer/labshare/PROJECTS/SPATIAL_STATS/j_function_results'

for sample in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sample)
    simulation_list = os.listdir(sim_path)
    max_sens_results['sample_name'].append(sample)
    infidelity_results['sample_name'].append(sample)

    for i, sim_type in enumerate(simulation_list):
        max_sens = []
        infidelity = []
        sim_name = sim_type.replace('.csv', '')
        baseline_j_function_auc = None
        sim_type_path = os.path.join(sim_path, sim_type)
        print(f'Processing {sim_name} ({i+1} of {len(simulation_list)})')
    
        for j in range(num_perturbations):
            print(f'Processing perturbation {j+1} of {num_perturbations}')
            perturb_matrix = (j != 0)

            j_functions, j_function_dict, summary_dict = calculate_j_function(data_path=sim_type_path,
                    distance_cols=['x', 'y'],
                    cell_a_label=0,
                    cell_b_label=1,
                    perturb_matrix=perturb_matrix,
                    roi_tile_size=175,
                    cell_label_col='label',
                    r_max=300,
                    dr=5,
                    return_bboxes=True,
                    plot_j_function=False,
                    return_summary=True)

            # Get AUC array
            j_function_auc_array = np.array([summary_dict['AUC']])

            if j == 0:
                baseline_j_function_auc = j_function_auc_array
                baseline_j_function_auc_shape = baseline_j_function_auc.shape
                infidelity = 0
                max_sens = 0
                continue
            else:
                # Check if shapes match and pad/trim if necessary
                if j_function_auc_array.shape != baseline_j_function_auc_shape:
                    # Pad with zeros or trim to match baseline shape
                    padded_j_function_aucs = np.zeros(baseline_j_function_auc_shape)
                    min_rows = min(baseline_j_function_auc_shape[0], j_function_auc_array.shape[0])
                    min_cols = min(baseline_j_function_auc_shape[1], j_function_auc_array.shape[1])
                    padded_j_function_aucs[:min_rows, :min_cols] = j_function_auc_array[:min_rows, :min_cols]
                    j_function_auc_array = padded_j_function_aucs
                # Calculate explainer
                infidelity = calculate_infidelity(baseline_j_function_auc, j_function_auc_array, infidelity)
                max_sens = calculate_max_sens(baseline_j_function_auc, j_function_auc_array, max_sens)

        # Add max sensitivity values to appropriate key in results dict
        if 'raw' in sim_name:
            max_sens_results['raw_simulation'].append(max_sens)
            infidelity_results['raw_simulation'].append(infidelity)
        elif 'high_infiltration' in sim_name:
            max_sens_results['high_immune_infiltration'].append(max_sens)
            infidelity_results['high_immune_infiltration'].append(infidelity)
        elif 'immune_exclusion' in sim_name:
            max_sens_results['immune_exclusion'].append(max_sens)
            infidelity_results['immune_exclusion'].append(infidelity)
        elif 'immune_ring' in sim_name:
            max_sens_results['immune_ring_formation'].append(max_sens)
            infidelity_results['immune_ring_formation'].append(infidelity)
        elif 'immune_surveillance' in sim_name:
            max_sens_results['scattered_immune_surveillance'].append(max_sens)
            infidelity_results['scattered_immune_surveillance'].append(infidelity)
        elif 'dense_cluster' in sim_name:
            max_sens_results['dense_tumor_clustering'].append(max_sens)
            infidelity_results['dense_tumor_clustering'].append(infidelity)
        elif 'diffuse_mixed' in sim_name:
            max_sens_results['diffuse_mixed_distribution'].append(max_sens)
            infidelity_results['diffuse_mixed_distribution'].append(infidelity)
        
    print("Max Sensitivity Results: ", max_sens_results, '\n')
    print("Infidelity Results: ", infidelity_results, '\n')
    
# Results to CSV
max_sens_results_df = pd.DataFrame(max_sens_results)
infidelity_results_df = pd.DataFrame(infidelity_results)

max_sens_results_df.to_csv(os.path.join(save_dir, f'max_sens_results.csv'))
infidelity_results_df.to_csv(os.path.join(save_dir, f'infidelity_results.csv'))

