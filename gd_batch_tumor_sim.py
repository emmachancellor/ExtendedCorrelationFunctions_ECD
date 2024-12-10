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

data_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/data'
simulation_sample_list = os.listdir(data_dir)

all_explainer_stats = []
markers = ['tumor_cell', 'immune_cell']
labels = {1: 'Tumor', 2: 'Immune'}
keep_cols = ['x', 'y']
num_perturbations = 2
delta_range = (-0.01, 0.01)
results = {
        'sample_name': [],
        'infidelity': []
    }
save_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/gd_batch_tumor_sim'

for sample in simulation_sample_list:
    data_path = os.path.join(data_dir, sample)
    simulation_list = os.listdir(data_path)

    for i, sample in enumerate(simulation_list):
        all_explainer = []
        sample_name = sample.replace('.csv', '')
        baseline_gd = None
        sim_type_path = os.path.join(data_path, sample)
        print(f'Processing {sample_name} ({i+1} of {len(simulation_list)})')
        
        for j in range(num_perturbations):
            perturb_matrix = (j != 0) 
        
            gd, cell_count_dict = calculate_gd(data_path=sim_type_path,
                             distance_cols=['x_centroid', 'y_centroid'],
                             intensity_cols=['sox2_mean', 'cd45_mean'],
                             perturb_matrix=True,
                             roi_tile_size=500,
                             cell_label_col='tumor_immune',
                             return_bboxes=False)
            gd = np.array(gd)

            if j == 0:
                baseline_gd = np.array(gd)
                baseline_shape = baseline_gd.shape
                explainer = 0
                continue
        else:
            # Check if shapes match and pad/trim if necessary
            if gd.shape != baseline_shape:
                # Pad with zeros or trim to match baseline shape
                padded_gd = np.zeros(baseline_shape)
                min_rows = min(baseline_shape[0], gd.shape[0])
                min_cols = min(baseline_shape[1], gd.shape[1])
                padded_gd[:min_rows, :min_cols] = gd[:min_rows, :min_cols]
                gd = padded_gd
            explainer = calculate_infidelity(baseline_gd, gd, explainer)
            all_explainer.append(explainer)
        # Add max sensitivity values to appropriate key in results dict
        results['sample_name'].append(sample_name)
        results['infidelity'].append(explainer)

        # Create line plot of max sensitivity values
        plt.figure(figsize=(10,6))
        sns.lineplot(data=all_explainer, markers='o')
        plt.xlabel('Perturbation Index')
        plt.ylabel('Infidelity')
        plt.title(f'Infidelity Over {num_perturbations} Perturbations\n{sample_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{sample_name}_infidelity_over_perturbations.png'))
        plt.close()
print(results)

# Convert results dictionary to DataFrame and save as CSV
results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(save_dir, f'gd_infidelity_results.csv'))