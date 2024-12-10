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
from ecd_j_function import *

# Define directories
simulations_dir = '/mnt/labshare/PROJECTS/SPATIAL_STATS/simulations'
simulation_sample_list = os.listdir(simulations_dir)

# Define perturbation parameters
num_perturbations = 25
delta_range = (-0.01, 0.01)

# Define results dictionary and save directory
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

save_dir = '/home/ecdyer/labshare/PROJECTS/SPATIAL_STATS/j_function_results'

for sample in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sample)
    simulation_list = os.listdir(sim_path)
    results['sample_name'].append(sample)

    for i, sim_type in enumerate(simulation_list):
        all_explainer = []
        sim_name = sim_type.replace('.csv', '')
        baseline_tcm = None
        sim_type_path = os.path.join(sim_path, sim_type)
        print(f'Processing {sim_name} ({i+1} of {len(simulation_list)})')
    
        for j in range(num_perturbations):
            perturb_matrix = (j != 0)

            



