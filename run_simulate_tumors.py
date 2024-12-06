import sys
import os
import numpy as np
import pandas as pd

from simulate_tumors import TumorCellSimulator
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from ecd_helperFunctions import *

btc_pt2_s2 = pd.read_excel("/Users/emmadyer/Desktop/PROJECTS/BTC/processed_cycif/BTC_P2_S2.xlsx")
# Tumor = 0, Immune = 1
btc_pt2_s2['tumor_immune'] = np.where(btc_pt2_s2['sox2_mean'] > btc_pt2_s2['cd_45_mean'], 0, 1)

# Example usage:
coordinates = btc_pt2_s2[['x_centroid_um', 'y_centroid_um']].to_numpy()
cell_labels = btc_pt2_s2['tumor_immune'].values

np.random.seed(42)

simulator = TumorCellSimulator(coordinates, cell_labels)

n_points_per_type = {
    0: 4500,
    1: 3200
}

n_points = 7700

# Raw Data Simulation

raw_points, raw_labels = simulator.simulate_raw(n_points_per_type=n_points_per_type,
                                                     noise_level=0.1)


# 2. Mixed/Tuned Simulation with different parameter sets
# Default parameters
mixed_points_default, mixed_labels_default = simulator.simulate(
    n_points=n_points,
    mode='mixed',
    component_ratio=0.7,
    n_components=3,
    overlap_density=0.5,
    interaction_scale=0.3
)

# Custom interaction parameters for more complex behavior
interaction_params = {
    ('tumor', 'tumor'): {'range': 10.0, 'attraction': 1.5, 'repulsion': 2.0},
    ('immune', 'immune'): {'range': 8.0, 'attraction': 1.0, 'repulsion': 1.5},
    ('tumor', 'immune'): {'range': 12.0, 'attraction': 0.8, 'repulsion': 1.2}
}


# Simulation with custom parameters
mixed_points_custom, mixed_labels_custom = simulator.simulate(
    n_points=n_points,
    mode='mixed',
    component_ratio=0.8,
    n_components=4,
    overlap_density=0.6,
    interaction_scale=0.4,
    interaction_params=interaction_params,
    n_iterations=1500,
    temperature=0.08
)