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

simulations_dir = '/home/ecdyer/PROJECTS/mIF_stats/simulations'
simulation_sample_list = os.listdir(simulations_dir)

markers = ['tumor_cell', 'immune_cell']
labels = {1: 'Tumor', 2: 'Immune'}

for sim in simulation_sample_list:
    sim_path = os.path.join(simulations_dir, sim)
    simulation_list = os.listdir(sim_path)
    for sim_type in simulation_list:
        sim_type_path = os.path.join(sim_path, sim_type)
        simulation_df = pd.read_csv(sim_type_path)
        # One hot encode the cell types
        simulation_df = create_cell_type_columns(simulation_df)

        # Generate pointcloud object
        generate_pointcloud(simulation_df, markers, labels)
