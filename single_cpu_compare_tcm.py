# Emma C. Dyer
# Last Updated: 10 July 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
import time
import logging
import sys
from functools import partial
from joblib import Parallel, delayed
from helperFunctions import *
from ecd_helperFunctions import *
from smallestEnclosingCircle import make_circle
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp

if __name__ == '__main__':
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    cwd = '/michorlab/ecdyer/multiplex_spatial/crc_grid_data/cph_grids'
    short_grid_files = os.listdir(cwd)
    grid_files = [os.path.join(cwd, f) for f in short_grid_files]
    markers = ['CD45_status', 'PD-L1_status']
    keep_cols = ['X_centroid', 'Y_centroid', 'CellID']
    labels  = {1: 'Lymphocytes',
                2: 'PD-L1'}
    rename_cols_dict = {'X_centroid': 'x', 
                        'Y_centroid': 'y'}
    save_tcm_plot_path = '/michorlab/ecdyer/multiplex_spatial/figures/tcm_plots/'
    save_csr_plot_path = '/michorlab/ecdyer/multiplex_spatial/figures/csr_tcm_plots/'
    save_ks_results_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/cd45_pdl1/'
    save_tcm_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/cd45_pdl1/'
    save_csr_tcm_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/cd45_pdl1/'

    for grid_file in grid_files:
        compare_tcm(grid_file, 
            markers,
            keep_cols,
            labels,
            visualiseStages=False,
            #save_tcm_plot_path=save_tcm_plot_path,
            #save_csr_plot_path=save_csr_plot_path,
            #save_tcm_path=save_tcm_path,
            #save_csr_tcm_path=save_csr_tcm_path,
            save_ks_results_path=save_ks_results_path,
            rename_cols_dict=rename_cols_dict,
            plot_point_cloud=False)