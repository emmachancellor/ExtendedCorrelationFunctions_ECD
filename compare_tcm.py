# Emma C. Dyer
# Last Updated: 8 July 2024

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
import time
from functools import partial
from joblib import Parallel, delayed
from helperFunctions import *
from ecd_helperFunctions import *
from smallestEnclosingCircle import make_circle
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp

grid_file = '/michorlab/ecdyer/multiplex_spatial/crc_grid_data/CRC17_tile_ROIs.h5'

markers = ['CD45_status', 'SMA_status']

keep_cols = ['X_centroid', 'Y_centroid', 'CellID']

labels  = {1: 'Lymphocytes',
            2: 'SMA'}

rename_cols_dict = {'X_centroid': 'x', 
                    'Y_centroid': 'y'}

save_csr_tcm_path = '/michorlab/ecdyer/multiplex_spatial/crc_grid_data/test_csr_files/'

save_tcm_plot_path = '/michorlab/ecdyer/multiplex_spatial/figures/tcm_plots/'

save_csr_plot_path = '/michorlab/ecdyer/multiplex_spatial/figures/csr_tcm_plots/'

save_ks_results_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/'

save_tcm_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/'

save_csr_tcm_path = '/michorlab/ecdyer/multiplex_spatial/tcm_results/'

ks_test_results = compare_tcm(grid_file, 
            markers,
            keep_cols,
            labels,
            visualiseStages=False,
            save_tcm_plot_path=save_tcm_plot_path,
            save_csr_plot_path=save_csr_plot_path,
            save_tcm_path=save_tcm_path,
            save_csr_tcm_path=save_csr_tcm_path,
            save_ks_results_path=save_ks_results_path,
            rename_cols_dict=rename_cols_dict,
            plot_point_cloud=True)