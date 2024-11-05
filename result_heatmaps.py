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
    grid_files = '/michorlab/ecdyer/multiplex_spatial/crc_grid_data/cph_grids/'
    sample_files = '/michorlab/labsyspharm_ORION-CRC/labsyspharm_ORION-CRC/aws_data/'
    results_directory = '/michorlab/ecdyer/multiplex_spatial/tcm_results/pdl1_cd45/'
    save_directory = '/michorlab/ecdyer/multiplex_spatial/figures/pdl1_cd45_heatmaps/'

    generate_grid_heatmap(grid_files,
                         sample_files,
                         results_directory,
                         save_directory=save_directory,
                         save_plot=True)
