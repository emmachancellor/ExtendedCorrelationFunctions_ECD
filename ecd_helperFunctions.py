import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
import time
import logging
from functools import partial
from helperFunctions import *
from smallestEnclosingCircle import make_circle
from sklearn.mixture import GaussianMixture
from scipy.stats import ks_2samp
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def gmm_classify(data, 
                 n_components=2, 
                 marker_name='marker_status', 
                 component_name='component'):
    """
    Classify data using Gaussian Mixture Model (GMM).

    Parameters:
        data (DataFrame): The input data to be classified.
        n_components (int): The number of components (clusters) in the GMM. Default is 2.
        marker_name (str): The name of the column in the DataFrame to store the marker status. Default is 'marker_status', which
            is added to the name of the column being normalized. Example 'CD4_status'.
        component_name (str): The name of the column in the DataFrame to store the component assignments. Default is 'component',
            which is added to the name of the column being normalized. Example 'CD4_component'.

    Returns:
        data (pd.DataFrame): The input data with the marker status and component assignments added.

    """
    # Fit GMM to the data
    gmm = GaussianMixture(n_components=n_components, random_state=0)
    gmm.fit(data)
    # Predict the component (cluster) for each intensity value
    components = gmm.predict(data)
    
    # Convert to DataFrame if necessary
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=[marker_name])
    
    # Add the component assignments to the DataFrame
    data[component_name] = components

    # Extract the means of the components
    means = gmm.means_.flatten()
    print('Means: ', means)

    # Determine which component is positive (assuming higher mean indicates positive)
    positive_component = np.argmax(means)
    print('Positive Component: ', positive_component)
    negative_component = np.argmin(means)

    # Map components to positive/negative
    data[marker_name] = data[component_name].apply(lambda x: 1 if x == positive_component else 0)
    return data

def apply_gmm(file_paths, 
              markers, 
              df_together=None, 
              zscore_normalize=True):
    """
    Applies Gaussian Mixture Model (GMM) classification to the given file paths and markers.

    Args:
        file_paths (list): A list of file paths.
        markers (list): A list of marker names.
        df_together (pandas.DataFrame, optional): A DataFrame to store the results. Defaults to None.

    Returns:
        pandas.DataFrame: The DataFrame containing the results of GMM classification.

    """
    for f in file_paths:
        df = pd.read_csv(f)
        # normalize
        if zscore_normalize is True:
            df = zscore_norm(df, markers)
        df = df[markers + ['X_centroid', 'Y_centroid', 'CellID']]
        for m in markers:
            component_name = m + '_component'
            marker_name = m + '_status'
            marker_data = df[m].values.reshape(-1, 1)
            df_gmm = gmm_classify(marker_data, n_components=2, marker_name=marker_name, component_name=component_name)
            if df_together is None:
                df_together = pd.concat([df, df_gmm[marker_name]], axis=1)
            else:
                df_together = pd.concat([df_together, df_gmm[marker_name]], axis=1)
    return df_together

def zscore_norm(df, markers):
    """
    Z-score normalization of the given markers in the DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame to normalize.
        markers (list): A list of marker names.

    Returns:
        pandas.DataFrame: The normalized DataFrame.

    """
    df[markers] = df[markers].apply(lambda x: (x - x.mean()) / x.std(), axis=0)
    return df

def save_grid_h5(grid_df_dict, save_path):
    """
    Save a dictionary of grid dataframes to an HDF5 file.

    Parameters:
        grid_df_dict (dict): A dictionary containing grid dataframes. The keys should be tuples representing the grid coordinates (x, y), and the values should be the corresponding dataframes.
        save_path (str): The path to save the HDF5 file.

    Returns:
        None
    """
    with pd.HDFStore(save_path, 'w') as store:
        for (x, y), df in grid_df_dict.items():
            # Use a key format that identifies each dataframe uniquely
            key = f'grid_{x}_{y}'
            store.put(key, df)
    return

def create_tile_rois(df, 
                     tile_size=1000, 
                     x_coord='X_centroid',
                     y_coord='Y_centroid', 
                     save=False, 
                     save_hdf5=None):
    """
    Create tile-based regions of interest (ROIs) from a dataframe of cell coordinates.

    Args:
        df (pandas.DataFrame): The dataframe containing cell coordinates.
        tile_size (int, optional): The size of each tile in pixels. Defaults to 1000.
        x_coord (str, optional): The column name for the X-coordinate of each cell. Defaults to 'X_centroid'.
        y_coord (str, optional): The column name for the Y-coordinate of each cell. Defaults to 'Y_centroid'.
        save (bool, optional): Whether to save the ROIs as individual files. Defaults to False.
        save_hdf5 (str, optional): Path to an HDF5 file to save grid dataframes for future use. Defaults to None.

    Returns:
        dict: A dictionary containing individual dataframes for each tile, grouped by grid indices.
    """

    # Calculate the grid indices for each cell
    df['X_grid'] = (df[x_coord] // tile_size).astype(int)
    df['Y_grid'] = (df[y_coord] // tile_size).astype(int)

    # Group the cells by grid indices
    grid_groups = df.groupby(['X_grid', 'Y_grid'])

    # Create a dictionary to store the individual grid dataframes
    grid_dataframes = {}

    # Iterate over the grid groups and create individual dataframes
    for (x, y), group in grid_groups:
        if len(group) > 0:  # Check if there are points in the grid
            grid_dataframes[(x, y)] = group.drop(['X_grid', 'Y_grid'], axis=1)
    if save:
        if save:
            if not save_hdf5 or not save_hdf5.endswith('.h5'):
                raise ValueError("Invalid save_hdf5 path. Path must be a valid string ending in '.h5'.")
            # Save each dataframe to the HDF5 file
            save_grid_h5(grid_dataframes, save_hdf5)
    return grid_dataframes

def load_tile_rois(hdf5_path):
    """
    Load tile ROIs from an HDF5 file.

    Parameters:
        hdf5_path (str): The path to the HDF5 file.

    Returns:
        loaded_grid_dataframes (dict): A dictionary containing the loaded dataframes, where 
            the keys are grid indices and the values are the corresponding dataframes.
    """

    # Initialize an empty dictionary to store the loaded dataframes
    loaded_grid_dataframes = {}

    # Open the HDF5 file for reading
    with pd.HDFStore(hdf5_path, 'r') as store:
        # Iterate over the keys in the HDF5 file
        for key in store.keys():
            # Remove the leading '/' from the key to match the format used when saving
            formatted_key = key[1:]
            # Split the key to get the grid indices
            _, x, y = formatted_key.split('_')
            # Convert the grid indices back to integers
            grid_indices = (int(x), int(y))
            # Load the dataframe and store it in the dictionary
            loaded_grid_dataframes[grid_indices] = store.get(key)
    return loaded_grid_dataframes

def generate_binary_pointcloud(df, 
                               markers, 
                               keep_cols, 
                               labels, 
                               rename_cols_dict=None):
    """
    Generate a binary point cloud dataframe based on the given input dataframe.

    Args:
        df (pandas.DataFrame): The input dataframe.
        markers (list): A list of column names representing the markers. Can only generate a pointcloud dataframe
            for two markers with this function.
        keep_cols (list): A list of column names to keep in the output dataframe.
        labels (dict): A dictionary mapping numeric values to cell type labels.
        rename_cols_dict (dict, optional): A dictionary mapping column names to new names. Defaults to None.

    Returns:
        pandas.DataFrame: The generated binary point cloud dataframe.
    """
    # Check if the number of markers is 2
    if len(markers) != 2:
        raise ValueError("This function can only generate a binary point cloud dataframe for two markers.")

    # Keep only the necessary columns
    df = df[keep_cols + markers]

    # Ensure that cells are only called as one cell type (remove redundant rows)
    # If marker 1 and marker 2 are both positive, set marker 2 to 0
    if markers is not None:
        if all(col in df.columns for col in markers):
            mask = df[markers[0]] == df[markers[1]]
            df.loc[mask, markers[1]] = 0

    df = df.copy()
    df.loc[:, 'Celltype_asNumeric'] = (df.loc[:, markers[0]] * 1) + (df.loc[:, markers[1]] * 2)
    df.loc[:, 'Celltype'] = df['Celltype_asNumeric'].map(labels)
    df = df.rename(columns=rename_cols_dict)
    df.drop(markers, axis=1, inplace=True)
    return df

def generate_csr_grid(pc_df, typea, typeb):
    """
    Generate a grid of points with a given number of points for each type.

    Parameters:
    pc_df (pandas.DataFrame): The input DataFrame containing the point cloud data.
    typea (str): The type of points for the first category. Must correspond to a valid string value
        of the Celltype column in the input DataFrame. 
    typeb (str): The type of points for the second category. Must correspond to a valid string value
        of the Celltype column in the input DataFrame. 

    Returns:
    pandas.DataFrame: The generated grid of points with the specified number of points for each type.
    """

    # Determine the number of points for each type
    num_points = pc_df['Celltype'].value_counts()
    num_points_a, num_points_b = num_points[typea], num_points[typeb]

    # Get the bounds of the ROI
    x_min, x_max = pc_df['x'].min(), pc_df['x'].max()
    y_min, y_max = pc_df['y'].min(), pc_df['y'].max()

    # Function to generate random points for a given type within the bounds
    def generate_points(num_points, celltype):
        x_points = np.random.uniform(x_min, x_max, num_points)
        y_points = np.random.uniform(y_min, y_max, num_points)
        points = np.column_stack((x_points, y_points))
        return pd.DataFrame(points, columns=['x', 'y']).assign(Celltype=celltype)

    # Generate CSR for both types and concatenate results
    csr_df = pd.concat([generate_points(num_points_a, typea), 
                        generate_points(num_points_b, typeb)], 
                    ignore_index=True)
    return csr_df

def save_tcm_plot(df, save_path=None):
    """
    Save a plot of the given DataFrame using a specific colormap and colorbar.

    Parameters:
        df (pandas.DataFrame): The DataFrame to be plotted.
        save_path (str, optional): The file path to save the plot. If not provided, the plot will be displayed instead.

    Returns:
        None
    """
    plt.figure(figsize=(20,20))
    l = int(np.ceil(np.max(np.abs([df.min(),df.max()]))))
    plt.imshow(df,cmap='RdBu_r',vmin=-l,vmax=l,origin='lower')
    plt.colorbar(label='$\Gamma_{C_1 C_2}(r=100)$')
    ax = plt.gca()
    ax.grid(False)
    plt.savefig(save_path, dpi=300)
    plt.close()
    return

def generate_pointcloud(pc_df,
                        sample_grid_name):
    """
    Generate a point cloud object from a DataFrame containing x, y coordinates and cell types.

    Args:
        pc_df (pandas.DataFrame): DataFrame containing x, y coordinates and cell types.
        sample_grid_name (str): Name of the sample grid.

    Returns:
        PointCloud: Point cloud object with x, y coordinates and cell types.

    """
    points = np.asarray([pc_df['x'], pc_df['y']]).transpose()

    # Convert pc_df['Celltype'] to a list
    celltype_list = pc_df['Celltype'].tolist()

    pc = generatePointCloud(sample_grid_name, points)
    pc.addLabels('Celltype', 'categorical', celltype_list, cmap='tab10')
    return pc

def get_crc_sample_label(input_string):
    # Find the index of 'CRC'
    crc_index = input_string.find('CRC')
    
    # Check if 'CRC' is found in the string
    if crc_index == -1:
        return None  # or raise an error, or return a default value
    
    end_index = crc_index + 5  # Get the next two characters
    return input_string[crc_index:end_index]

def compare_tcm(grid_file, 
                markers,
                keep_cols,
                labels,
                visualiseStages=False,
                save_tcm_plot_path=None,
                save_tcm_path=None,
                save_csr_plot_path=None,
                rename_cols_dict=None,
                save_csr_tcm_path=None,
                save_ks_results_path=None,
                plot_point_cloud=False,
                **kwargs):
    """
    Compare the topographical correlation maps (TCMs) between two cell types in a grid dataset.
    Allows one to save the results of a Kolmogorov-Smirnov test comparing the TCMs.

    Parameters:
        grid_files (list): List of paths to the grid dataset files.
        markers (list): The list of marker names to consider.
        keep_cols (list): The list of column names to keep in the grid dataset.
        labels (list): The list of labels for the two cell types to compare.
        visualiseStages (bool, optional): Whether to visualize the stages of the TCM calculation. Defaults to False.
        save_tcm_plot_path (str, optional): The path to a directory to save the TCM plots. Defaults to None.
        save_tcm_path (str, optional): The path to a directory to save the TCM grids. Defaults to None.
        save_csr_plot_path (str, optional): The path to a directory to save the CSR TCM plots. Defaults to None.
        save_ks_results_path (str, optional): The path to a directory to save the KS test results. Defaults to None.
        rename_cols_dict (dict, optional): A dictionary to rename the column names in the grid dataset. Defaults to None.
        save_csr_tcm_path (str, optional): The path to a directory to save the CSR TCM grids. Defaults to None.
        plot_point_cloud (bool, optional): Whether to plot the point cloud. Defaults to False.
        **kwargs: Additional keyword arguments for the TCM calculation.

    Returns:
        None
    """
    plt.rcParams['font.family'] = 'DejaVu Sans'
    sample_name = get_crc_sample_label(grid_file)
    # Load grid data
    grid_dataframes = load_tile_rois(grid_file)
    print(f'Total number of grids in {sample_name}:', len(grid_dataframes))
    csr_tcm_grids = {}
    tcm_grids = {}
    ks_test_results = {}
    typea = labels[1]
    typeb = labels[2]
    for index, (grid_name, grid) in enumerate(grid_dataframes.items()):
        #plt.pyplot.close()
        grid_string = '_'.join(str(x) for x in grid_name)
        sample_grid_name = f'{sample_name}_{grid_string}'
        # Format data for TCM
        pc_df = generate_binary_pointcloud(grid,
                                        markers,
                                        keep_cols,
                                        labels,
                                        rename_cols_dict=rename_cols_dict)
        # Get rid of rows of cells that have neither marker
        pc_df = pc_df[pc_df['Celltype_asNumeric'] != 0]

        # Ensure that both cell types are present in the given grid:
        if typeb not in pc_df['Celltype'].unique() or typea not in pc_df['Celltype'].unique():
            continue
        typea_count = pc_df['Celltype'].value_counts()[typea]
        typeb_count = pc_df['Celltype'].value_counts()[typeb]
        print(f"Grid {index} of {len(grid_dataframes)}")
        print(f"Number {typea} in {grid_name}: {typea_count}")
        print(f"Number of {typeb} in {grid_name}: {typeb_count} \n")

        # Generate PointCloud object for TCM
        pc = generate_pointcloud(pc_df, sample_grid_name)

        if plot_point_cloud is True:
            visualisePointCloud(pc, 'Celltype', markerSize=100)

        # Calculate TCM on tissue data
        tcm = topographicalCorrelationMap(pc, 'Celltype', typea, 'Celltype', typeb, 
                                        radiusOfInterest=100, 
                                        maxCorrelationThreshold=5.0, 
                                        kernelRadius=150, 
                                        kernelSigma=50, 
                                        visualiseStages=visualiseStages,
                                        **kwargs)
        # Add TCM to dictionary
        if save_tcm_path is not None:
            tcm_grids[grid_string] = tcm

        # Save TCM plot
        if save_tcm_plot_path is not None:
            directory_path = save_tcm_plot_path + sample_name 
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            save_path = directory_path + '/' + f'TCM_{grid_string}.png'
            save_tcm_plot(tcm, save_path)

        # Generate CSR grid based on the tissue grid
        csr_df = generate_csr_grid(pc_df, typea, typeb)
        
        # Generate PointCloud object for CSR TCM
        csr_pc = generate_pointcloud(csr_df, f'{sample_grid_name}_CSR')

        # Calculate TCM on the CSR dataset
        csr_tcm = topographicalCorrelationMap(csr_pc, 'Celltype', typea, 'Celltype', typeb, 
                                            radiusOfInterest=100, 
                                            maxCorrelationThreshold=5.0, 
                                            kernelRadius=150, 
                                            kernelSigma=50, 
                                            visualiseStages=visualiseStages,
                                            **kwargs)
        # Save CSR TCM plot
        if save_csr_plot_path is not None:
            directory_path = save_csr_plot_path + sample_name
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            save_path = directory_path + '/' + f'CSR_TCM_{grid_string}.png'
            save_tcm_plot(csr_tcm, save_path)

        # Add CSR TCM to dictionary to save later
        if save_csr_tcm_path is not None:
            csr_tcm_grids[grid_string] = csr_tcm

        # Flatten the TCMs for comparison
        csr_collapsed_tcm = csr_tcm.flatten()
        collapsed_tcm = tcm.flatten()

        pos_csr_collapsed_tcm = csr_collapsed_tcm[csr_collapsed_tcm > 0]
        pos_collapsed_tcm = collapsed_tcm[collapsed_tcm > 0]

        neg_csr_collapsed_tcm = csr_collapsed_tcm[csr_collapsed_tcm < 0]
        neg_collapsed_tcm = collapsed_tcm[collapsed_tcm < 0]

        # Perform a Kolmogorov-Smirnov test to compare the distributions
        # Whole distribution
        if len(csr_collapsed_tcm) > 0 and len(collapsed_tcm) > 0:
            ks_statistic, p_value = ks_2samp(csr_collapsed_tcm, collapsed_tcm)
            p_value_scientific = f"{p_value:.2e}"
        else:
            ks_statistic = None
            p_value_scientific = None
        # Positive values
        if len(pos_csr_collapsed_tcm) > 0 and len(pos_collapsed_tcm) > 0:
            ks_statistic_pos, p_value_pos = ks_2samp(pos_csr_collapsed_tcm, pos_collapsed_tcm)
            p_value_pos_scientific = f"{p_value_pos:.2e}"
        else: 
            ks_statistic_pos = None
            p_value_pos_scientific = None
        # Negative values
        if len(neg_csr_collapsed_tcm) > 0 and len(neg_collapsed_tcm) > 0:
            ks_statistic_neg, p_value_neg = ks_2samp(neg_csr_collapsed_tcm, neg_collapsed_tcm)
            p_value_neg_scientific = f"{p_value_neg:.2e}"
        else:
            ks_statistic_neg = None
            p_value_neg_scientific = None

        ks_test_results[grid_name] = (ks_statistic, p_value_scientific,
                                        ks_statistic_pos, p_value_pos_scientific,
                                    ks_statistic_neg, p_value_neg_scientific)
    print(f'Finished comparing TCMs for {sample_name}')
    # Save .h5 files for reproducibility
    if save_csr_tcm_path is not None:
        save_path = save_csr_tcm_path + f'{sample_name}_CSR_TCM.npz'
        np.savez(save_path, **csr_tcm_grids)

    if save_tcm_path is not None:
        save_path = save_tcm_path + f'{sample_name}_TCM.npz'
        np.savez(save_path, **tcm_grids)
    
    if save_ks_results_path is not None:
        # Extract keys and values
        keys = list(ks_test_results.keys())
        values = list(ks_test_results.values())
    
        # Create DataFrame
        df = pd.DataFrame(values, columns=['ks_stat', 'p_value', 'ks_stat_pos', 'p_value_pos', 'ks_stat_neg', 'p_value_neg'])
        df.insert(0, 'grid_location', keys)

        # Save DataFrame as .csv
        results_path = save_ks_results_path + f'{sample_name}_KS_results.csv'
        df.to_csv(results_path, index=False)

        print(f'KS test results saved to {results_path}')

    return

def multithread_compare_tcm(grid_files, 
                            markers, 
                            keep_cols, 
                            labels, 
                            visualiseStages=False, 
                            save_tcm_plot_path=None, 
                            save_tcm_path=None,
                            save_csr_plot_path=None, 
                            rename_cols_dict=None, 
                            save_csr_tcm_path=None, 
                            save_ks_results_path=None, 
                            plot_point_cloud=False, 
                            **kwargs):
    """
    Compare the TCM (Topological Coorelation Map) for multiple grid files in parallel using multithreading.

    Parameters:
    - grid_files (list): A list of grid file paths to compare.
    - markers (list): A list of markers to consider for the comparison.
    - keep_cols (list): A list of columns to keep in the comparison.
    - labels (list): A list of labels for the grid files.
    - visualiseStages (bool, optional): Whether to visualize the stages of the comparison. Defaults to False.
    - save_tcm_plot_path (str, optional): The file path to save the TCM plot. Defaults to None.
    - save_tcm_path (str, optional): The file path to save the TCM. Defaults to None.
    - save_csr_plot_path (str, optional): The file path to save the CSR (Complete Spatial Randomness) plot. Defaults to None.
    - rename_cols_dict (dict, optional): A dictionary to rename the columns in the comparison. Defaults to None.
    - save_csr_tcm_path (str, optional): The file path to save the CSR TCM. Defaults to None.
    - save_ks_results_path (str, optional): The file path to save the KS (Kolmogorov-Smirnov) test results. Defaults to None.
    - plot_point_cloud (bool, optional): Whether to plot the point cloud. Defaults to False.
    - **kwargs: Additional keyword arguments to pass to the compare_tcm function.

    Returns:
    - None
    """
    
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(compare_tcm, grid_file, markers, keep_cols, labels, visualiseStages, 
                                   save_tcm_plot_path, save_tcm_path, save_csr_plot_path, rename_cols_dict, 
                                   save_csr_tcm_path, save_ks_results_path, plot_point_cloud, **kwargs) 
                   for grid_file in grid_files]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            logging.info(f"Task completed with result: {result}")
        except Exception as e:
            logging.error(f"Task generated an exception: {e}")

    return

