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
from matplotlib.cm import ScalarMappable
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
                               labels, 
                               keep_cols=None,
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
    if keep_cols is not None:
        df = df[keep_cols + markers]
    else:
        df = df[markers]

    # Ensure that cells are only called as one cell type (remove redundant rows)
    # If marker 1 and marker 2 are both positive, set marker 2 to 0
    if markers is not None:
        if all(col in df.columns for col in markers):
            mask = df[markers[0]] == df[markers[1]]
            df.loc[mask, markers[1]] = 0

    df = df.copy()
    df.loc[:, 'Celltype_asNumeric'] = (df.loc[:, markers[0]] * 1) + (df.loc[:, markers[1]] * 2)
    df.loc[:, 'Celltype'] = df['Celltype_asNumeric'].map(labels)
    if rename_cols_dict is not None:
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

def rank_and_calculate_distance(ks_results_dir, 
                                grid_files,
                                compare_n = 20,
                                ks_stat='ks_stat',
                                save_dir=None,
                                log_scale_ks=False,
                                bounding_box=True):
    """
    Ranks and calculates distance for the top results based on the KS statistic.

    Parameters:
    - ks_results_dir (str): The directory path where the KS results are stored.
    - grid_files (str): The directory path where the grid files are stored.
    - compare_n (int, optional): The number of top results to consider. Default is 20.
    - ks_stat (str, optional): The column name of the KS statistic in the results. Default is 'ks_stat'.
    - save_dir (str, optional): The directory path to save the top results. Default is None.
    - log_scale_ks (bool, optional): Whether to apply logarithmic scaling to the KS statistic. Default is False.
    - bounding_box (bool, optional): Whether to calculate the bounding box coordinates. Default is True.

    Returns:
    - ks_rank_grid_dict (dict): A dictionary containing the top results for each sample.

    """
    grid_dict = {}
    ks_rank_grid_dict = {}
    grid_file_list = os.listdir(grid_files)
    for g in grid_file_list:
        grid_sample_name = get_crc_sample_label(g)
        grid_dict[grid_sample_name] = grid_files + '/' + g
    for f in os.listdir(ks_results_dir):
        if f.endswith('.csv'):
            ks_sample_name = f[:5]
            ks_results = pd.read_csv(os.path.join(ks_results_dir, f))
            top_ks_results = ks_results.nlargest(compare_n, ks_stat)
            ks_result_grid_file = grid_dict[ks_sample_name]
            ks_result_grid = load_tile_rois(ks_result_grid_file)
            # Initialize new columns in top_ks_results
            top_ks_results['X_centroid'] = None
            top_ks_results['Y_centroid'] = None

            # Convert keys from tuples to strings
            ks_result_grid = {str(k): v for k, v in ks_result_grid.items()}
            
            # Iterate over each row in top_ks_results
            for index, row in top_ks_results.iterrows():
                grid_location = row['grid_location']

                # Locate the corresponding DataFrame in ks_result_grid
                if grid_location in ks_result_grid.keys():
                    grid_data = ks_result_grid[grid_location]
                    
                    # Calculate the centroid values
                    x_centroid = grid_data['X_centroid'].mean()
                    y_centroid = grid_data['Y_centroid'].mean()
                    
                    # Add the centroid values to top_ks_results
                    top_ks_results.at[index, 'X_centroid'] = x_centroid
                    top_ks_results.at[index, 'Y_centroid'] = y_centroid

                    if bounding_box is True:
                        # Calculate the bounding box
                        min_x = float('inf')
                        min_y = float('inf')
                        max_x = float('-inf')
                        max_y = float('-inf')
                        # Iterate over each point in grid_location
                        for point in grid_data['grid_location']:
                            x, y = point
                            min_x = min(min_x, x)
                            min_y = min(min_y, y)
                            max_x = max(max_x, x)
                            max_y = max(max_y, y)

                        # Add the bounding box coordinates to top_ks_results
                        top_ks_results.at[index, 'min_x'] = min_x
                        top_ks_results.at[index, 'min_y'] = min_y
                        top_ks_results.at[index, 'max_x'] = max_x
                        top_ks_results.at[index, 'max_y'] = max_y
            
            # Calculate the distance from the previous row
            distances = [0]  # First row has a distance of 0
            for i in range(1, len(top_ks_results)):
                x1 = top_ks_results.iloc[i]['X_centroid']
                y1 = top_ks_results.iloc[i]['Y_centroid']
                x2 = top_ks_results.iloc[i - 1]['X_centroid']
                y2 = top_ks_results.iloc[i - 1]['Y_centroid']
                
                # Check for None and set to 0 if necessary
                if x1 is None or y1 is None or x2 is None or y2 is None:
                    distances.append(0)
                    continue

                x_diff = x1 - x2
                y_diff = y1 - y2
                distance = np.sqrt(x_diff**2 + y_diff**2)
                distances.append(distance)
            top_ks_results['distance_between_grids'] = distances

            if log_scale_ks is True:
                new_col = 'log_' + ks_stat
                top_ks_results[new_col] = np.log(top_ks_results[ks_stat]) * -1
            if save_dir is not None :
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                top_ks_results.to_csv(os.path.join(save_dir, ks_sample_name + '_top_ks_results.csv'))
            ks_rank_grid_dict[ks_sample_name] = top_ks_results
    return ks_rank_grid_dict

def combine_clinical_and_tcm(clinical_data_file,
                             results_directory,
                             save_path,
                             markers=None,
                             ks_stat='log_ks_stat'):
    """
    Combines clinical data with TCM (Tumor Cell Mapping) data and saves the combined data to a CSV file.

    Parameters:
    - clinical_data_file (str): The file path of the clinical data CSV file.
    - results_directory (str): The directory path containing the TCM data CSV files.
    - save_path (str): The file path to save the combined data CSV file.
    - markers (list or None): The list of marker names to consider. If None, all markers will be considered.
    - ks_stat (str): The name of the column in the TCM data representing the KS statistic. Default is 'log_ks_stat'.

    Returns:
    - None

    Raises:
    - ValueError: If a Specimen_ID from the TCM data is not found in the clinical data.

    """
    clinical_data = pd.read_csv(clinical_data_file)
    
    # Ensure the columns exist in the DataFrame
    for i in range(1, 21):
        col_name = f'ks_grid_{i}_{markers}'
        if col_name not in clinical_data.columns:
            clinical_data[col_name] = None

    for i in range(1, 20):
        col_name = f'distance_{i}_{markers}'
        if col_name not in clinical_data.columns:
            clinical_data[col_name] = None

    for file in os.listdir(results_directory):
        if file.endswith('.csv'):
            tcm_data = pd.read_csv(os.path.join(results_directory, file))
            tcm_data = tcm_data[['distance_between_grids', 'log_ks_stat']]
            tcm_data['Specimen_ID'] = file[2:5]

            # Extract the log_ks_stat column from tcm_data
            tcm_ks_stat = tcm_data[ks_stat].astype(str).str.split(',')
            distance = tcm_data['distance_between_grids'].astype(str).str.split(',')

            # Convert split series to lists
            tcm_ks_stat_list = tcm_ks_stat.tolist()
            distance_list = distance.tolist()

            # Find the row in clinical_data with the matching Specimen_ID
            specimen_ID = file[2:5]
            clinical_row_index = clinical_data[clinical_data['Specimen_ID'] == specimen_ID].index
            # Verify that clinical_row_index is not empty and is a valid index
            if clinical_row_index.empty:
                raise ValueError(f"Specimen_ID {specimen_ID} not found in clinical data")
            
            clinical_row_index = clinical_row_index[0]

            # Fill in the ks_grid_1 to ks_grid_20 values in clinical_data
            for i in range(1, 21):
                if i-1 < len(tcm_ks_stat_list):
                    value = tcm_ks_stat_list[i-1][0] if len(tcm_ks_stat_list[i-1]) > 0 else None
                    clinical_data.at[clinical_row_index, f'ks_grid_{i}_{markers}'] = float(value) if value else None
            for i in range(1, 20):
                if i-1 < len(distance_list):
                    value = distance_list[i-1][0] if len(distance_list[i-1]) > 0 else None
                    clinical_data.at[clinical_row_index, f'distance_{i}_{markers}'] = float(value) if value else None
    clinical_data = clinical_data.drop(columns=[col for col in clinical_data.columns if 'distance_1_' in col])
    clinical_data.to_csv(save_path, index=False)
    return

def get_sample_raw_files(data_dir,
                         file_extension='.csv'):
    """
    Retrieves the file paths of all CSV files within the specified data directory and its subdirectories.

    Args:
        data_dir (str): The path to the data directory.
        file_extension (str, optional): The file extension to search for. Defaults to '.csv'.
    Returns:
        list: A list of file paths for all files found with a specified file extension 
            within the data directory and its subdirectories.
    """
    data_parent_dir = os.listdir(data_dir)
    file_paths = []
    for i in range(len(data_parent_dir)):
        files = os.listdir(data_dir + data_parent_dir[i])
        csv_files = [file for file in files if file.endswith(file_extension)]
        for file in csv_files:
            file_path = os.path.join(data_dir, data_parent_dir[i], file)
            file_paths.append(file_path)
    return file_paths


def generate_grid_heatmap(grid_files,
                        sample_files,
                        results_directory,
                        grid_index_name='grid_location',
                        stat_col = 'ks_stat',
                        x_coord = 'X_centroid',
                        y_coord = 'Y_centroid',
                        display_plot=True,
                        save_plot=False,
                        bounding_box=True,
                        save_directory=None,
                        title_text = 'K-S Test', 
                        cmap='viridis',
                        **kwargs):
    grid_dict = {}
    # Create a dictionary that holds the grid file paths and a sample label
    grid_files_list = os.listdir(grid_files)
    grid_files_list = [os.path.join(grid_files, file) for file in grid_files_list]
    for g in grid_files_list:
        sample_name = get_crc_sample_label(g)
        grid_dict[sample_name] = g
    # Create a list of sample file paths
    if os.path.isdir(sample_files) is True:
        sample_files = get_sample_raw_files(sample_files)
    elif os.path.isfile(sample_files) is True:
        sample_files = [sample_files]
    else: 
        raise ValueError("Invalid sample_files path. Path must be a valid directory or file path.")
    print("Sample Files: ", sample_files)
    for f in os.listdir(results_directory):
        if f.endswith('.csv'):
            # Locate correct result file and original sample file for the sample grid
            ks_sample_name = f[:5]
            print('KS Sample Name: ', ks_sample_name)
            sample_file = next((file for file in sample_files if ks_sample_name in file), None)
            print('Sample File: ', sample_file)
            if sample_file is not None:
                sample_file_df = pd.read_csv(sample_file)
            else:
                raise ValueError(f"No sample file found for sample name: {sample_name}")
            ks_results = pd.read_csv(os.path.join(results_directory, f))
            ks_result_grid_file = grid_dict[ks_sample_name]
            ks_result_grid = load_tile_rois(ks_result_grid_file)
            # Convert keys from tuples to strings
            ks_result_grid = {str(k): v for k, v in ks_result_grid.items()}

            # Iterate over each result and match the results to a sample grid
            for index, row in ks_results.iterrows():
                grid_location = row[grid_index_name]
                # Locate the corresponding DataFrame in ks_result_grid
                if grid_location in ks_result_grid.keys():
                    grid_data=ks_result_grid[grid_location]
                    if bounding_box is True:
                        # Calculate the bounding box
                        min_x = grid_data['X_centroid'].min()
                        max_x = grid_data['X_centroid'].max()
                        min_y = grid_data['Y_centroid'].min()
                        max_y = grid_data['Y_centroid'].max()

                        # Add the bounding box coordinates to ks_results
                        ks_results.at[index, 'min_x'] = min_x
                        ks_results.at[index, 'min_y'] = min_y
                        ks_results.at[index, 'max_x'] = max_x
                        ks_results.at[index, 'max_y'] = max_y
            # Plot the heatmap for a given sample
            x_min = ks_results['min_x']
            x_max = ks_results['max_x']
            y_min = ks_results['min_y']
            y_max = ks_results['max_y']
            ks_stat = ks_results[stat_col]
            # Simple baseline plot function
            sns.set_style('white')
            fig, ax = plt.subplots()
            ax.plot(sample_file_df['X_centroid'], sample_file_df['Y_centroid'], 
                    'o', color='lightgray', markersize=0.015, alpha=0.5, **kwargs)
            
                # Overlay translucent rectangles based on bounding box coordinates and normalized ks_stat values
            for i in range(len(ks_results)):
                rect = plt.Rectangle((x_min[i], y_min[i]), x_max[i] - x_min[i], y_max[i] - y_min[i],
                                    color=sns.color_palette(cmap, as_cmap=True)(ks_stat[i]), alpha=0.3)
                ax.add_patch(rect)
            # Create a ScalarMappable and add the colorbar
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax)
            # Add plot title
            ax.set_title(f'{ks_sample_name} {title_text} Heatmap')
            if save_plot is True:
                if save_directory is not None:
                    plt.savefig(f'{save_directory}{ks_sample_name}_heatmap.png')
                else:
                    plt.savefig(f'{ks_sample_name}_heatmap.png', dpi=450)
            if display_plot is True:
                plt.show()
    return

def generate_heatmap(grid_files,
                        sample_files,
                        results_directory,
                        grid_index_name='grid_location',
                        stat_col = 'ks_stat',
                        x_coord = 'X_centroid',
                        y_coord = 'Y_centroid',
                        display_plot=True,
                        save_plot=False,
                        bounding_box=True,
                        save_directory=None,
                        title_text = 'K-S Test', 
                        cmap='viridis',
                        **kwargs):
    grid_dict = {}
    # Create a dictionary that holds the grid file paths and a sample label
    grid_files_list = os.listdir(grid_files)
    grid_files_list = [os.path.join(grid_files, file) for file in grid_files_list]
    for g in grid_files_list:
        sample_name = get_crc_sample_label(g)
        grid_dict[sample_name] = g
    # Create a list of sample file paths
    if os.path.isdir(sample_files) is True:
        sample_files = get_sample_raw_files(sample_files)
    elif os.path.isfile(sample_files) is True:
        sample_files = [sample_files]
    else: 
        raise ValueError("Invalid sample_files path. Path must be a valid directory or file path.")
    print("Sample Files: ", sample_files)
    for f in os.listdir(results_directory):
        if f.endswith('.csv'):
            # Locate correct result file and original sample file for the sample grid
            ks_sample_name = f[:5]
            print('KS Sample Name: ', ks_sample_name)
            sample_file = next((file for file in sample_files if ks_sample_name in file), None)
            print('Sample File: ', sample_file)
            if sample_file is not None:
                sample_file_df = pd.read_csv(sample_file)
            else:
                raise ValueError(f"No sample file found for sample name: {sample_name}")
            ks_results = pd.read_csv(os.path.join(results_directory, f))
            ks_result_grid_file = grid_dict[ks_sample_name]
            ks_result_grid = load_tile_rois(ks_result_grid_file)
            # Convert keys from tuples to strings
            ks_result_grid = {str(k): v for k, v in ks_result_grid.items()}

            # Iterate over each result and match the results to a sample grid
            for index, row in ks_results.iterrows():
                grid_location = row[grid_index_name]
                # Locate the corresponding DataFrame in ks_result_grid
                if grid_location in ks_result_grid.keys():
                    grid_data=ks_result_grid[grid_location]
                    print(grid_data.head())
                    if bounding_box is True:
                        # Calculate the bounding box
                        min_x = grid_data['X_centroid'].min()
                        max_x = grid_data['X_centroid'].max()
                        min_y = grid_data['Y_centroid'].min()
                        max_y = grid_data['Y_centroid'].max()

                        # Add the bounding box coordinates to ks_results
                        ks_results.at[index, 'min_x'] = min_x
                        ks_results.at[index, 'min_y'] = min_y
                        ks_results.at[index, 'max_x'] = max_x
                        ks_results.at[index, 'max_y'] = max_y
            # Plot the heatmap for a given sample
            x_min = ks_results['min_x']
            x_max = ks_results['max_x']
            y_min = ks_results['min_y']
            y_max = ks_results['max_y']
            ks_stat = ks_results[stat_col]
            # Simple baseline plot function
            sns.set_style('white')
            fig, ax = plt.subplots()
            ax.plot(sample_file_df['X_centroid'], sample_file_df['Y_centroid'], 
                    'o', color='lightgray', markersize=0.015, alpha=0.5, **kwargs)
            
                # Overlay translucent rectangles based on bounding box coordinates and normalized ks_stat values
            for i in range(len(ks_results)):
                rect = plt.Rectangle((x_min[i], y_min[i]), x_max[i] - x_min[i], y_max[i] - y_min[i],
                                    color=sns.color_palette(cmap, as_cmap=True)(ks_stat[i]), alpha=0.3)
                ax.add_patch(rect)
            # Create a ScalarMappable and add the colorbar
            sm = ScalarMappable(cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=ax)
            # Add plot title
            ax.set_title(f'{ks_sample_name} {title_text} Heatmap')
            if save_plot is True:
                if save_directory is not None:
                    plt.savefig(f'{save_directory}{ks_sample_name}_heatmap.png')
                else:
                    plt.savefig(f'{ks_sample_name}_heatmap.png', dpi=450)
            if display_plot is True:
                plt.show()
    return

def create_cell_type_columns(df: pd.DataFrame):
    """
    Create one-hot encoded columns for immune and tumor cells
    """
    df['immune_cell'] = (df['label'] == 0).astype(int)
    df['tumor_cell'] = (df['label'] == 1).astype(int)
    return df

def spatial_perturb_matrix(df, 
                            delta_range=(-1, 1),
                            perturb_columns = ['x', 'y']):
    """
    Perturbs the spatial location of points within a DataFrame by randomly shifting
    each point's coordinates within a specified delta range.
    
    Parameters:
    - df (pandas.DataFrame): The original DataFrame containing point coordinates
    - delta_range (tuple): A tuple specifying the min and max shift for the coordinates
    
    Returns:
    - perturbed_df (pandas.DataFrame): The DataFrame with perturbed coordinates
    """
    # Create a copy of the DataFrame to avoid modifying the original
    perturbed_df = df.copy()
    
    # Add random perturbations to columns
    for i in perturb_columns:
        perturbed_df[i] += np.random.uniform(delta_range[0], delta_range[1], size=len(df))
    
    return perturbed_df


def calculate_tcm_from_df(data_path,
                          markers,
                          labels,
                          keep_cols,
                          typea='Tumor',
                          typeb='Immune',
                          pointcloud_name='tumor_immune',
                          visualise=False,
                          df=None,
                          perturb_matrix=False):
    """
    Calculate the Topographical Correlation Map (TCM) from a dataframe or CSV file.

    Parameters:
    -----------
    data_path : str, optional
        Path to CSV file containing cell data. Required if df is None.
    markers : list
        List of marker names to use for cell type identification.
    labels : dict
        Dictionary mapping numerical labels to cell type names.
    keep_cols : list
        List of column names to keep for spatial coordinates (e.g. ['x', 'y']).
    typea : str, optional
        First cell type to correlate. Default is 'Tumor'.
    typeb : str, optional
        Second cell type to correlate. Default is 'Immune'.
    pointcloud_name : str, optional
        Name for the generated point cloud object. Default is 'tumor_immune'.
    visualise : bool, optional
        Whether to display visualization of the point cloud. Default is False.
    df : pandas.DataFrame, optional
        Input dataframe containing cell data. Required if data_path is None.
    perturb_matrix : bool, optional
        Whether to perturb the spatial coordinates of the points. Default is False.

    Returns:
    --------
    ndarray
        The topographical correlation map between the two specified cell types.

    Raises:
    -------
    ValueError
        If neither data_path nor df is provided.

    Notes:
    ------
    This function processes cell data to generate a TCM showing spatial relationships
    between two cell types. It handles data input either as a CSV file or dataframe,
    performs one-hot encoding of cell types, and generates a point cloud representation
    before calculating the correlation map.
    """
    if data_path is not None:
        df = pd.read_csv(data_path)
    elif df is not None:
        pass
    else:
        raise ValueError("No data path or dataframe provided")
    if perturb_matrix is True:
        df = spatial_perturb_matrix(df)
    # One hot encode the cell types
    df = create_cell_type_columns(df)

    # Generate pointcloud object
    pc_df = generate_binary_pointcloud(df, 
                                markers=markers, 
                                keep_cols=keep_cols,
                                labels=labels)
    # Convert points to numpy array
    points = np.asarray([pc_df['x'], pc_df['y']]).transpose()

    # Convert pc_df['Celltype'] to a list
    celltype_list = pc_df['Celltype'].tolist()

    pc = generatePointCloud(pointcloud_name, points)
    pc.addLabels('Celltype', 'categorical', celltype_list, cmap='tab10')

    if visualise is True:
        visualisePointCloud(pc, 'Celltype', markerSize=100)

    # Calculate TCM
    tcm = topographicalCorrelationMap(pc, 'Celltype', typea, 'Celltype', typeb, 
                                    radiusOfInterest=100, 
                                    maxCorrelationThreshold=5.0, 
                                    kernelRadius=150, 
                                    kernelSigma=50, 
                                    visualiseStages=visualise)
    return tcm

def calculate_max_sens(baseline_df,
                        perturbed_df,
                        max_sensitivity):
    # Calculate the L2-norm difference between the original and perturbed explanations
    # Normalise the difference by the L2-norm of the original explanation to compare across simulations/baselines
    # Measures the relative difference and can be compared accross different baseline matrices
    difference = np.linalg.norm(baseline_df - perturbed_df) / np.linalg.norm(baseline_df)
        
    # Update max sensitivity if the current difference is larger
    max_sensitivity = max(max_sensitivity, difference)
    return max_sensitivity


def calculate_infidelity(baseline_df, 
                         perturbed_df,
                         baseline_infidelity):
    """
    Calculate the infidelity metric directly from baseline and perturbed attributions.

    Parameters:
    ----------
    baseline : ndarray
        Array of baseline feature attributions.
    perturbed : ndarray
        Array of perturbed feature attributions.
    baseline_infidelity : float
        The baseline infidelity score.

    Returns:
    --------
    float
        The computed infidelity score.
    """
    if baseline_df.shape != perturbed_df.shape:
        raise ValueError("Baseline and perturbed DataFrames must have the same shape.")

    # Compute the infidelity: squared differences
    diff = (baseline_df - perturbed_df) ** 2
    max_infidelity = max(np.mean(diff), baseline_infidelity)
    # Return the mean infidelity score across all elements
    return max_infidelity

def mantel_test(matrix1, matrix2, permutations=1000):
    """
    Perform a Mantel test to calculate the correlation between two distance matrices.
    
    Parameters:
        matrix1: numpy array, first distance matrix
        matrix2: numpy array, second distance matrix
        permutations: int, number of permutations for significance testing
    
    Returns:
        mantel_r: Mantel correlation coefficient
        p_value: p-value for the test
    """
    # Flatten the upper triangle of both matrices to get pairwise distances
    dist1 = matrix1[np.triu_indices_from(matrix1, k=1)]
    dist2 = matrix2[np.triu_indices_from(matrix2, k=1)]
    
    # Calculate the Pearson correlation as the Mantel statistic
    mantel_r, _ = pearsonr(dist1, dist2)
    
    # Permutation test for significance
    permuted_rs = []
    for _ in range(permutations):
        permuted = np.random.permutation(dist2)
        permuted_r, _ = pearsonr(dist1, permuted)
        permuted_rs.append(permuted_r)
    
    # Calculate p-value based on permutations
    permuted_rs = np.array(permuted_rs)
    p_value = np.sum(np.abs(permuted_rs) >= np.abs(mantel_r)) / permutations
    
    return mantel_r, p_value

def calculate_gd(data_path,
                 markers,
                 labels,
                 keep_cols,
                 distance_cols=['x_centroid', 'y_centroid'],
                 intensity_cols=['sox2_mean', 'cd45_mean'],
                 perturb_matrix=False,
                 roi_tile_size=1000):
    """
    Calculate the Geodesic Distance (GD) between spatial and intensity features across ROIs.

    Parameters:
    -----------
    data_path : str, optional
        Path to CSV file containing cell data. Required if df is None.
    markers : list
        List of marker names to use for cell type identification.
    labels : dict
        Dictionary mapping numerical labels to cell type names.
    keep_cols : list
        List of column names to keep for spatial coordinates.
    distance_cols : list, optional
        Column names for spatial coordinates. Default is ['x_centroid', 'y_centroid'].
    intensity_cols : list, optional
        Column names for intensity features. Default is ['sox2_mean', 'cd45_mean'].
    perturb_matrix : bool, optional
        Whether to perturb the spatial coordinates and intensity values. Default is False.
    roi_tile_size : int, optional
        Size of ROI tiles for dividing the data. Default is 1000.

    Returns:
    --------
    list
        List of Mantel correlation coefficients between distance and intensity matrices for each ROI.

    Raises:
    -------
    ValueError
        If neither data_path nor df is provided.

    Notes:
    ------
    This function:
    1. Loads data from CSV or uses provided dataframe
    2. Divides data into ROIs based on tile_size
    3. Optionally perturbs spatial and intensity values
    4. Calculates Mantel correlation between distance and intensity matrices for each ROI
    """
    mantel_tests = []
    if data_path is not None:
        df = pd.read_csv(data_path)
    elif df is not None:
        pass
    else:
        raise ValueError("No data path or dataframe provided")
    
    # Get distance df
    dist_df = df[distance_cols]

    # Get intensity df
    intensity_df = df[intensity_cols]

    # Divide data into ROIs
    dist_rois = create_tile_rois(dist_df, 
                                 tile_size=roi_tile_size, 
                                 x_coord=distance_cols[0], 
                                 y_coord=distance_cols[1])
    intensity_rois = create_tile_rois(intensity_df, 
                                      tile_size=roi_tile_size, 
                                      x_coord=intensity_cols[0], 
                                      y_coord=intensity_cols[1])

    if perturb_matrix is True:
        for key in intensity_rois.keys():
            intensity_rois[key] = spatial_perturb_matrix(intensity_rois[key], perturb_columns=intensity_cols)
            dist_rois[key] = spatial_perturb_matrix(dist_rois[key], perturb_columns=distance_cols)
            mantel_r, _ = mantel_test(intensity_rois[key], dist_rois[key], permutations=1000)
            mantel_tests.append(mantel_r)
    
    return mantel_tests