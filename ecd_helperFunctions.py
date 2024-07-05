import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle
import random
import time
from functools import partial
from helperFunctions import *
from smallestEnclosingCircle import make_circle
from sklearn.mixture import GaussianMixture


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
            with pd.HDFStore(save_hdf5, 'w') as store:
                for (x, y), df in grid_dataframes.items():
                    # Use a key format that identifies each dataframe uniquely
                    key = f'grid_{x}_{y}'
                    store.put(key, df)
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

    # Get the width and height of the ROI
    width, height = pc_df['x'].max(), pc_df['y'].max()

    # Function to generate random points for a given type
    def generate_points(num_points, celltype):
        points = np.random.rand(num_points, 2) * [width, height]
        return pd.DataFrame(points, columns=['x', 'y']).assign(Celltype=celltype)

    # Generate CSR for both types and concatenate results
    csr_df = pd.DataFrame()
    csr_df = pd.concat([generate_points(num_points_a, typea), 
                        generate_points(num_points_b, typeb)], 
                    ignore_index=True)
    return csr_df