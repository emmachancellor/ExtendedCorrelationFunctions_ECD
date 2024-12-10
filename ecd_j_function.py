import numpy as np
import pandas as pd
from scipy.spatial import KDTree
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from matplotlib.colors import SymLogNorm
from ecd_helperFunctions import create_tile_rois, spatial_perturb_matrix

def j_function_edge_corrected(points_type1, points_type2, bbox, r_max, dr):
    """
    Calculates the J-function between two types of points in a 2D spatial distribution with edge correction.
    
    Edge correction is implemented using Ripley's isotropic correction.
    
    Parameters:
    -----------
    points_type1 : array_like
        An (N1, 2) array of coordinates for points of Type 1.
    points_type2 : array_like
        An (N2, 2) array of coordinates for points of Type 2.
    bbox : tuple
        A tuple defining the study area's bounds: (x_min, x_max, y_min, y_max).
    r_max : float
        Maximum radius to compute the functions.
    dr : float
        Width of the radial distance bins.

    Returns:
    --------
    r : numpy.ndarray
        Array of radial distance bin centers.
    J_r : numpy.ndarray
        Edge-corrected J-function values for each bin.
    """
    # Ensure inputs are NumPy arrays
    points_type1 = np.asarray(points_type1)
    points_type2 = np.asarray(points_type2)

    if len(points_type1) == 0 or len(points_type2) == 0:
        return np.array([]), np.array([])

    # Build KDTree for efficient neighbor searches
    tree_type1 = KDTree(points_type1)

    # Set up radial bins
    r_edges = np.arange(0, r_max + dr, dr)
    r_centers = (r_edges[:-1] + r_edges[1:]) / 2

    # Calculate G12(r): from Type 2 to nearest Type 1
    distances, _ = tree_type1.query(points_type2, k=1)

    # Compute edge correction weights for each point in points_type2
    x_min, x_max, y_min, y_max = bbox
    edge_distances = np.minimum.reduce([
        points_type2[:, 0] - x_min,
        x_max - points_type2[:, 0],
        points_type2[:, 1] - y_min,
        y_max - points_type2[:, 1]
    ])

    # For each distance, compute the weight w_i = w(edge_distance_i, distances_i)
    weights = np.ones_like(distances)
    valid = edge_distances > 0
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        correction_factors = np.minimum(1.0, edge_distances[valid] / distances[valid])
    weights[valid] = correction_factors

    # Bin the distances with weights
    G12_counts_weighted, _ = np.histogram(distances, bins=r_edges, weights=weights)
    G12_cumulative = np.cumsum(G12_counts_weighted) / np.sum(weights)

    # Calculate F12(r): from random points to nearest Type 1
    # Generate random points within the study area
    num_random_points = 1000  # Adjust as needed for accuracy
    random_x = np.random.uniform(x_min, x_max, num_random_points)
    random_y = np.random.uniform(y_min, y_max, num_random_points)
    random_points = np.column_stack((random_x, random_y))

    # Compute distances from random points to nearest Type 1 point
    distances_random, _ = tree_type1.query(random_points, k=1)

    # Edge correction weights for random points
    edge_distances_random = np.minimum.reduce([
        random_points[:, 0] - x_min,
        x_max - random_points[:, 0],
        random_points[:, 1] - y_min,
        y_max - random_points[:, 1]
    ])

    weights_random = np.ones_like(distances_random)
    valid_random = edge_distances_random > 0
    with np.errstate(divide='ignore', invalid='ignore'):
        correction_factors_random = np.minimum(1.0, edge_distances_random[valid_random] / distances_random[valid_random])
    weights_random[valid_random] = correction_factors_random

    # Bin the distances with weights
    F12_counts_weighted, _ = np.histogram(distances_random, bins=r_edges, weights=weights_random)
    F12_cumulative = np.cumsum(F12_counts_weighted) / np.sum(weights_random)

    # Compute J-function
    # Avoid division by zero by adding a small epsilon
    epsilon = 1e-10
    denominator = 1 - F12_cumulative + epsilon
    J_r = (1 - G12_cumulative + epsilon) / denominator

    return r_centers, J_r

def calculate_j_function_summary(points_type1, points_type2, bbox, r_max, dr):
    """
    Calculates the J-function and summary statistics between two types of points using edge correction.
    
    Returns:
    --------
    dict
        A dictionary containing:
        - 'r': Radial distances.
        - 'J_r': J-function values.
        - 'AUC': Area under the curve of J(r) - 1.
        - 'mean_deviation': Mean deviation of J(r) from 1.
        - 'max_deviation': Maximum absolute deviation of J(r) from 1.
    """
    # Use the edge-corrected j_function
    r, J_r = j_function_edge_corrected(points_type1, points_type2, bbox, r_max, dr)
    
    if len(r) == 0 or len(J_r) == 0:
        # Handle cases with insufficient data
        return {
            'r': np.array([]),
            'J_r': np.array([]),
            'AUC': np.nan,
            'mean_deviation': np.nan,
            'max_deviation': np.nan
        }
    
    # Compute the area under the curve (AUC)
    auc = simpson(y=J_r - 1, x=r)
    
    # Compute the mean deviation
    mean_deviation = np.mean(J_r - 1)
    
    # Compute the maximum deviation
    max_deviation = np.max(np.abs(J_r - 1))
    
    # Return the results as a dictionary
    return {
        'r': r,
        'J_r': J_r,
        'AUC': auc,
        'mean_deviation': mean_deviation,
        'max_deviation': max_deviation
    }

def calculate_j_function(data_path=None,
                         df=None,
                         distance_cols=['x_centroid', 'y_centroid'],
                         cell_a_label=0,
                         cell_b_label=1,
                         return_bboxes=False,
                         perturb_matrix=False,
                         roi_tile_size=1000,
                         cell_label_col='tumor_immune',
                         r_max=300,
                         dr=5,
                         plot_j_function=True,
                         return_summary=True):
    if data_path is not None:
        df = pd.read_csv(data_path)
    elif df is not None:
        pass
    else:
        raise ValueError("No data path or dataframe provided")

    # Divide data into ROIs
    roi_data, bboxes = create_tile_rois(df, 
                                 tile_size=roi_tile_size, 
                                 x_coord=distance_cols[0], 
                                 y_coord=distance_cols[1],
                                 return_bboxes=True)
    
    # Initialize dictionaries to store results
    j_functions = []
    j_function_dict = {}
    summary_dict = {
        'roi_coordinates': [],
        'bbox': [],
        'r': [],
        'J_r': [],
        'AUC': [],
        'mean_deviation': [],
        'max_deviation': []
    }

    for i, key in enumerate(roi_data.keys()):
        print(f"Calculating J-function for ROI: {key} ({i+1} of {len(roi_data.keys())} ROIs)")

        roi_df = roi_data[key]

        if perturb_matrix:
            roi_df = spatial_perturb_matrix(roi_df, perturb_columns=distance_cols)

        cell_a_points = roi_df[roi_df[cell_label_col] == cell_a_label][distance_cols].values
        cell_b_points = roi_df[roi_df[cell_label_col] == cell_b_label][distance_cols].values

        # Skip if there are not enough points
        if len(cell_a_points) == 0 or len(cell_b_points) == 0:
            continue

        bbox = bboxes[key]
        summary = calculate_j_function_summary(cell_a_points, cell_b_points, bbox, r_max, dr)

        # Append results
        j_functions.append(summary['J_r'])
        j_function_dict[key] = summary['J_r']

        if plot_j_function and len(summary['r']) > 0:
            # Plot the J-function
            plt.figure(figsize=(8, 6))
            plt.plot(summary['r'], summary['J_r'], label='J(r)')
            plt.axhline(y=1.0, color='r', linestyle='--', label='CSR (J(r) = 1)')
            plt.xlabel('Distance r')
            plt.ylabel('J-function J(r)')
            plt.title(f'J-function Between Two Cell Types in ROI {key}')
            plt.legend()
            plt.grid(True)
            plt.show()

        if return_summary:
            summary_dict['roi_coordinates'].append(key)
            summary_dict['bbox'].append(bbox)
            summary_dict['r'].append(summary['r'])
            summary_dict['J_r'].append(summary['J_r'])
            summary_dict['AUC'].append(summary['AUC'])
            summary_dict['mean_deviation'].append(summary['mean_deviation'])
            summary_dict['max_deviation'].append(summary['max_deviation'])

    if return_summary:
        return j_functions, j_function_dict, summary_dict
    else:
        return j_functions, j_function_dict

def plot_jfunc_auc_roi_explanations(data_dict, sample_df, bboxes,
                          x_col='x_centroid', y_col='y_centroid',
                          class_col='tumor_immune',
                          class_values=[0, 1], class_names=['Tumor', 'Immune'],
                          seaborn_style='darkgrid',
                          figure_size=(10, 8),
                          cmap=cm.RdBu_r,  # Reversed RdBu colormap
                          alpha=0.5, edgecolor='none', zorder_rect=2,
                          text_color='black', text_ha='center', text_va='center',
                          text_fontsize=8, text_zorder=3,
                          xlabel='X Position', ylabel='Y Position', title='GBM Sample S1 - ROI AUC Values',
                          colorbar_label='AUC Value',
                          save_path=None, dpi=300, bbox_inches='tight',
                          legend_kwargs=None, scatter_kwargs=None,
                          show_labels=True):
    # Set seaborn style
    sns.set_style(seaborn_style)
    
    # Extract AUC values from data_dict, excluding NaNs
    explainer_values = [value for value in data_dict.values() if not np.isnan(value)]
    
    # Handle edge cases with all NaNs or empty list
    if explainer_values:
        # Manually set vmin and vmax to cap extreme values
        vmin = -250  # Adjust as needed
        vmax = 1000  # Adjust as needed
        
        # Use SymLogNorm to handle both negative and positive values with extreme ranges
        linthresh = 50  # Linear range within +/- linthresh
        linscale = 1  # This affects the slope within the linear region
        
        norm = SymLogNorm(linthresh=linthresh, linscale=linscale, vmin=vmin, vmax=vmax)
    else:
        vmin, vmax = -1, 1
        linthresh = 0.1
        norm = SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax)
    
    print(f"Adjusted AUC value range: {vmin} to {vmax}")
    
    # Create figure with seaborn style
    plt.figure(figsize=figure_size)
    
    # Initialize default scatterplot options if not provided
    if scatter_kwargs is None:
        scatter_kwargs = {'s': 3, 'zorder': 1}
    else:
        # Ensure 's' and 'zorder' have default values if not provided
        scatter_kwargs.setdefault('s', 3)
        scatter_kwargs.setdefault('zorder', 1)
    
    # Scatter plot with separate classes for the legend
    for class_value, class_name in zip(class_values, class_names):
        subset = sample_df[sample_df[class_col] == class_value]
        sns.scatterplot(
            data=subset,
            x=x_col, y=y_col,
            label=class_name,
            **scatter_kwargs)
    
    # Initialize default legend options if not provided
    if legend_kwargs is None:
        legend_kwargs = {'title': 'Tumor/Immune', 'markerscale': 5, 'loc': 'upper right', 'fontsize': 8}
    else:
        # Ensure default values if not provided
        legend_kwargs.setdefault('title', 'Tumor/Immune')
        legend_kwargs.setdefault('markerscale', 5)
        legend_kwargs.setdefault('loc', 'upper right')
        legend_kwargs.setdefault('fontsize', 8)
    
    # Add legend for classes
    plt.legend(**legend_kwargs)
    
    # Draw heatmap rectangles
    for key in bboxes:
        min_x, max_x, min_y, max_y = bboxes[key]
        width = max_x - min_x
        height = max_y - min_y
    
        # Get AUC value, replacing NaN with zero
        explainer_value = data_dict.get(key, 0)
        explainer_value = explainer_value if not np.isnan(explainer_value) else 0
    
        # Clip the AUC value to vmin and vmax
        explainer_value = np.clip(explainer_value, vmin, vmax)
    
        # Determine color
        color = cmap(norm(explainer_value))
    
        # Draw rectangle
        rect = plt.Rectangle((min_x, min_y), width, height, facecolor=color,
                             alpha=alpha, edgecolor=edgecolor, zorder=zorder_rect)
        plt.gca().add_patch(rect)
    
        if show_labels:
            # Calculate the center of the rectangle
            center_x = min_x + width / 2
            center_y = min_y + height / 2
    
            # Add label at the center
            plt.text(center_x, center_y, f'{key}', color=text_color,
                     ha=text_ha, va=text_va, fontsize=text_fontsize, zorder=text_zorder)
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=plt.gca(), label=colorbar_label)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
    
    plt.show()