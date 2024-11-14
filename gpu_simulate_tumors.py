from simulate_tumors import TumorCellSimulator
from simulate_tumors import *
from ecd_helperFunctions import *

btc_pt2_s2 = pd.read_csv('/home/ecdyer/PROJECTS/mIF_stats/data/S2_BTC.csv')

# Get tumor cell and immune cell counts
btc_pt2_s2['tumor_immune'] = np.where(btc_pt2_s2['sox2_mean'] > btc_pt2_s2['cd45_mean'], 0, 1)

# Example usage:
coordinates = btc_pt2_s2[['centroid_x_um', 'centroid_y_um']].to_numpy()
cell_labels = btc_pt2_s2['tumor_immune'].values

np.random.seed(42)

simulator = TumorCellSimulator(coordinates, cell_labels)

n_points_per_type = {
    0: 4200,
    1: 3500
}

# Raw Data Simulation

raw_points, raw_labels = simulator.simulate_raw(n_points_per_type=n_points_per_type,
                                                     noise_level=0.1)


# 2. Mixed/Tuned Simulation with different parameter sets
# Default parameters
mixed_points_default, mixed_labels_default = simulator.simulate_gpu(
    n_points_per_type=n_points_per_type,
    mode='mixed',
    component_ratio=0.7,
    n_components=3,
    overlap_density=0.5,
    interaction_scale=0.3
)


# 1. High immune cell infiltration
high_infiltration_params = {
    ('tumor', 'tumor'): {'interaction_range': 8.0, 'attraction_strength': 2.0, 'repulsion_strength': 1.5},
    ('immune', 'immune'): {'interaction_range': 12.0, 'attraction_strength': 0.5, 'repulsion_strength': 1.0},
    ('tumor', 'immune'): {'interaction_range': 15.0, 'attraction_strength': 1.5, 'repulsion_strength': 0.8}
}

# 2. Immune exclusion
immune_exclusion_params = {
    ('tumor', 'tumor'): {'interaction_range': 8.0, 'attraction_strength': 2.0, 'repulsion_strength': 1.5},
    ('immune', 'immune'): {'interaction_range': 10.0, 'attraction_strength': 1.0, 'repulsion_strength': 1.0},
    ('tumor', 'immune'): {'interaction_range': 12.0, 'attraction_strength': 0.2, 'repulsion_strength': 2.5}
}

# 1. High immune cell infiltration:
high_immune_points, high_immune_labels = simulator.simulate_gpu(
    n_points_per_type=n_points_per_type,
    interaction_params=high_infiltration_params,
    overlap_density=0.7,
    interaction_scale=0.6
)

# 2. Immune exclusion:
immune_exclusion_points, immune_exclusion_labels = simulator.simulate_gpu(
    n_points_per_type=n_points_per_type,
    interaction_params=immune_exclusion_params,
    overlap_density=0.3,
    interaction_scale=0.2
)

colors = {
    0: 'blue',  # Immune cells
    1: 'red'    # Tumor cells
}

# Ensure cell_labels are integers
cell_labels = cell_labels.astype(int)
print(cell_labels)

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(15, 15))
fig.suptitle('Multi-Cell Tumor Simulation Comparison', fontsize=16, y=0.95)

# Plot original data
# Convert DataFrame to NumPy array
points = btc_pt2_s2[['centroid_x_um', 'centroid_y_um']].to_numpy()
plot_cell_distribution(points, cell_labels, 
                      'Original Data', axes[0, 0])

# Plot raw simulation
plot_cell_distribution(raw_points, raw_labels,
                      'Raw Simulation', axes[0, 1])

# Plot default mixed simulation
plot_cell_distribution(high_immune_points, high_immune_labels,
                      'High Immune Infiltration Parameters', axes[1, 0])

# Plot custom mixed simulation
plot_cell_distribution(immune_exclusion_points, immune_exclusion_labels,
                      'Immune Exclusion Parameters', axes[1, 1])

plt.tight_layout()
plt.show()

# Print some statistics about the distributions
def print_distribution_stats(points, labels, title):
    print(f"\n{title} Statistics:")
    labels = np.where(labels == 0, 'Immune', 'Tumor')
    for cell_type in np.unique(labels):
        mask = labels == cell_type
        cell_points = points[mask]
        print(f"\n{cell_type} CELLS:")
        print(f"Count: {np.sum(mask)}")
        print(f"Mean position: ({cell_points[:, 0].mean():.2f}, {cell_points[:, 1].mean():.2f})")
        print(f"Std deviation: ({cell_points[:, 0].std():.2f}, {cell_points[:, 1].std():.2f})")
        
        # Calculate average distance to nearest neighbor of same type
        from scipy.spatial import cKDTree
        tree = cKDTree(cell_points)
        distances, _ = tree.query(cell_points, k=2)  # k=2 to get nearest neighbor (first point is self)
        print(f"Mean distance to nearest neighbor: {distances[:, 1].mean():.2f}")

# Print statistics for all distributions
print_distribution_stats(points, cell_labels, "Original Data")
print_distribution_stats(raw_points, raw_labels, "Raw Simulation")
print_distribution_stats(high_immune_points, high_immune_labels, "High Immune Infiltration Parameters")
print_distribution_stats(immune_exclusion_points, immune_exclusion_labels, "Immune Exclusion Parameters")

fig.savefig('/home/ecdyer/PROJECTS/mIF_stats/figures/Multi_Cell_Tumor_Simulation_Comparison.png', dpi=300)