import numpy as np
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

class TumorCellSimulator:
    def __init__(self, real_data, cell_labels):
        """
        Initialize simulator with real tumor cell spatial distribution data and cell types
        
        Parameters:
        real_data: array-like, shape (n_samples, 2)
            Real spatial coordinates of cells
        cell_labels: array-like, shape (n_samples,)
            Labels indicating cell type for each coordinate pair
        """
        self.real_data = real_data
        self.cell_labels = cell_labels
        self.unique_cell_types = np.unique(cell_labels)
        
        # Create separate data arrays for each cell type
        self.cell_type_data = {
            cell_type: real_data[cell_labels == cell_type]
            for cell_type in self.unique_cell_types
        }
        
        # Calculate bounds for each cell type and overall bounds
        self.cell_type_bounds = {}
        overall_min_x = float('inf')
        overall_max_x = float('-inf')
        overall_min_y = float('inf')
        overall_max_y = float('-inf')
        
        for cell_type, data in self.cell_type_data.items():
            bounds = (
                data[:, 0].min(),
                data[:, 0].max(),
                data[:, 1].min(),
                data[:, 1].max()
            )
            self.cell_type_bounds[cell_type] = bounds
            
            overall_min_x = min(overall_min_x, bounds[0])
            overall_max_x = max(overall_max_x, bounds[1])
            overall_min_y = min(overall_min_y, bounds[2])
            overall_max_y = max(overall_max_y, bounds[3])
            
        self.bounds = (overall_min_x, overall_max_x, overall_min_y, overall_max_y)
        
        # Initialize GMMs for each cell type
        self.gmms = {}
        
        # Calculate kernel density estimation for each cell type
        self.kdes = {}
        for cell_type, data in self.cell_type_data.items():
            # Use Scott's rule for bandwidth selection
            n_samples = len(data)
            bandwidth = n_samples ** (-1/6) if n_samples > 1 else 1.0
            
            self.kdes[cell_type] = KernelDensity(
                bandwidth=bandwidth,
                kernel='gaussian'
            ).fit(data)
    
    def calculate_interaction_energy(self, points, **energy_params):
        """
        Calculate interaction energy between points
        
        Parameters:
        points: array-like
            Point coordinates
        **energy_params: dict
            interaction_range: float, default=10.0
                Range of interaction between points
            attraction_strength: float, default=1.0
                Strength of attractive force
            repulsion_strength: float, default=2.0
                Strength of repulsive force
        """
        # Set default parameters if not provided
        interaction_range = energy_params.get('interaction_range', 10.0)
        attraction_strength = energy_params.get('attraction_strength', 1.0)
        repulsion_strength = energy_params.get('repulsion_strength', 2.0)
        
        distances = squareform(pdist(points))
        np.fill_diagonal(distances, np.inf)
        
        attractive = -attraction_strength * np.exp(-distances / interaction_range)
        repulsive = repulsion_strength / (distances ** 2 + 1e-6)
        
        return attractive + repulsive
    
    def simulate_raw(self, n_points_per_type, noise_level=0.1):
        """
        Simulate raw data for each cell type based purely on the original distribution
        without additional parameters or interactions.
        
        Parameters:
        n_points_per_type: dict
            Dictionary mapping cell types to number of points to generate
        noise_level: float
            Standard deviation of Gaussian noise to add to sampled points
            
        Returns:
        tuple (points, labels):
            points: array of shape (n_total_points, 2) with simulated coordinates
            labels: array of shape (n_total_points,) with corresponding cell type labels
        """
        all_points = []
        all_labels = []
        
        for cell_type in self.unique_cell_types:
            n_points = n_points_per_type[cell_type]
            
            if len(self.cell_type_data[cell_type]) > 1:
                # Generate points using KDE
                sampled_points = self.kdes[cell_type].sample(n_points)
                
                # Add small random noise
                noise = np.random.normal(0, noise_level, size=sampled_points.shape)
                cell_points = sampled_points + noise
                
            else:
                # If only one point exists for this cell type, sample around it with noise
                original_point = self.cell_type_data[cell_type][0]
                noise = np.random.normal(0, noise_level, size=(n_points, 2))
                cell_points = original_point + noise
            
            all_points.extend(cell_points)
            all_labels.extend([cell_type] * n_points)
        
        return np.array(all_points), np.array(all_labels)

    def fit_gmm(self, n_components='auto'):
        """
        Fit GMM to the real data
        
        Parameters:
        n_components: int or 'auto'
            Number of GMM components. If 'auto', uses BIC to select optimal number
        """
        if n_components == 'auto':
            # Try different numbers of components and select best using BIC
            best_bic = np.inf
            best_n = 1
            
            for n in range(1, min(len(self.real_data) // 20, 20)):
                gmm = GaussianMixture(n_components=n)
                gmm.fit(self.real_data)
                bic = gmm.bic(self.real_data)
                if bic < best_bic:
                    best_bic = bic
                    best_n = n
                else:
                    break
            
            n_components = best_n
        
        self.gmm = GaussianMixture(n_components=n_components)
        self.gmm.fit(self.real_data)
        return self.gmm

    def generate_mixed_distribution(self, n_points_per_type, component_ratio=0.7, 
                                n_components=3, overlap_density=0.5):
        """
        Generate points with a mixture of clustered and co-localized distributions
        
        Parameters:
        n_points_per_type: dict
            Dictionary mapping cell types to number of points to generate
        component_ratio: float
            Proportion of points to be generated from GMM components (0-1)
        n_components: int
            Number of GMM components to use
        overlap_density: float
            Density parameter for co-localized points (0-1)
        """
        all_points = []
        all_labels = []
        
        for cell_type in self.unique_cell_types:
            # Fit GMM for this cell type
            n_points = n_points_per_type[cell_type]
            cell_data = self.cell_type_data[cell_type]
            
            gmm = GaussianMixture(n_components=n_components)
            gmm.fit(cell_data)
            
            # Calculate number of points for each distribution type
            n_component_points = int(n_points * component_ratio)
            n_random_points = n_points - n_component_points
            
            # Generate points from GMM components
            component_points = gmm.sample(n_component_points)[0]
            
            # Generate co-localized random points
            n_regions = max(1, int(n_random_points * overlap_density / 50))
            random_points = []
            
            points_per_region = n_random_points // n_regions
            remaining_points = n_random_points % n_regions
            
            for i in range(n_regions):
                center_x = np.random.uniform(self.bounds[0], self.bounds[1])
                center_y = np.random.uniform(self.bounds[2], self.bounds[3])
                
                region_size = (1 - overlap_density) * min(
                    self.bounds[1] - self.bounds[0],
                    self.bounds[3] - self.bounds[2]
                ) / 4
                
                n_points_region = points_per_region + (1 if i < remaining_points else 0)
                points = np.random.normal(
                    loc=[center_x, center_y],
                    scale=[region_size, region_size],
                    size=(n_points_region, 2)
                )
                random_points.append(points)
            
            if random_points:
                random_points = np.vstack(random_points)
                cell_points = np.vstack([component_points, random_points])
            else:
                cell_points = component_points
                
            all_points.extend(cell_points)
            all_labels.extend([cell_type] * len(cell_points))
        
        return np.array(all_points), np.array(all_labels)

    def simulate_pure_gmm(self, n_points_per_type):
        """Simulate points using only GMM fitted to real data"""
        all_points = []
        all_labels = []
        
        for cell_type in self.unique_cell_types:
            if not hasattr(self, 'gmms') or cell_type not in self.gmms:
                gmm = GaussianMixture(n_components=3)  # or whatever default you prefer
                gmm.fit(self.cell_type_data[cell_type])
                self.gmms[cell_type] = gmm
                
            points = self.gmms[cell_type].sample(n_points_per_type[cell_type])[0]
            all_points.extend(points)
            all_labels.extend([cell_type] * len(points))
        
        return np.array(all_points), np.array(all_labels)

    def simulate_pure_random(self, n_points_per_type):
        """Simulate points by random sampling from real data distribution"""
        all_points = []
        all_labels = []
        
        for cell_type in self.unique_cell_types:
            n_points = n_points_per_type[cell_type]
            cell_data = self.cell_type_data[cell_type]
            
            indices = np.random.choice(len(cell_data), size=n_points)
            sampled_points = cell_data[indices].copy()
            noise = np.random.normal(0, 0.1, size=sampled_points.shape)
            
            all_points.extend(sampled_points + noise)
            all_labels.extend([cell_type] * n_points)
        
        return np.array(all_points), np.array(all_labels)

    def simulate(self, n_points_per_type, mode='mixed', component_ratio=0.7, n_components=3, 
                overlap_density=0.5, n_iterations=1000, temperature=0.1, 
                **energy_params):
        """
        Simulate cell distribution with specified parameters
        
        Parameters:
        n_points_per_type: dict
            Dictionary mapping cell types to number of points to generate
        mode: str
            'mixed': Mixture of components and co-localization
            'gmm': Pure GMM-based simulation
            'random': Pure random sampling with noise
        component_ratio: float
            For mode='mixed': proportion of points from GMM components
        n_components: int or 'auto'
            Number of GMM components
        overlap_density: float
            For mode='mixed': density of co-localized regions
        n_iterations: int
            Number of optimization iterations
        temperature: float
            Temperature parameter for optimization
        **energy_params: dict
            Parameters for interaction energy calculation
        """
        if mode == 'gmm':
            return self.simulate_pure_gmm(n_points_per_type)
        elif mode == 'random':
            return self.simulate_pure_random(n_points_per_type)
        elif mode == 'mixed':
            # Generate initial distribution
            current_points, current_labels = self.generate_mixed_distribution(
                n_points_per_type, component_ratio, n_components, overlap_density
            )
            
            # Optimize using interaction energy with provided parameters
            current_energy = self.calculate_interaction_energy(
                current_points, **energy_params
            ).sum()
            
            n_total_points = len(current_points)
            
            for i in range(n_iterations):
                point_idx = np.random.randint(n_total_points)
                new_points = current_points.copy()
                new_points[point_idx] += np.random.normal(0, 0.1, 2)
                
                new_energy = self.calculate_interaction_energy(
                    new_points, **energy_params
                ).sum()
                
                delta_energy = new_energy - current_energy
                if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / temperature):
                    current_points = new_points
                    current_energy = new_energy
            
            return current_points, current_labels
        else:
            raise ValueError("Mode must be 'mixed', 'gmm', or 'random'")

def plot_cell_distribution(points, labels, title, ax=None):
    """Helper function to plot cell distributions"""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    
    # Create scatter plot with different colors for each cell type
    colors = {'tumor': 'red', 'immune': 'blue'}
    for cell_type in np.unique(labels):
        mask = labels == cell_type
        ax.scatter(points[mask, 0], points[mask, 1], 
                c=colors[cell_type], alpha=0.6, label=cell_type)
    
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    return ax