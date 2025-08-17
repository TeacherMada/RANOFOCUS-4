from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import numpy as np
import json
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.interpolate import griddata, RBFInterpolator
from scipy.ndimage import gaussian_filter
import io
import base64

geophysics_bp = Blueprint('geophysics', __name__)

class GeophysicalProcessor:
    """Advanced geophysical data processing class."""
    
    def __init__(self):
        self.data = None
        self.grid = None
        self.metadata = {}
    
    def load_data_from_array(self, data_array):
        """Load XYZ data from numpy array."""
        try:
            self.data = {
                'x': data_array[:, 0],
                'y': data_array[:, 1], 
                'z': data_array[:, 2]
            }
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def advanced_idw(self, nx=200, ny=200, power=2.5, k_neighbors=25, adaptive=True):
        """Advanced Inverse Distance Weighting with adaptive parameters."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        x, y, z = self.data['x'], self.data['y'], self.data['z']
        
        # Create regular grid
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Prepare data points
        points = np.column_stack((x, y))
        grid_points = np.column_stack((Xi.ravel(), Yi.ravel()))
        
        # Calculate distances
        distances = cdist(grid_points, points)
        
        # Find k nearest neighbors for each grid point
        Zi = np.zeros(len(grid_points))
        
        for i, (gx, gy) in enumerate(grid_points):
            # Get k nearest neighbors
            dist_to_point = distances[i]
            nearest_indices = np.argsort(dist_to_point)[:k_neighbors]
            nearest_distances = dist_to_point[nearest_indices]
            nearest_values = z[nearest_indices]
            
            # Adaptive power based on local density
            if adaptive:
                local_density = len(nearest_indices) / (np.mean(nearest_distances) + 1e-10)
                adaptive_power = power * (1 + 0.1 * np.log(local_density + 1))
            else:
                adaptive_power = power
            
            # Calculate weights (avoid division by zero)
            weights = 1.0 / (nearest_distances + 1e-10) ** adaptive_power
            
            # Weighted interpolation
            Zi[i] = np.sum(weights * nearest_values) / np.sum(weights)
        
        # Reshape to grid
        self.grid = {
            'x': xi.tolist(),
            'y': yi.tolist(),
            'z': Zi.reshape(ny, nx).tolist(),
            'method': 'advanced_idw'
        }
        
        return self.grid
    
    def geological_inversion(self, n_layers=5, regularization=0.01):
        """Simplified geological inversion using least squares with regularization."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        x, y, z = self.data['x'], self.data['y'], self.data['z']
        
        # Create depth-dependent model
        depths = np.unique(y)
        depths = depths[depths < 0]  # Only negative depths
        
        if len(depths) < n_layers:
            n_layers = len(depths)
        
        # Define layer boundaries
        layer_boundaries = np.linspace(depths.min(), depths.max(), n_layers + 1)
        
        # Initialize model parameters (resistivity for each layer)
        initial_resistivity = np.ones(n_layers) * np.median(z)
        
        def forward_model(resistivity_model):
            """Forward modeling function."""
            modeled_z = np.zeros_like(z)
            
            for i, (xi, yi) in enumerate(zip(x, y)):
                # Find which layer this point belongs to
                layer_idx = np.digitize(yi, layer_boundaries) - 1
                layer_idx = np.clip(layer_idx, 0, n_layers - 1)
                
                # Add depth-dependent variation
                depth_factor = np.exp(-abs(yi) / 50.0)  # Exponential decay with depth
                modeled_z[i] = resistivity_model[layer_idx] * depth_factor
            
            return modeled_z
        
        def objective_function(resistivity_model):
            """Objective function for inversion."""
            predicted = forward_model(resistivity_model)
            data_misfit = np.sum((z - predicted) ** 2)
            
            # Regularization term (smoothness)
            if len(resistivity_model) > 1:
                smoothness = np.sum(np.diff(resistivity_model) ** 2)
                regularization_term = regularization * smoothness
            else:
                regularization_term = 0
            
            return data_misfit + regularization_term
        
        # Perform inversion
        result = minimize(objective_function, initial_resistivity, 
                         method='L-BFGS-B', 
                         bounds=[(0.001, 100)] * n_layers)
        
        if result.success:
            inverted_resistivity = result.x
            
            return {
                'resistivity_model': inverted_resistivity.tolist(),
                'layer_boundaries': layer_boundaries.tolist(),
                'method': 'geological_inversion',
                'success': True,
                'misfit': result.fun
            }
        
        return {'success': False, 'error': 'Inversion failed'}
    
    def detect_anomalies(self, threshold_factor=2.0):
        """Detect geological anomalies based on statistical analysis."""
        if self.data is None:
            raise ValueError("No data loaded")
        
        z = self.data['z']
        
        # Calculate statistics
        mean_z = np.mean(z)
        std_z = np.std(z)
        
        # Define anomaly thresholds
        high_threshold = mean_z + threshold_factor * std_z
        low_threshold = mean_z - threshold_factor * std_z
        
        # Find anomalies
        high_anomalies = z > high_threshold
        low_anomalies = z < low_threshold
        
        anomalies = {
            'high_resistivity': {
                'indices': np.where(high_anomalies)[0].tolist(),
                'values': z[high_anomalies].tolist(),
                'threshold': float(high_threshold),
                'interpretation': 'Possible solid rock or dry formations'
            },
            'low_resistivity': {
                'indices': np.where(low_anomalies)[0].tolist(),
                'values': z[low_anomalies].tolist(),
                'threshold': float(low_threshold),
                'interpretation': 'Possible water-bearing formations or clay'
            },
            'statistics': {
                'mean': float(mean_z),
                'std': float(std_z),
                'n_high_anomalies': int(np.sum(high_anomalies)),
                'n_low_anomalies': int(np.sum(low_anomalies))
            }
        }
        
        return anomalies

@geophysics_bp.route('/process', methods=['POST'])
@cross_origin()
def process_data():
    """Process geophysical data with advanced interpolation."""
    try:
        data = request.get_json()
        
        # Extract parameters
        raw_data = np.array(data['data'])
        nx = data.get('nx', 150)
        ny = data.get('ny', 150)
        power = data.get('power', 2.2)
        k_neighbors = data.get('k_neighbors', 20)
        adaptive = data.get('adaptive', True)
        
        # Initialize processor
        processor = GeophysicalProcessor()
        processor.load_data_from_array(raw_data)
        
        # Perform interpolation
        grid = processor.advanced_idw(nx=nx, ny=ny, power=power, 
                                     k_neighbors=k_neighbors, adaptive=adaptive)
        
        return jsonify({
            'success': True,
            'grid': grid,
            'stats': {
                'n_points': len(raw_data),
                'x_range': [float(raw_data[:, 0].min()), float(raw_data[:, 0].max())],
                'y_range': [float(raw_data[:, 1].min()), float(raw_data[:, 1].max())],
                'z_range': [float(raw_data[:, 2].min()), float(raw_data[:, 2].max())]
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@geophysics_bp.route('/inversion', methods=['POST'])
@cross_origin()
def geological_inversion():
    """Perform geological inversion."""
    try:
        data = request.get_json()
        
        # Extract parameters
        raw_data = np.array(data['data'])
        n_layers = data.get('n_layers', 4)
        regularization = data.get('regularization', 0.01)
        
        # Initialize processor
        processor = GeophysicalProcessor()
        processor.load_data_from_array(raw_data)
        
        # Perform inversion
        result = processor.geological_inversion(n_layers=n_layers, 
                                              regularization=regularization)
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@geophysics_bp.route('/anomalies', methods=['POST'])
@cross_origin()
def detect_anomalies():
    """Detect geological anomalies."""
    try:
        data = request.get_json()
        
        # Extract parameters
        raw_data = np.array(data['data'])
        threshold_factor = data.get('threshold_factor', 2.0)
        
        # Initialize processor
        processor = GeophysicalProcessor()
        processor.load_data_from_array(raw_data)
        
        # Detect anomalies
        anomalies = processor.detect_anomalies(threshold_factor=threshold_factor)
        
        return jsonify({
            'success': True,
            'anomalies': anomalies
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@geophysics_bp.route('/sample-data', methods=['GET'])
@cross_origin()
def get_sample_data():
    """Generate sample geophysical data for testing."""
    try:
        # Generate synthetic test data
        np.random.seed(42)
        
        # Generate synthetic resistivity data
        x_stations = np.arange(1, 11)
        y_depths = np.arange(-3, -101, -3)
        
        data_points = []
        for x in x_stations:
            for y in y_depths:
                # Create realistic resistivity profile
                base_resistivity = 0.004
                
                # Add depth variation
                if y > -20:
                    z = base_resistivity + np.random.normal(0, 0.01)
                elif y > -60:
                    z = base_resistivity + 0.02 + np.random.normal(0, 0.02)
                else:
                    z = base_resistivity + 0.1 + np.random.normal(0, 0.05)
                
                # Add some anomalies
                if x == 4 and y == -93:
                    z = 8.225  # High resistivity anomaly
                elif x >= 8 and y < -80:
                    z += 0.2  # Resistive zone
                
                data_points.append([float(x), float(y), max(0.001, float(z))])  # Ensure positive values and convert to float
        
        return jsonify({
            'success': True,
            'data': data_points,
            'description': 'Synthetic geophysical data for testing'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

