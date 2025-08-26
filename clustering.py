


import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy import stats

def load_and_analyze_clustering(filename):
    """Load sister trajectory and analyze clustering patterns"""
    # Load the data
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
    
    if len(sister_trajectory) == 0:
        print("Empty trajectory")
        return None
    
    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    print(f"Analyzing trajectory: {num_steps} time points, {num_sisters} sisters")
    
    results = {
        'unique_positions': [],
        'clustering_coefficients': [],
        'ripley_k': [],
        'variance_in_positions': []
    }
    
    for t in range(num_steps):
        positions = sister_array[t, :]
        
        # 1. Count unique positions (basic clustering measure)
        unique_pos = len(np.unique(positions))
        results['unique_positions'].append(unique_pos)
        
        # 2. Calculate clustering coefficient (positions per sister)
        clustering_coeff = unique_pos / num_sisters
        results['clustering_coefficients'].append(clustering_coeff)
        
        # 3. Ripley's K statistic (for spatial clustering)
        k_stat = calculate_ripleys_k(positions)
        results['ripley_k'].append(k_stat)
        
        # 4. Variance in positions (spread measure)
        pos_variance = np.var(positions)
        results['variance_in_positions'].append(pos_variance)
    
    return results, sister_array

def calculate_ripleys_k(positions, r=1):
    """Calculate Ripley's K statistic for 1D clustering analysis"""
    n = len(positions)
    if n <= 1:
        return 0
    
    # For 1D case, count neighbors within distance r
    distances = np.abs(positions[:, np.newaxis] - positions)
    neighbors_within_r = np.sum(distances <= r, axis=1) - 1  # Exclude self
    
    # Ripley's K estimate
    k_estimate = np.mean(neighbors_within_r)
    
    # Expected K for random distribution in 1D is approximately 2*r
    expected_k = 2 * r
    
    # Return L(r) = sqrt(K(r)/pi) - r, which is easier to interpret
    # For 1D, we use K(r) - expected directly
    return k_estimate - expected_k

def calculate_nearest_neighbor_distances(positions):
    """Calculate mean nearest neighbor distance"""
    if len(positions) <= 1:
        return np.inf
    
    distances = pdist(positions.reshape(-1, 1))
    if len(distances) == 0:
        return np.inf
    
    distance_matrix = squareform(distances)
    np.fill_diagonal(distance_matrix, np.inf)  # Exclude self-distances
    
    nearest_distances = np.min(distance_matrix, axis=1)
    return np.mean(nearest_distances)

def plot_clustering_analysis(results, sister_array):
    """Plot various clustering metrics over time"""
    
    num_steps = len(results['unique_positions'])
    time_points = np.arange(num_steps)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Unique positions over time
    axes[0,0].plot([t * 50 for t in time_points], results['unique_positions'], 'b-', linewidth=2)
    axes[0,0].set_title('Number of unique sisters')
    axes[0,0].set_xlabel('Time Step [Extrusion Timestep Units]')
    axes[0,0].set_ylabel('Number of Unique Positions')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Clustering coefficient
    axes[0,1].plot(time_points, results['clustering_coefficients'], 'r-', linewidth=2)
    axes[0,1].axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='No clustering')
    axes[0,1].set_title('Clustering Coefficient')
    axes[0,1].set_xlabel('Time Step')
    axes[0,1].set_ylabel('Unique Positions / Total Sisters')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Ripley's K statistic
    axes[0,2].plot(time_points, results['ripley_k'], 'g-', linewidth=2)
    axes[0,2].axhline(y=0, color='k', linestyle='--', alpha=0.5, label='Random')
    axes[0,2].set_title("Ripley's K Statistic")
    axes[0,2].set_xlabel('Time Step')
    axes[0,2].set_ylabel('K - Expected K')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 5. Position variance
    axes[1,1].plot(time_points, results['variance_in_positions'], 'c-', linewidth=2)
    axes[1,1].set_title('Position Variance (Spread)')
    axes[1,1].set_xlabel('Time Step')
    axes[1,1].set_ylabel('Variance')
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Sister positions heatmap
    im = axes[1,2].imshow(sister_array.T, aspect='auto', cmap='viridis', origin='lower')
    axes[1,2].set_title('Sister Positions Over Time')
    axes[1,2].set_xlabel('Time Step')
    axes[1,2].set_ylabel('Sister ID')
    plt.colorbar(im, ax=axes[1,2], label='Position')
    
    plt.tight_layout()
    plt.show()
    
    return fig

def summarize_clustering_stats(results):
    """Print summary statistics of clustering analysis"""
    
    unique_pos = np.array(results['unique_positions'])
    clustering_coeff = np.array(results['clustering_coefficients'])
    ripley_k = np.array(results['ripley_k'])
    
    print("=== CLUSTERING ANALYSIS SUMMARY ===")
    print(f"  Mean: {np.mean(unique_pos):.2f}")
    print(f"  Std:  {np.std(unique_pos):.2f}")
    print(f"  Range: {np.min(unique_pos)} - {np.max(unique_pos)}")
    
    print(f"\nClustering Coefficient (1.0 = no clustering):")
    print(f"  Mean: {np.mean(clustering_coeff):.3f}")
    print(f"  Std:  {np.std(clustering_coeff):.3f}")
    
    print(f"\nRipley's K (negative = clustering, positive = dispersion):")
    print(f"  Mean: {np.mean(ripley_k):.3f}")
    print(f"  Std:  {np.std(ripley_k):.3f}")
    
    # Interpretation
    mean_cluster_coeff = np.mean(clustering_coeff)
    if mean_cluster_coeff < 0.8:
        print(f"\n>>> STRONG CLUSTERING detected (coeff = {mean_cluster_coeff:.3f})")
    elif mean_cluster_coeff < 0.95:
        print(f"\n>>> MODERATE CLUSTERING detected (coeff = {mean_cluster_coeff:.3f})")
    else:
        print(f"\n>>> WEAK/NO CLUSTERING detected (coeff = {mean_cluster_coeff:.3f})")

# Usage example
def run_clustering_analysis(filename):
    """Run the complete clustering analysis"""
    
    # Analyze the saved trajectory
    results, sister_array = load_and_analyze_clustering(filename)
    
    if results is None:
        return
    
    # Print summary statistics
    summarize_clustering_stats(results)
    
    # Create plots
    fig = plot_clustering_analysis(results, sister_array)
    
    return results, sister_array, fig

# Alternative: Simple unique position analysis
def quick_clustering_check(filename):
    """Quick check of clustering by counting unique positions"""
    
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
    
    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    print(f"Quick Clustering Analysis:")
    print(f"Total sisters: {num_sisters}")
    
    for t in range(min(10, num_steps)):  # Show first 10 time points
        positions = sister_array[t, :]
        unique_positions = len(np.unique(positions))
        clustering_ratio = unique_positions / num_sisters
        
        print(f"Time {t}: {unique_positions}/{num_sisters} unique positions (ratio: {clustering_ratio:.3f})")
        
        # Show actual positions
        unique_vals, counts = np.unique(positions, return_counts=True)
        clusters = [(pos, count) for pos, count in zip(unique_vals, counts) if count > 1]
        if clusters:
            print(f"  Clusters: {clusters}")
    
    return sister_array


results, sister_array, fig = run_clustering_analysis(filename = 'sister_trajectory_500_65000steps_dW.pkl')