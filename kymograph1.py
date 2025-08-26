


import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib.colors as mcolors

def load_sister_trajectory(filename='sister_trajectory.pkl'):
    """Load sister trajectory from pickle file"""
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
    return sister_trajectory

def find_cluster_regions(sister_trajectory, min_cluster_size=5, final_time_only=True):
    """Find regions where sisters cluster together"""
    
    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    if final_time_only:
        # Look at final time point to find clusters
        final_positions = sister_array[-1, :]
        pos_counts = Counter(final_positions)
        
        clusters = []
        for pos, count in pos_counts.items():
            if count >= min_cluster_size:
                clusters.append({
                    'position': pos,
                    'size': count,
                    'sisters': np.where(final_positions == pos)[0].tolist()
                })
        
        clusters.sort(key=lambda x: x['size'], reverse=True)
        
        print(f"Found {len(clusters)} clusters with ≥{min_cluster_size} sisters:")
        for i, cluster in enumerate(clusters):
            print(f"  Cluster {i+1}: {cluster['size']} sisters at position {cluster['position']}")
        
        return clusters

def plot_cluster_trajectories(sister_trajectory, cluster, sample_interval=50, 
                             show_all_sisters=True, alpha=0.6):
    """Plot individual sister trajectories as lines for a specific cluster"""
    
    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    # Create time axis
    time_axis = np.arange(num_steps) * sample_interval
    
    # Get sisters in this cluster
    sisters_in_cluster = cluster['sisters']
    
    # Show all sisters in the cluster
    if show_all_sisters:
        sisters_to_plot = sisters_in_cluster
        print(f"Showing all {len(sisters_in_cluster)} sisters in cluster")
    else:
        # Legacy behavior for backward compatibility
        max_lines = 30
        if len(sisters_in_cluster) > max_lines:
            sisters_to_plot = sisters_in_cluster[:max_lines]
            print(f"Showing first {max_lines} of {len(sisters_in_cluster)} sisters in cluster")
        else:
            sisters_to_plot = sisters_in_cluster
    
    # Create plot
    plt.figure(figsize=(16, 10))  # Larger figure for more sisters
    
    # Generate colors - use multiple colormaps for better distinction
    n_sisters = len(sisters_to_plot)
    if n_sisters <= 20:
        colors = plt.cm.tab20(np.linspace(0, 1, n_sisters))
    elif n_sisters <= 50:
        # Use two colormaps
        colors1 = plt.cm.tab20(np.linspace(0, 1, 20))
        colors2 = plt.cm.Set3(np.linspace(0, 1, n_sisters-20))
        colors = np.vstack([colors1, colors2])
    else:
        # For very large clusters, use continuous colormap
        colors = plt.cm.hsv(np.linspace(0, 1, n_sisters))
    
    # Plot each sister's trajectory
    for i, sister_id in enumerate(sisters_to_plot):
        trajectory = sister_array[:, sister_id]
        plt.plot(time_axis, trajectory, color=colors[i], 
                linewidth=1.2, alpha=alpha, label=f'Sister {sister_id}')
    
    plt.xlabel('Time Steps', fontsize=12)
    plt.ylabel('Lattice Position', fontsize=12)
    plt.title(f'ALL Sister Trajectories - Cluster at Final Position {cluster["position"]}\n'
              f'Showing all {cluster["size"]} sisters in cluster', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Add time markers
    max_time = time_axis[-1]
    time_markers = [0, max_time//4, max_time//2, 3*max_time//4, max_time]
    for tm in time_markers:
        plt.axvline(x=tm, color='gray', linestyle='--', alpha=0.3)
    
    # Only show legend if reasonable number of sisters (otherwise it's too crowded)
    if len(sisters_to_plot) <= 15:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    elif len(sisters_to_plot) <= 30:
        # Show legend with smaller font and multiple columns
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=6, ncol=2)
    else:
        # For large clusters, add text annotation instead of legend
        plt.text(0.02, 0.98, f'Cluster contains {len(sisters_to_plot)} sisters\n'
                              f'(Legend omitted for clarity)', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def plot_all_cluster_trajectories(sister_trajectory, sample_interval=50, 
                                 min_cluster_size=5, show_all_sisters=True):
    """Plot trajectory lines for all significant clusters, showing ALL sisters"""
    
    # Find clusters
    clusters = find_cluster_regions(sister_trajectory, min_cluster_size=min_cluster_size)
    
    if len(clusters) == 0:
        print("No significant clusters found!")
        return None
    
    sister_array = np.array(sister_trajectory)
    time_axis = np.arange(len(sister_array)) * sample_interval
    
    # Create subplots for each cluster
    n_clusters = len(clusters)
    fig, axes = plt.subplots(n_clusters, 1, figsize=(16, 6*n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    for i, cluster in enumerate(clusters):
        sisters_in_cluster = cluster['sisters']
        
        # Show ALL sisters in each cluster
        if show_all_sisters:
            sisters_to_plot = sisters_in_cluster
            print(f"Cluster {i+1}: Plotting all {len(sisters_in_cluster)} sisters")
        else:
            # Legacy mode - limit lines
            max_lines_per_cluster = 20
            sisters_to_plot = sisters_in_cluster[:max_lines_per_cluster]
        
        # Generate colors for this cluster
        n_sisters = len(sisters_to_plot)
        if n_sisters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, n_sisters))
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_sisters))
        
        # Plot trajectories
        for j, sister_id in enumerate(sisters_to_plot):
            trajectory = sister_array[:, sister_id]
            axes[i].plot(time_axis, trajectory, color=colors[j], 
                        linewidth=1.0, alpha=0.7)
        
        axes[i].set_xlabel('Time Steps', fontsize=11)
        axes[i].set_ylabel('Lattice Position', fontsize=11)
        axes[i].set_title(f'Cluster {i+1}: ALL {cluster["size"]} sisters → final position {cluster["position"]}', 
                         fontsize=12)
        axes[i].grid(True, alpha=0.3)
        
        # Add final position line
        axes[i].axhline(y=cluster['position'], color='red', linestyle=':', 
                       alpha=0.8, linewidth=2, label=f'Final position: {cluster["position"]}')
        axes[i].legend()
        
        # Add annotation with cluster info
        axes[i].text(0.02, 0.95, f'{len(sisters_to_plot)} sister trajectories', 
                    transform=axes[i].transAxes, fontsize=9,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    plt.tight_layout()
    plt.show()
    
    return fig, clusters

def plot_cluster_convergence(sister_trajectory, cluster, sample_interval=50):
    """Plot how sisters converge to a cluster over time"""
    
    sister_array = np.array(sister_trajectory)
    time_axis = np.arange(len(sister_array)) * sample_interval
    
    sisters_in_cluster = cluster['sisters']
    final_position = cluster['position']
    
    plt.figure(figsize=(14, 10))
    
    # Top subplot: Individual trajectories
    plt.subplot(2, 1, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sisters_in_cluster)))
    
    for i, sister_id in enumerate(sisters_in_cluster):
        trajectory = sister_array[:, sister_id]
        plt.plot(time_axis, trajectory, color=colors[i], linewidth=1.5, alpha=0.8)
    
    # Highlight final cluster position
    plt.axhline(y=final_position, color='red', linestyle='--', linewidth=3, 
               alpha=0.8, label=f'Final cluster position: {final_position}')
    
    plt.ylabel('Lattice Position')
    plt.title(f'Convergence to Cluster at Position {final_position}\n'
              f'{len(sisters_in_cluster)} sisters')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Bottom subplot: Distance from final position over time
    plt.subplot(2, 1, 2)
    
    mean_distances = []
    std_distances = []
    
    for t in range(len(sister_array)):
        positions = sister_array[t, sisters_in_cluster]
        distances = np.abs(positions - final_position)
        mean_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))
    
    mean_distances = np.array(mean_distances)
    std_distances = np.array(std_distances)
    
    plt.plot(time_axis, mean_distances, 'b-', linewidth=2, label='Mean distance')
    plt.fill_between(time_axis, 
                    mean_distances - std_distances,
                    mean_distances + std_distances,
                    alpha=0.3, color='blue', label='±1 std')
    
    plt.xlabel('Time Steps')
    plt.ylabel('Distance from Final Position')
    plt.title('Convergence Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return plt.gcf()

def analyze_cluster_transitions(filename='sister_trajectory.pkl', sample_interval=50):
    """Complete analysis of cluster formation with trajectory tracking"""
    
    # Load data
    sister_trajectory = load_sister_trajectory(filename)
    sister_array = np.array(sister_trajectory)
    
    print(f"Loaded trajectory: {len(sister_array)} time points, {sister_array.shape[1]} sisters")
    print(f"Time range: 0 to {(len(sister_array)-1) * sample_interval} steps")
    
    # Find clusters
    clusters = find_cluster_regions(sister_trajectory, min_cluster_size=5)
    
    if len(clusters) == 0:
        print("No significant clusters found!")
        return None
    
    # Plot trajectories for all clusters
    print("\nPlotting all cluster trajectories...")
    fig1, clusters = plot_all_cluster_trajectories(sister_trajectory, sample_interval)
    
    # Plot detailed view of largest cluster
    print(f"\nPlotting detailed view of largest cluster...")
    largest_cluster = clusters[0]
    fig2 = plot_cluster_trajectories(sister_trajectory, largest_cluster, sample_interval)
    
    # Plot convergence analysis for largest cluster
    print(f"\nAnalyzing convergence dynamics...")
    fig3 = plot_cluster_convergence(sister_trajectory, largest_cluster, sample_interval)
    
    return fig1, fig2, fig3, clusters

def plot_single_cluster_detailed(filename='sister_trajectory.pkl', cluster_index=0, 
                                sample_interval=50):
    """Plot detailed trajectory analysis for a single cluster"""
    
    sister_trajectory = load_sister_trajectory(filename)
    clusters = find_cluster_regions(sister_trajectory, min_cluster_size=5)
    
    if cluster_index >= len(clusters):
        print(f"Only {len(clusters)} clusters found, cannot plot cluster {cluster_index}")
        return None
    
    cluster = clusters[cluster_index]
    
    # Plot individual trajectories
    fig1 = plot_cluster_trajectories(sister_trajectory, cluster, sample_interval)
    
    # Plot convergence analysis
    fig2 = plot_cluster_convergence(sister_trajectory, cluster, sample_interval)
    
    return fig1, fig2, cluster

# Main execution
if __name__ == "__main__":
    # Run complete cluster transition analysis
    fig1, fig2, fig3, clusters = analyze_cluster_transitions('sister_trajectory.pkl', sample_interval=50)

    