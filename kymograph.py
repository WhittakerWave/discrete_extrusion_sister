

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

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
    
    else:
        # Find clusters across all time points (more complex)
        # This could be implemented if needed
        pass

def plot_focused_kymograph(sister_trajectory, cluster_region=None, sister_subset=None, 
                          sample_interval=50, max_sisters_display=50):
    """Create focused kymograph with proper time scaling and sister subset"""
    
    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    # Create proper time axis (multiply by sampling interval)
    time_axis = np.arange(num_steps) * sample_interval
    max_time = time_axis[-1]
    
    print(f"Original data: {num_steps} time points, {num_sisters} sisters")
    print(f"Time range: 0 to {max_time} steps")
    
    # Determine which sisters to display
    if sister_subset is not None:
        # Use provided subset
        sisters_to_show = sister_subset[:max_sisters_display]
        data_to_plot = sister_array[:, sisters_to_show]
        title_suffix = f"(Sisters {min(sisters_to_show)}-{max(sisters_to_show)})"
    elif cluster_region is not None:
        # Use sisters from specified cluster
        sisters_to_show = cluster_region['sisters'][:max_sisters_display]
        data_to_plot = sister_array[:, sisters_to_show]
        title_suffix = f"(Cluster at pos {cluster_region['position']}, {len(sisters_to_show)} sisters)"
    else:
        # Use first N sisters
        sisters_to_show = list(range(min(max_sisters_display, num_sisters)))
        data_to_plot = sister_array[:, sisters_to_show]
        title_suffix = f"(First {len(sisters_to_show)} sisters)"
    
    # Create the focused kymograph
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[3, 1])
    
    # Main kymograph
    im = ax1.imshow(data_to_plot.T, aspect='auto', cmap='viridis', 
                    origin='lower', interpolation='nearest', 
                    extent=[0, max_time, 0, len(sisters_to_show)])
    
    ax1.set_title(f'Sister Position Kymograph {title_suffix}')
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Sister Index')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, label='Lattice Position')
    
    # Add time markers
    time_markers = [0, max_time//4, max_time//2, 3*max_time//4, max_time]
    for tm in time_markers:
        ax1.axvline(x=tm, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
    
    # Clustering evolution plot below
    unique_positions = []
    cluster_sizes = []
    
    for t in range(num_steps):
        if sister_subset is not None:
            positions = sister_array[t, sisters_to_show]
        else:
            positions = sister_array[t, :]
        
        unique_pos = len(np.unique(positions))
        unique_positions.append(unique_pos)
        
        pos_counts = Counter(positions)
        max_cluster = max(pos_counts.values()) if pos_counts else 1
        cluster_sizes.append(max_cluster)
    
    ax2.plot(time_axis, unique_positions, 'b-', linewidth=2, label='Unique positions')
    ax2.plot(time_axis, cluster_sizes, 'r-', linewidth=2, label='Largest cluster size')
    ax2.axhline(y=len(sisters_to_show), color='k', linestyle='--', alpha=0.5, label='Total sisters shown')
    
    ax2.set_title('Clustering Evolution (for displayed sisters)')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return fig, sisters_to_show

def multi_cluster_kymograph(sister_trajectory, sample_interval=50, clusters_to_show=3):
    """Show kymographs for multiple cluster regions side by side"""
    
    # Find clusters
    clusters = find_cluster_regions(sister_trajectory, min_cluster_size=5)
    
    if len(clusters) == 0:
        print("No significant clusters found!")
        return None
    
    # Limit to requested number of clusters
    clusters_to_plot = clusters[:min(clusters_to_show, len(clusters))]
    
    sister_array = np.array(sister_trajectory)
    time_axis = np.arange(len(sister_array)) * sample_interval
    
    fig, axes = plt.subplots(len(clusters_to_plot), 1, figsize=(14, 4*len(clusters_to_plot)))
    if len(clusters_to_plot) == 1:
        axes = [axes]
    
    for i, cluster in enumerate(clusters_to_plot):
        sisters_in_cluster = cluster['sisters']
        data_to_plot = sister_array[:, sisters_in_cluster]
        
        im = axes[i].imshow(data_to_plot.T, aspect='auto', cmap='viridis',
                           origin='lower', interpolation='nearest',
                           extent=[0, time_axis[-1], 0, len(sisters_in_cluster)])
        
        axes[i].set_title(f'Cluster {i+1}: {cluster["size"]} sisters at position {cluster["position"]}')
        axes[i].set_xlabel('Time Steps')
        axes[i].set_ylabel('Sister Index')
        
        plt.colorbar(im, ax=axes[i], label='Lattice Position')
    
    plt.tight_layout()
    plt.show()
    
    return fig, clusters_to_plot

def analyze_clustering_dynamics(filename='sister_trajectory.pkl', sample_interval=50):
    """Complete clustering analysis with proper time scaling"""
    
    # Load data
    sister_trajectory = load_sister_trajectory(filename)
    sister_array = np.array(sister_trajectory)
    
    print(f"Loaded trajectory: {len(sister_array)} time points, {sister_array.shape[1]} sisters")
    print(f"Time range: 0 to {(len(sister_array)-1) * sample_interval} steps")
    
    # Find clusters
    clusters = find_cluster_regions(sister_trajectory)
    
    # Plot focused kymograph for largest cluster
    if clusters:
        largest_cluster = clusters[0]
        fig1, displayed_sisters = plot_focused_kymograph(sister_trajectory, 
                                                        cluster_region=largest_cluster,
                                                        sample_interval=sample_interval)
        
        # Plot multiple clusters
        fig2 = multi_cluster_kymograph(sister_trajectory, sample_interval=sample_interval)
        
        return fig1, fig2, clusters
    else:
        print("No clusters found, showing subset of sisters")
        fig1, displayed_sisters = plot_focused_kymograph(sister_trajectory,
                                                        sample_interval=sample_interval,
                                                        max_sisters_display=50)
        return fig1, None, []

def quick_focused_plot(filename='sister_trajectory.pkl', num_sisters=50, sample_interval=50):
    """Quick plot showing just first N sisters with proper time scaling"""
    
    sister_trajectory = load_sister_trajectory(filename)
    sister_array = np.array(sister_trajectory)
    
    # Take subset
    data_subset = sister_array[:, :num_sisters]
    time_axis = np.arange(len(sister_array)) * sample_interval
    
    plt.figure(figsize=(14, 8))
    im = plt.imshow(data_subset.T, aspect='auto', cmap='viridis',
                   origin='lower', interpolation='nearest',
                   extent=[0, time_axis[-1], 0, num_sisters])
    
    plt.colorbar(im, label='Lattice Position')
    plt.title(f'Sister Position Kymograph (First {num_sisters} sisters)')
    plt.xlabel('Time Steps')
    plt.ylabel('Sister ID')
    
    # Add time markers every 10,000 steps
    for t in range(0, int(time_axis[-1])+1, 10000):
        plt.axvline(x=t, color='white', linestyle='--', alpha=0.5, linewidth=0.8)
    
    plt.show()

# Main execution
if __name__ == "__main__":
    # Run focused analysis
    fig1, fig2, clusters = analyze_clustering_dynamics('sister_trajectory.pkl', sample_interval=50)

    

    