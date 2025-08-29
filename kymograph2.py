

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_sister_trajectory(filename='sister_trajectory.pkl'):
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
    return sister_trajectory

def find_cluster_regions(sister_trajectory, min_cluster_size=5, final_time_only=True):
    sister_array = np.array(sister_trajectory)
    if final_time_only:
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
                              show_all_sisters=True, alpha=0.6, ctcf_positions=None):
    sister_array = np.array(sister_trajectory)
    time_axis = np.arange(sister_array.shape[0]) * sample_interval
    sisters_to_plot = cluster['sisters']
    
    plt.figure(figsize=(16, 10))
    colors = plt.cm.hsv(np.linspace(0, 1, len(sisters_to_plot)))
    
    for i, sister_id in enumerate(sisters_to_plot):
        trajectory = sister_array[:, sister_id]
        plt.plot(time_axis, trajectory, color=colors[i], linewidth=1.2, alpha=alpha)

    plt.xlabel('Time Steps')
    plt.ylabel('Lattice Position')
    plt.title(f'Sister Trajectories - Cluster at Final Position {cluster["position"]}')
    plt.grid(True, alpha=0.3)

    # Add CTCF horizontal lines if within visible range
    if ctcf_positions is not None:
        y_min, y_max = plt.ylim()
        for pos in ctcf_positions:
            if y_min <= pos <= y_max:
                plt.axhline(y=pos, color='purple', linestyle='--', alpha=0.5)
                plt.text(time_axis[-1]*0.98, pos, 'CTCF', color='purple',
                         fontsize=8, va='center', ha='right', alpha=0.7)

    plt.tight_layout()
    plt.show()
    return plt.gcf()

def plot_cluster_convergence(sister_trajectory, cluster, sample_interval=50, ctcf_positions=None):
    sister_array = np.array(sister_trajectory)
    time_axis = np.arange(len(sister_array)) * sample_interval
    sisters_in_cluster = cluster['sisters']
    final_position = cluster['position']
    
    plt.figure(figsize=(14, 10))

    # Top: Individual trajectories
    plt.subplot(2, 1, 1)
    colors = plt.cm.viridis(np.linspace(0, 1, len(sisters_in_cluster)))
    for i, sister_id in enumerate(sisters_in_cluster):
        trajectory = sister_array[:, sister_id]
        plt.plot(time_axis, trajectory, color=colors[i], linewidth=1.5, alpha=0.8)
    plt.axhline(y=final_position, color='red', linestyle='--', linewidth=3, alpha=0.8, label='Final cluster pos')
    plt.ylabel('Lattice Position')
    plt.title('Convergence to Cluster')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Add CTCF lines if within visible range
    if ctcf_positions is not None:
        y_min, y_max = plt.ylim()
        for pos in ctcf_positions:
            if y_min <= pos <= y_max:
                plt.axhline(y=pos, color='purple', linestyle='--', alpha=0.5)
                plt.text(time_axis[-1]*0.98, pos, 'CTCF', color='purple',
                         fontsize=8, va='center', ha='right', alpha=0.7)

    # Bottom: Distance over time
    plt.subplot(2, 1, 2)
    mean_distances, std_distances = [], []
    for t in range(len(sister_array)):
        positions = sister_array[t, sisters_in_cluster]
        distances = np.abs(positions - final_position)
        mean_distances.append(np.mean(distances))
        std_distances.append(np.std(distances))

    mean_distances = np.array(mean_distances)
    std_distances = np.array(std_distances)

    plt.plot(time_axis, mean_distances, 'b-', linewidth=2, label='Mean distance')
    plt.fill_between(time_axis, mean_distances - std_distances, mean_distances + std_distances,
                     alpha=0.3, color='blue', label='±1 std')
    plt.xlabel('Time Steps')
    plt.ylabel('Distance from Final Position')
    plt.title('Convergence Dynamics')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
    return plt.gcf()

def analyze_cluster_transitions(filename='sister_trajectory.pkl', sample_interval=50,
                                ctcf_left_file='ctcf_left_positions.npy'):
    sister_trajectory = load_sister_trajectory(filename)
    sister_array = np.array(sister_trajectory)
    
    print(f"Loaded trajectory: {len(sister_array)} time points, {sister_array.shape[1]} sisters")
    print(f"Time range: 0 to {(len(sister_array)-1) * sample_interval} steps")

    # Load CTCF positions
    ctcf_left_positions = np.load(ctcf_left_file)
    ctcf_positions = np.unique(ctcf_left_positions)  # mirror if needed

    # Find clusters
    clusters = find_cluster_regions(sister_trajectory, min_cluster_size=5)
    if len(clusters) == 0:
        print("No significant clusters found!")
        return None

    # Plot all cluster trajectories (optional)
    print("\nPlotting detailed view of largest cluster...")
    largest_cluster = clusters[0]
    fig1 = plot_cluster_trajectories(sister_trajectory, largest_cluster,
                                     sample_interval, ctcf_positions=ctcf_positions)

    print("\nAnalyzing convergence dynamics...")
    fig2 = plot_cluster_convergence(sister_trajectory, largest_cluster,
                                    sample_interval, ctcf_positions=ctcf_positions)

    return fig1, fig2, clusters


analyze_cluster_transitions(
        filename='CTCF_dynamic/sister_trajectory_500_65000steps_dW_CTCF_dynamic.pkl',
        sample_interval=50,
        ctcf_left_file='CTCF_dynamic/ctcf_left_positions.npy'
    )

