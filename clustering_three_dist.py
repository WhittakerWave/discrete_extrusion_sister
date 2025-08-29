

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_sister_trajectory(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_cluster_size_distribution(sister_trajectory):
    arr = np.array(sister_trajectory)
    final_positions = arr[-1, :]
    counts = np.bincount(final_positions)
    # Exclude zeros if position 0 isn't meaningful; adjust as needed
    cluster_sizes = counts[counts > 0]
    return cluster_sizes

def plot_cluster_size_distribution(filenames, labels, bins='auto'):
    plt.figure(figsize=(10, 6))
    
    for fname, label in zip(filenames, labels):
        traj = load_sister_trajectory(fname)
        cluster_sizes = compute_cluster_size_distribution(traj)
        num_clusters = len(cluster_sizes)
        plt.hist(cluster_sizes, bins=bins, alpha=0.6, 
                 label=f'{label} (n={num_clusters})')
    
    plt.xlabel('Cluster Size [Number of sisters in one lattice]', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.title('Cluster Size Distribution at 65000 steps, dynamical CTCF with tau 100s', fontsize=20)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

filenames = [
    'sister_trajectory_500_65000steps_dW_CTCF_dynamic.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_dynamic.pkl',
    'sister_trajectory_500_65000steps_WT_CTCF_dynamic.pkl'
]
labels = ['dW99%', 'dN75%', 'WT']

plot_cluster_size_distribution(filenames, labels)

