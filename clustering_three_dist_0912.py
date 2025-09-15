

import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def load_sister_trajectory(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def compute_cluster_size_distribution(sister_trajectory):
    arr = np.array(sister_trajectory['sister'])
    final_positions = arr[-1, :]
    counts = np.bincount(final_positions)
    # Exclude zeros if position 0 isn't meaningful; adjust as needed
    cluster_sizes = counts[counts > 0]
    return cluster_sizes

def plot_cluster_size_distribution(filenames, labels, bins='auto'):
    plt.figure(figsize=(8, 6))
    
    for fname, label in zip(filenames, labels):
        traj = load_sister_trajectory(fname)
        cluster_sizes = compute_cluster_size_distribution(traj)
        num_clusters = len(cluster_sizes)
        plt.hist(cluster_sizes, bins=bins, alpha=0.6, 
                 label=f'{label} (n={num_clusters})')
    
    plt.xlabel('Cluster Size [Number of sisters in one lattice]', fontsize=20)
    plt.ylabel('Number of Clusters', fontsize=20)
    plt.title('Cluster Size Distribution at 65000 steps \n CTCF Static', fontsize=20)
    plt.legend(fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

filenames = [
    'CTCF_dynamics_0912/dW_CTCF_Static_sister1.pkl',
    'CTCF_dynamics_0912/dN_CTCF_Static_sister1.pkl',
    'CTCF_dynamics_0912/WT_CTCF_Static_sister1.pkl'
]
labels = ['dW99%', 'dN75%', 'WT']

plot_cluster_size_distribution(filenames, labels)

