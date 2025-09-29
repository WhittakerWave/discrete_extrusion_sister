
import pickle
import numpy as np
from math import sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt


def long_time_clusters(M, N, k):
    """
    Long-time mean-field expected number of unique collapsed points (clusters).
    
    Parameters
    ----------
    M : int
        Number of initial points
    N : int
        Lattice length (drops out of long-time limit, included for clarity)
    k : int
        Number of extruders
    
    Returns
    -------
    float
        Expected number of clusters as t → ∞
    """
    if k <= 1:  # one extruder = one gap
        return 2.0 - np.exp(-M)
    
    f = lambda x: (1 - np.exp(-M * x)) * (k - 1) * (1 - x) ** (k - 2)
    val, _ = quad(f, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9) 
    return k * val 


def extract_unique_positions(filename):
    """Extract unique positions over time from a trajectory file"""
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
        
    if len(sister_trajectory) == 0:
        print(f"{filename} is empty.")
        return None, None

    sister_array = np.array(sister_trajectory['sister'])
    num_steps, num_sisters = sister_array.shape
    
    unique_positions = []
    for t in range(num_steps):
        positions = sister_array[t, :]
        unique_pos = len(np.unique(positions))
        unique_positions.append(unique_pos)
    
    return np.array(unique_positions), sister_array

def extract_consecutive_positions(filename):
    """Extract consecutive position clusters over time from a trajectory file"""
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
        
    if len(sister_trajectory) == 0:
        print(f"{filename} is empty.")
        return None, None

    sister_array = np.array(sister_trajectory['sister'])
    num_steps, num_sisters = sister_array.shape
    
    consecutive_positions = []
    for t in range(num_steps):
        positions = sister_array[t, :]
        # Sort positions to identify consecutive ranges
        sorted_positions = np.sort(positions)
        
        # Merge consecutive positions
        merged_count = 1  # Start with first position
        for i in range(1, len(sorted_positions)):
            # If current position is not consecutive to previous, count as new cluster
            if sorted_positions[i] - sorted_positions[i-1] > 1:
                merged_count += 1
        
        consecutive_positions.append(merged_count)
    
    return np.array(consecutive_positions), sister_array


# Example: filenames for multiple runs
filenames = [
    "test_simple_formula_0926/test_case_simple_alpha=10_k=100_run1.pkl",
    "test_simple_formula_0926/test_case_simple_alpha=10_k=100_run2.pkl",
    "test_simple_formula_0926/test_case_simple_alpha=10_k=100_run3.pkl",
]

labels = [
    "Sim 1",
    "Sim 2",
    "Sim 3",
]

# Compute long-time cluster limit
M = 500
N = 32000
k = 100
A = long_time_clusters(M=M, N=N, k=k)
print("Long-time expected clusters:", A)

def plot_simulation_with_long_time(filenames, labels=None):
    """Plot simulation trajectories + long-time mean-field limit"""
    plt.figure(figsize=(8,6))

    # ---- Simulation data ----
    for i, filename in enumerate(filenames):
        label = labels[i] if labels else f"Dataset {i+1}"
        consecutive_positions, _ = extract_consecutive_positions(filename)
        if consecutive_positions is None:
            continue
        time_steps = np.arange(len(consecutive_positions)) * 100
        plt.plot(time_steps, consecutive_positions, label=label, linewidth=2)

    # ---- Long-time limit ----
    plt.axhline(A, color='k', linestyle='--', linewidth=2.5, label=f"Theoretical Long-time limit ~ {A:.1f}")

    # ---- Formatting ----
    plt.title("Simulation vs Mean Field Limit\n[$M$=500, $N$=32000, $k=100$, $\\alpha=10$]", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=24)
    plt.ylabel("Unique Sister Positions", fontsize=24)
    plt.ylim(0,500)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    plt.show()

# Plot
plot_simulation_with_long_time(filenames, labels)


