

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
    "test_0930_all_mutants/alpha100_tau20h/dWdS_trajectory1.pkl",
    "test_0930_all_mutants/alpha100_tau20h/dWdS_trajectory2.pkl",
    "test_0930_all_mutants/alpha100_tau20h/dW_trajectory1.pkl",
    "test_0930_all_mutants/alpha100_tau20h/dW_trajectory2.pkl",
]

labels = [
    "dWdS sister1",
    "dWdS sister2",
    "dW sister1",
    "dW sister2",
]

# Compute long-time cluster limit
M = 500
N = 32000
k = 500
A = long_time_clusters(M=M, N=N, k=k)
print("Long-time expected clusters:", A)


def plot_simulation_with_long_time(filenames, labels=None, colors=None):
    """Plot simulation trajectories + long-time mean-field limit"""
    plt.figure(figsize=(8,6))

    # ---- Simulation data ----
    for i, filename in enumerate(filenames):
        label = labels[i] if labels else f"Dataset {i+1}"
        consecutive_positions, _ = extract_consecutive_positions(filename)
        if consecutive_positions is None:
            continue
        time_steps = np.arange(len(consecutive_positions)) * 100/3600
        color = colors[i] if colors else None  # use provided color or default
        plt.plot(time_steps, consecutive_positions, 
                 label=label, linewidth=2, color=color)

    # ---- Long-time limit ----
    # plt.axhline(A, color='k', linestyle='--', linewidth=2.5,
    #             label=f"Theoretical Long-time limit ~ {A:.1f}")

    # ---- Formatting ----
    plt.title("dW vs dWdS\n[$M$=776, $N$=32000, a=500, using CRN LE]", fontsize=20)
    plt.xlabel("Time Step [h]", fontsize=24)
    plt.ylabel("Unique Sister Positions", fontsize=24)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=16, loc='best')
    plt.tight_layout()
    plt.show()

def plot_simulation_with_decay(filenames, labels, colors, N0, tau, T, alpha):
    """Plot simulation trajectories + ODE decay on the same figure"""
    plt.figure(figsize=(8,6))

    # ---- Simulation data ----
    for i, filename in enumerate(filenames):
        label = labels[i] if labels else f"Dataset {i+1}"
        consecutive_positions, _ = extract_consecutive_positions(filename)
        if consecutive_positions is None:
            continue
        time_steps = np.arange(len(consecutive_positions)) * 100/3600
        color = colors[i] if colors else None
        plt.plot(time_steps, consecutive_positions, 
                 label=label, linewidth=2, color=color)

    # ---- ODE Decay Curve ----
    t = np.linspace(0, T, 500)
    N = N0 * np.exp(-t / tau)
    plt.plot(t/3600, N, '--', lw=2.5, color = "#6BADD7",
             label=fr"Decay-only: $N(t)={N0} e^{{-t/\tau}}, \tau={tau/3600:.0f}h$")

    # ---- Formatting ----
    plt.title(f"dW vs dWdS + ODE\n[$M$=776, $N$=32000, a={alpha}, using CRN LE]", fontsize=20)
    plt.xlabel("Time [h]", fontsize=24)
    plt.ylabel("Unique Sister Positions", fontsize=24)
    plt.ylim(0, 800)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend(fontsize=14, loc='best')
    plt.tight_layout()
    plt.show()


# plot_decay_ode(N0=7765, tau=10*3600, T=20*3600)
# Example usage
residence_time = 20
colors = ["#e66d50", "#f3a361", "#8ab07c", "#299d8f"]  # blue, orange, green, red
plot_simulation_with_decay(filenames, labels=labels, colors=colors, 
        N0=776, tau=residence_time*3600, T=18*3600, alpha=100)


