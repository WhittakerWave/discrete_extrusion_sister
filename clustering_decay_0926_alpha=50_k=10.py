

import pickle
import numpy as np
from math import sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt

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

# 
def Ns(t, rho, alpha, rho_c=0.0):
    if t <= 0:
        return 0.0
    a = alpha
    b = 1.0 + alpha * rho_c
    if a == 0.0:
        return rho * t
    disc = b*b + 4.0 * a * rho * t
    return (-b + sqrt(disc)) / (2.0 * a)

# Single-side growth
def Ns(t, rho, alpha, rho_c=0.0):
    if t <= 0:
        return 0.0
    a = alpha
    b = 1.0 + alpha * rho_c
    if a == 0.0:
        return rho * t
    disc = b*b + 4.0*a*rho*t
    return (-b + sqrt(disc)) / (2.0*a)

# Collision-aware total N(t)
def N_total_collision(t_array, M, L, k, rho, alpha, rho_c=None):
    if rho_c is None:
        rho_c = rho
    out = np.zeros_like(t_array, dtype=float)
    max_per_extruder = L / k
    for i, t in enumerate(t_array):
        N_per = Ns(t, rho, alpha, rho_c)
        N_per = min(N_per, max_per_extruder)   # limit due to collisions
        out[i] = min(k * N_per, M)             # total cannot exceed M
    return out



def expected_N_ex_per_extruder(t, rho, alpha, rho_c, rho_e, v=1.0):
    lam = 2.0 * rho_e * v
    if lam == 0.0:
        return rho_c + 2.0 * Ns(t, rho, alpha, rho_c)
    term1 = np.exp(-lam * t) * Ns(t, rho, alpha, rho_c)
    integrand = lambda tau: Ns(tau, rho, alpha, rho_c) * lam * np.exp(-lam * tau)
    term2, _ = quad(integrand, 0.0, t, epsabs=1e-8, epsrel=1e-6, limit=200)
    return rho_c + 2.0 * (term1 + term2)

def N_total_array(t_array, M, L, k, rho, alpha, rho_c=None, v=1.0):
    if rho_c is None:
        rho_c = rho
    rho_e = float(k) / float(L)
    out = np.zeros_like(t_array, dtype=float)
    for i, t in enumerate(t_array):
        N_ex = expected_N_ex_per_extruder(t, rho, alpha, rho_c, rho_e, v=v)
        out[i] = k * N_ex
    out = np.minimum(out, M)
    return out

def plot_unique_positions_combined_v2(filenames, labels=None, t=None, M=None, L=None, k=None, rho=None, alpha=None, v=1.0):
    """Plot simulation trajectories + analytic extruder curves for given k and alpha."""
    plt.figure(figsize=(8,6))

    # ---- Simulation data ----
    for i, filename in enumerate(filenames):
        label = labels[i] if labels else f"Dataset {i+1}"
        unique_positions, _ = extract_consecutive_positions(filename)
        if unique_positions is None:
            continue
        time_steps = np.arange(len(unique_positions)) * 100  # match your timestep unit
        plt.plot(time_steps, unique_positions, label=label, linewidth=2)

    # ---- Analytic curve ----
    if all(v is not None for v in [t, M, L, k, rho, alpha]):
        N_analytic = M - N_total_collision(t, M, L, k, rho, alpha, rho_c=None)
        plt.plot(t, N_analytic, 'k--', linewidth=2.5, label=f"Analytic k={k}, alpha={alpha}")

    # ---- Formatting ----
    plt.title("Simulation vs Mean Field Behavior\n[$M$=500, $N$=32000, $k=50$, $\\alpha=10$]", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=18)
    plt.ylabel("Unique Sister Positions", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()


filenames = [
    "test_simple_formula_0926/test_case_simple_alpha=10_k=50_run1.pkl",
    "test_simple_formula_0926/test_case_simple_alpha=10_k=50_run2.pkl",
    "test_simple_formula_0926/test_case_simple_alpha=10_k=50_run3.pkl",
]

labels = [
    "Sim 1",
    "Sim 2",
    "Sim 3",
]

k_list = 50
alpha_list = 10

# Example usage
t = np.linspace(0, 65000, 200)
M = 500
L = 32000
rho = M/L

plot_unique_positions_combined_v2(filenames, labels, t, M, L, k_list, rho, alpha=alpha_list, v=1.0)



