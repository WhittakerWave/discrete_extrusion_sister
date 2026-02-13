

import pickle
import numpy as np
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



def plot_unique_positions_combined(filenames, labels=None, t=None, rho=None, alpha=None, M=None):
    """Plot unique positions for trajectories + analytic extruder curve."""
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    plt.figure(figsize=(8, 6))


    # ---- Simulation data ---- 
    for i, filename in enumerate(filenames):
        # Determine label
        label = labels[i] if labels else f"Dataset {i+1}"
        # Extract unique positions
        unique_positions, _ = extract_consecutive_positions(filename)
        if unique_positions is None:
            continue
        # Compute time steps
        time_steps = np.arange(len(unique_positions)) * 100
        # Plot
        plt.plot(time_steps, unique_positions, label=label, linewidth=2)
    # ---- Simulation data ----
    if t is not None and rho is not None and alpha is not None and M is not None:
        t = np.array(t)
        # Early-time: t roughly 0 → 5000
        t_early = t[t <= 500]
        N_early = rho * t_early
        plt.plot(t_early, M - N_early, 'b--', linewidth=2.5, label="Early-time Scaling: $M - M/N t$")

        # Late-time scaling law
        t_late = t[t >= 50000]
        N_late = np.sqrt(2 * rho * t_late / alpha)
        plt.plot(
            t_late,
            M - N_late,
            'r--',
            linewidth=2.5,
            label=r"Late-time Scaling: $M - \sqrt{2Mt/(N\alpha)}$")
    
      # ---- Analytic curve ----
    if t is not None and rho is not None and alpha is not None and M is not None:
        N = (-1 + np.sqrt(1 + 2 * alpha * rho * np.array(t))) / alpha
        remaining = M - N
        plt.plot(t, remaining, 'k--', linewidth=2.5, label="Analytic M - N(t)")

    # ---- Formatting ----
    plt.title("Simulation vs Mean Field Behavior\n[$M$=500, $N$=32000, $\\alpha=0.01$]", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=18)
    plt.ylabel("Unique Sister Positions", fontsize=18)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=14)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()


filenames = [
    "test_case_mean_field_alpha.pkl",
]

labels = [
    "Sim 1",
]
# Example usage
t = np.linspace(0, 10000, 100)
rho = 500/32000
alpha = 0.01
M = 500

plot_unique_positions_combined(filenames, labels=labels, t=t, rho=rho, alpha=alpha, M=M)



