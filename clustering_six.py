

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

    sister_array = np.array(sister_trajectory)
    num_steps, num_sisters = sister_array.shape
    
    unique_positions = []
    for t in range(num_steps):
        positions = sister_array[t, :]
        unique_pos = len(np.unique(positions))
        unique_positions.append(unique_pos)
    
    return np.array(unique_positions), sister_array

def plot_unique_positions_across_files(filenames, labels=None):
    """Plot unique positions for multiple trajectory files"""
    plt.figure(figsize=(8, 6))

    # Define consistent colors for WT, dN75%, dW
    base_labels = ['WT', 'dN75%', 'dW']
    color_map = {
        'WT': '#1f77b4',      # blue
        'dN75%': '#ff7f0e',   # orange
        'dW': '#2ca02c'       # green
    }

    for i, filename in enumerate(filenames):
        unique_positions, _ = extract_unique_positions(filename)
        if unique_positions is None:
            continue

        time_steps = np.arange(len(unique_positions)) * 50  # assuming timestep units
        label = labels[i] if labels else f"Dataset {i+1}"

        # Determine base label (without CTCF) for color mapping
        for base in base_labels:
            if base in label:
                color = color_map[base]
                break
        else:
            color = 'black'  # fallback color if no match

        linestyle = '--' if 'CTCF' in label else '-'

        plt.plot(time_steps, unique_positions, linewidth=2, label=label, linestyle=linestyle, color=color)
    
    plt.title("Number of Unique Sisters Over Time", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=20)
    plt.ylabel("Unique Positions", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

# Example usage
filenames = [
    'sister_trajectory_500_65000steps_WT_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast_dynamic.pkl'

]

labels = ['WT+CTCF', 'dN75%+CTCF', 'dW+CTCF', 'WT', 'dN75%', 'dW']  # optional, for legend clarity
plot_unique_positions_across_files(filenames, labels)


