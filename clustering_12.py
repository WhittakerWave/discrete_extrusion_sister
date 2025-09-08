

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

def plot_unique_positions_combined(filenames, labels=None):
    """Plot unique positions for static and dynamic trajectory files"""
    import matplotlib.pyplot as plt
    import numpy as np
    import pickle

    def extract_unique_positions(filename):
        with open(filename, 'rb') as f:
            sister_trajectory = pickle.load(f)

        if len(sister_trajectory) == 0:
            print(f"{filename} is empty.")
            return None, None

        sister_array = np.array(sister_trajectory)
        unique_positions = [len(np.unique(sister_array[t, :])) for t in range(sister_array.shape[0])]
        return np.array(unique_positions), sister_array

    plt.figure(figsize=(10, 8))

    base_labels = ['WT', 'dN75%', 'dW']
    color_map = {
        'WT': '#1f77b4',      # blue
        'dN75%': '#ff7f0e',   # orange
        'dW': '#2ca02c'       # green
    }
    marker_map = {
        'WT': '^',      # triangle up
        'dN75%': 's',   # square
        'dW': 'v'       # triangle down
    }

    for i, filename in enumerate(filenames):
        label = labels[i] if labels else f"Dataset {i+1}"
        unique_positions, _ = extract_unique_positions(filename)
        if unique_positions is None:
            continue

        time_steps = np.arange(len(unique_positions)) * 100

        # Determine color
        for base in base_labels:
            if base in label:
                color = color_map[base]
                marker = marker_map[base]
                break
        else:
            color = 'black'
            marker = 'o'

        is_dynamic = 'dynamic' in filename or 'dynamic' in label
        is_ctcf = 'CTCF' in label
        is_1000s = '1000s' in label

        if is_dynamic:
            # Different marker styles for 500 vs 1000s dynamic
            if is_1000s:
                # Filled markers for 1000s dynamic
                plt.scatter(time_steps, unique_positions, label=label, color=color,
                           marker=marker, s=10, alpha=0.9, edgecolors='black', linewidths=1)
            else:
                # Hollow markers for 500 dynamic
                plt.scatter(time_steps, unique_positions, label=label, color='white',
                           marker=marker, s=10, alpha=0.8, edgecolors=color, linewidths=2)
        else:
            linestyle = '--' if is_ctcf else '-'
            plt.plot(time_steps, unique_positions, label=label, color=color,
                     linestyle=linestyle, linewidth=2)

    # Final plot formatting
    plt.title("Number of Unique Sisters Over Time", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=20)
    plt.ylabel("Unique Positions", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=16)
    plt.legend(fontsize=12, loc='best')
    plt.tight_layout()
    plt.show()

# Updated file list with the three additional files
filenames = [
    # Original files with CTCF
    'sister_trajectory_500_65000steps_WT_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_WT_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_WT_CTCF_fast_dynamic_1000s.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast_dynamic_1000s.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast_dynamic_1000s.pkl',
    # New files without CTCF
    'sister_trajectory_500_65000steps_WT_fast.pkl',
    'sister_trajectory_500_65000steps_dN_fast.pkl',
    'sister_trajectory_500_65000steps_dW_fast.pkl',
]

# Updated labels with the three additional files
labels = [
    'WT+CTCF [static]', 'dN75%+CTCF [static]', 'dW+CTCF [static]',
    'WT+CTCF [dynamic 100s]', 'dN75%+CTCF [dynamic 100s]', 'dW+CTCF [dynamic 100s]',
    'WT+CTCF [dynamic 1000s]', 'dN75%+CTCF [dynamic 1000s]', 'dW+CTCF [dynamic 1000s]',
    'WT', 'dN75%', 'dW'
]

plot_unique_positions_combined(filenames, labels)