

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

def plot_fast_vs_slow_comparison(fast_files, slow_files, labels=None):
    """Plot comparison between fast and slow implementations"""
    plt.figure(figsize=(14, 10))
    
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

    # Plot fast implementation results
    for i, filename in enumerate(fast_files):
        label = labels[i] if labels else f"Fast Dataset {i+1}"
        unique_positions, _ = extract_unique_positions(filename)
        if unique_positions is None:
            continue

        time_steps = np.arange(len(unique_positions)) * 100

        # Determine color and marker based on condition
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

        if is_dynamic:
            # Plot with markers only for dynamic
            plt.scatter(time_steps, unique_positions, 
                       label=f"{label} (Fast)", 
                       color=color, marker=marker, s=100, alpha=0.8)
        else:
            # Solid line for fast static
            plt.plot(time_steps, unique_positions, 
                    label=f"{label} (Fast)", 
                    color=color, linestyle='-', linewidth=2)

    # Plot slow implementation results
    for i, filename in enumerate(slow_files):
        label = labels[i] if labels else f"Slow Dataset {i+1}"
        unique_positions, _ = extract_unique_positions(filename)
        if unique_positions is None:
            continue

        time_steps = np.arange(len(unique_positions)) * 50  # Slow uses *50 timestep

        # Determine color and marker based on condition
        for base in base_labels:
            if base in label:
                color = color_map[base]
                marker = marker_map[base]
                break
        else:
            color = 'black'
            marker = 'o'

        is_dynamic = 'dynamic' in filename or 'dynamic' in label

        if is_dynamic:
            # Plot with hollow markers for slow dynamic
            plt.scatter(time_steps, unique_positions, 
                       label=f"{label} (Slow)", 
                       color='white', edgecolors=color, 
                       marker=marker, s=10, linewidth=2, alpha=0.8)
        else:
            # Dashed line for slow static
            plt.plot(time_steps, unique_positions, 
                    label=f"{label} (Slow)", 
                    color=color, linestyle='--', linewidth=2, alpha=0.7)

    # Final plot formatting
    plt.title("Fast vs Slow Implementation: Sister Trajectories Comparison", fontsize=20)
    plt.xlabel("Time Step [Extrusion Timestep Units]", fontsize=20)
    plt.ylabel("Number of Unique Sister Positions", fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', labelsize=16)
    
    # Organize legend
    handles, legend_labels = plt.gca().get_legend_handles_labels()
    plt.legend(handles, legend_labels, fontsize=12, ncol=2, 
               bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()

def calculate_similarity_metrics(fast_files, slow_files, labels):
    """Calculate quantitative similarity metrics between fast and slow implementations"""
    print("=== SIMILARITY ANALYSIS ===\n")
    
    for i, (fast_file, slow_file) in enumerate(zip(fast_files, slow_files)):
        label = labels[i] if labels else f"Condition {i+1}"
        
        fast_unique, _ = extract_unique_positions(fast_file)
        slow_unique, _ = extract_unique_positions(slow_file)
        
        if fast_unique is None or slow_unique is None:
            print(f"{label}: Missing data")
            continue
        
        # Ensure same length for comparison
        min_len = min(len(fast_unique), len(slow_unique))
        fast_unique = fast_unique[:min_len]
        slow_unique = slow_unique[:min_len]
        
        # Calculate metrics
        correlation = np.corrcoef(fast_unique, slow_unique)[0, 1]
        mean_abs_diff = np.mean(np.abs(fast_unique - slow_unique))
        max_abs_diff = np.max(np.abs(fast_unique - slow_unique))
        relative_diff = mean_abs_diff / np.mean(slow_unique) * 100
        
        print(f"{label}:")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  Mean Absolute Difference: {mean_abs_diff:.2f}")
        print(f"  Max Absolute Difference: {max_abs_diff:.2f}")
        print(f"  Relative Difference: {relative_diff:.2f}%")
        print(f"  Fast Final: {fast_unique[-1]}, Slow Final: {slow_unique[-1]}")
        print()

# Define file lists
fast_files = [
    'sister_trajectory_500_65000steps_WT_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast.pkl',
    'sister_trajectory_500_65000steps_WT_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dN_CTCF_fast_dynamic.pkl',
    'sister_trajectory_500_65000steps_dW_CTCF_fast_dynamic.pkl',
]

slow_files = [
    'CTCF/sister_trajectory_500_65000steps_WT_CTCF.pkl',
    'CTCF/sister_trajectory_500_65000steps_dN_CTCF.pkl',
    'CTCF/sister_trajectory_500_65000steps_dW_CTCF.pkl',
    'CTCF_dynamic/sister_trajectory_500_65000steps_WT_CTCF_dynamic.pkl',
    'CTCF_dynamic/sister_trajectory_500_65000steps_dN_CTCF_dynamic.pkl',
    'CTCF_dynamic/sister_trajectory_500_65000steps_dW_CTCF_dynamic.pkl',
]

labels = ['WT+CTCF', 'dN75%+CTCF', 'dW+CTCF', 
         'WT+CTCF (dynamic)', 'dN75%+CTCF (dynamic)', 'dW+CTCF (dynamic)']

# Generate comparison plot
plot_fast_vs_slow_comparison(fast_files, slow_files, labels)

# Calculate quantitative similarity
calculate_similarity_metrics(fast_files, slow_files, labels)