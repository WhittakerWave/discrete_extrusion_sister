
import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_translocator(filename='translocator_trajectory.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)   # {"sister": ..., "lef": ..., "ctcf": ...}
    return data


def plot_sisters_and_lefs_with_ctcf(data, sample_interval=1, ctcf_left_positions=None, ctcf_right_positions=None, 
                                    max_timepoints=None, max_lefs_per_time=None, sister_indices=None, 
                                    plot_lefs=False, plot_ctcf=False):
    sister_array = np.array(data["sister"])  # shape (time, n_sisters)
    lef_list = data["lef"]                   # ragged: list over time of lists of [left, right]
    time_axis = np.arange(len(data['lef'])) * sample_interval

    if max_timepoints is None:
        max_timepoints = len(time_axis)

    plt.figure(figsize=(14, 8))

    # --- Sisters: continuous trajectories ---
    # If sister_indices is specified, only plot those sisters
    if sister_indices is not None:
        # Ensure indices are valid
        sister_indices = [i for i in sister_indices if 0 <= i < sister_array.shape[1]]
        indices_to_plot = sister_indices
    else:
        # Plot all sisters if no specific indices given
        indices_to_plot = range(sister_array.shape[1])
    
    for i in indices_to_plot:
        plt.plot(time_axis[:max_timepoints], sister_array[:max_timepoints, i],
            lw=1, alpha=0.5, color="steelblue", label=f"Sister {i}" if len(indices_to_plot) <= 10 else None)

    # --- LEFs: dashed vertical lines at each time ---
    if plot_lefs:
        for t, t_val in enumerate(time_axis[:max_timepoints]):
            lefs_at_t = lef_list[t]  # list of [left, right] pairs at this time
            if max_lefs_per_time is not None:
                lefs_at_t = lefs_at_t[:max_lefs_per_time]
            for left_leg, right_leg in lefs_at_t:
                if left_leg >= 0 and right_leg >= 0:
                    plt.plot([t_val, t_val], [left_leg, right_leg],
                             linestyle="--", color="darkorange", lw=0.9, alpha=0.9)

    # --- CTCF: horizontal dashed lines ---
    if ctcf_left_positions is not None:
        y_min, y_max = plt.ylim()
        for pos in np.unique(ctcf_left_positions):
            if y_min <= pos <= y_max:
                plt.axhline(y=pos, color="purple", linestyle="--", alpha=0.5)
                plt.text(time_axis[-1] * 0.98, pos, "CTCF Left",
                         color="purple", fontsize=8, va="center", ha="right")
    
    if ctcf_right_positions is not None:
        y_min, y_max = plt.ylim()
        for pos in np.unique(ctcf_right_positions):
            if y_min <= pos <= y_max:
                plt.axhline(y=pos, color="blue", linestyle="--", alpha=0.5)
                plt.text(time_axis[-1] * 0.98, pos, "CTCF Right",
                         color="blue", fontsize=8, va="center", ha="right")

    plt.xlabel("Time Steps")
    plt.ylabel("Lattice Position")
    
    # Update title to reflect which sisters are plotted
    if sister_indices is not None:
        if len(sister_indices) == 1:
            plt.title(f"Sister {sister_indices[0]} + LEFs (dashed links) + CTCF sites")
        else:
            plt.title(f"Sisters {sister_indices} + LEFs (dashed links) + CTCF sites")
    else:
        plt.title("Sisters + LEFs (dashed links) + CTCF sites")
    
    # Add legend if plotting few enough sisters
    if sister_indices is not None and len(sister_indices) <= 10:
        plt.legend()
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage

dN_static_sister1 = load_translocator("CTCF_dynamics_0912/dN_CTCF_Static_sister1.pkl")['sister'][-1]
dN_static_sister2 = load_translocator("CTCF_dynamics_0912/dN_CTCF_Static_sister2.pkl")['sister'][-1]
static_diff = 2.5*(dN_static_sister2 - dN_static_sister1)
max_val = np.max(static_diff)
max_idx = np.argmax(static_diff)

data1 = load_translocator("CTCF_dynamics_0912/dN_CTCF_Static_sister1.pkl")
data2 = load_translocator("CTCF_dynamics_0912/dN_CTCF_Static_sister2.pkl")
ctcf_left_positions = np.load("ctcf_left_positions.npy")
ctcf_right_positions = np.load("ctcf_right_positions.npy")

# Example 1: Plot only sister with maximum difference
plot_sisters_and_lefs_with_ctcf(data1, sample_interval=100, 
                                ctcf_left_positions=ctcf_left_positions,
                                ctcf_right_positions=ctcf_right_positions, 
                                max_timepoints=650, max_lefs_per_time=200,
                                sister_indices=[max_idx],plot_lefs=False, plot_ctcf=False)  

plot_sisters_and_lefs_with_ctcf(data2, sample_interval=100, 
                                ctcf_left_positions=ctcf_left_positions,
                                ctcf_right_positions=ctcf_right_positions, 
                                max_timepoints=650, max_lefs_per_time=200,
                                sister_indices=[max_idx],plot_lefs=False, plot_ctcf=False)  





