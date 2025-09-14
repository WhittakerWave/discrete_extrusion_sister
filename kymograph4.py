

import pickle
import numpy as np
import matplotlib.pyplot as plt


def load_translocator(filename='translocator_trajectory.pkl'):
    with open(filename, 'rb') as f:
        data = pickle.load(f)   # {"sister": ..., "lef": ..., "ctcf": ...}
    return data


def plot_sisters_and_lefs_with_ctcf(data, sample_interval=1, ctcf_left_positions=None, ctcf_right_positions=None, 
                                    max_timepoints=None, max_lefs_per_time=None):
    sister_array = np.array(data["sister"])  # shape (time, n_sisters)
    lef_list = data["lef"]                   # ragged: list over time of lists of [left, right]
    time_axis = np.arange(len(data['lef'])) * sample_interval

    if max_timepoints is None:
        max_timepoints = len(time_axis)

    plt.figure(figsize=(14, 8))

    # --- Sisters: continuous trajectories ---
    for i in range(sister_array.shape[1]):
        plt.plot(time_axis[:max_timepoints], sister_array[:max_timepoints, i],
            lw=1, alpha=0.5, color="steelblue")

    # --- LEFs: dashed vertical lines at each time ---
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
                         color="purple", fontsize=8, va="center", ha="right")

    plt.xlabel("Time Steps")
    plt.ylabel("Lattice Position")
    plt.title("Sisters + LEFs (dashed links) + CTCF sites")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Example usage
data = load_translocator("CTCF_dynamics_0912/dN_CTCF_Static_sister1.pkl")
ctcf_left_positions = np.load("ctcf_left_positions.npy")
ctcf_right_positions = np.load("ctcf_right_positions.npy")
plot_sisters_and_lefs_with_ctcf(data, sample_interval=100, ctcf_left_positions = ctcf_left_positions,
                                ctcf_right_positions = ctcf_right_positions, 
                                max_timepoints=650, max_lefs_per_time=200)





