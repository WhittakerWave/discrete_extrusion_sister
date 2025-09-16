

import pickle
import numpy as np
import matplotlib.pyplot as plt

def load_translocator_data(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)  # dict with keys "sister", "lef", "ctcf"
    return data

def plot_sisters_with_lefs(sister_array, lef_array, sample_interval=1, cmap='viridis'):
    sister_array = np.array(sister_array)   # shape: (time, n_sisters)
    lef_array    = np.array(lef_array)      # shape: (time, 2)

    time_axis = np.arange(sister_array.shape[0]) * sample_interval

    plt.figure(figsize=(8, 6))

    # kymograph of sisters
    plt.imshow(
        sister_array.T, aspect='auto', origin='lower',
        extent=[time_axis[0], time_axis[-1], 0, sister_array.shape[1]],
        cmap=cmap, interpolation='nearest', alpha=0.8
    )
    # plt.colorbar(label="Lattice Position (sisters)")

    # overlay LEF legs
    plt.plot(time_axis, sister_array[:,0], color='#B0B0B0', marker='^', label='Sister 1')
    plt.plot(time_axis, sister_array[:,1], color='#B0B0B0', marker='^', label='Sister 2')
    plt.plot(time_axis, lef_array[:, 0, 0], color='#56B4E9', linewidth=2, label='LEF leg 1')
    plt.plot(time_axis, lef_array[:, 0, 1], color='#CC79A7', linewidth=2, label='LEF leg 2')

    plt.xlabel("Time Steps", fontsize=25)
    plt.ylabel("Lattice Position", fontsize=25)
    plt.title("Kymograph for LEFs and sisters \n  alpha=1 \n CTCF 10s", fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.tight_layout()
    plt.show()

# --- Usage ---
data = load_translocator_data('translocator_test_cases_with_fall_off/test_case_V.pkl')
plot_sisters_with_lefs(data["sister"], data["lef"], sample_interval=1)


