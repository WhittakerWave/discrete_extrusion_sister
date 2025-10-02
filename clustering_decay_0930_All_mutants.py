

import pickle
import numpy as np
from math import sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt
from scipy.stats import norm

def extract_consecutive_positions_last(filename):
    """Extract consecutive position clusters over time from the *last frame* of a trajectory file"""
    with open(filename, 'rb') as f:
        sister_trajectory = pickle.load(f)
        
    if len(sister_trajectory) == 0:
        print(f"{filename} is empty.")
        return None, None

    # Only take the last snapshot
    sister_array = np.array(sister_trajectory['sister'][-1])

    # ---- Remove -1 entries ----
    valid_positions = sister_array[sister_array != -1]
    # ---- Sort ----
    sorted_positions = np.sort(valid_positions)
    return sorted_positions


def plot_sister_difference_histograms(filenames, labels, bins=50):
    """
    Plot histograms of (sister2 - sister1) for each mutant.
    
    Parameters
    ----------
    filenames : list of str
        List of trajectory filenames [WT1, WT2, dN1, dN2]
    labels : list of str
        List of labels corresponding to each filename
    bins : int
        Number of bins for the histogram
    """
    # (sister1, sister2)
    mutants = {
        "WT": (filenames[0], filenames[1]),   
        "S8h": (filenames[2], filenames[3]), 
        "dS": (filenames[4], filenames[5]), 
        "dN": (filenames[6], filenames[7]), 
        "dW": (filenames[8], filenames[9]), 
        "dWdS": (filenames[10], filenames[11]), 
    }

    plt.figure(figsize=(10,5))

    for i, (mutant, (file1, file2)) in enumerate(mutants.items(), 1):
        # Load data
        data1  =  extract_consecutive_positions_last(file1)
        data2  =  extract_consecutive_positions_last(file2)
        if data1 is None or data2 is None:
            continue
  
        # Compute difference sister2 - sister1        
        diff = 2.5*(data2 - data1)
       
        # Estimate sigma (std dev) with mean fixed at 0
        sigma = np.std(diff)
        # Fit Gaussian(0, sigma) PDF
        x = np.linspace(min(diff), max(diff), 200)
        pdf = norm.pdf(x, loc=0, scale=sigma)
        # Plot histogram
        plt.subplot(2, 3, i)
        plt.hist(diff, bins=bins, alpha=0.75, color="#1f77b4", edgecolor="k")
        # Overlay Gaussian fit
        # plt.plot(x, pdf, "r-", lw=2, label=f"N(0, {sigma:.2f}²)")
        plt.title(f"{mutant}: Sister2 - Sister1\n σ = {sigma:.2f} [kb]", fontsize=18)
        plt.xlabel("Difference [kb]", fontsize=14)
        plt.ylabel("Count", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', labelsize=12)

    plt.tight_layout()
    plt.show()


# Example: filenames for multiple runs
filenames = [
    "test_0930_all_mutants/alpha50_tau10h/WT_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/WT_trajectory2.pkl",
    "test_0930_all_mutants/alpha50_tau10h/S2h_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/S2h_trajectory2.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dS_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dS_trajectory2.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dN_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dN_trajectory2.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dW_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dW_trajectory2.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dWdS_trajectory1.pkl",
    "test_0930_all_mutants/alpha50_tau10h/dWdS_trajectory2.pkl",
]

labels = []


plot_sister_difference_histograms(filenames, labels, bins=50)

