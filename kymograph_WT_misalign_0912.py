

import pickle
import numpy as np
import matplotlib.pyplot as plt



def load_translocator(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)   # {"sister": ..., "lef": ..., "ctcf": ...}
    return data



WT_static_sister1 = load_translocator("CTCF_dynamics_0912/WT_CTCF_Static_sister1.pkl")['sister'][-1]
WT_static_sister2 = load_translocator("CTCF_dynamics_0912/WT_CTCF_Static_sister2.pkl")['sister'][-1]
WT_100s_sister1 = load_translocator("CTCF_dynamics_0912/WT_CTCF100s_sister1.pkl")['sister'][-1]
WT_100s_sister2 = load_translocator("CTCF_dynamics_0912/WT_CTCF100s_sister2.pkl")['sister'][-1]
WT_1000s_sister1 = load_translocator("CTCF_dynamics_0912/WT_CTCF1000s_sister1.pkl")['sister'][-1]
WT_1000s_sister2 = load_translocator("CTCF_dynamics_0912/WT_CTCF1000s_sister2.pkl")['sister'][-1]


static_diff = 2.5*(WT_static_sister2 - WT_static_sister1)
s100_diff = 2.5*(WT_100s_sister2 - WT_100s_sister1)
s1000_diff = 2.5*(WT_1000s_sister2 - WT_1000s_sister1)

# Create histogram
plt.figure(figsize=(8, 6))

# Plot histograms with transparency
plt.hist(static_diff, bins=100, alpha=0.7, label='Static CTCF', color='blue')
plt.hist(s100_diff, bins=100, alpha=0.7, label='100s CTCF', color='orange')
plt.hist(s1000_diff, bins=100, alpha=0.7, label='1000s CTCF', color='green')
plt.tick_params(axis='both', which='major', labelsize=20)
plt.xlabel('Sister Position Difference [sister2 - sister1] [kb]', fontsize=20)
plt.ylabel('Count', fontsize=20)
plt.title('Distribution of Sisters Differences for WT', fontsize=18)
plt.legend(fontsize=20)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


