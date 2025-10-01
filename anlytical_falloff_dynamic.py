
import numpy as np

def E_simple(kon, koff, kappa, s, N, M):
    f = kon/(kon+koff)
    tau = 1.0/koff
    alpha = s * tau * N
    keff = f * kappa
    aeff = f * alpha
    return ( (keff + aeff) * (keff + aeff + M) ) / (2.0*M)

# params (example)
kappa = 500
s = 1.0
N = 32000
M = 500.0
koff = 1/822
kons = np.logspace(-1, 1.5, 40)   # vary kon relative to koff

vals = [E_simple(k, koff, kappa, s, N, M) for k in kons]

import matplotlib.pyplot as plt
plt.loglog(kons/koff, vals)
plt.xlabel('k_on / k_off')
plt.ylabel('E_infty (simple mf)')
plt.title('Mean-field E_infty vs k_on/k_off')
plt.grid(True)
plt.show()



