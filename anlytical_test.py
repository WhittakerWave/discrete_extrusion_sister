

import numpy as np
import matplotlib.pyplot as plt

# parameters
L = 32000.0
M = 500.0
rho = M / L
a = 0.01
k = 10
t = np.linspace(0.0, L*4, 200)

def N_t(a, rho, t):
    return (-1.0 + np.sqrt(1.0 + 4.0 * a * rho * t)) / a

def N_final_x(a, M, x):
    return (-1.0 + np.sqrt(1.0 + 2.0 * a * M * x)) / a

def remaining_with_xmin(tvals, M, k, a, rho, x_min=0.0, n_quad=600, corrected=False):
    """
    Compute remaining vs time.
    If corrected=True, avoid double-counting k in per_finished.
    """
    L = M / rho
    xs_full = np.linspace(0.0, 1.0, n_quad)
    Nx_full = N_final_x(a, M, xs_full)
    w_full = (1.0 - xs_full)**(k - 1)
    remaining = np.zeros_like(tvals, dtype=float)
    
    for idx, tval in enumerate(tvals):
        y = 2.0 * tval / L
        Nt = N_t(a, rho, tval)
        
        # finished integral from lower=x_min to upper=min(y,1)
        lower = x_min
        upper = min(y, 1.0)
        if upper <= lower:
            integral = 0.0
        else:
            j0 = int(np.searchsorted(xs_full, lower, side='left'))
            j1 = int(np.searchsorted(xs_full, upper, side='right'))
            xs = xs_full[j0:j1]
            if len(xs) < 2:
                integral = 0.0
            else:
                integrand = Nx_full[j0:j1] * w_full[j0:j1]
                integral = np.trapz(integrand, xs)
        
        if corrected:
            per_finished = integral
        else:
            per_finished = k * integral

        cut = max(y, x_min)
        prob_inprog = 0.0 if cut >= 1.0 else (1.0 - cut)**k
        per_inprog = Nt * prob_inprog

        per_total = per_finished + per_inprog
        total_captured = k * per_total
        remaining[idx] = M - total_captured
    
    return remaining

# compute both versions
remaining_original = remaining_with_xmin(t, M, k, a, rho, x_min=0.0, corrected=False)
remaining_corrected = remaining_with_xmin(t, M, k, a, rho, x_min=0.0, corrected=True)

# plot
plt.figure(figsize=(8,6))
plt.plot(t, remaining_original, lw=2, label="Original (with k*k factor)")
plt.plot(t, remaining_corrected, lw=2, label="Corrected (single k factor)")
plt.xlabel("time t")
plt.ylabel("Remaining")
plt.title("Remaining vs Time")
plt.legend()
plt.grid(True)
plt.show()


