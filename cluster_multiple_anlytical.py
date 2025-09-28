

import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt

# --- Original ODE solution ---
def N_single_extruder(t, a, rho):
    """ODE solution for single extruder before meeting time."""
    return (-1 + np.sqrt(1 + 4*a*rho*t)) / a

def N_final_cell(x, a, M):
    """Final captured for a cell fraction x."""
    return (-1 + np.sqrt(1 + 2*a*M*x)) / a

# --- Beta-distribution mean capture with 2-point minimum ---
def E_captured_per_extruder_2point(t, k, M, L, a):
    rho = M / L
    y = 2 * t / L
    y = np.clip(y, 0, 1)  # clip threshold fraction

    # N(t) for in-progress cells
    Nt = N_single_extruder(t, a, rho)

    # Integrand with minimum 2-point rule
    def integrand(x):
        points_in_cell = M * x
        if points_in_cell < 2:
            return 0.0  # ignore tiny cells
        else:
            return N_final_cell(x, a, M) * k * (1 - x)**(k - 1)

    I_y, _ = quad(integrand, 0, y)

    # Contribution from in-progress cells (x > y)
    def inprogress_integrand(x):
        points_in_cell = M * x
        if points_in_cell < 2:
            return 0.0
        else:
            return Nt * k * (1 - x)**(k - 1)

    I_inprogress, _ = quad(inprogress_integrand, y, 1)
    
    # Mean captured per extruder
    E_per = I_y + I_inprogress
    return E_per / k  # per-extruder

# --- Total captured and remaining ---
def total_captured_2point(t_array, k, M, L, a):
    return np.array([k * E_captured_per_extruder_2point(t, k, M, L, a) for t in t_array])

def remaining_points_2point(t_array, k, M, L, a):
    return M - total_captured_2point(t_array, k, M, L, a)

# --- Example usage ---
if __name__ == "__main__":
    L = 32000
    M = 500 
    k = 1  # try larger k to see suppression
    a = 0.001
    t_array = np.linspace(0, 65000, 300)

    total = total_captured_2point(t_array, k, M, L, a)
    remaining = remaining_points_2point(t_array, k, M, L, a)

    plt.figure(figsize=(8,6))
    plt.plot(t_array, total, label='Total Captured (2-point min)', lw=2)
    plt.plot(t_array, remaining, label='Remaining Points', lw=2)
    plt.xlabel('Time')
    plt.ylabel('Points')
    plt.title(f'Beta Formula with 2-point Minimum, k={k}, M={M}')
    plt.legend()
    plt.grid(True)
    plt.show()

plt.show()



import numpy as np
from math import sqrt
from scipy.integrate import quad
import matplotlib.pyplot as plt

# ---------- Core functions (mean-field analytic) ----------
def Ns(t, rho, alpha, rho_c=0.0):
    """Per-side solution Ns(t) from quadratic root; returns scalar t>=0."""
    if t <= 0:
        return 0.0
    a = alpha
    b = 1.0 + alpha * rho_c
    # handle alpha==0 limit
    if a == 0.0:
        # dNs/dt = rho/ (1 + alpha*(rho_c + 2 Ns)) -> alpha=0 -> dNs/dt = rho => Ns = rho t
        return rho * t
    disc = b*b + 4.0 * a * rho * t
    return (-b + sqrt(disc)) / (2.0 * a)

def expected_N_ex_per_extruder(t, rho, alpha, rho_c, rho_e, v=1.0):
    """Expected number in one extruder (center + two sides), averaged over Voronoi half-length."""
    lam = 2.0 * rho_e * v  # rate for tau distribution
    # If lam==0 => extruders isolated, just return full-time growth
    if lam == 0.0:
        return rho_c + 2.0 * Ns(t, rho, alpha, rho_c)
    # term for tau >= t:
    term1 = np.exp(-lam * t) * Ns(t, rho, alpha, rho_c)
    # integral term: integrate Ns(tau) * lam * exp(-lam*tau) from 0 to t
    integrand = lambda tau: Ns(tau, rho, alpha, rho_c) * lam * np.exp(-lam * tau)
    term2, _ = quad(integrand, 0.0, t, epsabs=1e-8, epsrel=1e-6, limit=200)
    return rho_c + 2.0 * (term1 + term2)

def N_total_array(t_array, M, L, k, rho, alpha, rho_c=None, v=1.0):
    """Compute analytic expected total N(t) for array of times."""
    if rho_c is None:
        rho_c = rho
    rho_e = float(k) / float(L)
    out = np.zeros_like(t_array, dtype=float)
    for i, t in enumerate(t_array):
        N_ex = expected_N_ex_per_extruder(t, rho, alpha, rho_c, rho_e, v=v)
        out[i] = k * N_ex
    # Cap at M (cannot collect more than total points)
    out = np.minimum(out, M)
    return out

# ---------- Example usage ----------
M = 500
L = 32000
rho = M / L
t_end = 65000
tgrid = np.linspace(0, t_end, 400)  # or match your simulation output times

k_list = [20, 50, 100, 500]
alpha_list = [10]
# alpha_list = [0.0, 1e-4, 1.0, 5.0, 20.0]

# compute curves
analytic_curves = {}
for alpha in alpha_list:
    analytic_curves[alpha] = {}
    for k in k_list:
        analytic_curves[alpha][k] = N_total_array(tgrid, M, L, k, rho, alpha)

plt.figure(figsize=(8,6))
for alpha in alpha_list:
    for k in k_list:
        plt.plot(tgrid, analytic_curves[alpha][k] / M, label=f'k={k}, alpha={alpha}')
plt.xlabel('t')
plt.ylabel('Fraction collected N(t)/M')
plt.title('Analytic N(t)/M')
plt.legend()
plt.grid(True)
plt.show()
