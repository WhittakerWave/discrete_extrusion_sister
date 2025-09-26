


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

k_list = [1, 5, 10, 50, 100, 500]
alpha_list = [0.0, 5, 100]
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
