

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def rhs(t, y, params):
    N, F = y
    v0, a, Ns, tau_att, tau_free = params
    v = v0 / (1 + a * N)
    attach = v * (F / Ns)
    detach = N / tau_att
    decay = F / tau_free if np.isfinite(tau_free) else 0.0
    dN = attach - detach
    dF = -attach + detach - decay
    return [dN, dF]

def simulate(params, M, N0, F0, t_span, t_eval=None):
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 1000)
    sol = solve_ivp(rhs, t_span, [N0, F0], args=(params,), t_eval=t_eval, rtol=1e-7, atol=1e-9)
    return sol.t, sol.y[0], sol.y[1]

# Example parameters
N_sites = 32000   # lattice length
M_points = 500    # initial points
N0 = 0            # initially attached
F0 = M_points     # initially free
v0 = 1.0
a = 0.1
tau_att = 10*3600    # mean detach time
tau_free = 10*3600  # mean free-point fall-off time (set np.inf for no decay)

params = (v0, a, N_sites, tau_att, tau_free)

t, N_t, F_t = simulate(params, M_points, N0, F0, t_span=(0,65000), t_eval=np.linspace(0,65000,65001))

plt.figure(figsize=(8,6))
plt.plot(t, N_t, label='N(t) attached')
plt.plot(t, F_t, label='F(t) free')
plt.plot(t, N_t + F_t, '--', label='Total remaining (N+F)')
plt.plot(t, F_t - N_t, '--', label='Total remaining (F-N)')
plt.xlabel('t')
plt.ylabel('counts')
plt.title(f"Attachment + free decay (tau_free={tau_free})")
plt.legend()
plt.grid(True)
plt.show()
