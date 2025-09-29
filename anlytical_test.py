



import numpy as np
import matplotlib.pyplot as plt
from math import exp
from scipy.integrate import quad


def long_time_clusters(M, N, k):
    """
    Long-time mean-field expected number of unique collapsed points (clusters).
    
    Parameters
    ----------
    M : int
        Number of initial points
    N : int
        Lattice length (drops out of long-time limit, included for clarity)
    k : int
        Number of extruders
    
    Returns
    -------
    float
        Expected number of clusters as t → ∞
    """
    if k <= 1:  # one extruder = one gap
        return 2.0 - np.exp(-M)
    
    f = lambda x: (1 - np.exp(-M * x)) * (k - 1) * (1 - x) ** (k - 2)
    val, _ = quad(f, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9) 
    return k * val 


A = long_time_clusters(M=500, N=320000, k=500)

print(A)

def E_clusters_infty(M, k):
    if k <= 1:
        return 1.0 - np.exp(-M)   # single gap
    f = lambda x: (1.0 - np.exp(-M * x)) * (k - 1) * (1 - x) ** (k - 2)
    val, _ = quad(f, 0.0, 1.0, epsabs=1e-9, epsrel=1e-9)
    return k * val

def approx_smallM(M, k):
    # M << k: approx E ≈ M (but cannot exceed k)
    return min(M, k)

def approx_largeM(M, k):
    if M == 0:
        return 0.0
    return k * (1.0 - (k - 1) / M)  # valid for M >> k

# example sweep
k = 5000
Ms = np.logspace(-1, 4, 500)   # M from 0.1 to 1e4
E_vals = np.array([E_clusters_infty(int(round(M)), k) for M in Ms])

plt.figure(figsize=(8,5))
plt.loglog(Ms, E_vals, label="E_clusters_infty (exact numeric)")
plt.loglog(Ms, np.minimum(Ms, k), ls='--', label="small-M approx (min(M,k))")
plt.loglog(Ms, np.maximum(0, k * (1 - (k-1)/Ms)), ls=':', label="large-M approx")
plt.xlabel("M (number of points)")
plt.ylabel("E[infty clusters]")
plt.title(f"Long-time expected clusters vs M (k={k})")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.show()




import numpy as np
import matplotlib.pyplot as plt

class SimpleUnifiedModel:
    """
    Simple unified model combining all key physics:
    
    Key insight: Use single extruder solution as base, then apply
    scaling factor that depends on k to capture interference effects.
    
    Total_absorbed(t) = k * Single_absorbed(t/τ(k)) * η(k)
    
    Where:
    - τ(k) = time scaling (collision effects)  
    - η(k) = efficiency factor (interference effects)
    """
    
    def __init__(self, M, L, k, a):
        if M > L:
            raise ValueError(f"M ({M}) cannot be greater than L ({L})")
            
        self.M = M
        self.L = L
        self.k = k  
        self.a = a
        self.rho = M / L
        
        # Precompute scaling factors
        self.time_scale = self._compute_time_scale()
        self.efficiency_factor = self._compute_efficiency_factor()
    
    def single_extruder_absorbed(self, t):
        """Base single extruder solution"""
        if self.a == 0:
            return min(2 * self.rho * t, self.M)
        
        absorbed = (-1 + np.sqrt(1 + 4 * self.a * self.M * t / self.L)) / self.a
        return min(absorbed, self.M)
    
    def _compute_time_scale(self):
        """
        Time scaling factor τ(k):
        - k=1: τ=1 (no scaling)
        - k large: τ→0 (fast collisions, short effective time)
        """
        if self.k <= 1:
            return 1.0
        
        # Collision time scale based on domain overlap
        # When k extruders are placed, average spacing ~ L/k
        # Collision happens when domains of size ~ L/k overlap
        # Time to collision ~ (L/k) / velocity ~ L/k
        
        collision_time = self.L / self.k
        absorption_time = self.L / 2  # Single extruder time scale
        
        # Effective time scaling
        tau = collision_time / absorption_time
        return min(tau, 1.0)  # Can't be faster than single extruder
    
    def _compute_efficiency_factor(self):
        """
        Efficiency factor η(k):
        - k=1: η=1 (full efficiency)
        - k small: η≈1 (little interference)  
        - k large: η→0 (high interference)
        """
        if self.k <= 1:
            return 1.0
        
        # Key insight: efficiency depends on how much "space" each extruder gets
        # Optimal spacing gives each extruder domain ~ L/k
        # Points per domain ~ M/k
        
        # Three regimes:
        lambda_param = self.M / self.k  # Points per extruder
        
        if lambda_param >= 10:
            # Sparse regime: plenty of points per extruder
            return 1.0
        elif lambda_param >= 1:
            # Intermediate regime: some competition
            return 0.5 + 0.5 * (lambda_param / 10)
        else:
            # Dense regime: severe competition
            # Each extruder fights over < 1 point on average
            return lambda_param * 0.5
    
    def total_absorbed(self, t):
        """
        Unified formula:
        Total_absorbed(t) = k * Single_absorbed(t * τ(k)) * η(k)
        """
        # Effective time due to collision effects
        t_effective = t * self.time_scale
        
        # Base absorption (what one extruder would do)
        single_absorbed = self.single_extruder_absorbed(t_effective)
        
        # Scale by number of extruders and efficiency
        total = self.k * single_absorbed * self.efficiency_factor
        
        # Physical bounds
        return min(total, self.M)
    
    def remaining_points(self, t):
        """Points remaining at time t"""
        return self.M - self.total_absorbed(t)
    
    def final_efficiency(self):
        """Asymptotic efficiency as t→∞"""
        # In the limit, single extruder absorbs all M points
        # So total = k * M * η(k), but capped at M
        max_possible = self.k * self.M * self.efficiency_factor
        final_absorbed = min(max_possible, self.M)
        return final_absorbed / self.M
    
    def analyze_regime(self):
        """Classify the regime based on k"""
        lambda_param = self.M / self.k
        
        if self.k == 1:
            regime = "Single extruder"
            description = f"Complete absorption (η={self.efficiency_factor:.2f})"
        elif lambda_param >= 10:
            regime = "Sparse extruders" 
            description = f"High efficiency (λ={lambda_param:.1f}, η={self.efficiency_factor:.2f})"
        elif lambda_param >= 1:
            regime = "Intermediate density"
            description = f"Moderate efficiency (λ={lambda_param:.1f}, η={self.efficiency_factor:.2f})"
        else:
            regime = "Dense extruders"
            description = f"Low efficiency (λ={lambda_param:.2f}, η={self.efficiency_factor:.2f})"
        
        print(f"k={self.k}: {regime}")
        print(f"  {description}")
        print(f"  Time scale: τ={self.time_scale:.3f}")
        print(f"  Final efficiency: {self.final_efficiency():.1%}")
        
        return regime


def test_simple_model():
    """Test the simple unified model"""
    
    # Your parameters
    L = 32000.0
    M = 500.0  
    a = 0.01
    k_list = [1, 50, 500, 2000, 8000]
    
    t = np.linspace(0, L, 100)
    
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Remaining points vs time
    plt.subplot(2, 3, 1)
    
    print("=== Simple Unified Model Results ===")
    for k in k_list:
        model = SimpleUnifiedModel(M, L, k, a)
        remaining = [model.remaining_points(time) for time in t]
        
        plt.plot(t, remaining, linewidth=2, label=f'k={k}')
        
        # Analyze regime
        model.analyze_regime()
        print()
    
    plt.xlabel('Time')
    plt.ylabel('Remaining points')
    plt.title('Simple Unified Model')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Efficiency factors
    plt.subplot(2, 3, 2)
    k_range = np.logspace(0, 4, 50)
    efficiency_factors = []
    time_scales = []
    
    for k in k_range:
        model = SimpleUnifiedModel(M, L, int(k), a)
        efficiency_factors.append(model.efficiency_factor)
        time_scales.append(model.time_scale)
    
    plt.loglog(k_range, efficiency_factors, 'r-', linewidth=2, label='Efficiency η(k)')
    plt.loglog(k_range, time_scales, 'b-', linewidth=2, label='Time scale τ(k)')
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Scaling factor')
    plt.title('Scaling Factors vs k')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Final efficiency vs k
    plt.subplot(2, 3, 3)
    final_efficiencies = []
    
    for k in k_range:
        model = SimpleUnifiedModel(M, L, int(k), a)
        final_efficiencies.append(model.final_efficiency())
    
    plt.semilogx(k_range, final_efficiencies, 'g-', linewidth=2)
    
    # Mark the test points
    for k in k_list:
        model = SimpleUnifiedModel(M, L, k, a)
        eff = model.final_efficiency()
        plt.plot(k, eff, 'ro', markersize=8)
        plt.annotate(f'{eff:.1%}', (k, eff), xytext=(5, 5), 
                    textcoords='offset points')
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Final efficiency')
    plt.title('Final Efficiency vs k')
    plt.grid(True)
    
    # Plot 4: Points per extruder (λ = M/k)
    plt.subplot(2, 3, 4)
    lambda_values = M / k_range
    
    plt.loglog(k_range, lambda_values, 'purple', linewidth=2)
    plt.axhline(1, color='red', linestyle='--', label='λ=1 (critical)')
    plt.axhline(10, color='orange', linestyle='--', label='λ=10 (sparse)')
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Points per extruder λ=M/k')
    plt.title('Resource Competition')
    plt.legend()
    plt.grid(True)
    
    # Plot 5: Regime classification
    plt.subplot(2, 3, 5)
    regimes = []
    colors = []
    
    for k in k_range:
        lambda_param = M / k
        if k == 1:
            regimes.append('Single')
            colors.append('black')
        elif lambda_param >= 10:
            regimes.append('Sparse')
            colors.append('green')
        elif lambda_param >= 1:
            regimes.append('Intermediate')
            colors.append('orange')
        else:
            regimes.append('Dense')
            colors.append('red')
    
    # Plot efficiency colored by regime
    for regime, color in [('Single', 'black'), ('Sparse', 'green'), 
                         ('Intermediate', 'orange'), ('Dense', 'red')]:
        mask = [r == regime for r in regimes]
        if any(mask):
            k_regime = k_range[mask]
            eff_regime = np.array(final_efficiencies)[mask]
            plt.scatter(k_regime, eff_regime, c=color, s=50, 
                       alpha=0.7, label=regime)
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Final efficiency')
    plt.title('Efficiency by Regime')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    
    # Plot 6: Early time behavior
    plt.subplot(2, 3, 6)
    t_early = np.linspace(0, L/10, 100)
    
    for k in [1, 10, 100, 1000]:
        model = SimpleUnifiedModel(M, L, k, a)
        remaining = [model.remaining_points(time) for time in t_early]
        plt.plot(t_early, remaining, linewidth=2, label=f'k={k}')
    
    plt.xlabel('Time')
    plt.ylabel('Remaining points')
    plt.title('Early Time Behavior')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Summary
    print("=== SUMMARY ===")
    for k in k_list:
        model = SimpleUnifiedModel(M, L, k, a)
        final_remaining = model.remaining_points(L)
        print(f"k={k:4d}: {final_remaining:6.1f} points remaining ({final_remaining/M:5.1%})")


def compare_all_models():
    """Compare simple model with your original code behavior"""
    
    L = 32000.0
    M = 500.0
    a = 0.01
    k_list = [1, 50, 500, 2000, 8000]
    
    plt.figure(figsize=(12, 8))
    
    # Test different time ranges
    for subplot, (t_max, title) in enumerate([(L/4, 'Early Time'), (L, 'Full Time')]):
        plt.subplot(2, 2, subplot + 1)
        
        t = np.linspace(0, t_max, 100)
        
        for k in k_list:
            model = SimpleUnifiedModel(M, L, k, a)
            remaining = [model.remaining_points(time) for time in t]
            plt.plot(t, remaining, linewidth=2, label=f'k={k}')
        
        plt.xlabel('Time')
        plt.ylabel('Remaining points')
        plt.title(f'Simple Model - {title}')
        plt.legend()
        plt.grid(True)
    
    # Key metrics
    plt.subplot(2, 2, 3)
    
    metrics = {
        'k': [],
        'final_remaining': [],
        'efficiency': [],
        'regime': []
    }
    
    for k in k_list:
        model = SimpleUnifiedModel(M, L, k, a)
        final_rem = model.remaining_points(2*L)  # Long time
        efficiency = (M - final_rem) / M
        regime = model.analyze_regime()
        
        metrics['k'].append(k)
        metrics['final_remaining'].append(final_rem)
        metrics['efficiency'].append(efficiency)
        metrics['regime'].append(regime)
    
    plt.semilogx(metrics['k'], metrics['efficiency'], 'ro-', 
                linewidth=3, markersize=8)
    
    for k, eff in zip(metrics['k'], metrics['efficiency']):
        plt.annotate(f'{eff:.1%}', (k, eff), xytext=(5, 5), 
                    textcoords='offset points', fontsize=10)
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Final efficiency')
    plt.title('Efficiency Summary')
    plt.grid(True)
    
    # Expected behavior check
    plt.subplot(2, 2, 4)
    
    expected = {
        1: "~100% (complete absorption)",
        50: "High (80%+)",
        500: "Moderate (40-60%)", 
        2000: "Low (10-20%)",
        8000: "Very low (<5%)"
    }
    
    actual_text = []
    for k, eff in zip(metrics['k'], metrics['efficiency']):
        actual_text.append(f"k={k}: {eff:.1%}")
        print(f"k={k:4d}: Expected {expected.get(k, '?')}, Got {eff:.1%}")
    
    # Text summary
    plt.text(0.1, 0.9, "Expected vs Actual:", transform=plt.gca().transAxes, 
            fontsize=12, fontweight='bold')
    
    for i, (k, exp_text) in enumerate(expected.items()):
        actual_eff = metrics['efficiency'][metrics['k'].index(k)]
        plt.text(0.1, 0.8 - i*0.12, f"k={k}: Expected {exp_text}", 
                transform=plt.gca().transAxes, fontsize=10)
        plt.text(0.1, 0.76 - i*0.12, f"      Got {actual_eff:.1%}", 
                transform=plt.gca().transAxes, fontsize=10, 
                color='red' if actual_eff < 0.9 and k == 1 else 'green')
    
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.axis('off')
    plt.title('Expected vs Actual Check')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Simple Unified Extruder Model ===\n")
    
    test_simple_model()
    
    print("\n" + "="*60)
    print("Comparing with expected behavior...")
    compare_all_models()


import numpy as np
from scipy.integrate import quad

# -------------------------
# Core math functions
# -------------------------
def N_final_x(x, a, M):
    """
    Final captured in a cell of fraction x:
      N_final(x) = (-1 + sqrt(1 + 2 a M x)) / a
    """
    return ( -1.0 + np.sqrt(1.0 + 2.0 * a * M * x) ) / a

def remaining_infty_modelB(M, k, a, rho, x_min=0.0, enforce_floor=False, eps=1e-12):
    """
    Compute the long-time remaining number of distinct A points under Model B
    (cells with x <= x_min are blocked and capture nothing).
    
    Parameters
    ----------
    M : int or float
        Total number of A points.
    k : int
        Number of extruders.
    a : float
        Pushing parameter.
    rho : float
        Density M/L.
    x_min : float, optional
        Minimum active cell fraction (0 <= x_min < 1). Cells with x <= x_min are blocked.
    enforce_floor : bool, optional
        If True, enforce final aggregates at least 2*k (practical small-k correction).
    eps : float
        small epsilon for numerical stability when x_min ~ 1.
    
    Returns
    -------
    remaining : float
        Expected number of remaining distinct A's at long time.
    info : dict
        Diagnostic info: integral value, total_captured, per_extruder_expected.
    """
    # basic checks
    if x_min >= 1.0 - eps:
        # everything blocked -> no capture
        remaining = float(M)
        return remaining, {'integral': 0.0, 'total_captured': 0.0, 'per_extruder_expected': 0.0}
    if x_min <= 0.0:
        x_min = 0.0

    # integrand for the Beta-weighted average:
    # per-extruder expectation = k * integral_{x_min}^{1} N_final(x) * (1-x)^(k-1) dx
    # total_captured = k * per_extruder = k^2 * integral_{x_min}^{1} N_final(x) (1-x)^(k-1) dx
    def integrand(x):
        return N_final_x(x, a, M) * (1.0 - x)**(k - 1)

    # evaluate integral (numerical quadrature)
    integral_val, integral_err = quad(integrand, x_min, 1.0, epsabs=1e-9, epsrel=1e-9, limit=200)
    per_extruder_expected = k * integral_val
    total_captured = k * per_extruder_expected  # = k^2 * integral

    # numerical safeguards
    total_captured = float(np.clip(total_captured, 0.0, float(M)))
    remaining = float(M - total_captured)

    # optional small-k floor: cannot have fewer aggregates than 2k
    if enforce_floor:
        remaining = max(remaining, 2.0 * k)

    info = {
        'integral': integral_val,
        'integral_err': integral_err,
        'per_extruder_expected': per_extruder_expected,
        'total_captured': total_captured,
    }
    return remaining, info

# Optional: Poisson-sum alternative (discrete count form) for comparison
def remaining_infty_poisson(M, k, a, use_poisson=True, nmax_extra=50):
    """
    Poisson-sum calculation for Remaining_infty:
      Remaining = M - k * sum_{n>=0} P(n) * S(n)
    where S(n) = sum_{i=0}^{n-1} 1/(1+a i), and P(n) is Poisson(lambda=M/k).
    """
    lam = M / k
    # truncate at nmax
    nmax = max(200, int(lam + 10 * np.sqrt(max(1.0, lam))) + nmax_extra)
    # generate Poisson probabilities iteratively
    probs = []
    p = np.exp(-lam)
    probs.append(p)
    for n in range(1, nmax + 1):
        p = p * lam / n
        probs.append(p)
    # compute S(n)
    S = np.zeros(nmax + 1, dtype=float)
    for n in range(1, nmax + 1):
        # incremental sum avoids repeated loops
        S[n] = S[n - 1] + 1.0 / (1.0 + a * (n - 1))
    expected_per = 0.0
    for n, p in enumerate(probs):
        expected_per += p * S[n]
    total_captured = k * expected_per
    total_captured = float(np.clip(total_captured, 0.0, M))
    remaining = float(M - total_captured)
    return remaining, {'lambda': lam, 'nmax': nmax, 'expected_per_extruder': expected_per, 'total_captured': total_captured}

# -------------------------
# Example usage & prints
# -------------------------
if __name__ == "__main__":
    # Example parameters (replace with your desired values)
    M = 500.0
    L = 32000.0
    rho = M / L
    a = 0.01

    examples = [
        {'k': 1, 'x_min': 0.0},
        {'k': 50, 'x_min': 0.0},
        {'k': 500, 'x_min': 0.0},
        {'k': 5000, 'x_min': 0.0},
        # with blocking threshold (cells smaller than x_min blocked)
        {'k': 5000, 'x_min': 1e-3},
        {'k': 5000, 'x_min': 1e-2},
    ]

    for ex in examples:
        k = ex['k']
        x_min = ex['x_min']
        rem, info = remaining_infty_modelB(M=M, k=k, a=a, rho=rho, x_min=x_min, enforce_floor=True)
        rem_pois, info_p = remaining_infty_poisson(M=M, k=k, a=a)
        print(f"k={k:6d}, x_min={x_min:8.4g}  => Remaining (ModelB integral, floor2k) = {rem:7.3f}, "
              f"Remaining (Poisson approx) = {rem_pois:7.3f}")
        # print diagnostics for the last example
        if k == 5000 and x_min == 1e-2:
            print("  diagnostic (integral):", info)
            print("  diagnostic (poisson):", info_p)






import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.stats import beta
import warnings
warnings.filterwarnings('ignore')

class VoronoiExtruderModel:
    """
    Multi-extruder model based on Voronoi cell approach from the document.
    
    Key insight: k randomly placed extruders create Voronoi cells with 
    cell fraction x ~ Beta(1, k-1), so each cell has length Lx.
    
    For each cell:
    - Meeting time: t_meet(x) = Lx/2
    - While sweeping: dN/dt = 2ρ/(1 + aN(t))
    - Final capture: N_final(x) = (-1 + sqrt(1 + 2aMx))/a
    """
    
    def __init__(self, M, L, k, a):
        # Validate constraints
        if M > L:
            raise ValueError(f"M ({M}) cannot be greater than lattice size L ({L})")
        
        self.M = M          # Initial points
        self.L = L          # Lattice size  
        self.k = k          # Number of extruders
        self.a = a          # Slowing parameter
        self.rho = M / L    # Point density
    
    def beta_pdf(self, x):
        """PDF of cell fraction: p(x) = k(1-x)^(k-1)"""
        return self.k * (1 - x)**(self.k - 1)
    
    def meeting_time(self, x):
        """Meeting time for cell with fraction x: t_meet(x) = Lx/2"""
        return self.L * x / 2
    
    def N_conditional(self, t, x):
        """
        Expected capture in cell with fraction x at time t.
        
        If t <= t_meet(x): N(t|x) = (-1 + sqrt(1 + 4aρt))/a
        If t > t_meet(x): N(t|x) = N_final(x)
        """
        t_meet = self.meeting_time(x)
        
        if t <= t_meet:
            # Still sweeping
            if self.a == 0:
                return 2 * self.rho * t
            return (-1 + np.sqrt(1 + 4 * self.a * self.rho * t)) / self.a
        else:
            # Finished sweeping
            return self.N_final(x)
    
    def N_final(self, x):
        """Final capture for cell with fraction x"""
        if self.a == 0:
            return self.M * x  # All points in cell
        return (-1 + np.sqrt(1 + 2 * self.a * self.M * x)) / self.a
    
    def expected_per_extruder(self, t):
        """
        Expected capture per extruder at time t using Dirichlet averaging.
        
        E[N_per(t)] = k∫[0 to y] N_final(x) * (1-x)^(k-1) dx + N(t) * (1-y)^k
        
        where y = 2t/L
        """
        y = 2 * t / self.L
        
        # Term 1: Integral over finished cells (x ≤ y)
        if y > 0:
            def integrand(x):
                return self.N_final(x) * (1 - x)**(self.k - 1)
            
            upper_limit = min(y, 1.0)
            integral_term, _ = quad(integrand, 0, upper_limit, limit=100)
            integral_term *= self.k
        else:
            integral_term = 0
        
        # Term 2: In-progress cells (x > y)
        if y < 1:
            N_current = self.N_conditional(t, y)  # Current capture for in-progress cells
            prob_in_progress = (1 - min(y, 1.0))**self.k
            in_progress_term = N_current * prob_in_progress
        else:
            in_progress_term = 0
        
        return integral_term + in_progress_term
    
    def total_captured(self, t):
        """Total points captured by all k extruders"""
        return self.k * self.expected_per_extruder(t)
    
    def remaining_points(self, t):
        """Points remaining at time t"""
        return self.M - self.total_captured(t)
    
    def plug_in_approximation(self, t):
        """
        Fast plug-in mean-gap approximation: replace random x by E[x] = 1/k
        
        N_per(t) ≈ {(-1 + sqrt(1 + 4aρt))/a,     t ≤ L/(2k)
                   {(-1 + sqrt(1 + 2aλ))/a,      t ≥ L/(2k)
        
        where λ = M/k
        """
        lambda_param = self.M / self.k
        transition_time = self.L / (2 * self.k)
        
        if t <= transition_time:
            # Still sweeping
            if self.a == 0:
                return self.k * 2 * self.rho * t
            N_per = (-1 + np.sqrt(1 + 4 * self.a * self.rho * t)) / self.a
        else:
            # Finished sweeping
            if self.a == 0:
                N_per = lambda_param
            else:
                N_per = (-1 + np.sqrt(1 + 2 * self.a * lambda_param)) / self.a
        
        return self.k * N_per
    
    def small_time_approximation(self, t):
        """
        Small time expansion: N_total(t) ≈ 2Mt/L
        """
        return 2 * self.M * t / self.L
    
    def get_time_series(self, t_max, n_points=200):
        """Generate time series using exact Dirichlet averaging"""
        times = np.linspace(0, t_max, n_points)
        
        total_captured = np.array([self.total_captured(t) for t in times])
        remaining = self.M - total_captured
        per_extruder = total_captured / self.k
        
        # Also compute approximations
        plug_in = np.array([self.plug_in_approximation(t) for t in times])
        small_time = self.small_time_approximation(times)
        
        return {
            'times': times,
            'total_captured': total_captured,
            'remaining': remaining,
            'per_extruder': per_extruder,
            'plug_in_approx': plug_in,
            'small_time_approx': small_time
        }
    
    def analyze_regimes(self):
        """Analyze different regimes based on k"""
        print(f"=== Voronoi Cell Analysis ===")
        print(f"Parameters: M={self.M}, L={self.L}, k={self.k}, a={self.a}")
        print(f"Point density: ρ = {self.rho:.3f}")
        print(f"Mean cell fraction: E[x] = 1/k = {1/self.k:.4f}")
        print(f"Mean cell size: E[Lx] = L/k = {self.L/self.k:.2f}")
        print()
        
        # Time scales
        mean_meeting_time = self.L / (2 * self.k)
        lambda_param = self.M / self.k
        
        print(f"Time scales:")
        print(f"  Mean meeting time: L/(2k) = {mean_meeting_time:.2f}")
        print(f"  Points per extruder: λ = M/k = {lambda_param:.2f}")
        
        if self.a > 0:
            slowing_time = 1 / (self.a * self.rho)
            print(f"  Slowing time scale: 1/(aρ) = {slowing_time:.2f}")
        print()
        
        # Regime classification
        if self.k <= 5:
            regime = "Sparse extruders"
            description = "Large cells, high efficiency per extruder"
        elif self.k >= 50:
            regime = "Dense extruders" 
            description = "Small cells, potential collision effects"
        else:
            regime = "Intermediate density"
            description = "Moderate cell sizes"
        
        print(f"Regime: {regime}")
        print(f"  {description}")
        
        # Final capture estimate
        if self.a == 0:
            final_per_extruder = lambda_param
        else:
            final_per_extruder = (-1 + np.sqrt(1 + 2 * self.a * lambda_param)) / self.a
        
        final_efficiency = (self.k * final_per_extruder) / self.M
        print(f"  Expected final efficiency: {final_efficiency:.1%}")
        
        return regime, final_efficiency


def compare_k_values():
    """Compare different numbers of extruders k"""
    
    # Parameters
    M, L, a = 80, 100, 1.0
    k_values = [1, 2, 5, 10, 20, 50, 100]
    
    plt.figure(figsize=(18, 12))
    
    # Colors for different k values
    colors = plt.cm.viridis(np.linspace(0, 1, len(k_values)))
    
    # Solve for each k
    results = {}
    for k in k_values:
        model = VoronoiExtruderModel(M, L, k, a)
        result = model.get_time_series(t_max=100)
        results[k] = result
    
    # Plot 1: Remaining points over time
    plt.subplot(2, 4, 1)
    for k, color in zip(k_values, colors):
        result = results[k]
        plt.plot(result['times'], result['remaining'], 
                color=color, linewidth=2, label=f'k={k}')
    
    plt.xlabel('Time t')
    plt.ylabel('Remaining points')
    plt.title('Point Evolution vs k')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    
    # Plot 2: Exact vs plug-in approximation (for k=10)
    plt.subplot(2, 4, 2)
    k_test = 10
    result = results[k_test]
    plt.plot(result['times'], result['remaining'], 'b-', linewidth=2, label='Exact')
    plt.plot(result['times'], M - result['plug_in_approx'], 'r--', linewidth=2, label='Plug-in approx')
    plt.plot(result['times'], M - result['small_time_approx'], 'g:', linewidth=2, label='Small-t approx')
    
    plt.xlabel('Time t')
    plt.ylabel('Remaining points')
    plt.title(f'Approximation Quality (k={k_test})')
    plt.legend()
    plt.grid(True)
    
    # Plot 3: Final efficiency vs k
    plt.subplot(2, 4, 3)
    final_efficiencies = []
    for k in k_values:
        result = results[k]
        final_eff = result['total_captured'][-1] / M
        final_efficiencies.append(final_eff)
    
    plt.semilogx(k_values, final_efficiencies, 'ro-', markersize=8, linewidth=2)
    plt.xlabel('Number of extruders k')
    plt.ylabel('Final absorption efficiency')
    plt.title('Efficiency vs Extruder Density')
    plt.grid(True)
    
    # Plot 4: Per-extruder efficiency
    plt.subplot(2, 4, 4)
    per_extruder_eff = np.array(final_efficiencies) / np.array(k_values) * M
    plt.loglog(k_values, per_extruder_eff, 'bs-', markersize=8, linewidth=2)
    plt.xlabel('Number of extruders k')
    plt.ylabel('Points per extruder')
    plt.title('Per-Extruder Efficiency')
    plt.grid(True)
    
    # Plot 5: Cell size distribution for different k
    plt.subplot(2, 4, 5)
    x_range = np.linspace(0.001, 0.999, 200)
    
    for k in [1, 5, 20, 100]:
        model = VoronoiExtruderModel(M, L, k, a)
        pdf_vals = [model.beta_pdf(x) for x in x_range]
        plt.plot(x_range, pdf_vals, linewidth=2, label=f'k={k}')
    
    plt.xlabel('Cell fraction x')
    plt.ylabel('Probability density p(x)')
    plt.title('Cell Size Distribution: k(1-x)^(k-1)')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 6: Meeting time distribution
    plt.subplot(2, 4, 6)
    for k in [1, 5, 20, 100]:
        meeting_times = L * x_range / 2
        model = VoronoiExtruderModel(M, L, k, a)
        pdf_vals = [model.beta_pdf(x) for x in x_range]
        plt.plot(meeting_times, pdf_vals, linewidth=2, label=f'k={k}')
    
    plt.xlabel('Meeting time t_meet = Lx/2')
    plt.ylabel('Probability density')
    plt.title('Meeting Time Distribution')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # Plot 7: Early vs late time behavior
    plt.subplot(2, 4, 7)
    times_short = np.linspace(0, 20, 100)
    
    for k in [5, 20, 100]:
        model = VoronoiExtruderModel(M, L, k, a)
        remaining_exact = [model.remaining_points(t) for t in times_short]
        remaining_linear = M - model.small_time_approximation(times_short)
        
        plt.plot(times_short, remaining_exact, '-', linewidth=2, label=f'k={k} exact')
        if k == 20:  # Show linear approx for one case
            plt.plot(times_short, remaining_linear, '--', linewidth=2, 
                    alpha=0.7, label='Linear approx')
    
    plt.xlabel('Time t')
    plt.ylabel('Remaining points')
    plt.title('Early Time Behavior')
    plt.legend()
    plt.grid(True)
    
    # Plot 8: Regime analysis
    plt.subplot(2, 4, 8)
    regimes = []
    efficiencies = []
    
    k_fine = np.logspace(0, 2, 30)
    for k in k_fine:
        model = VoronoiExtruderModel(M, L, int(k), a)
        _, efficiency = model.analyze_regimes()
        efficiencies.append(efficiency)
        
        if k <= 5:
            regimes.append(0)  # Sparse
        elif k >= 50:
            regimes.append(2)  # Dense
        else:
            regimes.append(1)  # Intermediate
    
    # Color points by regime
    colors_regime = ['green', 'orange', 'red']
    for i, regime in enumerate([0, 1, 2]):
        mask = np.array(regimes) == regime
        if np.any(mask):
            plt.scatter(k_fine[mask], np.array(efficiencies)[mask], 
                       c=colors_regime[regime], s=50, alpha=0.7,
                       label=['Sparse', 'Intermediate', 'Dense'][regime])
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Final efficiency')
    plt.title('Regime Classification')
    plt.xscale('log')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Find optimal k
    optimal_idx = np.argmax(final_efficiencies)
    optimal_k = k_values[optimal_idx]
    max_efficiency = final_efficiencies[optimal_idx]
    
    print(f"\n=== RESULTS SUMMARY ===")
    print(f"Optimal k = {optimal_k}")
    print(f"Maximum efficiency = {max_efficiency:.1%}")
    
    return k_values, final_efficiencies


def parameter_sensitivity():
    """Study sensitivity to parameters M, L, a"""
    
    plt.figure(figsize=(15, 10))
    
    # Base parameters
    M_base, L_base, k_base, a_base = 80, 100, 20, 1.0
    
    # Study 1: Effect of M
    plt.subplot(2, 3, 1)
    M_values = [20, 40, 60, 80]
    times = np.linspace(0, 80, 100)
    
    for M in M_values:
        model = VoronoiExtruderModel(M, L_base, k_base, a_base)
        remaining = [model.remaining_points(t) for t in times]
        efficiency = [(M - r)/M for r in remaining]
        plt.plot(times, efficiency, linewidth=2, label=f'M={M}')
    
    plt.xlabel('Time t')
    plt.ylabel('Absorption efficiency')
    plt.title('Effect of Initial Points M')
    plt.legend()
    plt.grid(True)
    
    # Study 2: Effect of L
    plt.subplot(2, 3, 2)
    L_values = [50, 100, 200, 400]
    
    for L in L_values:
        model = VoronoiExtruderModel(M_base, L, k_base, a_base)
        remaining = [model.remaining_points(t) for t in times]
        plt.plot(times, remaining, linewidth=2, label=f'L={L}')
    
    plt.xlabel('Time t')
    plt.ylabel('Remaining points')
    plt.title('Effect of Lattice Size L')
    plt.legend()
    plt.grid(True)
    
    # Study 3: Effect of a
    plt.subplot(2, 3, 3)
    a_values = [0, 0.5, 1.0, 2.0, 5.0]
    
    for a in a_values:
        model = VoronoiExtruderModel(M_base, L_base, k_base, a)
        remaining = [model.remaining_points(t) for t in times]
        plt.plot(times, remaining, linewidth=2, label=f'a={a}')
    
    plt.xlabel('Time t')
    plt.ylabel('Remaining points')
    plt.title('Effect of Slowing Parameter a')
    plt.legend()
    plt.grid(True)
    
    # Study 4: Scaling with density ρ = M/L
    plt.subplot(2, 3, 4)
    densities = [0.2, 0.4, 0.6, 0.8]
    L_fixed = 100
    
    for rho in densities:
        M = int(rho * L_fixed)
        model = VoronoiExtruderModel(M, L_fixed, k_base, a_base)
        remaining = [model.remaining_points(t) for t in times]
        efficiency = [(M - r)/M for r in remaining]
        plt.plot(times, efficiency, linewidth=2, label=f'ρ={rho}')
    
    plt.xlabel('Time t')
    plt.ylabel('Absorption efficiency')
    plt.title('Effect of Density ρ = M/L')
    plt.legend()
    plt.grid(True)
    
    # Study 5: Final efficiency surface (k vs a)
    plt.subplot(2, 3, 5)
    k_range = np.logspace(0, 2, 20)
    a_range = np.logspace(-1, 1, 20)
    K, A = np.meshgrid(k_range, a_range)
    
    efficiency_surface = np.zeros_like(K)
    for i in range(len(a_range)):
        for j in range(len(k_range)):
            model = VoronoiExtruderModel(M_base, L_base, int(K[i,j]), A[i,j])
            final_captured = model.total_captured(100)  # Large time
            efficiency_surface[i,j] = final_captured / M_base
    
    contour = plt.contourf(K, A, efficiency_surface, levels=20, cmap='viridis')
    plt.colorbar(contour, label='Final efficiency')
    plt.xlabel('Number of extruders k')
    plt.ylabel('Slowing parameter a')
    plt.title('Efficiency Surface')
    plt.xscale('log')
    plt.yscale('log')
    
    # Study 6: Time to reach efficiency levels
    plt.subplot(2, 3, 6)
    efficiency_targets = [0.5, 0.7, 0.9, 0.95]
    k_range_time = [1, 5, 10, 20, 50]
    
    for eff_target in efficiency_targets:
        times_to_target = []
        
        for k in k_range_time:
            model = VoronoiExtruderModel(M_base, L_base, k, a_base)
            
            # Binary search for time to reach target efficiency
            t_low, t_high = 0, 200
            for _ in range(20):  # Binary search iterations
                t_mid = (t_low + t_high) / 2
                captured = model.total_captured(t_mid)
                current_eff = captured / M_base
                
                if current_eff < eff_target:
                    t_low = t_mid
                else:
                    t_high = t_mid
            
            times_to_target.append((t_low + t_high) / 2)
        
        plt.semilogy(k_range_time, times_to_target, 'o-', linewidth=2, 
                    label=f'{eff_target:.0%} efficiency')
    
    plt.xlabel('Number of extruders k')
    plt.ylabel('Time to reach efficiency')
    plt.title('Absorption Speed vs k')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("=== Voronoi Cell-Based Multi-Extruder Model ===\n")
    
    # Example analysis
    M, L, k, a = 80, 100, 20, 1.0
    
    model = VoronoiExtruderModel(M, L, k, a)
    regime, efficiency = model.analyze_regimes()
    
    print(f"\nSingle case result:")
    print(f"  Regime: {regime}")
    print(f"  Expected efficiency: {efficiency:.1%}")
    
    print("\n" + "="*60)
    print("1. Comparing different k values...")
    compare_k_values()
    
    print("\n" + "="*60)
    print("2. Parameter sensitivity analysis...")
    


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


