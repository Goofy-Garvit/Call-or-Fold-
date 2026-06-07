import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

# Parameters
N = 1000   # true population
T = 100    # tagged animals
n = 150    # second capture size
num_sims = 10000

# Simulation
rng = np.random.default_rng(42)
population = np.array([1]*T + [0]*(N-T))  # 1 = tagged, 0 = untagged

k_simulated = np.array([
    rng.choice(population, size=n, replace=False).sum()
    for _ in range(num_sims)
])

# Theoretical PMF (Hypergeometric)
k_min = max(0, n + T - N)
k_max = min(n, T)
k_vals = np.arange(k_min, k_max + 1)

pmf_theoretical = np.array([
    comb(T, k, exact=True) * comb(N - T, n - k, exact=True) / comb(N, n, exact=True)
    for k in k_vals
])

# Empirical PMF
k_counts = np.bincount(k_simulated, minlength=k_max+2)
k_all = np.arange(len(k_counts))
k_probs = k_counts / num_sims

# Plot range
plot_range = np.where(k_probs > 1e-4)[0]
lo, hi = plot_range[0], plot_range[-1]

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(k_all[lo:hi+1], k_probs[lo:hi+1],
       color='steelblue', alpha=0.65, label='Empirical PMF (10,000 simulations)', width=0.6)
ax.plot(k_vals, pmf_theoretical, 'ro-', markersize=5, linewidth=1.8,
        label='Theoretical PMF (Hypergeometric)')

mean_sim = k_simulated.mean()
mean_theo = n * T / N
ax.axvline(mean_sim, color='steelblue', linestyle='--', alpha=0.8, label=f'Sim mean = {mean_sim:.2f}')
ax.axvline(mean_theo, color='red', linestyle='--', alpha=0.8, label=f'Theo mean = {mean_theo:.2f}')

ax.set_xlabel('k  (tagged animals in second capture)', fontsize=13)
ax.set_ylabel('Probability', fontsize=13)
ax.set_title('Capture-Recapture: Empirical vs Theoretical PMF\n'
             f'N={N}, T={T}, n={n}', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('capture_recapture.png', dpi=150)
print(f"Simulation mean:    {mean_sim:.4f}")
print(f"Theoretical mean:   {mean_theo:.4f}")
print(f"Max PMF deviation:  {max(abs(k_probs[k_vals] - pmf_theoretical)):.5f}")
