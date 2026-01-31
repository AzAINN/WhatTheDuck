import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters ------------------------------------------------------------------#
# np.random.seed(42)  # reproducibility

mu = 0.7      # mean daily return (0.1%)
sigma = 0.13  # daily volatility (2%)
confidence_level = 0.95
num_samples_list = [n for n in range(10, 10**8 + 1, 100)]


# simulate_returns - a --------------------------------------------------------#			
def simulate_returns(mu, sigma, num_samples):
    """
    Simulate daily returns from a Gaussian distribution.
    """
    return np.random.normal(mu, sigma, num_samples)

# compute_var - b -------------------------------------------------------------#
def compute_var(returns, confidence_level):
    """
    Compute classical Monte Carlo Value at Risk (VaR) at given confidence level.
    VaR is defined as the alpha-quantile of losses (positive number).
    """
    losses = -returns  # convert returns to losses
    var = np.quantile(losses, confidence_level)
    return var

# Run monte carlo - c ---------------------------------------------------------#
mc_results = []
errors = []

# Theoretical VaR for Gaussian
theoretical_var = - (mu + sigma * norm.ppf(1 - confidence_level))
print(f"Theoretical VaR({int(confidence_level*100)}%) = {theoretical_var:.5f}")

for N in num_samples_list:
    returns = simulate_returns(mu, sigma, N)
    var_estimate = compute_var(returns, confidence_level)
    mc_results.append(var_estimate)
    
    # Monte Carlo error: |estimate - theory|
    error = abs(var_estimate - theoretical_var)
    errors.append(error)
    
    print(f"Samples: {N:>7}, VaR ≈ {var_estimate:.5f}, Error ≈ {error:.5f}")

# Plot convergence and demonstrate O(1/ε²) scaling - d ------------------------#
plt.figure(figsize=(12,5))

# Plot 1: Convergence of VaR estimate
plt.subplot(1,2,1)
plt.plot(num_samples_list, mc_results, marker='o', label='Monte Carlo VaR estimate')
plt.axhline(y=theoretical_var, color='r', linestyle='--', label='Theoretical VaR')
plt.xscale('log')
plt.xlabel("Number of Samples (log scale)")
plt.ylabel(f"VaR at {int(confidence_level*100)}% confidence")
plt.title("Convergence of Classical Monte Carlo VaR")
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()

# Plot 2: Error scaling vs samples
plt.subplot(1,2,2)
plt.plot(num_samples_list, errors, marker='o', label='|VaR estimate - Theory|')
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Number of Samples (log scale)")
plt.ylabel("Absolute Error")
plt.title("Monte Carlo Error Scaling")

# Reference line: O(1/sqrt(N)) relation
ref_line_sqrt = errors[0] * np.sqrt(num_samples_list[0]) / np.sqrt(np.array(num_samples_list))
plt.plot(num_samples_list, ref_line_sqrt, 'k--', label=r'O(1/$\sqrt{N}$)')

# Reference line: samples vs precision ε (O(1/ε²))
# eps_list = np.array(errors)
# ref_line_eps = num_samples_list[0] * (eps_list[0] / eps_list)**2
# plt.plot(num_samples_list, eps_list, 'o', alpha=0)  # just to align data points
# plt.plot(num_samples_list, ref_line_eps, 'g--', label=r'O(1/$\varepsilon^2$)')

plt.grid(True, which='both', ls='--', alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()
