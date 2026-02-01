import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from pathlib import Path

def run_classical_monte_carlo(mu=0.15, sigma=0.20, confidence_level=0.95, samples_list=None):
    if samples_list is None:
        samples_list = [100, 1000, 10000, 100000, 1000000]
    
    # Theoretical VaR for a Gaussian (Analytical Baseline)
    # VaR_alpha = mu + sigma * stats.norm.ppf(1 - alpha)
    # Note: For P&L, VaR is typically the threshold where P&L <= threshold
    alpha = 1 - confidence_level
    theoretical_var = stats.norm.ppf(alpha, loc=mu, scale=sigma)
    
    results = []
    print(f"Theoretical VaR ({confidence_level*100}%): {theoretical_var:.6f}")
    
    for n in samples_list:
        # Generate random samples from Gaussian distribution
        samples = np.random.normal(mu, sigma, n)
        
        # Estimate VaR: the (1-alpha) percentile of the loss distribution
        # or simply the alpha quantile of the P&L distribution
        estimated_var = np.quantile(samples, alpha)
        error = abs(estimated_var - theoretical_var)
        
        results.append({
            "samples": n,
            "estimated_var": estimated_var,
            "error": error
        })
        print(f"Samples: {n:7} | Estimated: {estimated_var:.6f} | Error: {error:.6f}")
        
    return results, theoretical_var

# Run and Plot
mc_results, true_val = run_classical_monte_carlo()