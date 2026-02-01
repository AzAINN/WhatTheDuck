import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_comparative_scaling(out_dir="results2"):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Classical Scaling (Theoretical O(1/sqrt(N)))
    # Error epsilon ~ 1/sqrt(Samples)
    n_classical = np.geomspace(100, 1e6, 10)
    err_classical = 1 / np.sqrt(n_classical)

    # 2. Quantum Scaling (Theoretical O(1/N))
    # Error epsilon ~ 1/Oracle_Queries
    n_quantum = np.geomspace(100, 1e4, 10)
    err_quantum = 1 / n_quantum

    plt.figure(figsize=(10, 6))
    
    # Classical Plot
    plt.loglog(n_classical, err_classical, 'o-', color='tab:blue', label='Classical Monte Carlo $O(1/\sqrt{N})$')
    
    # Quantum Plot
    plt.loglog(n_quantum, err_quantum, 's-', color='tab:orange', label='Quantum IQAE $O(1/N)$')

    # Add reference slopes
    plt.loglog(n_classical, 0.5/np.sqrt(n_classical), '--', color='gray', alpha=0.5, label='Slope -1/2 (Classical)')
    plt.loglog(n_quantum, 0.5/n_quantum, ':', color='gray', alpha=0.5, label='Slope -1 (Quantum)')

    plt.title("Benchmarking: Classical vs. Quantum Scaling", fontsize=14)
    plt.xlabel("Number of Samples / Oracle Queries (N)", fontsize=12)
    plt.ylabel("Estimation Error ($\epsilon$)", fontsize=12)
    plt.legend()
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    plt.savefig(out_path / "classical_vs_quantum_scaling.png", dpi=200)
    plt.show()

plot_comparative_scaling()