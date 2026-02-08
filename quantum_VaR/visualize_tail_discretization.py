"""
Visualize tail-focused discretization vs uniform discretization.

Shows how bins are allocated to concentrate resolution in the tail region
where VaR calculations are most sensitive.
"""

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Same parameters as VaR_Quantum.py
NUM_QUBITS = 7
MU = 0.15
SIGMA = 0.20
STUDENT_T_DF = 4
SKEW_NC = -1.0
ALPHA = 0.05

GRID_LO = MU - 3 * SIGMA
GRID_HI = MU + 3 * SIGMA


def build_uniform_pmf_from_dist(dist, num_qubits, lo, hi):
    """Uniform grid discretization."""
    n = 2 ** num_qubits
    grid = np.linspace(lo, hi, n)
    pdf = dist.pdf(grid)
    probs = (pdf / pdf.sum()).tolist()
    return grid, probs


def build_tail_focused_pmf_from_dist(
    dist,
    num_qubits: int,
    tail_alpha: float = 0.01,
    tail_mass: float = 0.30,
    tail_bin_frac: float = 0.70,
    clip_mass: float = 1e-6,
    pmf_mode: str = "cdf_diff",
):
    """Non-uniform tail-focused discretization."""
    N = 2 ** num_qubits

    # Split bins: more bins for tail region, fewer for the rest
    N_tail = int(np.round(N * tail_bin_frac))
    N_tail = max(4, min(N - 4, N_tail))
    N_body = N - N_tail

    # Define quantile ranges in CDF space
    u_tail = np.linspace(clip_mass, tail_mass, N_tail, endpoint=False)
    u_body = np.linspace(tail_mass, 1.0 - clip_mass, N_body)

    u = np.concatenate([u_tail, u_body])
    u = np.clip(u, clip_mass, 1.0 - clip_mass)

    # Grid points at those quantiles
    x = dist.ppf(u)

    # Ensure monotonic and finite
    x = np.nan_to_num(x, neginf=dist.ppf(clip_mass), posinf=dist.ppf(1.0 - clip_mass))
    x = np.sort(x)

    # Build probabilities on this nonuniform grid
    if pmf_mode == "cdf_diff":
        edges = np.empty(N + 1)
        edges[1:-1] = 0.5 * (x[1:] + x[:-1])
        edges[0] = dist.ppf(clip_mass)
        edges[-1] = dist.ppf(1.0 - clip_mass)

        cdf_edges = dist.cdf(edges)
        probs = np.diff(cdf_edges)
        probs = np.maximum(probs, 0.0)
        probs = probs / probs.sum()
        return x, probs.tolist(), edges

    else:
        raise ValueError("Only cdf_diff supported in this visualization")


def plot_tail_discretization_comparison():
    """Create comprehensive visualization of tail-focused discretization."""

    # Build distribution (heavy-tailed, skewed Student-t)
    dist = scipy.stats.nct(df=STUDENT_T_DF, nc=SKEW_NC, loc=MU, scale=SIGMA)

    # Get uniform discretization
    grid_uniform, probs_uniform = build_uniform_pmf_from_dist(
        dist, NUM_QUBITS, GRID_LO, GRID_HI
    )

    # Get tail-focused discretization
    grid_tail, probs_tail, edges_tail = build_tail_focused_pmf_from_dist(
        dist,
        num_qubits=NUM_QUBITS,
        tail_alpha=ALPHA,
        tail_mass=0.30,
        tail_bin_frac=0.70,
        pmf_mode="cdf_diff",
    )

    # Create figure with 4 subplots
    fig = plt.figure(figsize=(16, 12))

    # ---------- Plot 1: Full distribution with bin markers ----------
    ax1 = fig.add_subplot(2, 2, 1)

    # Continuous PDF
    x_cont = np.linspace(GRID_LO, GRID_HI, 500)
    pdf_cont = dist.pdf(x_cont)
    ax1.plot(x_cont, pdf_cont, 'k-', linewidth=2, label='True PDF', alpha=0.7)

    # Uniform bins (vertical lines)
    for i, gx in enumerate(grid_uniform[::4]):  # Every 4th for clarity
        ax1.axvline(gx, color='blue', alpha=0.3, linewidth=1, linestyle='--')

    # Tail-focused bins (vertical lines)
    for i, gx in enumerate(grid_tail[::4]):
        ax1.axvline(gx, color='red', alpha=0.3, linewidth=1, linestyle='-')

    ax1.axvline(0, color='blue', alpha=0.5, linewidth=2, linestyle='--', label='Uniform bins')
    ax1.axvline(0, color='red', alpha=0.5, linewidth=2, linestyle='-', label='Tail-focused bins')

    ax1.set_xlabel('P&L Return', fontsize=12)
    ax1.set_ylabel('Probability Density', fontsize=12)
    ax1.set_title('Distribution with Discretization Bins (every 4th shown)', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # ---------- Plot 2: Zoomed tail region ----------
    ax2 = fig.add_subplot(2, 2, 2)

    # Find tail region (left 30% CDF mass)
    tail_threshold = dist.ppf(0.30)
    x_tail = np.linspace(GRID_LO, tail_threshold, 500)
    pdf_tail = dist.pdf(x_tail)

    ax2.fill_between(x_tail, 0, pdf_tail, color='lightcoral', alpha=0.3, label='Tail region (30% mass)')
    ax2.plot(x_tail, pdf_tail, 'k-', linewidth=2, alpha=0.7)

    # Mark uniform bins in tail
    uniform_in_tail = [g for g in grid_uniform if g <= tail_threshold]
    for gx in uniform_in_tail:
        ax2.axvline(gx, color='blue', alpha=0.5, linewidth=1.5, linestyle='--')

    # Mark tail-focused bins in tail
    tail_in_tail = [g for g in grid_tail if g <= tail_threshold]
    for gx in tail_in_tail:
        ax2.axvline(gx, color='red', alpha=0.5, linewidth=1.5, linestyle='-')

    ax2.axvline(0, color='blue', alpha=0.7, linewidth=2, linestyle='--',
                label=f'Uniform ({len(uniform_in_tail)} bins)')
    ax2.axvline(0, color='red', alpha=0.7, linewidth=2, linestyle='-',
                label=f'Tail-focused ({len(tail_in_tail)} bins)')

    ax2.set_xlabel('P&L Return', fontsize=12)
    ax2.set_ylabel('Probability Density', fontsize=12)
    ax2.set_title('ZOOMED: Left Tail (30% CDF mass)', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # ---------- Plot 3: Bin spacing analysis ----------
    ax3 = fig.add_subplot(2, 2, 3)

    # Compute bin widths
    dx_uniform = np.diff(grid_uniform)
    dx_tail = np.diff(grid_tail)

    # Plot bin widths vs position
    ax3.semilogy(grid_uniform[:-1], dx_uniform, 'o-', color='blue',
                 alpha=0.6, label='Uniform', markersize=4)
    ax3.semilogy(grid_tail[:-1], dx_tail, 's-', color='red',
                 alpha=0.6, label='Tail-focused', markersize=4)

    ax3.axvline(tail_threshold, color='green', linestyle=':', linewidth=2,
                label=f'Tail boundary (30% CDF)')

    ax3.set_xlabel('P&L Return', fontsize=12)
    ax3.set_ylabel('Bin Width (log scale)', fontsize=12)
    ax3.set_title('Bin Width vs Position', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # ---------- Plot 4: CDF and VaR location ----------
    ax4 = fig.add_subplot(2, 2, 4)

    # CDF
    x_cdf = np.linspace(GRID_LO, GRID_HI, 1000)
    cdf = dist.cdf(x_cdf)
    ax4.plot(x_cdf, cdf, 'k-', linewidth=2, label='True CDF')

    # Empirical CDF from discretizations
    cdf_uniform = np.cumsum(probs_uniform)
    cdf_tail = np.cumsum(probs_tail)

    ax4.step(grid_uniform, cdf_uniform, where='post', color='blue',
             alpha=0.6, linewidth=1.5, label='Uniform discretization')
    ax4.step(grid_tail, cdf_tail, where='post', color='red',
             alpha=0.6, linewidth=1.5, label='Tail-focused discretization')

    # Mark VaR level
    var_true = dist.ppf(ALPHA)
    ax4.axhline(ALPHA, color='green', linestyle='--', linewidth=2,
                label=f'VaR level (α={ALPHA})')
    ax4.axvline(var_true, color='green', linestyle=':', linewidth=2, alpha=0.7)

    # Mark tail region
    ax4.axvspan(GRID_LO, tail_threshold, color='lightcoral', alpha=0.2)

    ax4.set_xlabel('P&L Return', fontsize=12)
    ax4.set_ylabel('Cumulative Probability', fontsize=12)
    ax4.set_title('CDF Comparison & VaR Location', fontsize=14, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig('results/tail_discretization_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/tail_discretization_comparison.png")

    # ---------- Additional plot: Histogram-style comparison ----------
    fig2, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Continuous PDF for reference
    ax_a.plot(x_cont, pdf_cont, 'k-', linewidth=2, label='True PDF', zorder=10)
    ax_b.plot(x_cont, pdf_cont, 'k-', linewidth=2, label='True PDF', zorder=10)

    # Uniform discretization as histogram
    bin_edges_uniform = np.empty(len(grid_uniform) + 1)
    bin_edges_uniform[1:-1] = 0.5 * (grid_uniform[1:] + grid_uniform[:-1])
    bin_edges_uniform[0] = GRID_LO
    bin_edges_uniform[-1] = GRID_HI

    heights_uniform = np.array(probs_uniform) / np.diff(bin_edges_uniform)
    for i in range(len(grid_uniform)):
        rect = Rectangle(
            (bin_edges_uniform[i], 0),
            bin_edges_uniform[i+1] - bin_edges_uniform[i],
            heights_uniform[i],
            facecolor='blue', edgecolor='darkblue', alpha=0.4
        )
        ax_a.add_patch(rect)

    # Tail-focused discretization as histogram
    heights_tail = np.array(probs_tail) / np.diff(edges_tail)
    for i in range(len(grid_tail)):
        rect = Rectangle(
            (edges_tail[i], 0),
            edges_tail[i+1] - edges_tail[i],
            heights_tail[i],
            facecolor='red', edgecolor='darkred', alpha=0.4
        )
        ax_b.add_patch(rect)

    ax_a.set_ylabel('Probability Density', fontsize=12)
    ax_a.set_title('UNIFORM Discretization (equal bin widths)',
                   fontsize=14, fontweight='bold')
    ax_a.legend(fontsize=10)
    ax_a.grid(True, alpha=0.3)
    ax_a.set_xlim([GRID_LO, GRID_HI])

    ax_b.set_xlabel('P&L Return', fontsize=12)
    ax_b.set_ylabel('Probability Density', fontsize=12)
    ax_b.set_title('TAIL-FOCUSED Discretization (70% bins in left 30% mass)',
                   fontsize=14, fontweight='bold')
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3)
    ax_b.set_xlim([GRID_LO, GRID_HI])

    # Mark tail region on both
    ax_a.axvspan(GRID_LO, tail_threshold, color='yellow', alpha=0.2,
                 label='Tail region (30% CDF)')
    ax_b.axvspan(GRID_LO, tail_threshold, color='yellow', alpha=0.2,
                 label='Tail region (30% CDF)')

    plt.tight_layout()
    plt.savefig('results/tail_histogram_comparison.png', dpi=300, bbox_inches='tight')
    print("Saved: results/tail_histogram_comparison.png")
    plt.close('all')  # Close all figures instead of showing

    # Print statistics
    print("\n" + "="*60)
    print("DISCRETIZATION STATISTICS")
    print("="*60)
    print(f"Total bins: {2**NUM_QUBITS}")
    print(f"Tail region: up to {tail_threshold:.4f} (30% CDF mass)")
    print(f"\nUNIFORM:")
    print(f"  Bins in tail: {len(uniform_in_tail)} ({100*len(uniform_in_tail)/len(grid_uniform):.1f}%)")
    print(f"  Mean bin width in tail: {np.mean([dx_uniform[i] for i in range(len(uniform_in_tail)-1)]):.6f}")
    print(f"\nTAIL-FOCUSED:")
    print(f"  Bins in tail: {len(tail_in_tail)} ({100*len(tail_in_tail)/len(grid_tail):.1f}%)")
    print(f"  Mean bin width in tail: {np.mean([dx_tail[i] for i in range(len(tail_in_tail)-1)]):.6f}")
    print(f"\nResolution improvement in tail: {len(tail_in_tail)/len(uniform_in_tail):.2f}x more bins")
    print("="*60)


if __name__ == "__main__":
    plot_tail_discretization_comparison()
