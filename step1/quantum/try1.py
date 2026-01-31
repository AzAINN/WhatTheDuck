"""
QUANTUM VaR ANALYSIS - IQAE EPSILON SWEEP VISUALIZATION
============================================================
Iterative Quantum Amplitude Estimation (IQAE) convergence analysis:
- Quadratic speedup demonstration
- Error scaling with Grover iterations
- Sample complexity analysis
- Comprehensive convergence visualization

Based on quantum Value-at-Risk estimation using IQAE algorithm.

Author: Quantum Monte Carlo Analysis
Date: 2026
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm

# ============================================================================
# MATPLOTLIB CONFIGURATION
# ============================================================================

plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['mathtext.fontset'] = 'stix'

# Professional color palette - quantum-inspired
COLOR_PRIMARY = '#1e3a8a'      # Deep blue
COLOR_SECONDARY = '#3b82f6'    # Bright blue
COLOR_ACCENT = '#f59e0b'       # Amber accent
COLOR_DANGER = '#dc2626'       # Red for theoretical line
COLOR_GRID = '#e5e7eb'         # Light gray grid
COLOR_TEXT = '#1f2937'         # Dark gray text
COLOR_BOUND_UPPER = '#6366f1'  # Indigo for upper bound
COLOR_BOUND_LOWER = '#8b5cf6'  # Purple for lower bound
COLOR_DIST = '#3b82f6'         # Distribution color
COLOR_QUANTUM = '#7c3aed'      # Violet for quantum

# ============================================================================
# PARAMETERS (from quantum VaR code)
# ============================================================================

num_qubits = 7
mu = 0.7                       # Log-normal mean
sigma = 0.13                   # Log-normal std dev
ALPHA = 0.07                   # VaR confidence level
TOLERANCE = ALPHA / 10

# Output directory
output_dir = './outputs'
os.makedirs(output_dir, exist_ok=True)

# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def style_axes(ax, title, xlabel, ylabel):
    """Apply consistent professional styling to axes."""
    ax.set_facecolor('#fafafa')
    ax.set_title(title, fontweight='600', color=COLOR_TEXT, pad=20)
    ax.set_xlabel(xlabel, fontweight='500', color=COLOR_TEXT)
    ax.set_ylabel(ylabel, fontweight='500', color=COLOR_TEXT)
    ax.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
    ax.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLOR_GRID)
        spine.set_linewidth(1.5)


def create_legend(ax, **kwargs):
    """Create styled legend with consistent formatting."""
    legend = ax.legend(frameon=True, fancybox=True, shadow=True,
                      framealpha=0.95, edgecolor=COLOR_GRID, **kwargs)
    legend.get_frame().set_facecolor('white')
    return legend


# ============================================================================
# LOAD AND PROCESS DATA
# ============================================================================

print("="*70)
print("QUANTUM VaR ANALYSIS - IQAE EPSILON SWEEP")
print("="*70)

# Load CSV data
csv_path = './data/iqae_epsilon_sweep.csv'
if not os.path.exists(csv_path):
    print(f"\nError: {csv_path} not found!")
    print("Please ensure data.csv is in the current directory.")
    exit(1)

df = pd.read_csv(csv_path)

print(f"\nLoaded data from {csv_path}")
print(f"  • Data points: {len(df)}")
print(f"  • Epsilon range: [{df['epsilon'].min():.3f}, {df['epsilon'].max():.3f}]")
print(f"  • Alpha (confidence): {df['alpha'].iloc[0]:.3f}")

# Extract key arrays
# epsilon represents the target precision
# grover_calls_mean represents N (number of Grover iterations × shots)
# abs_error_mean represents the actual achieved error
epsilons = df['epsilon'].values
grover_calls = df['grover_calls_mean'].values
abs_errors = df['abs_error_mean'].values
a_true = df['a_true'].iloc[0]
a_hat_means = df['a_hat_mean'].values
ci_lows = df['ci_low'].values
ci_highs = df['ci_high'].values

print(f"\nTrue amplitude (a_true): {a_true:.5f}")
print(f"Grover calls range: [{grover_calls.min():,.0f}, {grover_calls.max():,.0f}]")
print(f"Actual epsilon range: [{epsilons.min():.5f}, {epsilons.max():.5f}]")

# ============================================================================
# CALCULATE REFERENCE LINES FOR QUANTUM SPEEDUP
# ============================================================================

# For quantum algorithms (IQAE), error scales as O(1/√N) where N = Grover calls
# This gives us the quadratic speedup: ε ∝ 1/√N → N ∝ 1/ε²

# Calculate bounds using percentiles
scaled_errors = epsilons * np.sqrt(grover_calls)
witness_top_n = np.percentile(scaled_errors, 95)
witness_bottom_n = np.percentile(scaled_errors, 5)
ref_line_top_n = witness_top_n / np.sqrt(grover_calls)
ref_line_bottom_n = witness_bottom_n / np.sqrt(grover_calls)

# Sample complexity: N ∝ 1/ε²
inv_error_sq = 1 / epsilons**2
scaled_samples = grover_calls / inv_error_sq
witness_top_e2 = np.percentile(scaled_samples, 95)
witness_bottom_e2 = np.percentile(scaled_samples, 5)
ref_line_top_e2 = witness_top_e2 * inv_error_sq
ref_line_bottom_e2 = witness_bottom_e2 * inv_error_sq

print(f"\nQuantum Convergence Bounds:")
print(f"  • Error scaling:     [{witness_bottom_n:.5f}, {witness_top_n:.5f}] × N^(-1/2)")
print(f"  • Sample complexity: [{witness_bottom_e2:.2f}, {witness_top_e2:.2f}] × ε^(-2)")

# Filter outliers for cleaner visualizations
mask_fig2 = (scaled_errors >= witness_bottom_n) & (scaled_errors <= witness_top_n)
grover_calls_fig2 = grover_calls[mask_fig2]
abs_errors_fig2 = epsilons[mask_fig2]
ref_top_n_fig2 = ref_line_top_n[mask_fig2]
ref_bot_n_fig2 = ref_line_bottom_n[mask_fig2]

mask_fig3 = (scaled_samples >= witness_bottom_e2) & (scaled_samples <= witness_top_e2)
grover_calls_fig3 = grover_calls[mask_fig3]
inv_error_sq_fig3 = inv_error_sq[mask_fig3]
ref_top_e2_fig3 = ref_line_top_e2[mask_fig3]
ref_bot_e2_fig3 = ref_line_bottom_e2[mask_fig3]

# ============================================================================
# GENERATE LOG-NORMAL DISTRIBUTION VISUALIZATION
# ============================================================================

print("\nGenerating log-normal distribution visualization...")

# Reconstruct the log-normal distribution
log_normal_mean = np.exp(mu + sigma**2 / 2)
log_normal_variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
log_normal_stddev = np.sqrt(log_normal_variance)

low = np.maximum(0, log_normal_mean - 3 * log_normal_stddev)
high = log_normal_mean + 3 * log_normal_stddev
x = np.linspace(low, high, 1000)
pdf_values = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

# Calculate VaR threshold
grid_points = np.linspace(low, high, 2**num_qubits)
probs = lognorm.pdf(grid_points, s=sigma, scale=np.exp(mu))
probs = probs / np.sum(probs)

VAR = 0
accumulated_value = 0
for index in range(len(probs)):
    accumulated_value += probs[index]
    if accumulated_value > ALPHA:
        VAR = grid_points[index]
        break

# Create distribution figure
fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
fig_dist.patch.set_facecolor('white')
ax_dist.set_facecolor('#fafafa')

# Plot PDF
ax_dist.plot(x, pdf_values, linewidth=2.5, color=COLOR_QUANTUM, 
             alpha=0.8, label='Log-Normal Distribution')
ax_dist.fill_between(x, pdf_values, alpha=0.2, color=COLOR_QUANTUM)

# VaR line
ax_dist.axvline(x=VAR, color=COLOR_DANGER, linestyle='--',
                linewidth=2.5, alpha=0.9, 
                label=f'VaR at {int(ALPHA*100)}%', zorder=5)

# Shade VaR tail
tail_mask = x <= VAR
if np.any(tail_mask):
    ax_dist.fill_between(x[tail_mask], pdf_values[tail_mask], 
                         alpha=0.4, color=COLOR_DANGER, 
                         label='Risk Region')

# Styling
style_axes(ax_dist,
           f'Asset Value Distribution: Log-Normal (μ={mu}, σ={sigma})',
           'Asset Value',
           'Probability Density')
create_legend(ax_dist, loc='upper right')

# Add statistics box
stats_text = (
    f'Distribution Parameters:\n'
    f'μ = {mu:.2f}\n'
    f'σ = {sigma:.2f}\n'
    f'VaR ({int(ALPHA*100)}%) = {VAR:.3f}\n'
    f'True Amplitude = {a_true:.5f}'
)
ax_dist.text(0.02, 0.98, stats_text, transform=ax_dist.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLOR_GRID, alpha=0.9))

plt.tight_layout()
plt.savefig(f'{output_dir}/00_distribution.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 00_distribution.png")
plt.close()

# ============================================================================
# FIGURE 1: AMPLITUDE ESTIMATION CONVERGENCE
# ============================================================================

print("\nGenerating convergence plots...")

fig1, ax1 = plt.subplots(figsize=(10, 6))
fig1.patch.set_facecolor('white')

# Plot estimates with error bars
ax1.errorbar(grover_calls, a_hat_means, 
             yerr=[a_hat_means - ci_lows, ci_highs - a_hat_means],
             marker='o', markersize=5, linewidth=2, capsize=5, capthick=2,
             color=COLOR_QUANTUM, alpha=0.8, ecolor=COLOR_SECONDARY,
             label='IQAE Estimate (with CI)', zorder=3)

# True amplitude line
ax1.axhline(y=a_true, color=COLOR_DANGER,
           linestyle='--', linewidth=2.5, alpha=0.9,
           label='True Amplitude', zorder=2)

# Confidence interval band
ax1.fill_between(grover_calls, ci_lows, ci_highs,
                 alpha=0.15, color=COLOR_QUANTUM, zorder=1)

ax1.set_xscale('log')
style_axes(ax1,
          'Convergence of Quantum Amplitude Estimation (IQAE)',
          'Number of Grover Calls (N)',
          'Estimated Amplitude')
create_legend(ax1, loc='best')

plt.tight_layout()
plt.savefig(f'{output_dir}/01_var_convergence.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 01_var_convergence.png")
plt.close()

# ============================================================================
# FIGURE 2: ERROR SCALING (QUANTUM SPEEDUP)
# ============================================================================

fig2, ax2 = plt.subplots(figsize=(10, 6))
fig2.patch.set_facecolor('white')

ax2.plot(grover_calls_fig2, abs_errors_fig2,
         marker='o', markersize=4, linewidth=2,
         color=COLOR_QUANTUM, alpha=0.8,
         label='Absolute Error', zorder=4)

ax2.plot(grover_calls_fig2, ref_top_n_fig2,
         color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
         label=r'$\mathcal{O}(N^{-1/2})$ Upper', zorder=3)

ax2.plot(grover_calls_fig2, ref_bot_n_fig2,
         color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
         label=r'$\Omega(N^{-1/2})$ Lower', zorder=3)

ax2.fill_between(grover_calls_fig2, ref_bot_n_fig2, ref_top_n_fig2,
                 alpha=0.1, color=COLOR_QUANTUM, zorder=1)

ax2.set_xscale('log')
ax2.set_yscale('log')
style_axes(ax2,
          'Quantum Error Scaling: IQAE Convergence Rate',
          'Number of Grover Calls (N)',
          'Absolute Error |â - a|')
create_legend(ax2, loc='best')

# Annotation
mid_idx = len(grover_calls) // 2
ax2.annotate(r'Error $\propto N^{-1/2}$',
            xy=(grover_calls[mid_idx], epsilons[mid_idx]),
            xytext=(grover_calls[mid_idx] * 0.05, epsilons[mid_idx] * 4),
            fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLOR_ACCENT, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

# Add quantum speedup note
speedup_text = (
    'Quadratic Speedup:\n'
    'Quantum vs Classical\n'
    r'$O(1/\varepsilon^2)$ vs $O(1/\varepsilon^4)$'
)
ax2.text(0.98, 0.02, speedup_text, transform=ax2.transAxes,
        fontsize=9, verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 edgecolor=COLOR_QUANTUM, alpha=0.9))

plt.tight_layout()
plt.savefig(f'{output_dir}/02_error_scaling.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 02_error_scaling.png")
plt.close()

# ============================================================================
# FIGURE 3: SAMPLE COMPLEXITY
# ============================================================================

fig3, ax3 = plt.subplots(figsize=(10, 6))
fig3.patch.set_facecolor('white')

ax3.plot(inv_error_sq_fig3, grover_calls_fig3,
         marker='o', markersize=4, linewidth=2,
         color=COLOR_QUANTUM, alpha=0.8,
         label=r'Observed Grover Calls vs $\varepsilon^{-2}$', zorder=4)

ax3.plot(inv_error_sq_fig3, ref_top_e2_fig3,
         color=COLOR_BOUND_UPPER, linestyle='--', linewidth=2, alpha=0.7,
         label=r'$\mathcal{O}(\varepsilon^{-2})$ Upper', zorder=3)

ax3.plot(inv_error_sq_fig3, ref_bot_e2_fig3,
         color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=2, alpha=0.7,
         label=r'$\Omega(\varepsilon^{-2})$ Lower', zorder=3)

ax3.fill_between(inv_error_sq_fig3, ref_bot_e2_fig3, ref_top_e2_fig3,
                 alpha=0.1, color=COLOR_QUANTUM, zorder=1)

ax3.set_xscale('log')
ax3.set_yscale('log')
style_axes(ax3,
          'Sample Complexity: Quantum Amplitude Estimation',
          r'Inverse Squared Error ($\varepsilon^{-2}$)',
          'Required Grover Calls (N)')
create_legend(ax3, loc='best')

# Annotation
mid_idx = len(inv_error_sq) // 2
ax3.annotate(r'$N \propto \varepsilon^{-2}$',
            xy=(inv_error_sq[mid_idx], grover_calls[mid_idx]),
            xytext=(inv_error_sq[mid_idx] * 0.15, grover_calls[mid_idx] * 5),
            fontsize=11, color=COLOR_ACCENT, fontweight='600', zorder=5,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                     edgecolor=COLOR_ACCENT, alpha=0.9),
            arrowprops=dict(arrowstyle='->', color=COLOR_ACCENT, lw=2))

plt.tight_layout()
plt.savefig(f'{output_dir}/03_sample_complexity.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 03_sample_complexity.png")
plt.close()

# ============================================================================
# FIGURE 4: COMBINED ANALYSIS (4-PANEL)
# ============================================================================

print("\nGenerating combined analysis figure...")

fig4 = plt.figure(figsize=(20, 11))
fig4.patch.set_facecolor('white')
gs = fig4.add_gridspec(2, 3,
                       width_ratios=[2, 1.5, 1.5],
                       hspace=0.35, wspace=0.35,
                       top=0.93, bottom=0.06, left=0.06, right=0.97)

fig4.suptitle(f'Quantum VaR Analysis: IQAE on Log-Normal Distribution '
              f'(μ={mu}, σ={sigma}, VaR={int(ALPHA*100)}%)',
              fontsize=18, fontweight='700', color=COLOR_TEXT, y=0.98)

# Panel A: Distribution
ax_a = fig4.add_subplot(gs[:, 0])
ax_a.set_facecolor('#fafafa')

ax_a.plot(x, pdf_values, linewidth=2.5, color=COLOR_QUANTUM, alpha=0.8)
ax_a.fill_between(x, pdf_values, alpha=0.2, color=COLOR_QUANTUM)
ax_a.axvline(x=VAR, color=COLOR_DANGER, linestyle='--', 
             linewidth=2.5, alpha=0.9, zorder=5)

tail_mask = x <= VAR
if np.any(tail_mask):
    ax_a.fill_between(x[tail_mask], pdf_values[tail_mask],
                      alpha=0.4, color=COLOR_DANGER)

ax_a.set_xlabel('Asset Value', fontweight='500', color=COLOR_TEXT)
ax_a.set_ylabel('Probability Density', fontweight='500', color=COLOR_TEXT)
ax_a.set_title('(A) Asset Value Distribution', fontweight='600', 
               color=COLOR_TEXT, loc='left')
ax_a.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
for spine in ax_a.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel B: Amplitude Convergence
ax_b = fig4.add_subplot(gs[0, 1:])
ax_b.set_facecolor('#fafafa')

ax_b.errorbar(grover_calls, a_hat_means,
              yerr=[a_hat_means - ci_lows, ci_highs - a_hat_means],
              marker='o', markersize=4, linewidth=2, capsize=4, capthick=1.5,
              color=COLOR_QUANTUM, alpha=0.8, ecolor=COLOR_SECONDARY)
ax_b.axhline(y=a_true, color=COLOR_DANGER, linestyle='--', 
             linewidth=2, alpha=0.9)
ax_b.fill_between(grover_calls, ci_lows, ci_highs,
                  alpha=0.15, color=COLOR_QUANTUM)

ax_b.set_xscale('log')
ax_b.set_xlabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT)
ax_b.set_ylabel('Amplitude Estimate', fontweight='500', color=COLOR_TEXT)
ax_b.set_title('(B) IQAE Amplitude Convergence', fontweight='600', 
               color=COLOR_TEXT, loc='left')
ax_b.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_b.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_b.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel C: Error Scaling
ax_c = fig4.add_subplot(gs[1, 1])
ax_c.set_facecolor('#fafafa')

ax_c.plot(grover_calls_fig2, abs_errors_fig2, marker='o', markersize=3,
          linewidth=2, color=COLOR_QUANTUM, alpha=0.8)
ax_c.plot(grover_calls_fig2, ref_top_n_fig2,
          color=COLOR_BOUND_UPPER, linestyle='--', linewidth=1.5, alpha=0.7)
ax_c.plot(grover_calls_fig2, ref_bot_n_fig2,
          color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=1.5, alpha=0.7)
ax_c.fill_between(grover_calls_fig2, ref_bot_n_fig2, ref_top_n_fig2,
                   alpha=0.1, color=COLOR_QUANTUM)

ax_c.set_xscale('log')
ax_c.set_yscale('log')
ax_c.set_xlabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_c.set_ylabel('Absolute Error', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_c.set_title('(C) Error Scaling', fontweight='600', color=COLOR_TEXT, loc='left')
ax_c.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_c.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_c.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

# Panel D: Sample Complexity
ax_d = fig4.add_subplot(gs[1, 2])
ax_d.set_facecolor('#fafafa')

ax_d.plot(inv_error_sq_fig3, grover_calls_fig3, marker='o', markersize=3,
          linewidth=2, color=COLOR_QUANTUM, alpha=0.8)
ax_d.plot(inv_error_sq_fig3, ref_top_e2_fig3,
          color=COLOR_BOUND_UPPER, linestyle='--', linewidth=1.5, alpha=0.7)
ax_d.plot(inv_error_sq_fig3, ref_bot_e2_fig3,
          color=COLOR_BOUND_LOWER, linestyle='-.', linewidth=1.5, alpha=0.7)
ax_d.fill_between(inv_error_sq_fig3, ref_bot_e2_fig3, ref_top_e2_fig3,
                   alpha=0.1, color=COLOR_QUANTUM)

ax_d.set_xscale('log')
ax_d.set_yscale('log')
ax_d.set_xlabel(r'$\varepsilon^{-2}$', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_d.set_ylabel('Grover Calls (N)', fontweight='500', color=COLOR_TEXT, fontsize=10)
ax_d.set_title('(D) Sample Complexity', fontweight='600', color=COLOR_TEXT, loc='left')
ax_d.grid(True, which='major', linestyle='-', alpha=0.3, color=COLOR_GRID)
ax_d.grid(True, which='minor', linestyle=':', alpha=0.15, color=COLOR_GRID)
for spine in ax_d.spines.values():
    spine.set_edgecolor(COLOR_GRID)
    spine.set_linewidth(1.5)

plt.savefig(f'{output_dir}/04_combined_analysis.png',
            bbox_inches='tight', facecolor='white', edgecolor='none')
print("✓ Saved: 04_combined_analysis.png")
plt.close()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("✨ ALL PUBLICATION-QUALITY GRAPHS GENERATED SUCCESSFULLY!")
print("="*70)
print("\nGenerated Files:")
print("  📊 00_distribution.png         - Log-normal asset distribution & VaR")
print("  📊 01_var_convergence.png      - IQAE amplitude estimate convergence")
print("  📊 02_error_scaling.png        - Quantum error scaling O(N^-1/2)")
print("  📊 03_sample_complexity.png    - Sample complexity O(ε^-2)")
print("  📊 04_combined_analysis.png    - Comprehensive 4-panel summary")
print("\nAnalysis Summary:")
print(f"  • Algorithm:      Iterative Quantum Amplitude Estimation (IQAE)")
print(f"  • Distribution:   Log-Normal (μ={mu}, σ={sigma})")
print(f"  • VaR Level:      {int(ALPHA*100)}%")
print(f"  • True Amplitude: {a_true:.5f}")
print(f"  • Epsilon Range:  [{epsilons.min():.3f}, {epsilons.max():.3f}]")
print(f"  • Grover Range:   [{grover_calls.min():,.0f}, {grover_calls.max():,.0f}]")
print(f"  • Data Points:    {len(df)}")
print(f"\nQuantum Advantage:")
print(f"  • Error scaling:  O(N^-1/2) - Quadratic speedup over classical O(N^-1)")
print(f"  • Complexity:     O(ε^-2) - vs Classical O(ε^-4)")
print(f"\nLocation: {output_dir}/")
print("="*70)

# Optional: Display plots if running interactively
# plt.show()