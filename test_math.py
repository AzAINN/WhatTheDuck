"""
Mathematical verification tests for VaR_Quantum_highD.py

Run: python test_math.py

Tests each mathematical component against known analytical results
to verify correctness before running on GPU clusters.
"""

import numpy as np
import scipy.stats

# Import from the main module
from quantum_VaR.VaR_Quantum_highD import (
    FactorModel,
    _equicorrelated_normal,
    build_analytic_student_t_pmf,
    build_tail_aware_pmf,
    build_pmf_from_samples,
    classical_var,
    mc_warm_start_from_losses,
)

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


# =========================================================================
# Test 1: Equicorrelated normal — covariance structure
# =========================================================================
print("=" * 60)
print("Test 1: Equicorrelated normal sampling")
print("=" * 60)

rng = np.random.default_rng(0)
d, N = 5, 500_000
rho = 0.6
G = _equicorrelated_normal(rng, N, d, rho, backend="numpy")

# E[G_i] = 0
means = G.mean(axis=0)
check("E[G_i] ~= 0", np.allclose(means, 0, atol=0.02),
      f"max |mean| = {np.abs(means).max():.4f}")

# Var[G_i] = (1-rho) + rho = 1
variances = G.var(axis=0)
check("Var[G_i] ~= 1", np.allclose(variances, 1.0, atol=0.02),
      f"variances = {variances}")

# Cov[G_i, G_j] = rho for i != j
cov_matrix = np.cov(G.T)
off_diag = cov_matrix[np.triu_indices(d, k=1)]
check(f"Cov[G_i, G_j] ~= {rho} (off-diag)",
      np.allclose(off_diag, rho, atol=0.02),
      f"mean off-diag = {off_diag.mean():.4f}")

diag = np.diag(cov_matrix)
check("Cov[G_i, G_i] ~= 1 (diagonal)",
      np.allclose(diag, 1.0, atol=0.02),
      f"diag = {diag}")


# =========================================================================
# Test 2: Student-t sampling — moments match theory
# =========================================================================
print()
print("=" * 60)
print("Test 2: 1D Student-t sampling (numpy fallback)")
print("=" * 60)

df = 4
rng2 = np.random.default_rng(42)
# Manually replicate what gpu_sample_1d_student_t does (numpy path)
N2 = 1_000_000
G2 = rng2.standard_normal(N2)
U2 = rng2.chisquare(df, size=N2)
samples_t = G2 * np.sqrt(df / U2)

# E[t] = 0 for df > 1
check("E[t(4)] ~= 0", abs(samples_t.mean()) < 0.01,
      f"mean = {samples_t.mean():.4f}")

# Var[t(df)] = df/(df-2) for df > 2
expected_var = df / (df - 2)  # = 2.0
check(f"Var[t(4)] ~= {expected_var:.1f}",
      abs(samples_t.var() - expected_var) < 0.05,
      f"var = {samples_t.var():.4f}, expected = {expected_var}")

# Kurtosis of t(df) = 6/(df-4) for df > 4; for df=4 it's infinite
# But excess kurtosis should be large (>> 0, fatter than Gaussian)
kurt = scipy.stats.kurtosis(samples_t)
check("Excess kurtosis >> 0 (heavy tails)", kurt > 3.0,
      f"kurtosis = {kurt:.2f}")

# Check quantile matches scipy.stats.t.ppf
alpha = 0.05
mc_var = np.percentile(samples_t, alpha * 100)
analytic_var = scipy.stats.t.ppf(alpha, df=df)
check(f"MC 5th percentile ~= analytic ppf ({analytic_var:.4f})",
      abs(mc_var - analytic_var) < 0.05,
      f"MC = {mc_var:.4f}, analytic = {analytic_var:.4f}")


# =========================================================================
# Test 3: Multivariate Student-t via factor model — marginal is Student-t
# =========================================================================
print()
print("=" * 60)
print("Test 3: Factor model — identity B, single asset")
print("=" * 60)

# d=1, M=1, B=[[1]], w=[1] => loss = -Z where Z ~ Student-t(df)
# So loss distribution should be Student-t(df) (negated)
model_trivial = FactorModel(d=1, M=1, df=4,
                             B=np.array([[1.0]]),
                             w=np.array([1.0]))

rng3 = np.random.default_rng(99)
N3 = 500_000
G3 = rng3.standard_normal((N3, 1))
U3 = rng3.chisquare(df, size=(N3, 1))
Z3 = G3 * np.sqrt(df / U3)
r3 = Z3 @ model_trivial.B.T   # (N, 1)
P3 = r3 @ model_trivial.w     # (N,)
L3 = -P3                       # losses

# L = -Z ~ Student-t(4) (negated), so VaR at alpha=0.05 of L
# = -ppf(1-alpha, df) = ppf(alpha, df) ... wait:
# P(L <= x) = P(-Z <= x) = P(Z >= -x) = 1 - F_t(-x) = F_t(x) by symmetry
# So quantile of L at alpha is same as ppf(alpha, df) = negative number
# Actually: L = -Z, so P(L <= x) = P(Z >= -x).
# For symmetric t: P(Z >= -x) = P(Z <= x) = F_t(x).
# So the alpha-quantile of L = ppf(alpha, df).
mc_var_L = np.percentile(L3, alpha * 100)
analytic_var_L = scipy.stats.t.ppf(alpha, df=df)
check(f"Trivial model: MC VaR ~= analytic ({analytic_var_L:.4f})",
      abs(mc_var_L - analytic_var_L) < 0.05,
      f"MC = {mc_var_L:.4f}")

# Variance of L should be df/(df-2)
check(f"Trivial model: Var[L] ~= {expected_var:.1f}",
      abs(L3.var() - expected_var) < 0.1,
      f"var = {L3.var():.4f}")


# =========================================================================
# Test 4: Factor model — equal weight portfolio variance
# =========================================================================
print()
print("=" * 60)
print("Test 4: Factor model — equal weight portfolio, rho=0")
print("=" * 60)

# With rho=0, factors are independent.
# d=M, B=I (identity), w=1/M.
# P = (1/M) * sum(Z_i) where Z_i ~ iid t(df)
# For N iid t(df): Var[sum/M] = Var[Z]/M = (df/(df-2))/M
M_test = 10
d_test = M_test
model_id = FactorModel(d=d_test, M=M_test, df=4,
                        B=np.eye(M_test),
                        w=np.full(M_test, 1.0 / M_test))

rng4 = np.random.default_rng(7)
N4 = 500_000
# rho=0 => factors are independent
G4 = rng4.standard_normal((N4, d_test))  # no correlation
U4 = rng4.chisquare(df, size=(N4, 1))
Z4 = G4 * np.sqrt(df / U4)
r4 = Z4 @ model_id.B.T
P4 = r4 @ model_id.w
L4 = -P4

# Note: Z_i share the SAME chi2 divisor, so they're NOT independent —
# they're marginally t(df) but jointly multivariate-t.
# For multivariate t with identity scale, equal weights:
# P = (1/M) * sum(Z_i), Var[P] = (1/M^2) * (M * df/(df-2)) = df/(M*(df-2))
# But because they share the chi2, there's extra correlation.
# Actually for multivariate t with Sigma=I:
# Var[w^T Z] = (df/(df-2)) * w^T Sigma w = (df/(df-2)) * (1/M)
expected_var_portfolio = (df / (df - 2)) / M_test  # = 2.0/10 = 0.2
check(f"Equal-weight portfolio Var ~= {expected_var_portfolio:.3f}",
      abs(L4.var() - expected_var_portfolio) < 0.02,
      f"var = {L4.var():.4f}")


# =========================================================================
# Test 5: Uniform PMF discretization — CDF correctness
# =========================================================================
print()
print("=" * 60)
print("Test 5: Uniform PMF discretization")
print("=" * 60)

grid, pmf, lo, hi = build_analytic_student_t_pmf(df=4, num_qubits=7)

# PMF should sum to 1
check("PMF sums to 1", abs(sum(pmf) - 1.0) < 1e-8,
      f"sum = {sum(pmf):.10f}")

# PMF should be non-negative
check("PMF all >= 0", all(p >= 0 for p in pmf))

# Grid should have 128 points
check("Grid has 128 points", len(grid) == 128, f"len = {len(grid)}")

# Classical VaR should be close to analytic
ref_idx, ref_var = classical_var(grid, pmf, 0.05)
analytic = scipy.stats.t.ppf(0.05, df=4)
check(f"Classical CDF VaR ~= analytic ({analytic:.4f})",
      abs(ref_var - analytic) < 0.15,
      f"classical = {ref_var:.4f}, analytic = {analytic:.4f}")

# CDF at the VaR index should be close to alpha
cdf_at_var = sum(pmf[:ref_idx + 1])
check(f"CDF at VaR index ~= 0.05",
      abs(cdf_at_var - 0.05) < 0.02,
      f"cdf = {cdf_at_var:.4f}")


# =========================================================================
# Test 6: Tail-aware PMF — finer resolution
# =========================================================================
print()
print("=" * 60)
print("Test 6: Tail-aware PMF discretization")
print("=" * 60)

grid_t, pmf_t, lo_t, hi_t = build_tail_aware_pmf(df=4, num_qubits=7)

# PMF should sum to 1
check("Tail PMF sums to 1", abs(sum(pmf_t) - 1.0) < 1e-8,
      f"sum = {sum(pmf_t):.10f}")

# Grid range should be narrower than uniform
_, _, lo_u, hi_u = build_analytic_student_t_pmf(df=4, num_qubits=7)
bin_width_uniform = (hi_u - lo_u) / 128
bin_width_tail = (hi_t - lo_t) / 128
check(f"Tail bins are finer ({bin_width_tail:.4f} < {bin_width_uniform:.4f})",
      bin_width_tail < bin_width_uniform,
      f"tail={bin_width_tail:.4f} uniform={bin_width_uniform:.4f}")

# Classical VaR on tail grid should be closer to analytic
ref_idx_t, ref_var_t = classical_var(grid_t, pmf_t, 0.05)
err_uniform = abs(ref_var - analytic)
err_tail = abs(ref_var_t - analytic)
check(f"Tail VaR error ({err_tail:.4f}) <= uniform error ({err_uniform:.4f})",
      err_tail <= err_uniform + 0.01,  # small tolerance
      f"tail_err={err_tail:.4f} uniform_err={err_uniform:.4f}")

# VaR index should be in a reasonable range (not at edge)
check(f"VaR index in middle of grid (idx={ref_idx_t})",
      5 < ref_idx_t < 123,
      f"idx = {ref_idx_t}")


# =========================================================================
# Test 7: Warm start bracket — should contain the true VaR index
# =========================================================================
print()
print("=" * 60)
print("Test 7: Warm start bracket correctness")
print("=" * 60)

rng7 = np.random.default_rng(42)
samples_7 = rng7.standard_t(df=4, size=100_000)

# Use uniform grid
grid_7, pmf_7, _, _ = build_analytic_student_t_pmf(df=4, num_qubits=7)
ref_idx_7, ref_var_7 = classical_var(grid_7, pmf_7, 0.05)

ws = mc_warm_start_from_losses(samples_7, grid_7, alpha=0.05, confidence=0.99)

check(f"Bracket contains classical VaR index ({ref_idx_7})",
      ws["lo"] <= ref_idx_7 <= ws["hi"],
      f"bracket=[{ws['lo']}, {ws['hi']}], ref_idx={ref_idx_7}")

check(f"Bracket width is small (< 20)",
      ws["bracket_width"] < 20,
      f"width = {ws['bracket_width']}")

check(f"MC VaR estimate is reasonable",
      abs(ws["var_est_value"] - ref_var_7) < 0.5,
      f"mc_est={ws['var_est_value']:.4f}, ref={ref_var_7:.4f}")


# =========================================================================
# Test 8: PMF from samples matches analytic
# =========================================================================
print()
print("=" * 60)
print("Test 8: PMF from samples vs analytic PMF")
print("=" * 60)

rng8 = np.random.default_rng(0)
samples_8 = rng8.standard_t(df=4, size=1_000_000)

grid_s, pmf_s, _, _ = build_pmf_from_samples(samples_8, num_qubits=7,
                                               lo_pct=0.1, hi_pct=99.9)
grid_a, pmf_a, _, _ = build_analytic_student_t_pmf(df=4, num_qubits=7)

# The sample-based and analytic PMFs won't align exactly (different grid ranges)
# but the VaR should be close
ref_idx_s, ref_var_s = classical_var(grid_s, pmf_s, 0.05)
ref_idx_a, ref_var_a = classical_var(grid_a, pmf_a, 0.05)
analytic_ppf = scipy.stats.t.ppf(0.05, df=4)

check(f"Sample-based VaR ~= analytic ppf",
      abs(ref_var_s - analytic_ppf) < 0.2,
      f"sample={ref_var_s:.4f}, analytic={analytic_ppf:.4f}")


# =========================================================================
# Test 9: Factor model with equicorrelated factors — portfolio variance
# =========================================================================
print()
print("=" * 60)
print("Test 9: Equicorrelated factor model — portfolio variance")
print("=" * 60)

# With rho > 0, Sigma_factor = (1-rho)I + rho*11^T
# For multivariate t with scale Sigma:
#   Var[w^T B Z] = (df/(df-2)) * w^T B Sigma B^T w
# With B=I, w=1/M:
#   = (df/(df-2)) * (1/M^2) * 1^T Sigma 1
#   = (df/(df-2)) * (1/M^2) * (M(1-rho) + M^2*rho)
#   = (df/(df-2)) * ((1-rho)/M + rho)

rho_test = 0.3
d9, M9, df9 = 10, 10, 4
model_eq = FactorModel(d=d9, M=M9, df=df9,
                        B=np.eye(M9),
                        w=np.full(M9, 1.0 / M9))

rng9 = np.random.default_rng(123)
N9 = 500_000
G9 = _equicorrelated_normal(rng9, N9, d9, rho_test, backend="numpy")
U9 = rng9.chisquare(df9, size=(N9, 1))
Z9 = G9 * np.sqrt(df9 / U9)
r9 = Z9 @ model_eq.B.T
P9 = r9 @ model_eq.w
L9 = -P9

expected_var_eq = (df9 / (df9 - 2)) * ((1 - rho_test) / M9 + rho_test)
# = 2.0 * (0.07 + 0.3) = 2.0 * 0.37 = 0.74
check(f"Equicorrelated portfolio Var ~= {expected_var_eq:.4f}",
      abs(L9.var() - expected_var_eq) < 0.03,
      f"var = {L9.var():.4f}, expected = {expected_var_eq:.4f}")


# =========================================================================
# Summary
# =========================================================================
print()
print("=" * 60)
total = PASS + FAIL
print(f"RESULTS: {PASS}/{total} passed, {FAIL}/{total} failed")
print("=" * 60)
if FAIL > 0:
    exit(1)
