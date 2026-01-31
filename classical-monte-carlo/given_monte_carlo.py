# Example end-to-end calls for VaR using:
#  1) Classical Monte Carlo estimate of CDF(index) inside the same binary-search VaR solver
#  2) Quantum IQAE estimate of CDF(index) inside the same binary-search VaR solver
#
# Uses the SAME variable names as the provided notebook:
#   num_qubits, mu, sigma, ALPHA, TOLERANCE, grid_points, probs, VAR
# and keeps the same signatures:
#   calc_alpha(index: int, probs: list[float]) -> float
#   calc_alpha_quantum(index: int, probs: list[float]) -> float

import numpy as np
import scipy

from classiq import *
from classiq.applications.iqae.iqae import IQAE

# Problem definition
num_qubits = 7
mu = 0.7
sigma = 0.13
ALPHA = 0.07
TOLERANCE = ALPHA / 10


def get_log_normal_probabilities(mu_normal, sigma_normal, num_points):
    # Use the passed parameters (fixes the common bug of using globals)
    log_normal_mean = np.exp(mu_normal + sigma_normal**2 / 2)
    log_normal_variance = (np.exp(sigma_normal**2) - 1) * np.exp(
        2 * mu_normal + sigma_normal**2
    )
    log_normal_stddev = np.sqrt(log_normal_variance)

    low = np.maximum(0, log_normal_mean - 3 * log_normal_stddev)
    high = log_normal_mean + 3 * log_normal_stddev
    x = np.linspace(low, high, num_points)
    return x, scipy.stats.lognorm.pdf(x, s=sigma_normal, scale=np.exp(mu_normal))


grid_points, probs = get_log_normal_probabilities(mu, sigma, 2**num_qubits)
probs = (probs / np.sum(probs)).tolist()

# "Ground truth" VaR on the discretized grid (same approach as notebook)
VAR = 0.0
accumulated_value = 0.0
for index in range(len(probs)):
    accumulated_value += probs[index]
    if accumulated_value > ALPHA:
        VAR = float(grid_points[index])
        break
print(f"[Reference] Value at risk at {int(ALPHA*100)}%: {VAR}")

# Solver
def calc_alpha(index: int, probs: list[float]) -> float:
    # Exact CDF on the discretized grid (prefix sum); this is NOT Monte Carlo.
    # We keep it because the notebook uses it as the classical baseline.
    return float(sum(probs[:index]))


def update_index(index: int, required_alpha: float, alpha_v: float, search_size: int) -> int:
    if alpha_v < required_alpha:
        return index + search_size
    return index - search_size


def print_status(v, alpha_v, search_size, index):
    print(f"v: {v}, alpha_v: {alpha_v}")
    print(f"{search_size=}")
    print(f"{index=}")
    print("------------------------")


def print_results(grid_points, index, probs):
    print(f"Value at risk at {ALPHA*100}%: {grid_points[index]})")
    global VAR
    print("Real VaR", VAR)
    return index


def value_at_risk(required_alpha, index, calc_alpha_func=calc_alpha):
    # Fix the notebook bug: v should be the asset value, not probs[index]
    v = float(grid_points[index])
    alpha_v = float(calc_alpha_func(index, probs))
    search_size = index // 2
    print_status(v, alpha_v, search_size, index)

    while (not np.isclose(alpha_v, required_alpha, atol=TOLERANCE)) and search_size > 0:
        index = update_index(index, required_alpha, alpha_v, search_size)
        index = max(0, min(index, len(probs) - 1))  # safety clamp
        search_size //= 2

        v = float(grid_points[index])
        alpha_v = float(calc_alpha_func(index, probs))
        print_status(v, alpha_v, search_size, index)

    return print_results(grid_points, index, probs)


def get_initial_index():
    return int(2**num_qubits) // 4


# Classical MONTE CARLO estimator for alpha(index)
# Hyperparameters
MC_SAMPLES_PER_QUERY = 10_000
MC_SEED = 1234

# Precompute categorical sampler inputs
_support = np.arange(len(probs))
_probs_np = np.array(probs, dtype=float)
_cdf_np = np.cumsum(_probs_np)  # used only for debugging


def calc_alpha_mc(index: int, probs: list[float]) -> float:
    """
    Monte Carlo estimate of alpha_v = P(bin < index) by sampling from the discrete
    distribution defined by `probs`.
    Cost = MC_SAMPLES_PER_QUERY samples (one sample = one probability query).
    """
    rng = np.random.default_rng(MC_SEED + index)  # index-dependent seed for repeatability
    samples = rng.choice(_support, size=MC_SAMPLES_PER_QUERY, p=_probs_np)
    # Event: sampled bin is in {0, 1, ..., index-1}
    return float(np.mean(samples < index))


print("\n=== VaR using CLASSICAL MONTE CARLO inside the same binary search ===")
index0 = get_initial_index()
var_index_mc = value_at_risk(ALPHA, index0, calc_alpha_mc)
print(f"[MC] var_index={var_index_mc}, var_value={grid_points[var_index_mc]}")

# Quantum IQAE estimator for alpha(index)
written_qmod = False
GLOBAL_INDEX = 0  # used by the payoff oracle


@qfunc(synthesize_separately=True)
def state_preparation(asset: QArray[QBit], ind: QBit):
    load_distribution(asset=asset)
    payoff(asset=asset, ind=ind)


@qfunc
def load_distribution(asset: QNum):
    inplace_prepare_state(probs, bound=0, target=asset)


@qperm
def payoff(asset: Const[QNum], ind: QBit):
    # Mark bins with asset < GLOBAL_INDEX (i.e. index threshold)
    ind ^= asset < GLOBAL_INDEX


# Hyperparameters
IQAE_EPSILON = 0.05
IQAE_ALPHA = 0.01


def calc_alpha_quantum(index: int, probs: list[float]) -> float:
    global GLOBAL_INDEX, written_qmod
    GLOBAL_INDEX = index

    iqae = IQAE(
        state_prep_op=state_preparation,
        problem_vars_size=num_qubits,
        constraints=Constraints(max_width=28),
        preferences=Preferences(machine_precision=num_qubits),
    )

    qprog = iqae.get_qprog()
    if not written_qmod:
        written_qmod = True
        show(qprog)

    iqae_res = iqae.run(epsilon=IQAE_EPSILON, alpha=IQAE_ALPHA)

    measured_payoff = float(iqae_res.estimation)
    confidence_interval = np.array([x for x in iqae_res.confidence_interval], dtype=float)
    print("Measured Payoff:", measured_payoff)
    print("Confidence Interval:", confidence_interval)
    return measured_payoff


print("\n=== VaR using QUANTUM IQAE inside the same binary search ===")
index0 = get_initial_index()
var_index_q = value_at_risk(ALPHA, index0, calc_alpha_quantum)
print(f"[IQAE] var_index={var_index_q}, var_value={grid_points[var_index_q]}")
