from pathlib import Path

import numpy as np

from qae_sweep import _make_sampler_v2, compile_stateprep_to_qasm, _load_stateprep_qasm, solve_var_bisect_quantum


def estimate_var_quantum(
        probs: np.ndarray,
        grid_points: np.ndarray,
        alpha: float = 0.05,
        epsilon: float = 0.01,
        alpha_fail: float = 0.01,
        prob_tol: float = None,
        max_steps: int = 64,
        seed: int = 42,
        device: str = "GPU",
        method: str = "statevector",
        shots: int = 1024,
) -> dict:
    """
    Estimate VaR using quantum amplitude estimation (IQAE).

    Args:
        probs: Discretized probability distribution
        grid_points: Corresponding grid values
        alpha: Target tail probability (e.g., 0.05 for 5% VaR)
        epsilon: IQAE precision parameter
        alpha_fail: IQAE confidence parameter (1-confidence)
        prob_tol: CI ambiguity tolerance (default: epsilon/2)
        max_steps: Max bisection iterations
        seed: Random seed
        device: "GPU" or "CPU"
        method: Aer simulation method
        shots: Default shots for sampler

    Returns:
        dict with keys:
            - var: Estimated VaR value
            - var_index: Grid index of VaR
            - total_oracle_queries: Total oracle queries used (the cost)
            - epsilon: The epsilon used
            - bisection_steps: Number of bisection iterations
    """
    if prob_tol is None:
        prob_tol = epsilon / 2.0

    # Create sampler
    sampler_v2 = _make_sampler_v2(
        device=device,
        method=method,
        seed=seed,
        default_shots=shots,
    )

    # Compile state preparation circuit
    num_qubits = int(np.log2(len(grid_points)))
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=".qasm", delete=False, mode='w') as f:
        qasm_path = Path(f.name)

    compile_stateprep_to_qasm(
        probs=probs,
        num_qubits=num_qubits,
        out_qasm_path=qasm_path,
        debug_mode=False,
        qasm3=False,
    )

    stateprep = _load_stateprep_qasm(qasm_path)
    qasm_path.unlink()  # Clean up temp file

    # Run bisection
    var, var_idx, total_cost = solve_var_bisect_quantum(
        sampler_v2=sampler_v2,
        stateprep_asset_only=stateprep,
        grid_points=grid_points,
        probs=probs,
        alpha_target=alpha,
        epsilon=epsilon,
        alpha_fail=alpha_fail,
        prob_tol=prob_tol,
        max_steps=max_steps,
    )

    return {
        "var": var,
        "var_index": var_idx,
        "total_oracle_queries": total_cost,
        "epsilon": epsilon,
        "bisection_steps": max_steps,  # Actual count would need tracking in solve_var_bisect_quantum
    }
