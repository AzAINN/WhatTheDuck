from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import time

import numpy as np

from qae_sweep import (
    _make_sampler_v2,
    compile_stateprep_to_qasm,
    _load_stateprep_qasm,
    solve_var_bisect_quantum,
    get_distribution,
    compute_true_var,
    _estimate_tail_prob_iae,
)


def run_classical_mc(probs, threshold_idx, n_samples, seed):
    """Single classical MC estimate."""
    rng = np.random.default_rng(seed)
    samples = rng.choice(len(probs), size=n_samples, p=probs)
    p_hat = np.mean(samples < threshold_idx)
    return p_hat


def run_quantum_ae(sampler, stateprep, num_qubits, threshold_idx, epsilon, alpha_fail=0.05):
    """Single quantum AE estimate."""
    est = _estimate_tail_prob_iae(
        sampler_v2=sampler,
        stateprep_asset_only=stateprep,
        num_asset_qubits=num_qubits,
        threshold_index=threshold_idx,
        epsilon=epsilon,
        alpha_fail=alpha_fail,
    )
    return est.p_hat, est.cost_oracle_queries


if __name__ == '__main__':
    n_samples_list = np.unique(np.logspace(1, 7, 500).astype(int))
    epsilons = np.logspace(-3.5, -0.3, 150)  # 0.0003 to 0.5
    alpha = 0.05
    seed = 42
    num_qubits = 7
    num_points = 2 ** num_qubits

    grid_types = {
        "normal": get_distribution("normal", num_points, mu=0.0, sigma=1.0),
        "lognormal": get_distribution("lognormal", num_points, mu=0.7, sigma=0.13),
        "student_t": get_distribution("student_t", num_points, df=3.0, loc=0.0, scale=1.0),
        "skew_normal": get_distribution("skew_normal", num_points, skew=4.0, loc=0.0, scale=1.0),
        "beta": get_distribution("beta", num_points, a=2.0, b=5.0),
    }

    # Precompile circuits once per distribution
    print("Precompiling state preparation circuits...")
    precompiled = {}
    for dist_name, (grid, probs) in grid_types.items():
        qasm_path = Path(f"/tmp/{dist_name}_stateprep.qasm")
        compile_stateprep_to_qasm(
            probs=probs,
            num_qubits=num_qubits,
            out_qasm_path=qasm_path,
        )
        stateprep = _load_stateprep_qasm(qasm_path)
        true_var, ref_idx = compute_true_var(grid, probs, alpha)
        true_p = float(np.sum(probs[:ref_idx]))
        precompiled[dist_name] = {
            "grid": grid,
            "probs": probs,
            "stateprep": stateprep,
            "ref_idx": ref_idx,
            "true_var": true_var,
            "true_p": true_p,
        }
        print(f"  {dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}")

    results = []

    # =========== CLASSICAL MC ===========
    print(f"\nRunning classical MC ({len(n_samples_list)} x {len(grid_types)} = {len(n_samples_list)*len(grid_types)} tasks)...")
    t0 = time.time()

    def classical_task(dist_name, n_samples):
        data = precompiled[dist_name]
        p_hat = run_classical_mc(data["probs"], data["ref_idx"], n_samples, seed)
        err = abs(p_hat - data["true_p"])
        return {
            "method": "classical",
            "dist": dist_name,
            "queries": int(n_samples),
            "epsilon": None,
            "p_hat": p_hat,
            "true_p": data["true_p"],
            "error": err,
        }

    classical_tasks = [
        (dist_name, n) for dist_name in grid_types for n in n_samples_list
    ]

    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = [pool.submit(classical_task, d, n) for d, n in classical_tasks]
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 500 == 0:
                print(f"  Classical: {i+1}/{len(futures)}")

    print(f"Classical done in {time.time()-t0:.1f}s")

    print(f"\nRunning quantum AE ({len(epsilons)} x {len(grid_types)} = {len(epsilons)*len(grid_types)} tasks)...")
    t0 = time.time()

    # Create one sampler per thread (thread-local via separate creation)
    def quantum_task(dist_name, eps):
        data = precompiled[dist_name]
        sampler = _make_sampler_v2(
            device="GPU",
            method="statevector",
            seed=seed,
            default_shots=1024,
        )
        p_hat, queries = run_quantum_ae(
            sampler=sampler,
            stateprep=data["stateprep"],
            num_qubits=num_qubits,
            threshold_idx=data["ref_idx"],
            epsilon=eps,
        )
        err = abs(p_hat - data["true_p"])
        return {
            "method": "quantum",
            "dist": dist_name,
            "queries": int(queries),
            "epsilon": eps,
            "p_hat": p_hat,
            "true_p": data["true_p"],
            "error": err,
        }

    quantum_tasks = [
        (dist_name, eps) for dist_name in grid_types for eps in epsilons
    ]

    # Fewer workers for quantum (GPU-bound)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(quantum_task, d, e) for d, e in quantum_tasks]
        for i, f in enumerate(as_completed(futures)):
            results.append(f.result())
            if (i + 1) % 50 == 0:
                print(f"  Quantum: {i+1}/{len(futures)}")

    print(f"Quantum done in {time.time()-t0:.1f}s")

    out_path = Path("results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nWrote {len(results)} rows to {out_path}")
