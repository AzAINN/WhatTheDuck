from pathlib import Path
import csv
import argparse
import numpy as np
from tqdm import tqdm

from qae_sweep import (
    _make_sampler_v2,
    _load_stateprep_qasm,
    _estimate_tail_prob_iae,
)


def run_all_dists(indir, outfile, seed=42):
    """All distributions, fewer points per dist."""
    n_samples_list = np.unique(np.logspace(1, 7, 80).astype(int))
    epsilons = np.clip(np.logspace(-2, -0.3, 60), 1e-6, 0.49)

    precompiled = {}
    for data_path in sorted(indir.glob("*.data.npz")):
        blob = np.load(data_path)
        dist_name = str(blob["dist_name"])
        probs = blob["probs"].astype(np.float64)
        num_qubits = int(blob["num_qubits"])
        ref_idx = int(blob["ref_idx"])
        true_p = float(np.sum(probs[:ref_idx]))
        stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
        precompiled[dist_name] = (probs, stateprep, num_qubits, ref_idx, true_p)
        print(f"{dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}")

    sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])

        rng = np.random.default_rng(seed)
        for dist_name, (probs, _, _, ref_idx, true_p) in tqdm(precompiled.items(), desc="Classical"):
            for n in tqdm(n_samples_list, desc=dist_name, leave=False):
                samples = rng.choice(len(probs), size=n, p=probs)
                p_hat = np.mean(samples < ref_idx)
                writer.writerow(["classical", dist_name, n, "", p_hat, true_p, abs(p_hat - true_p)])
            f.flush()

        for dist_name, (probs, stateprep, num_qubits, ref_idx, true_p) in tqdm(precompiled.items(), desc="Quantum"):
            for eps in tqdm(epsilons, desc=dist_name, leave=False):
                est = _estimate_tail_prob_iae(
                    sampler_v2=sampler,
                    stateprep_asset_only=stateprep,
                    num_asset_qubits=num_qubits,
                    threshold_index=ref_idx,
                    epsilon=eps,
                    alpha_fail=0.05,
                )
                writer.writerow(["quantum", dist_name, est.cost_oracle_queries, eps, est.p_hat, true_p, abs(est.p_hat - true_p)])
            f.flush()

    print(f"Done: {outfile}")


def run_single_dist(indir, outfile, dist_name="normal", seed=42):
    """Single distribution, many points for detailed scaling."""
    n_samples_list = np.unique(np.logspace(1, 7, 500).astype(int))
    epsilons = np.clip(np.logspace(-3, -0.2, 400), 1e-6, 0.49)

    data_path = indir / f"{dist_name}.data.npz"
    blob = np.load(data_path)
    probs = blob["probs"].astype(np.float64)
    num_qubits = int(blob["num_qubits"])
    ref_idx = int(blob["ref_idx"])
    true_p = float(np.sum(probs[:ref_idx]))
    stateprep = _load_stateprep_qasm(indir / f"{dist_name}.stateprep.qasm")
    print(f"{dist_name}: ref_idx={ref_idx}, true_p={true_p:.6f}, num_qubits={num_qubits}")

    sampler = _make_sampler_v2("GPU", "statevector", seed, 1024)

    with open(outfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "dist", "queries", "epsilon", "p_hat", "true_p", "error"])

        rng = np.random.default_rng(seed)
        for n in tqdm(n_samples_list, desc="Classical"):
            samples = rng.choice(len(probs), size=n, p=probs)
            p_hat = np.mean(samples < ref_idx)
            writer.writerow(["classical", dist_name, n, "", p_hat, true_p, abs(p_hat - true_p)])
        f.flush()

        for eps in tqdm(epsilons, desc="Quantum"):
            est = _estimate_tail_prob_iae(
                sampler_v2=sampler,
                stateprep_asset_only=stateprep,
                num_asset_qubits=num_qubits,
                threshold_index=ref_idx,
                epsilon=eps,
                alpha_fail=0.05,
            )
            writer.writerow(["quantum", dist_name, est.cost_oracle_queries, eps, est.p_hat, true_p, abs(est.p_hat - true_p)])
            f.flush()

    print(f"Done: {outfile}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "single"], required=True)
    parser.add_argument("--dist", default="normal", help="Distribution for single mode")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    indir = Path("build_qasm_10")

    if args.mode == "all":
        run_all_dists(indir, "scaling_all_dists_10q.csv", args.seed)
    else:
        run_single_dist(indir, f"scaling_{args.dist}_10q.csv", args.dist, args.seed)