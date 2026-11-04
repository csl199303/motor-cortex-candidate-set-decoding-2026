"""Calibrate within-set decoding with label permutations and Poisson injections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_gnode_within_set_decoding import (
    aggregate_sessions, combine_conditions, fixed_folds, positive_clusters,
    session_decode, stable_rng, t_curve, target_sets,
)
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]


def batched_set_decode(features: np.ndarray, labels: np.ndarray, directions: tuple) -> np.ndarray:
    """Vectorize the original decoder; features have replicate/trial/unit/time axes."""
    if features.shape[0] == 1 and len(labels) > 1:
        features = np.broadcast_to(features, (len(labels), *features.shape[1:]))
    folds = np.empty_like(labels)
    for direction in directions:
        selected = labels == direction
        folds[selected] = ((np.cumsum(selected, axis=1) - 1) % 5)[selected]
    predicted = np.full((*labels.shape, features.shape[-1]), -1, dtype=int)
    for fold in range(5):
        train = folds != fold
        count = train.sum(axis=1)[:, None, None]
        total = np.einsum("pn,pnut->put", train, features)
        square = np.einsum("pn,pnut->put", train, features ** 2)
        variance = np.maximum((square - total ** 2 / count) / (count - 1), 0)
        variance[variance < 1e-16] = 1.0
        inverse = 1 / variance
        distances = []
        for ci, direction in enumerate(directions):
            selected = train & (labels == direction)
            class_count = selected.sum(axis=1)[:, None, None]
            centroid = np.einsum("pn,pnut->put", selected, features) / class_count
            distance = np.einsum("pnut,put->pnt", features ** 2, inverse)
            distance -= 2 * np.einsum("pnut,put->pnt", features, centroid * inverse)
            distance += np.sum(centroid ** 2 * inverse, axis=1)[:, None, :]
            distances.append(distance + ci * 1e-10)
        candidate = np.asarray(directions)[np.argmin(np.stack(distances, axis=2), axis=2)]
        test = folds == fold
        predicted[test] = candidate[test]
    scores = []
    for direction in directions:
        selected = labels == direction
        scores.append(np.sum((predicted == direction) * selected[:, :, None], axis=1)
                      / selected.sum(axis=1)[:, None])
    return np.mean(scores, axis=0) - 1 / len(directions)


def wilson(successes: int, n: int) -> list[float]:
    p = successes / n
    denominator = 1 + 1.96 ** 2 / n
    center = (p + 1.96 ** 2 / (2*n)) / denominator
    radius = 1.96 * np.sqrt(p*(1-p)/n + 1.96**2/(4*n*n)) / denominator
    return [float(center-radius), float(center+radius)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=499)
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gnode_decoder_calibration")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(ROOT / "data/processed/gnode_rs_sliding_rates.npz")
    centers_all = archive["centers_ms"].astype(float)
    tested = ((centers_all >= -350) & (centers_all <= -30)) | ((centers_all >= 0) & (centers_all <= 500))
    centers = centers_all[tested]
    rates = archive["rates_hz"].astype(float)[:, tested]
    metadata = pd.read_csv(ROOT / "data/processed/gnode_rs_sliding_metadata.csv").reset_index(names="rate_row")
    metadata = metadata[metadata.condition.isin([2, 3])]
    pre = centers < 0
    amplitudes = (0.0, 0.1, 0.25, 0.5, 1.0)
    null_sessions, observed_sessions, simulation_sessions, rows = [], [], [], []
    max_error = 0.0
    for si, ((monkey, condition, session), group) in enumerate(metadata.groupby(["monkey", "condition", "session"], sort=True)):
        lookup = group.pivot(index="trial", columns="global_neuron", values="rate_row")
        features = rates[lookup.to_numpy(int)]
        labels = group[["trial", "direction"]].drop_duplicates().set_index("trial").loc[lookup.index, "direction"].to_numpy(int)
        rng = stable_rng(f"decoder-calibration-{monkey}-{condition}-{session}")
        observed = session_decode(features, labels, int(condition))
        null_curve = np.zeros((args.permutations, len(centers)))
        simulated = np.zeros((len(amplitudes), args.simulations))
        for directions in target_sets(int(condition)):
            selected = np.isin(labels, directions)
            x, y = features[selected], labels[selected]
            actual = batched_set_decode(x[None], y[None], directions)[0]
            # Compare against an independently expressed single-set implementation.
            predictions = np.full((len(y), len(centers)), -1)
            folds = fixed_folds(y)
            for fold in range(5):
                train, test = folds != fold, folds == fold
                mean, scale = x[train].mean(axis=0), x[train].std(axis=0, ddof=1)
                scale[scale < 1e-8] = 1
                centroids = np.asarray([((x[train][y[train] == d]-mean)/scale).mean(axis=0) for d in directions])
                distance = np.sum((((x[test]-mean)/scale)[:, None]-centroids[None])**2, axis=2)
                distance += np.arange(len(directions))[None, :, None] * 1e-10
                predictions[test] = np.asarray(directions)[distance.argmin(axis=1)]
            expected = np.mean([(predictions[y == d] == d).mean(axis=0) for d in directions], axis=0) - 1/len(directions)
            max_error = max(max_error, float(np.max(np.abs(actual-expected))))
            for start in range(0, args.permutations, args.batch_size):
                stop = min(start+args.batch_size, args.permutations)
                shuffled = np.asarray([rng.permutation(y) for _ in range(stop-start)])
                null_curve[start:stop] += batched_set_decode(x[None], shuffled, directions) / len(target_sets(int(condition)))
            baseline_hz = np.maximum(x[:, :, pre].mean(axis=(0, 2)), 0.2)
            preferred = rng.uniform(0, 2*np.pi, x.shape[1])
            theta = 2*np.pi*(np.asarray(directions)-1)/6
            tuning = np.cos(theta[:, None] - preferred[None])
            class_index = np.asarray([directions.index(int(d)) for d in y])
            for ai, amplitude in enumerate(amplitudes):
                gain = np.exp(amplitude*tuning)
                gain /= gain.mean(axis=0, keepdims=True)
                mean_per_bin = baseline_hz[None, :, None] * gain[class_index, :, None] * 0.01
                for start in range(0, args.simulations, args.batch_size):
                    stop = min(start+args.batch_size, args.simulations)
                    spikes = rng.poisson(mean_per_bin[None], size=(stop-start, len(y), x.shape[1], 37))
                    cumulative = np.pad(spikes.cumsum(axis=-1), ((0,0),(0,0),(0,0),(1,0)))
                    counts = cumulative[..., 5:] - cumulative[..., :-5]
                    decoded = batched_set_decode(counts.astype(float), np.broadcast_to(y, (stop-start, len(y))), directions)
                    simulated[ai, start:stop] += decoded.mean(axis=1) / len(target_sets(int(condition)))
        null_sessions.append(null_curve)
        observed_sessions.append(observed)
        simulation_sessions.append(simulated)
        rows.append((monkey, int(condition), int(session), group.recording_component.iloc[0], x.shape[1], len(labels)))
        if (si+1) % 20 == 0:
            print(f"Calibrated {si+1}/160 sessions", flush=True)
    session_meta = pd.DataFrame(rows, columns=["monkey", "condition", "session", "recording_component", "n_units", "n_trials"])
    arrays = {}
    component_meta = None
    for name, data in (("observed", observed_sessions), ("permuted", null_sessions), ("simulated", simulation_sessions)):
        values, meta = aggregate_sessions(np.asarray(data), session_meta)
        arrays[name], current_meta = combine_conditions(values, meta)
        if component_meta is not None and not current_meta.equals(component_meta):
            raise ValueError("Calibration component axes differ")
        component_meta = current_meta
    results, power_rows, null_rows = {}, [], []
    for monkey in ("joe", "lili"):
        selected = component_meta.monkey.eq(monkey).to_numpy()
        observed = arrays["observed"][selected, 2]
        permuted = arrays["permuted"][selected, 2]
        null_mean = permuted.mean(axis=0)
        observed_pre = float(observed[:, pre].mean())
        null_pre = null_mean[:, pre].mean(axis=1)
        post = centers >= 0
        threshold = float(stats.t.ppf(0.95, len(observed)-1))
        clusters = positive_clusters(t_curve(observed[:, post]), threshold)
        max_mass = []
        for pi in range(args.permutations):
            found = positive_clusters(t_curve(permuted[:, pi, post]), threshold)
            max_mass.append(max((c["mass"] for c in found), default=0.0))
        for cluster in clusters:
            cluster["start_ms"] = float(centers[post][cluster["start_index"]])
            cluster["stop_ms"] = float(centers[post][cluster["stop_index"]])
            cluster["p_label_permutation"] = float((1+np.sum(np.asarray(max_mass)>=cluster["mass"]))/(args.permutations+1))
        simulation_mean = arrays["simulated"][selected, 2].mean(axis=0)
        # Split the zero-signal simulations: calibration and false-positive assessment are disjoint.
        split = args.simulations // 2
        critical = float(np.quantile(simulation_mean[0, :split], 0.95))
        for ai, amplitude in enumerate(amplitudes):
            values = simulation_mean[ai, split:] if ai == 0 else simulation_mean[ai]
            successes = int(np.sum(values > critical))
            power_rows.append(dict(monkey=monkey, log_rate_modulation=amplitude, n_simulations=len(values),
                                   mean_advantage=float(values.mean()), detection_fraction=successes/len(values),
                                   detection_ci_low=wilson(successes,len(values))[0], detection_ci_high=wilson(successes,len(values))[1]))
        results[monkey] = {"n_components": int(len(observed)), "pre_mean": observed_pre,
                           "pre_empirical_null_95interval": np.quantile(null_pre,[0.025,0.975]).tolist(),
                           "pre_label_permutation_p": float((1+np.sum(null_pre>=observed_pre))/(args.permutations+1)),
                           "post_test_interval_ms": [0,500], "post_clusters": clusters,
                           "simulation_critical_mean_advantage": critical}
        for pi, value in enumerate(null_pre):
            null_rows.append(dict(monkey=monkey, permutation=pi, pre_mean=value, post_max_cluster_mass=max_mass[pi]))
    pd.DataFrame(power_rows).to_csv(args.output_dir / "injection_detection.csv", index=False)
    pd.DataFrame(null_rows).to_csv(args.output_dir / "permutation_null.csv", index=False)
    session_meta.to_csv(args.output_dir / "session_inventory.csv", index=False)
    np.savez_compressed(args.output_dir / "calibration_arrays.npz", centers_ms=centers,
                        monkey=component_meta.monkey.to_numpy(dtype=str), **arrays)
    summary = {"analysis_status": "secondary decoder calibration", "n_permutations": args.permutations,
               "n_simulations": args.simulations, "decoder_implementation_max_error": max_error,
               "simulation_scope": "independent Poisson rate codes, empirical session/set/unit/trial layout and mean rates, overlapping 50-ms counts from 10-ms bins; does not calibrate every possible neural code",
               "injection_test": "mean pre-RS accuracy advantage; critical value calibrated on half of zero-signal simulations",
               "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for monkey, color in (("joe", "#0072B2"), ("lili", "#D55E00")):
        null = np.asarray([r["pre_mean"] for r in null_rows if r["monkey"]==monkey])
        axes[0].hist(null,bins=25,alpha=0.45,color=color,label=monkey.capitalize(),density=True)
        axes[0].axvline(results[monkey]["pre_mean"],color=color,lw=2)
        power = pd.DataFrame(power_rows).query("monkey == @monkey")
        axes[1].plot(power.log_rate_modulation,power.detection_fraction,"o-",color=color,label=monkey.capitalize())
    axes[0].set_xlabel("Pre-RS mean accuracy advantage")
    axes[0].set_ylabel("Permutation density")
    axes[1].set_xlabel("Injected log-rate modulation")
    axes[1].set_ylabel("Detection fraction")
    axes[1].set_ylim(-0.02,1.02)
    for ax in axes:
        ax.spines[["top","right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "decoder_calibration.png",dpi=240)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
