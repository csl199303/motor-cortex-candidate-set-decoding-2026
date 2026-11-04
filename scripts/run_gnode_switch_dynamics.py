"""Time-resolved switching and cross-temporal generalization for G-Node rz77m8."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ("2", "3", "pooled")
N_FLIPS = 10_000
N_BOOTSTRAPS = 5_000
MIN_SUSTAINED_POINTS = 11


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def direction_vector(direction: np.ndarray) -> np.ndarray:
    theta = 2.0 * np.pi * (direction.astype(float) - 1.0) / 6.0
    return np.column_stack((np.cos(theta), np.sin(theta)))


def design(direction: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(direction)), direction_vector(direction)))


def allowed_set(condition: int, direction: int) -> tuple[int, ...]:
    if condition == 2:
        return (1, 2) if direction in (1, 2) else (3, 4) if direction in (3, 4) else (5, 6)
    return (6, 1, 2) if direction in (6, 1, 2) else (3, 4, 5)


def expected_design(condition: np.ndarray, direction: np.ndarray) -> np.ndarray:
    vectors = []
    for cond, final_direction in zip(condition.astype(int), direction.astype(int)):
        targets = np.asarray(allowed_set(int(cond), int(final_direction)))
        vectors.append(direction_vector(targets).mean(axis=0))
    return np.column_stack((np.ones(len(vectors)), np.asarray(vectors)))


def normalized_delta(observed: np.ndarray, chosen: np.ndarray, target_set: np.ndarray) -> np.ndarray:
    variance = np.var(observed, axis=0)
    numerator = np.mean((observed - chosen) ** 2, axis=0) - np.mean(
        (observed - target_set) ** 2, axis=0
    )
    return np.divide(numerator, variance, out=np.full_like(numerator, np.nan), where=variance > 1e-12)


def cross_temporal_delta(
    observed: np.ndarray, chosen_prediction: np.ndarray, set_prediction: np.ndarray
) -> np.ndarray:
    n = len(observed)
    y2 = np.mean(observed**2, axis=0)[:, None]
    variance = np.var(observed, axis=0)[:, None]
    chosen_mse = y2 - 2.0 * observed.T @ chosen_prediction / n + np.mean(
        chosen_prediction**2, axis=0
    )[None, :]
    set_mse = y2 - 2.0 * observed.T @ set_prediction / n + np.mean(
        set_prediction**2, axis=0
    )[None, :]
    return np.divide(
        chosen_mse - set_mse,
        variance,
        out=np.full_like(chosen_mse, np.nan),
        where=variance > 1e-12,
    )


def aggregate_hierarchy(
    values: np.ndarray, neuron_meta: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    triplet_values, triplet_rows = [], []
    for (monkey, triplet), indices in neuron_meta.groupby(
        ["monkey", "recording_triplet"], sort=True
    ).groups.items():
        triplet_values.append(np.nanmedian(values[np.asarray(list(indices))], axis=0))
        component = neuron_meta.loc[list(indices), "recording_component"].iloc[0]
        triplet_rows.append((monkey, triplet, component))
    triplet_values = np.asarray(triplet_values)
    triplet_meta = pd.DataFrame(
        triplet_rows, columns=["monkey", "recording_triplet", "recording_component"]
    )

    component_values, component_rows = [], []
    for (monkey, component), indices in triplet_meta.groupby(
        ["monkey", "recording_component"], sort=True
    ).groups.items():
        component_values.append(np.nanmedian(triplet_values[np.asarray(list(indices))], axis=0))
        component_rows.append((monkey, component))
    return np.asarray(component_values), pd.DataFrame(
        component_rows, columns=["monkey", "recording_component"]
    )


def find_clusters(t_values: np.ndarray, threshold: np.ndarray) -> list[dict[str, object]]:
    clusters = []
    for sign in (1, -1):
        active = sign * t_values > threshold
        padded = np.r_[False, active, False]
        starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
        stops = np.flatnonzero(np.diff(padded.astype(int)) == -1)
        for start, stop in zip(starts, stops):
            clusters.append(
                {
                    "start_index": int(start),
                    "stop_index": int(stop - 1),
                    "sign": int(sign),
                    "mass": float(np.sum(np.abs(t_values[start:stop]))),
                }
            )
    return clusters


def t_values(data: np.ndarray) -> np.ndarray:
    counts = np.sum(np.isfinite(data), axis=0)
    mean = np.nanmean(data, axis=0)
    sem = np.nanstd(data, axis=0, ddof=1) / np.sqrt(counts)
    return np.divide(mean, sem, out=np.zeros_like(mean), where=sem > 0)


def cluster_permutation(data: np.ndarray, centers: np.ndarray, label: str) -> dict[str, object]:
    counts = np.sum(np.isfinite(data), axis=0)
    threshold = stats.t.ppf(0.975, counts - 1)
    observed_t = t_values(data)
    observed_clusters = find_clusters(observed_t, threshold)
    rng = stable_rng(label)
    null_max = np.zeros(N_FLIPS)
    for permutation in range(N_FLIPS):
        signs = rng.choice((-1.0, 1.0), size=(len(data), 1))
        clusters = find_clusters(t_values(data * signs), threshold)
        null_max[permutation] = max((cluster["mass"] for cluster in clusters), default=0.0)
    for cluster in observed_clusters:
        cluster["start_ms"] = float(centers[cluster["start_index"]])
        cluster["stop_ms"] = float(centers[cluster["stop_index"]])
        cluster["p_corrected"] = float(
            (1 + np.sum(null_max >= cluster["mass"])) / (N_FLIPS + 1)
        )
    return {
        "n_components": len(data),
        "cluster_forming_t_range": [float(np.nanmin(threshold)), float(np.nanmax(threshold))],
        "clusters": observed_clusters,
    }


def sustained_latency(mean_curve: np.ndarray, centers: np.ndarray) -> float:
    eligible = centers >= 0
    negative = mean_curve[eligible] < 0
    run = np.convolve(negative.astype(int), np.ones(MIN_SUSTAINED_POINTS, dtype=int), mode="valid")
    hits = np.flatnonzero(run == MIN_SUSTAINED_POINTS)
    return float(centers[eligible][hits[0]]) if len(hits) else np.nan


def latency_bootstrap(data: np.ndarray, centers: np.ndarray, label: str) -> dict[str, object]:
    rng = stable_rng(label)
    observed = sustained_latency(np.nanmean(data, axis=0), centers)
    latencies = np.empty(N_BOOTSTRAPS)
    for index in range(N_BOOTSTRAPS):
        sample = rng.integers(0, len(data), len(data))
        latencies[index] = sustained_latency(np.nanmean(data[sample], axis=0), centers)
    finite = latencies[np.isfinite(latencies)]
    return {
        "observed_ms": observed,
        "bootstrap_median_ms": float(np.median(finite)) if len(finite) else np.nan,
        "bootstrap_95ci_ms": [float(value) for value in np.quantile(finite, [0.025, 0.975])]
        if len(finite)
        else [np.nan, np.nan],
        "censored_fraction": float(1.0 - len(finite) / N_BOOTSTRAPS),
    }


def latency_contrast(
    condition_two: np.ndarray, condition_three: np.ndarray, centers: np.ndarray, label: str
) -> dict[str, object]:
    rng = stable_rng(label)
    contrasts = []
    for _ in range(N_BOOTSTRAPS):
        sample = rng.integers(0, len(condition_two), len(condition_two))
        latency_two = sustained_latency(np.nanmean(condition_two[sample], axis=0), centers)
        latency_three = sustained_latency(np.nanmean(condition_three[sample], axis=0), centers)
        if np.isfinite(latency_two) and np.isfinite(latency_three):
            contrasts.append(latency_three - latency_two)
    contrasts = np.asarray(contrasts)
    observed = sustained_latency(np.nanmean(condition_three, axis=0), centers) - sustained_latency(
        np.nanmean(condition_two, axis=0), centers
    )
    return {
        "observed_c3_minus_c2_ms": float(observed),
        "bootstrap_median_ms": float(np.median(contrasts)) if len(contrasts) else np.nan,
        "bootstrap_95ci_ms": [float(value) for value in np.quantile(contrasts, [0.025, 0.975])]
        if len(contrasts)
        else [np.nan, np.nan],
        "censored_fraction": float(1.0 - len(contrasts) / N_BOOTSTRAPS),
    }


def plot_timecourses(
    corrected: np.ndarray,
    component_meta: pd.DataFrame,
    centers: np.ndarray,
    clusters: dict[str, object],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), sharey=True)
    colors = {"2": "#0072B2", "3": "#D55E00", "pooled": "#222222"}
    for ax, monkey in zip(axes, ["joe", "lili"]):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            data = corrected[selected, evaluation_index]
            mean = np.nanmean(data, axis=0)
            sem = stats.sem(data, axis=0, nan_policy="omit")
            ax.plot(centers, mean, color=colors[evaluation], linewidth=2, label=evaluation)
            ax.fill_between(centers, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[evaluation], alpha=0.12)
        pooled_clusters = clusters[monkey]["pooled"]["clusters"]
        levels = {1: 0.205, -1: -0.205}
        for cluster in pooled_clusters:
            if cluster["p_corrected"] < 0.05:
                ax.plot(
                    [cluster["start_ms"], cluster["stop_ms"]],
                    [levels[cluster["sign"]]] * 2,
                    color="#009E73" if cluster["sign"] > 0 else "#CC3311",
                    linewidth=5,
                    solid_capstyle="butt",
                )
        ax.axhline(0, color="#777777", linewidth=1)
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xlim(centers[0], centers[-1])
        ax.set_xlabel("Time from response signal (ms)")
        ax.set_title(monkey.capitalize())
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Baseline-corrected normalized improvement")
    axes[1].legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_cross_temporal(matrices: dict[str, np.ndarray], centers: np.ndarray, output: Path) -> None:
    pooled = (matrices["joe"] + matrices["lili"]) / 2.0
    display = {"Joe": matrices["joe"], "Lili": matrices["lili"], "Equal-weight": pooled}
    limit = float(np.nanquantile(np.abs(np.stack(list(display.values()))), 0.90))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.3), sharex=True, sharey=True)
    for ax, (title, matrix) in zip(axes, display.items()):
        image = ax.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            extent=[centers[0], centers[-1], centers[0], centers[-1]],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
        )
        ax.axvline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(title)
        ax.set_xlabel("Train time from RS (ms)")
    axes[0].set_ylabel("Test time from RS (ms)")
    fig.subplots_adjust(left=0.07, right=0.88, bottom=0.14, top=0.86, wspace=0.1)
    colorbar_axis = fig.add_axes([0.905, 0.2, 0.015, 0.58])
    fig.colorbar(
        image,
        cax=colorbar_axis,
        label="Normalized MSE improvement",
        extend="both",
    )
    fig.suptitle(f"Cross-temporal generalization (colors clipped at +/-{limit:.2f})", y=0.98)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rates", type=Path, default=ROOT / "data" / "processed" / "gnode_rs_sliding_rates.npz"
    )
    parser.add_argument(
        "--metadata", type=Path, default=ROOT / "data" / "processed" / "gnode_rs_sliding_metadata.csv"
    )
    parser.add_argument(
        "--fixed-components",
        type=Path,
        default=ROOT / "results" / "gnode_target_set" / "component_metrics.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gnode_switch_dynamics"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.rates)
    rates = archive["rates_hz"].astype(float)
    centers = archive["centers_ms"].astype(float)
    metadata = pd.read_csv(args.metadata)
    neuron_arrays, cross_arrays, neuron_rows = [], [], []

    for (monkey, global_neuron), indices in metadata.groupby(
        ["monkey", "global_neuron"], sort=True
    ).groups.items():
        indices = np.asarray(list(indices))
        neuron = metadata.iloc[indices]
        c1_mask = neuron["condition"].to_numpy() == 1
        c1_direction = neuron.loc[c1_mask, "direction"].to_numpy(int)
        beta = np.linalg.lstsq(design(c1_direction), rates[indices[c1_mask]], rcond=None)[0]
        effects, pooled_cross = [], None
        for evaluation in EVALUATIONS:
            conditions = [int(evaluation)] if evaluation != "pooled" else [2, 3]
            test_mask = neuron["condition"].isin(conditions).to_numpy()
            test_meta = neuron.loc[test_mask]
            observed = rates[indices[test_mask]]
            chosen_prediction = design(test_meta["direction"].to_numpy(int)) @ beta
            set_prediction = expected_design(
                test_meta["condition"].to_numpy(int), test_meta["direction"].to_numpy(int)
            ) @ beta
            effects.append(normalized_delta(observed, chosen_prediction, set_prediction))
            if evaluation == "pooled":
                pooled_cross = cross_temporal_delta(observed, chosen_prediction, set_prediction)
        neuron_arrays.append(np.asarray(effects))
        cross_arrays.append(pooled_cross)
        first = neuron.iloc[0]
        neuron_rows.append(
            (monkey, int(global_neuron), first.recording_triplet, first.recording_component)
        )

    neuron_values = np.asarray(neuron_arrays)
    neuron_meta = pd.DataFrame(
        neuron_rows,
        columns=["monkey", "global_neuron", "recording_triplet", "recording_component"],
    )
    component_values, component_meta = aggregate_hierarchy(neuron_values, neuron_meta)
    component_cross, cross_meta = aggregate_hierarchy(np.asarray(cross_arrays), neuron_meta)
    if not component_meta.equals(cross_meta):
        raise ValueError("Component order differs between timecourse and cross-temporal results")

    fixed = pd.read_csv(args.fixed_components)
    fixed["evaluation"] = fixed["evaluation"].astype(str)
    corrected = component_values.copy()
    for component_index, component in component_meta.iterrows():
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            match = fixed[
                (fixed["monkey"] == component.monkey)
                & (fixed["recording_component"] == component.recording_component)
                & (fixed["window"] == "baseline")
                & (fixed["evaluation"] == evaluation)
            ]
            if len(match) != 1:
                raise ValueError(f"Missing baseline for {component.to_dict()}, {evaluation}")
            corrected[component_index, evaluation_index] -= float(match.iloc[0]["delta"])

    confirmatory = (centers >= -350) & (centers <= 650)
    clusters: dict[str, object] = {}
    latencies: dict[str, object] = {}
    raw_crossings: dict[str, object] = {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        clusters[monkey] = {}
        latencies[monkey] = {}
        raw_crossings[monkey] = {}
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            data = corrected[selected, evaluation_index]
            clusters[monkey][evaluation] = cluster_permutation(
                data[:, confirmatory], centers[confirmatory], f"cluster-{monkey}-{evaluation}"
            )
            latencies[monkey][evaluation] = latency_bootstrap(
                data, centers, f"latency-{monkey}-{evaluation}"
            )
            raw_crossings[monkey][evaluation] = latency_bootstrap(
                component_values[selected, evaluation_index], centers,
                f"raw-crossing-{monkey}-{evaluation}",
            )
        latencies[monkey]["c3_minus_c2"] = latency_contrast(
            corrected[selected, 0], corrected[selected, 1], centers, f"contrast-{monkey}"
        )

    success_by_monkey = {}
    for monkey in ("joe", "lili"):
        significant = [
            cluster
            for cluster in clusters[monkey]["pooled"]["clusters"]
            if cluster["p_corrected"] < 0.05
        ]
        positive_pre = any(cluster["sign"] > 0 and cluster["start_ms"] < 0 for cluster in significant)
        negative_post = any(cluster["sign"] < 0 and cluster["stop_ms"] >= 0 for cluster in significant)
        finite_switch = np.isfinite(latencies[monkey]["pooled"]["observed_ms"])
        success_by_monkey[monkey] = bool(positive_pre and negative_post and finite_switch)

    long_rows = []
    for component_index, component in component_meta.iterrows():
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            for time_index, center in enumerate(centers):
                long_rows.append(
                    {
                        "monkey": component.monkey,
                        "recording_component": component.recording_component,
                        "evaluation": evaluation,
                        "center_ms": center,
                        "delta_raw": component_values[component_index, evaluation_index, time_index],
                        "delta_baseline_corrected": corrected[component_index, evaluation_index, time_index],
                    }
                )
    pd.DataFrame(long_rows).to_csv(args.output_dir / "component_timecourses.csv", index=False)

    cross_group = {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        cross_group[monkey] = np.nanmean(component_cross[selected], axis=0)
    np.savez_compressed(
        args.output_dir / "cross_temporal_matrices.npz",
        centers_ms=centers.astype(np.float32),
        joe=cross_group["joe"].astype(np.float32),
        lili=cross_group["lili"].astype(np.float32),
    )

    summary = {
        "protocol": "switch_dynamics_protocol.md",
        "n_sign_flips": N_FLIPS,
        "n_bootstraps": N_BOOTSTRAPS,
        "component_counts": component_meta["monkey"].value_counts().to_dict(),
        "missing_component_timepoints": {
            monkey: {
                evaluation: int(
                    np.isnan(
                        component_values[
                            component_meta["monkey"].eq(monkey).to_numpy(), evaluation_index
                        ]
                    ).sum()
                )
                for evaluation_index, evaluation in enumerate(EVALUATIONS)
            }
            for monkey in ("joe", "lili")
        },
        "confirmatory_range_ms": [-350, 650],
        "success_by_monkey": success_by_monkey,
        "switching_claim_pass": bool(all(success_by_monkey.values())),
        "clusters": clusters,
        "latencies": latencies,
        "raw_crossings": raw_crossings,
        "crossing_scope": "descriptive sustained MSE crossing; neither loss of set information nor physiological onset",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
    )
    plot_timecourses(
        corrected,
        component_meta,
        centers,
        clusters,
        args.output_dir / "switch_timecourses.png",
    )
    plot_cross_temporal(
        cross_group, centers, args.output_dir / "cross_temporal_generalization.png"
    )
    print(
        json.dumps(
            {
                "switching_claim_pass": summary["switching_claim_pass"],
                "success_by_monkey": success_by_monkey,
                "latencies": latencies,
                "significant_pooled_clusters": {
                    monkey: [
                        cluster
                        for cluster in clusters[monkey]["pooled"]["clusters"]
                        if cluster["p_corrected"] < 0.05
                    ]
                    for monkey in ("joe", "lili")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
