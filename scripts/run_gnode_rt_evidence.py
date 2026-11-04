"""Relate held-out selected-direction evidence to reaction time."""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from run_gnode_within_set_decoding import fixed_folds, target_sets


ROOT = Path(__file__).resolve().parents[1]
EVALUATIONS = ("2", "3", "pooled")
N_FLIPS = 10_000
N_BOOTSTRAPS = 5_000
MO_CENTERS = np.arange(-200.0, 201.0, 10.0)


def quiet_nanmean(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def quiet_nanmedian(values: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis)


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def held_out_evidence(features: np.ndarray, labels: np.ndarray, condition: int) -> np.ndarray:
    evidence = np.full((len(labels), features.shape[2]), np.nan)
    for directions in target_sets(condition):
        selected_indices = np.flatnonzero(np.isin(labels, directions))
        x = features[selected_indices]
        y = labels[selected_indices]
        folds = fixed_folds(y)
        for fold in range(5):
            train = folds != fold
            test = folds == fold
            if not np.any(test):
                continue
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0, ddof=1)
            scale[scale < 1e-8] = 1.0
            train_z = (x[train] - mean) / scale
            test_z = (x[test] - mean) / scale
            centroids = np.asarray(
                [train_z[y[train] == direction].mean(axis=0) for direction in directions]
            )
            distances = np.sum(
                (test_z[:, None, :, :] - centroids[None, :, :, :]) ** 2, axis=2
            )
            distances += np.arange(len(directions))[None, :, None] * 1e-10
            class_lookup = {direction: index for index, direction in enumerate(directions)}
            correct_index = np.asarray([class_lookup[int(value)] for value in y[test]])
            trial_index = np.arange(np.sum(test))
            correct_distance = distances[trial_index, correct_index]
            incorrect = distances.copy()
            incorrect[trial_index, correct_index] = np.inf
            nearest_incorrect = np.min(incorrect, axis=1)
            margin = (nearest_incorrect - correct_distance) / (
                nearest_incorrect + correct_distance + 1e-12
            )
            evidence[selected_indices[test]] = margin
    if np.isnan(evidence).any():
        raise ValueError("Some held-out trials did not receive direction evidence")
    return evidence


def spearman_curve(rt: np.ndarray, evidence: np.ndarray) -> np.ndarray:
    if len(rt) < 5 or np.ptp(rt) == 0:
        return np.full(evidence.shape[1], np.nan)
    rt_rank = stats.rankdata(rt).astype(float)
    rt_rank -= rt_rank.mean()
    evidence_rank = stats.rankdata(evidence, axis=0).astype(float)
    evidence_rank -= evidence_rank.mean(axis=0)
    numerator = np.sum(rt_rank[:, None] * evidence_rank, axis=0)
    denominator = np.sqrt(np.sum(rt_rank**2) * np.sum(evidence_rank**2, axis=0))
    return np.divide(
        numerator,
        denominator,
        out=np.full(evidence.shape[1], np.nan),
        where=denominator > 0,
    )


def session_rt_correlation(
    rt: np.ndarray,
    evidence: np.ndarray,
    labels: np.ndarray,
    condition: int,
    fixed_window: np.ndarray,
) -> tuple[np.ndarray, float]:
    set_curves, set_fixed = [], []
    fixed_evidence = evidence[:, fixed_window].mean(axis=1)
    for directions in target_sets(condition):
        direction_curves, direction_fixed = [], []
        for direction in directions:
            selected = labels == direction
            curve = spearman_curve(rt[selected], evidence[selected])
            fixed = spearman_curve(rt[selected], fixed_evidence[selected, None])[0]
            if np.isfinite(curve).any() and np.isfinite(fixed):
                direction_curves.append(np.arctanh(np.clip(curve, -0.999999, 0.999999)))
                direction_fixed.append(np.arctanh(np.clip(fixed, -0.999999, 0.999999)))
        if direction_curves:
            set_curves.append(quiet_nanmean(np.asarray(direction_curves), axis=0))
            set_fixed.append(np.mean(direction_fixed))
    if not set_curves:
        return np.full(evidence.shape[1], np.nan), np.nan
    return np.tanh(quiet_nanmean(np.asarray(set_curves), axis=0)), float(
        np.tanh(np.mean(set_fixed))
    )


def speed_groups(rt: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fast = np.zeros(len(rt), dtype=bool)
    slow = np.zeros(len(rt), dtype=bool)
    for direction in np.unique(labels):
        indices = np.flatnonzero(labels == direction)
        ordered = indices[np.argsort(rt[indices], kind="stable")]
        split = len(ordered) // 2
        fast[ordered[:split]] = True
        slow[ordered[split:]] = True
    return fast, slow


def interpolate_to_movement(
    evidence: np.ndarray, centers: np.ndarray, rt: np.ndarray
) -> np.ndarray:
    aligned = np.empty((len(rt), len(MO_CENTERS)))
    for trial in range(len(rt)):
        aligned[trial] = np.interp(
            MO_CENTERS + rt[trial], centers, evidence[trial], left=np.nan, right=np.nan
        )
    return aligned


def aggregate_sessions(values: np.ndarray, session_meta: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    records, rows = [], []
    for (monkey, condition, component), indices in session_meta.groupby(
        ["monkey", "condition", "recording_component"], sort=True
    ).groups.items():
        records.append(quiet_nanmedian(values[np.asarray(list(indices))], axis=0))
        rows.append((monkey, int(condition), component))
    return np.asarray(records), pd.DataFrame(
        rows, columns=["monkey", "condition", "recording_component"]
    )


def combine_conditions(values: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    records, rows = [], []
    for (monkey, component), indices in metadata.groupby(
        ["monkey", "recording_component"], sort=True
    ).groups.items():
        indices = list(indices)
        group = metadata.loc[indices]
        if set(group["condition"]) != {2, 3}:
            continue
        ordered = [indices[group["condition"].tolist().index(condition)] for condition in (2, 3)]
        records.append(
            np.asarray(
                [values[ordered[0]], values[ordered[1]], quiet_nanmean(values[ordered], axis=0)]
            )
        )
        rows.append((monkey, component))
    return np.asarray(records), pd.DataFrame(rows, columns=["monkey", "recording_component"])


def sign_flip_summary(values: np.ndarray, label: str) -> dict[str, object]:
    values = values[np.isfinite(values)]
    rng = stable_rng(label)
    observed = float(np.mean(values))
    null = (
        rng.choice((-1.0, 1.0), size=(N_FLIPS, len(values))) * values[None, :]
    ).mean(axis=1)
    bootstrap = rng.choice(values, size=(N_BOOTSTRAPS, len(values)), replace=True).mean(axis=1)
    return {
        "n_components": len(values),
        "mean_r": observed,
        "median_r": float(np.median(values)),
        "negative_components": int(np.sum(values < 0)),
        "sign_flip_p_negative": float((1 + np.sum(null <= observed)) / (N_FLIPS + 1)),
        "bootstrap_95ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
    }


def negative_clusters(t_values: np.ndarray, threshold: float) -> list[dict[str, object]]:
    active = -t_values > threshold
    padded = np.r_[False, active, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    return [
        {
            "start_index": int(start),
            "stop_index": int(stop - 1),
            "mass": float(np.sum(-t_values[start:stop])),
        }
        for start, stop in zip(starts, stops)
    ]


def t_curve(data: np.ndarray) -> np.ndarray:
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
    return np.divide(data.mean(axis=0), sem, out=np.zeros(data.shape[1]), where=sem > 0)


def cluster_test(data: np.ndarray, centers: np.ndarray, label: str) -> dict[str, object]:
    threshold = float(stats.t.ppf(0.95, len(data) - 1))
    observed = negative_clusters(t_curve(data), threshold)
    rng = stable_rng(label)
    null_max = np.zeros(N_FLIPS)
    for permutation in range(N_FLIPS):
        signs = rng.choice((-1.0, 1.0), size=(len(data), 1))
        clusters = negative_clusters(t_curve(data * signs), threshold)
        null_max[permutation] = max((cluster["mass"] for cluster in clusters), default=0.0)
    for cluster in observed:
        cluster["start_ms"] = float(centers[cluster["start_index"]])
        cluster["stop_ms"] = float(centers[cluster["stop_index"]])
        cluster["p_corrected"] = float(
            (1 + np.sum(null_max >= cluster["mass"])) / (N_FLIPS + 1)
        )
    return {"n_components": len(data), "forming_t": threshold, "clusters": observed}


def plot_correlations(
    curves: np.ndarray,
    component_meta: pd.DataFrame,
    centers: np.ndarray,
    clusters: dict[str, object],
    output: Path,
) -> None:
    colors = {"2": "#0072B2", "3": "#D55E00", "pooled": "#222222"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for ax, monkey in zip(axes, ("joe", "lili")):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            data = curves[selected, evaluation_index]
            mean = np.nanmean(data, axis=0)
            sem = stats.sem(data, axis=0, nan_policy="omit")
            ax.plot(centers, mean, color=colors[evaluation], linewidth=2, label=evaluation)
            ax.fill_between(centers, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[evaluation], alpha=0.12)
        for cluster in clusters[monkey]["pooled"]["clusters"]:
            if cluster["p_corrected"] < 0.05:
                ax.plot([cluster["start_ms"], cluster["stop_ms"]], [-0.125, -0.125], color="#CC3311", linewidth=5)
        ax.axhline(0, color="#777777", linewidth=1)
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_xlim(-350, 500)
        ax.set_ylim(-0.14, 0.14)
        ax.set_title(monkey.capitalize())
        ax.set_xlabel("Time from response signal (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Direction-controlled Spearman r (evidence vs RT)")
    axes[1].legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_speed_groups(
    fast_rs: np.ndarray,
    slow_rs: np.ndarray,
    fast_mo: np.ndarray,
    slow_mo: np.ndarray,
    component_meta: pd.DataFrame,
    centers: np.ndarray,
    output: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharey=True)
    for row, monkey in enumerate(("joe", "lili")):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        for ax, fast, slow, time, title in (
            (axes[row, 0], fast_rs[selected, 2], slow_rs[selected, 2], centers, "RS aligned"),
            (axes[row, 1], fast_mo[selected, 2], slow_mo[selected, 2], MO_CENTERS, "MO aligned"),
        ):
            ax.plot(time, np.nanmean(fast, axis=0), color="#D55E00", linewidth=2, label="Fast RT")
            ax.plot(time, np.nanmean(slow, axis=0), color="#0072B2", linewidth=2, label="Slow RT")
            ax.axhline(0, color="#777777", linewidth=1)
            ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
            ax.set_title(f"{monkey.capitalize()} - {title}")
            ax.set_xlabel("Time from alignment event (ms)")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_xlim(-200, 500)
    axes[1, 0].set_xlim(-200, 500)
    axes[0, 0].set_ylabel("Held-out correct-direction evidence")
    axes[1, 0].set_ylabel("Held-out correct-direction evidence")
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
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
        "--output-dir", type=Path, default=ROOT / "results" / "gnode_rt_evidence"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.rates)
    rates = archive["rates_hz"].astype(float)
    centers = archive["centers_ms"].astype(float)
    movement_onset = archive["movement_onset_ms"].astype(float)
    metadata = pd.read_csv(args.metadata).reset_index(names="rate_row")
    metadata = metadata[metadata["condition"].isin([2, 3])].copy()
    fixed_window = (centers >= 100) & (centers <= 200)

    session_curves, session_fixed, session_fixed_premovement = [], [], []
    session_fast_rs, session_slow_rs = [], []
    session_fast_mo, session_slow_mo = [], []
    session_rows = []
    excluded_trials = 0
    retained_trials = 0
    for (monkey, condition, session), group in metadata.groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        lookup = group.pivot(index="trial", columns="global_neuron", values="rate_row")
        row_indices = lookup.to_numpy(int)
        representative_rows = row_indices[:, 0]
        rt = movement_onset[representative_rows]
        valid = (rt >= 50) & (rt <= 600)
        excluded_trials += int(np.sum(~valid))
        retained_trials += int(np.sum(valid))
        lookup = lookup.iloc[valid]
        row_indices = lookup.to_numpy(int)
        rt = rt[valid]
        features = rates[row_indices]
        labels = (
            group[["trial", "direction"]]
            .drop_duplicates()
            .set_index("trial")
            .loc[lookup.index, "direction"]
            .to_numpy(int)
        )
        evidence = held_out_evidence(features, labels, int(condition))
        curve, fixed = session_rt_correlation(rt, evidence, labels, int(condition), fixed_window)
        strictly_premovement = rt >= 225
        _, fixed_premovement = session_rt_correlation(
            rt[strictly_premovement],
            evidence[strictly_premovement],
            labels[strictly_premovement],
            int(condition),
            fixed_window,
        )
        fast, slow = speed_groups(rt, labels)
        movement_evidence = interpolate_to_movement(evidence, centers, rt)
        session_curves.append(curve)
        session_fixed.append(fixed)
        session_fixed_premovement.append(fixed_premovement)
        session_fast_rs.append(np.mean(evidence[fast], axis=0))
        session_slow_rs.append(np.mean(evidence[slow], axis=0))
        session_fast_mo.append(np.nanmean(movement_evidence[fast], axis=0))
        session_slow_mo.append(np.nanmean(movement_evidence[slow], axis=0))
        session_rows.append(
            (
                monkey,
                int(condition),
                int(session),
                group["recording_component"].iloc[0],
                len(rt),
                float(np.median(rt[fast])),
                float(np.median(rt[slow])),
            )
        )

    session_meta = pd.DataFrame(
        session_rows,
        columns=[
            "monkey",
            "condition",
            "session",
            "recording_component",
            "n_trials",
            "fast_median_rt_ms",
            "slow_median_rt_ms",
        ],
    )
    arrays = {}
    for name, values in {
        "correlation": session_curves,
        "fixed": session_fixed,
        "fixed_premovement_sensitivity": session_fixed_premovement,
        "fast_rs": session_fast_rs,
        "slow_rs": session_slow_rs,
        "fast_mo": session_fast_mo,
        "slow_mo": session_slow_mo,
    }.items():
        by_condition, condition_meta = aggregate_sessions(np.asarray(values), session_meta)
        arrays[name], component_meta = combine_conditions(by_condition, condition_meta)

    confirmatory = (centers >= 80) & (centers <= 300)
    primary, clusters, premovement_sensitivity, success = {}, {}, {}, {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        primary[monkey], clusters[monkey], premovement_sensitivity[monkey] = {}, {}, {}
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            primary[monkey][evaluation] = sign_flip_summary(
                arrays["fixed"][selected, evaluation_index], f"primary-{monkey}-{evaluation}"
            )
            clusters[monkey][evaluation] = cluster_test(
                arrays["correlation"][selected, evaluation_index][:, confirmatory],
                centers[confirmatory],
                f"cluster-{monkey}-{evaluation}",
            )
            premovement_sensitivity[monkey][evaluation] = sign_flip_summary(
                arrays["fixed_premovement_sensitivity"][selected, evaluation_index],
                f"premovement-{monkey}-{evaluation}",
            )
        pooled = primary[monkey]["pooled"]
        success[monkey] = bool(
            pooled["median_r"] < 0 and pooled["sign_flip_p_negative"] < 0.05
        )

    component_rows = []
    for component_index, component in component_meta.iterrows():
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            component_rows.append(
                {
                    "monkey": component.monkey,
                    "recording_component": component.recording_component,
                    "evaluation": evaluation,
                    "fixed_window_r": arrays["fixed"][component_index, evaluation_index],
                }
            )
    pd.DataFrame(component_rows).to_csv(args.output_dir / "component_primary_correlations.csv", index=False)
    session_meta.to_csv(args.output_dir / "session_rt_inventory.csv", index=False)
    np.savez_compressed(
        args.output_dir / "component_curves.npz",
        centers_ms=centers.astype(np.float32),
        mo_centers_ms=MO_CENTERS.astype(np.float32),
        monkey=component_meta["monkey"].to_numpy(),
        component=component_meta["recording_component"].to_numpy(),
        correlation=arrays["correlation"].astype(np.float32),
        fast_rs=arrays["fast_rs"].astype(np.float32),
        slow_rs=arrays["slow_rs"].astype(np.float32),
        fast_mo=arrays["fast_mo"].astype(np.float32),
        slow_mo=arrays["slow_mo"].astype(np.float32),
    )

    summary = {
        "protocol": "rt_evidence_protocol.md",
        "retained_trials_across_sessions": retained_trials,
        "excluded_rt_trials_across_sessions": excluded_trials,
        "rt_bounds_ms": [50, 600],
        "fixed_evidence_window_ms": [100, 200],
        "n_sign_flips": N_FLIPS,
        "n_bootstraps": N_BOOTSTRAPS,
        "component_counts": component_meta["monkey"].value_counts().to_dict(),
        "primary": primary,
        "posthoc_rt_at_least_225ms_sensitivity": premovement_sensitivity,
        "time_resolved_clusters": clusters,
        "success_by_monkey": success,
        "rt_association_claim_pass": bool(all(success.values())),
        "session_rt_medians_ms": {
            monkey: {
                "fast": float(group["fast_median_rt_ms"].median()),
                "slow": float(group["slow_median_rt_ms"].median()),
            }
            for monkey, group in session_meta.groupby("monkey")
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_correlations(
        arrays["correlation"],
        component_meta,
        centers,
        clusters,
        args.output_dir / "rt_correlation_timecourses.png",
    )
    plot_speed_groups(
        arrays["fast_rs"],
        arrays["slow_rs"],
        arrays["fast_mo"],
        arrays["slow_mo"],
        component_meta,
        centers,
        args.output_dir / "fast_slow_evidence_alignment.png",
    )
    print(
        json.dumps(
            {
                "rt_association_claim_pass": summary["rt_association_claim_pass"],
                "success_by_monkey": success,
                "primary_pooled": {monkey: primary[monkey]["pooled"] for monkey in ("joe", "lili")},
                "significant_pooled_clusters": {
                    monkey: [
                        cluster
                        for cluster in clusters[monkey]["pooled"]["clusters"]
                        if cluster["p_corrected"] < 0.05
                    ]
                    for monkey in ("joe", "lili")
                },
                "session_rt_medians_ms": summary["session_rt_medians_ms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
