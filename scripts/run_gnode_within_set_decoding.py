"""Decode selected direction within preparatory target sets using simultaneous units."""

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
N_FOLDS = 5
N_FLIPS = 10_000
N_BOOTSTRAPS = 5_000
SESOI = 0.05


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def target_sets(condition: int) -> tuple[tuple[int, ...], ...]:
    return ((1, 2), (3, 4), (5, 6)) if condition == 2 else ((6, 1, 2), (3, 4, 5))


def fixed_folds(labels: np.ndarray) -> np.ndarray:
    folds = np.empty(len(labels), dtype=int)
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        folds[indices] = np.arange(len(indices)) % N_FOLDS
    return folds


def session_decode(features: np.ndarray, labels: np.ndarray, condition: int) -> np.ndarray:
    set_scores = []
    for directions in target_sets(condition):
        selected = np.isin(labels, directions)
        x = features[selected]
        y = labels[selected]
        folds = fixed_folds(y)
        predictions = np.full((len(y), features.shape[2]), -1, dtype=int)
        for fold in range(N_FOLDS):
            train = folds != fold
            test = folds == fold
            if not np.any(test):
                continue
            mean = x[train].mean(axis=0)
            scale = x[train].std(axis=0, ddof=1)
            scale[scale < 1e-8] = 1.0
            train_z = (x[train] - mean) / scale
            test_z = (x[test] - mean) / scale
            centroids = np.asarray([train_z[y[train] == direction].mean(axis=0) for direction in directions])
            distances = np.sum(
                (test_z[:, None, :, :] - centroids[None, :, :, :]) ** 2, axis=2
            )
            distances += np.arange(len(directions))[None, :, None] * 1e-10
            predictions[test] = np.asarray(directions)[np.argmin(distances, axis=1)]
        recalls = np.asarray(
            [(predictions[y == direction] == direction).mean(axis=0) for direction in directions]
        )
        set_scores.append(recalls.mean(axis=0) - 1.0 / len(directions))
    return np.mean(set_scores, axis=0)


def session_cross_temporal(
    features: np.ndarray, labels: np.ndarray, condition: int, grid: np.ndarray
) -> np.ndarray:
    set_scores = []
    for directions in target_sets(condition):
        selected = np.isin(labels, directions)
        x = features[selected][:, :, grid]
        y = labels[selected]
        folds = fixed_folds(y)
        correct = np.zeros((len(directions), len(grid), len(grid)), dtype=float)
        counts = np.zeros(len(directions), dtype=int)
        for direction_index, direction in enumerate(directions):
            counts[direction_index] = np.sum(y == direction)
        for fold in range(N_FOLDS):
            train = folds != fold
            test = folds == fold
            if not np.any(test):
                continue
            for train_time in range(len(grid)):
                train_slice = x[train, :, train_time]
                mean = train_slice.mean(axis=0)
                scale = train_slice.std(axis=0, ddof=1)
                scale[scale < 1e-8] = 1.0
                centroids = np.asarray(
                    [
                        ((train_slice[y[train] == direction] - mean) / scale).mean(axis=0)
                        for direction in directions
                    ]
                )
                test_z = (x[test] - mean[None, :, None]) / scale[None, :, None]
                distances = np.sum(
                    (test_z[:, None, :, :] - centroids[None, :, :, None]) ** 2, axis=2
                )
                distances += np.arange(len(directions))[None, :, None] * 1e-10
                predicted = np.asarray(directions)[np.argmin(distances, axis=1)]
                for direction_index, direction in enumerate(directions):
                    class_test = y[test] == direction
                    if np.any(class_test):
                        correct[direction_index, :, train_time] += np.sum(
                            predicted[class_test] == direction, axis=0
                        )
        recalls = correct / counts[:, None, None]
        set_scores.append(recalls.mean(axis=0) - 1.0 / len(directions))
    return np.mean(set_scores, axis=0)


def positive_clusters(t_values: np.ndarray, threshold: float) -> list[dict[str, object]]:
    active = t_values > threshold
    padded = np.r_[False, active, False]
    starts = np.flatnonzero(np.diff(padded.astype(int)) == 1)
    stops = np.flatnonzero(np.diff(padded.astype(int)) == -1)
    return [
        {
            "start_index": int(start),
            "stop_index": int(stop - 1),
            "mass": float(np.sum(t_values[start:stop])),
        }
        for start, stop in zip(starts, stops)
    ]


def t_curve(data: np.ndarray) -> np.ndarray:
    sem = np.std(data, axis=0, ddof=1) / np.sqrt(len(data))
    return np.divide(data.mean(axis=0), sem, out=np.zeros(data.shape[1]), where=sem > 0)


def cluster_test(data: np.ndarray, centers: np.ndarray, label: str) -> dict[str, object]:
    threshold = float(stats.t.ppf(0.95, len(data) - 1))
    observed = positive_clusters(t_curve(data), threshold)
    rng = stable_rng(label)
    null_max = np.zeros(N_FLIPS)
    for permutation in range(N_FLIPS):
        signs = rng.choice((-1.0, 1.0), size=(len(data), 1))
        clusters = positive_clusters(t_curve(data * signs), threshold)
        null_max[permutation] = max((cluster["mass"] for cluster in clusters), default=0.0)
    for cluster in observed:
        cluster["start_ms"] = float(centers[cluster["start_index"]])
        cluster["stop_ms"] = float(centers[cluster["stop_index"]])
        cluster["p_corrected"] = float(
            (1 + np.sum(null_max >= cluster["mass"])) / (N_FLIPS + 1)
        )
    return {"n_components": len(data), "forming_t": threshold, "clusters": observed}


def pre_equivalence(data: np.ndarray, label: str) -> dict[str, object]:
    component_means = data.mean(axis=1)
    rng = stable_rng(label)
    bootstrap = rng.choice(
        component_means, size=(N_BOOTSTRAPS, len(component_means)), replace=True
    ).mean(axis=1)
    interval = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "mean_advantage": float(component_means.mean()),
        "bootstrap_95ci": [float(value) for value in interval],
        "sesoi": SESOI,
        "upper_below_sesoi": bool(interval[1] < SESOI),
        "two_sided_95ci_within_bounds": bool(interval[0] > -SESOI and interval[1] < SESOI),
        "estimand": "component-mean decoding advantage averaged over the pre-RS interval",
        "test": "one-sided upper-bound assessment; not a test of absence at every time",
    }


def aggregate_sessions(
    session_values: np.ndarray, session_meta: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    component_values, rows = [], []
    for (monkey, condition, component), indices in session_meta.groupby(
        ["monkey", "condition", "recording_component"], sort=True
    ).groups.items():
        component_values.append(np.median(session_values[np.asarray(list(indices))], axis=0))
        rows.append((monkey, int(condition), component))
    return np.asarray(component_values), pd.DataFrame(
        rows, columns=["monkey", "condition", "recording_component"]
    )


def combine_conditions(
    values: np.ndarray, metadata: pd.DataFrame
) -> tuple[np.ndarray, pd.DataFrame]:
    records, rows = [], []
    for (monkey, component), indices in metadata.groupby(
        ["monkey", "recording_component"], sort=True
    ).groups.items():
        group = metadata.loc[list(indices)]
        if set(group["condition"]) != {2, 3}:
            continue
        ordered = [list(indices)[group["condition"].tolist().index(condition)] for condition in (2, 3)]
        records.append(np.asarray([values[ordered[0]], values[ordered[1]], np.mean(values[ordered], axis=0)]))
        rows.append((monkey, component))
    return np.asarray(records), pd.DataFrame(rows, columns=["monkey", "recording_component"])


def plot_timecourses(
    values: np.ndarray,
    component_meta: pd.DataFrame,
    centers: np.ndarray,
    inference: dict[str, object],
    output: Path,
) -> None:
    colors = {"2": "#0072B2", "3": "#D55E00", "pooled": "#222222"}
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for ax, monkey in zip(axes, ("joe", "lili")):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            data = values[selected, evaluation_index]
            mean = data.mean(axis=0)
            sem = stats.sem(data, axis=0)
            ax.plot(centers, mean, color=colors[evaluation], linewidth=2, label=evaluation)
            ax.fill_between(centers, mean - 1.96 * sem, mean + 1.96 * sem, color=colors[evaluation], alpha=0.12)
        for interval, y in (("pre", 0.25), ("post", 0.235)):
            for cluster in inference[monkey]["pooled"][interval]["clusters"]:
                if cluster["p_corrected"] < 0.05:
                    ax.plot([cluster["start_ms"], cluster["stop_ms"]], [y, y], color="#009E73", linewidth=5)
        ax.axhline(0, color="#777777", linewidth=1)
        ax.axvline(0, color="#777777", linewidth=1, linestyle="--")
        ax.set_title(monkey.capitalize())
        ax.set_xlabel("Time from response signal (ms)")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Within-set balanced accuracy above chance")
    axes[1].legend(title="Condition", frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_cross_temporal(matrices: dict[str, np.ndarray], centers: np.ndarray, output: Path) -> None:
    pooled = (matrices["joe"] + matrices["lili"]) / 2.0
    display = {"Joe": matrices["joe"], "Lili": matrices["lili"], "Equal-weight": pooled}
    limit = max(0.05, float(np.nanquantile(np.abs(np.stack(list(display.values()))), 0.98)))
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
    fig.colorbar(image, cax=colorbar_axis, label="Balanced accuracy above chance")
    fig.suptitle("Within-set selected-direction cross-temporal decoding", y=0.98)
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
        "--output-dir", type=Path, default=ROOT / "results" / "gnode_within_set_decoding"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.rates)
    rates = archive["rates_hz"].astype(float)
    centers = archive["centers_ms"].astype(float)
    metadata = pd.read_csv(args.metadata).reset_index(names="rate_row")
    metadata = metadata[metadata["condition"].isin([2, 3])].copy()
    cross_grid = np.arange(0, len(centers), 5)

    session_curves, session_cross, session_rows = [], [], []
    for (monkey, condition, session), group in metadata.groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        lookup = group.pivot(index="trial", columns="global_neuron", values="rate_row")
        if lookup.isna().any().any():
            raise ValueError(f"Incomplete unit-trial matrix in {monkey}, {condition}, {session}")
        row_indices = lookup.to_numpy(int)
        features = np.transpose(rates[row_indices], (0, 1, 2))
        trial_labels = (
            group[["trial", "direction"]]
            .drop_duplicates()
            .set_index("trial")
            .loc[lookup.index, "direction"]
            .to_numpy(int)
        )
        session_curves.append(session_decode(features, trial_labels, int(condition)))
        session_cross.append(
            session_cross_temporal(features, trial_labels, int(condition), cross_grid)
        )
        session_rows.append(
            (
                monkey,
                int(condition),
                int(session),
                group["recording_component"].iloc[0],
                lookup.shape[1],
                lookup.shape[0],
            )
        )

    session_curves = np.asarray(session_curves)
    session_cross = np.asarray(session_cross)
    session_meta = pd.DataFrame(
        session_rows,
        columns=["monkey", "condition", "session", "recording_component", "n_units", "n_trials"],
    )
    component_by_condition, condition_meta = aggregate_sessions(session_curves, session_meta)
    component_values, component_meta = combine_conditions(component_by_condition, condition_meta)
    component_cross_by_condition, cross_condition_meta = aggregate_sessions(session_cross, session_meta)
    component_cross, cross_component_meta = combine_conditions(
        component_cross_by_condition, cross_condition_meta
    )
    if not component_meta.equals(cross_component_meta):
        raise ValueError("Component ordering mismatch")

    pre = (centers >= -350) & (centers <= -30)
    post = (centers >= 80) & (centers <= 500)
    inference: dict[str, object] = {}
    success = {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        inference[monkey] = {}
        for evaluation_index, evaluation in enumerate(EVALUATIONS):
            data = component_values[selected, evaluation_index]
            inference[monkey][evaluation] = {
                "pre": cluster_test(data[:, pre], centers[pre], f"pre-{monkey}-{evaluation}"),
                "post": cluster_test(data[:, post], centers[post], f"post-{monkey}-{evaluation}"),
                "pre_equivalence": pre_equivalence(data[:, pre], f"equivalence-{monkey}-{evaluation}"),
            }
        pooled = inference[monkey]["pooled"]
        pre_significant = any(
            cluster["p_corrected"] < 0.05 for cluster in pooled["pre"]["clusters"]
        )
        post_significant = any(
            cluster["p_corrected"] < 0.05 for cluster in pooled["post"]["clusters"]
        )
        success[monkey] = bool(
            not pre_significant
            and pooled["pre_equivalence"]["upper_below_sesoi"]
            and post_significant
        )

    session_output = pd.concat(
        [
            session_meta.reset_index(drop=True),
            pd.DataFrame(
                session_curves,
                columns=[f"t_{int(center)}" for center in centers],
            ),
        ],
        axis=1,
    )
    session_output.to_csv(args.output_dir / "session_timecourses_wide.csv", index=False)
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
                        "accuracy_above_chance": component_values[
                            component_index, evaluation_index, time_index
                        ],
                    }
                )
    pd.DataFrame(long_rows).to_csv(args.output_dir / "component_timecourses.csv", index=False)

    cross_matrices = {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        cross_matrices[monkey] = component_cross[selected, 2].mean(axis=0)
    np.savez_compressed(
        args.output_dir / "cross_temporal_decoding.npz",
        centers_ms=centers[cross_grid].astype(np.float32),
        joe=cross_matrices["joe"].astype(np.float32),
        lili=cross_matrices["lili"].astype(np.float32),
    )

    summary = {
        "protocol": "within_set_decoding_protocol.md",
        "n_folds": N_FOLDS,
        "n_sign_flips": N_FLIPS,
        "n_bootstraps": N_BOOTSTRAPS,
        "sesoi": SESOI,
        "session_counts": {
            f"{monkey}_condition_{condition}": int(count)
            for (monkey, condition), count in session_meta.groupby(
                ["monkey", "condition"]
            ).size().items()
        },
        "component_counts": component_meta["monkey"].value_counts().to_dict(),
        "success_by_monkey": success,
        "emergence_claim_pass": bool(all(success.values())),
        "inference": inference,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_timecourses(
        component_values,
        component_meta,
        centers,
        inference,
        args.output_dir / "within_set_decoding_timecourses.png",
    )
    plot_cross_temporal(
        cross_matrices,
        centers[cross_grid],
        args.output_dir / "within_set_cross_temporal.png",
    )
    print(
        json.dumps(
            {
                "emergence_claim_pass": summary["emergence_claim_pass"],
                "success_by_monkey": success,
                "pooled_inference": {
                    monkey: inference[monkey]["pooled"] for monkey in ("joe", "lili")
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
