"""Evaluate classifier, equivalence-bound, and unit-count decoding sensitivities."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gnode_within_set_decoding import (
    aggregate_sessions,
    cluster_test,
    combine_conditions,
    fixed_folds,
    pre_equivalence,
    target_sets,
)


ROOT = Path(__file__).resolve().parents[1]
SESOI_GRID = (0.005, 0.01, 0.02, 0.03, 0.04, 0.05)
MINIMUM_UNITS = (2, 3)
VARIANCE_FLOOR = 1e-3


def session_gaussian_nb(
    features: np.ndarray, labels: np.ndarray, condition: int
) -> np.ndarray:
    """Decode each time point with held-out diagonal Gaussian likelihoods."""
    set_scores = []
    for directions in target_sets(condition):
        selected = np.isin(labels, directions)
        x = features[selected]
        y = labels[selected]
        folds = fixed_folds(y)
        predictions = np.full((len(y), features.shape[2]), -1, dtype=int)
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
            scores = []
            for direction_index, direction in enumerate(directions):
                class_values = train_z[y[train] == direction]
                class_mean = class_values.mean(axis=0)
                class_variance = np.maximum(
                    class_values.var(axis=0, ddof=1), VARIANCE_FLOOR
                )
                log_likelihood = -0.5 * np.sum(
                    np.log(class_variance)[None, :, :]
                    + (test_z - class_mean[None, :, :]) ** 2
                    / class_variance[None, :, :],
                    axis=1,
                )
                scores.append(log_likelihood - direction_index * 1e-10)
            predictions[test] = np.asarray(directions)[
                np.argmax(np.stack(scores, axis=1), axis=1)
            ]
        recalls = np.asarray(
            [
                (predictions[y == direction] == direction).mean(axis=0)
                for direction in directions
            ]
        )
        set_scores.append(recalls.mean(axis=0) - 1.0 / len(directions))
    return np.mean(set_scores, axis=0)


def infer_pooled(
    component_values: np.ndarray,
    component_meta: pd.DataFrame,
    centers: np.ndarray,
    label: str,
) -> dict[str, object]:
    pre = (centers >= -350) & (centers <= -30)
    post = (centers >= 80) & (centers <= 500)
    result: dict[str, object] = {}
    for monkey in ("joe", "lili"):
        selected = component_meta["monkey"].eq(monkey).to_numpy()
        data = component_values[selected, 2]
        pre_result = pre_equivalence(data[:, pre], f"{label}-pre-{monkey}")
        post_result = cluster_test(
            data[:, post], centers[post], f"{label}-post-{monkey}"
        )
        significant_post = [
            cluster
            for cluster in post_result["clusters"]
            if cluster["p_corrected"] < 0.05
        ]
        result[monkey] = {
            "n_components": int(len(data)),
            "pre_equivalence": pre_result,
            "sesoi_grid": {
                f"{bound:.3f}": bool(
                    pre_result["bootstrap_95ci"][1] < bound
                )
                for bound in SESOI_GRID
            },
            "two_sided_95ci_grid": {
                f"{bound:.3f}": bool(
                    pre_result["bootstrap_95ci"][0] > -bound
                    and pre_result["bootstrap_95ci"][1] < bound
                )
                for bound in SESOI_GRID
            },
            "post": post_result,
            "first_significant_post_center_ms": (
                float(significant_post[0]["start_ms"])
                if significant_post
                else None
            ),
        }
    return result


def load_main_component_values(
    result_dir: Path, centers: np.ndarray
) -> tuple[np.ndarray, pd.DataFrame]:
    frame = pd.read_csv(result_dir / "component_timecourses.csv")
    records, rows = [], []
    for (monkey, component), group in frame.groupby(
        ["monkey", "recording_component"], sort=True
    ):
        curves = []
        for evaluation in ("2", "3", "pooled"):
            curve = (
                group[group["evaluation"].astype(str).eq(evaluation)]
                .set_index("center_ms")
                .reindex(centers)["accuracy_above_chance"]
                .to_numpy(float)
            )
            curves.append(curve)
        records.append(curves)
        rows.append((monkey, component))
    return np.asarray(records), pd.DataFrame(
        rows, columns=["monkey", "recording_component"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rates",
        type=Path,
        default=ROOT / "data" / "processed" / "gnode_rs_sliding_rates.npz",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=ROOT / "data" / "processed" / "gnode_rs_sliding_metadata.csv",
    )
    parser.add_argument(
        "--main-result-dir",
        type=Path,
        default=ROOT / "results" / "gnode_within_set_decoding",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "gnode_decoding_robustness",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    archive = np.load(args.rates)
    rates = archive["rates_hz"].astype(float)
    centers = archive["centers_ms"].astype(float)
    metadata = pd.read_csv(args.metadata).reset_index(names="rate_row")
    metadata = metadata[metadata["condition"].isin([2, 3])].copy()

    gnb_session_curves, session_rows = [], []
    for (monkey, condition, session), group in metadata.groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        lookup = group.pivot(index="trial", columns="global_neuron", values="rate_row")
        if lookup.isna().any().any():
            raise ValueError(
                f"Incomplete unit-trial matrix in {monkey}, {condition}, {session}"
            )
        row_indices = lookup.to_numpy(int)
        features = rates[row_indices]
        trial_labels = (
            group[["trial", "direction"]]
            .drop_duplicates()
            .set_index("trial")
            .loc[lookup.index, "direction"]
            .to_numpy(int)
        )
        gnb_session_curves.append(
            session_gaussian_nb(features, trial_labels, int(condition))
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

    gnb_session_curves = np.asarray(gnb_session_curves)
    session_meta = pd.DataFrame(
        session_rows,
        columns=[
            "monkey",
            "condition",
            "session",
            "recording_component",
            "n_units",
            "n_trials",
        ],
    )
    gnb_by_condition, gnb_condition_meta = aggregate_sessions(
        gnb_session_curves, session_meta
    )
    gnb_components, gnb_component_meta = combine_conditions(
        gnb_by_condition, gnb_condition_meta
    )

    main_components, main_component_meta = load_main_component_values(
        args.main_result_dir, centers
    )
    results: dict[str, object] = {
        "nearest_centroid": infer_pooled(
            main_components, main_component_meta, centers, "robustness-main"
        ),
        "gaussian_naive_bayes": infer_pooled(
            gnb_components, gnb_component_meta, centers, "robustness-gnb"
        ),
    }

    main_sessions = pd.read_csv(args.main_result_dir / "session_timecourses_wide.csv")
    time_columns = [column for column in main_sessions if column.startswith("t_")]
    session_curves = main_sessions[time_columns].to_numpy(float)
    session_metadata = main_sessions[
        [
            "monkey",
            "condition",
            "session",
            "recording_component",
            "n_units",
            "n_trials",
        ]
    ].copy()
    for minimum_units in MINIMUM_UNITS:
        retained = session_metadata["n_units"].ge(minimum_units).to_numpy()
        by_condition, condition_meta = aggregate_sessions(
            session_curves[retained], session_metadata[retained].reset_index(drop=True)
        )
        components, component_meta = combine_conditions(by_condition, condition_meta)
        results[f"nearest_centroid_min_{minimum_units}_units"] = infer_pooled(
            components,
            component_meta,
            centers,
            f"robustness-min-{minimum_units}",
        )

    rows = []
    for analysis, animal_results in results.items():
        for monkey, values in animal_results.items():
            post_clusters = [
                cluster
                for cluster in values["post"]["clusters"]
                if cluster["p_corrected"] < 0.05
            ]
            rows.append(
                {
                    "analysis": analysis,
                    "monkey": monkey,
                    "n_components": values["n_components"],
                    "pre_mean": values["pre_equivalence"]["mean_advantage"],
                    "pre_ci_low": values["pre_equivalence"]["bootstrap_95ci"][0],
                    "pre_ci_high": values["pre_equivalence"]["bootstrap_95ci"][1],
                    "upper_below_0_005": values["sesoi_grid"]["0.005"],
                    "two_sided_95ci_within_0_005": values["two_sided_95ci_grid"]["0.005"],
                    "post_cluster_start_ms": values[
                        "first_significant_post_center_ms"
                    ],
                    "post_cluster_p": (
                        post_clusters[0]["p_corrected"] if post_clusters else np.nan
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(args.output_dir / "robustness_summary.csv", index=False)
    pd.DataFrame(
        gnb_session_curves, columns=[f"t_{int(center)}" for center in centers]
    ).join(session_meta).to_csv(
        args.output_dir / "gaussian_nb_session_timecourses.csv", index=False
    )

    summary = {
        "classifier": {
            "alternative": "diagonal Gaussian naive Bayes",
            "variance_floor_after_training_standardization": VARIANCE_FLOOR,
            "fold_assignment": "identical deterministic five-fold labels as main analysis",
        },
        "sesoi_grid": list(SESOI_GRID),
        "minimum_unit_thresholds": list(MINIMUM_UNITS),
        "results": results,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps({"results": results}, indent=2))


if __name__ == "__main__":
    main()
