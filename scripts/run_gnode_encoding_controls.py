"""Separate prediction averaging from candidate-set predictive validity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_gnode_target_set_replication import design_from_directions, set_vector, stable_rng


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("baseline", "prep_late")
MEASURES = ("delta", "dispersion", "cross_term", "set_vs_intercept", "set_r2")


def aggregate(values: np.ndarray, metadata: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    triplets, rows = [], []
    for (monkey, triplet, component), ids in metadata.groupby(
        ["monkey", "recording_triplet", "recording_component"], sort=True
    ).groups.items():
        triplets.append(np.nanmedian(values[list(ids)], axis=0))
        rows.append((monkey, component))
    frame = pd.DataFrame(rows, columns=["monkey", "recording_component"])
    components, rows = [], []
    for (monkey, component), ids in frame.groupby(list(frame.columns), sort=True).groups.items():
        components.append(np.nanmedian(np.asarray(triplets)[list(ids)], axis=0))
        rows.append((monkey, component))
    return np.asarray(components), pd.DataFrame(rows, columns=list(frame.columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/gnode_encoding_controls")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rates = pd.read_csv(ROOT / "data/processed/gnode_uncertainty_rates.csv")
    cache = {}
    for keys, group in rates[rates.condition.isin([2, 3])].groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        trials = np.sort(group.trial.unique())
        rng = stable_rng(f"cue-permutations-{keys}")
        cache[keys] = np.asarray([rng.permutation(len(trials)) for _ in range(args.permutations)])

    measures, nulls, rows = [], [], []
    maximum_error = 0.0
    for (monkey, unit), group in rates.groupby(["monkey", "global_neuron"], sort=True):
        unit_measures, unit_nulls = [], []
        for window in WINDOWS:
            train = group[group.condition.eq(1)].dropna(subset=[window])
            beta = np.linalg.lstsq(
                design_from_directions(train.direction.to_numpy()), train[window].to_numpy(), rcond=None
            )[0]
            test = group[group.condition.isin([2, 3])].sort_values(["condition", "trial"])
            observed = test[window].to_numpy(float)
            chosen = design_from_directions(test.direction.to_numpy()) @ beta
            set_design = np.column_stack((np.ones(len(test)), np.asarray([
                set_vector(int(c), int(d)) for c, d in zip(test.condition, test.direction)
            ])))
            predicted = set_design @ beta
            variance = np.var(observed)
            mse_set = np.mean((observed - predicted) ** 2)
            mse_zero = np.mean((observed - beta[0]) ** 2)
            delta = (np.mean((observed - chosen) ** 2) - mse_set) / variance
            dispersion = np.mean((chosen - predicted) ** 2) / variance
            cross = -2 * np.mean((observed - predicted) * (chosen - predicted)) / variance
            maximum_error = max(maximum_error, abs(delta - dispersion - cross))
            unit_measures.append([delta, dispersion, cross, (mse_zero - mse_set) / variance,
                                  1 - mse_set / variance])
            shuffled_sse = np.zeros(args.permutations)
            for condition, subset in test.groupby("condition", sort=True):
                positions = np.flatnonzero(test.condition.to_numpy() == condition)
                permutation = cache[(monkey, condition, int(subset.session.iloc[0]))]
                if len(positions) != permutation.shape[1]:
                    raise ValueError("Cue permutation and unit trial axes differ")
                shuffled = predicted[positions][permutation]
                shuffled_sse += np.sum((observed[positions][None, :] - shuffled) ** 2, axis=1)
            unit_nulls.append((mse_zero - shuffled_sse / len(test)) / variance)
        measures.append(unit_measures)
        nulls.append(unit_nulls)
        first = group.iloc[0]
        rows.append((monkey, int(unit), first.recording_triplet, first.recording_component))
    metadata = pd.DataFrame(rows, columns=["monkey", "global_neuron", "recording_triplet", "recording_component"])
    values, component_meta = aggregate(np.asarray(measures), metadata)
    permutation_values, null_meta = aggregate(np.asarray(nulls), metadata)
    if not component_meta.equals(null_meta):
        raise ValueError("Control and permutation component axes differ")
    table = []
    results = {}
    for monkey in ("joe", "lili"):
        selected = component_meta.monkey.eq(monkey).to_numpy()
        data = values[selected]
        rng = stable_rng(f"encoding-controls-{monkey}")
        draws = rng.integers(len(data), size=(5000, len(data)))
        results[monkey] = {}
        for wi, window in enumerate(WINDOWS):
            result = {}
            for mi, name in enumerate(MEASURES):
                result[name] = {
                    "mean": float(data[:, wi, mi].mean()),
                    "bootstrap_95ci": np.quantile(data[draws, wi, mi].mean(axis=1), [0.025, 0.975]).tolist(),
                }
            observed = data[:, wi, 3].mean()
            null = permutation_values[selected, wi].mean(axis=0)
            result["cue_label_permutation_p"] = float((1 + np.sum(null >= observed)) / (len(null) + 1))
            result["n_components"] = int(len(data))
            result["shuffled_gain_mean"] = float(null.mean())
            results[monkey][window] = result
        for index in np.flatnonzero(selected):
            for wi, window in enumerate(WINDOWS):
                table.append(dict(monkey=monkey, recording_component=component_meta.iloc[index].recording_component,
                                  window=window, **dict(zip(MEASURES, values[index, wi]))))
    pd.DataFrame(table).to_csv(args.output_dir / "component_controls.csv", index=False)
    np.savez_compressed(args.output_dir / "cue_permutation_gains.npz", values=permutation_values,
                        monkey=component_meta.monkey.to_numpy(dtype=str))
    summary = {"analysis_status": "secondary prediction-control analysis", "n_permutations": args.permutations,
               "n_bootstraps": 5000, "decomposition_max_abs_error": maximum_error,
               "permutation_unit": "cue labels shuffled within session and condition; identical trial permutations across simultaneous units",
               "aggregation": "median within triplet, median within component, mean within animal; separately aggregated decomposition terms are not additive",
               "results": results}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    colors = ("#0072B2", "#D55E00", "#009E73")
    for mi, (name, label) in enumerate((("delta", "Original contrast"), ("dispersion", "Prediction dispersion"),
                                       ("set_vs_intercept", "Set vs C1 intercept"))):
        for ai, monkey in enumerate(("joe", "lili")):
            r = results[monkey]["prep_late"][name]
            mean, ci = r["mean"], r["bootstrap_95ci"]
            ax = axes[0] if mi < 2 else axes[1]
            position = ai + (mi - 0.5) * 0.18 if mi < 2 else ai
            ax.errorbar(position, mean, yerr=[[mean-ci[0]], [ci[1]-mean]], fmt="o",
                        color=colors[mi], capsize=3, label=label if ai == 0 else None)
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.8)
        ax.set_xticks([0, 1], ["Joe", "Lili"])
        ax.set_xlim(-0.5, 1.5)
        ax.set_ylabel("Normalized error contrast")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_title("Averaging contribution")
    axes[1].set_title("Candidate-set predictive gain")
    fig.tight_layout()
    fig.savefig(args.output_dir / "encoding_controls.png", dpi=240)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
