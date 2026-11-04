"""Run the frozen target-set expectation replication on G-Node rz77m8."""

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
WINDOWS = ["baseline", "prep_early", "prep_middle", "prep_late", "post_reveal", "movement"]
WINDOW_LABELS = ["Baseline", "Early", "Middle", "Late", "Post-RS", "Movement"]
EVALUATIONS = ["2", "3", "pooled"]
N_SIGN_FLIPS = 10_000


def direction_vector(direction: np.ndarray | int) -> np.ndarray:
    theta = 2.0 * np.pi * (np.asarray(direction, dtype=float) - 1.0) / 6.0
    return np.column_stack((np.cos(theta), np.sin(theta)))


def allowed_set(condition: int, direction: int) -> tuple[int, ...]:
    if condition == 2:
        return (1, 2) if direction in (1, 2) else (3, 4) if direction in (3, 4) else (5, 6)
    if condition == 3:
        return (6, 1, 2) if direction in (6, 1, 2) else (3, 4, 5)
    raise ValueError(f"No ambiguous target set for condition {condition}")


def set_vector(condition: int, direction: int, rotate: int = 0) -> np.ndarray:
    targets = allowed_set(condition, direction)
    targets = tuple(((target - 1 + rotate) % 6) + 1 for target in targets)
    return direction_vector(np.asarray(targets)).mean(axis=0)


def design_from_directions(directions: np.ndarray) -> np.ndarray:
    return np.column_stack((np.ones(len(directions)), direction_vector(directions)))


def cv_r2(condition_one: pd.DataFrame, window: str) -> float:
    usable = condition_one[["direction", window]].dropna().reset_index(drop=True)
    if len(usable) < 12 or usable["direction"].value_counts().min() < 2:
        return np.nan
    fold = np.empty(len(usable), dtype=int)
    for direction, indices in usable.groupby("direction").groups.items():
        ordered = np.asarray(sorted(indices))
        fold[ordered] = np.arange(len(ordered)) % 5
    observed = usable[window].to_numpy(float)
    predicted = np.full(len(usable), np.nan)
    for held_out in range(5):
        test = fold == held_out
        train = ~test
        if test.sum() == 0:
            continue
        beta = np.linalg.lstsq(
            design_from_directions(usable.loc[train, "direction"].to_numpy()),
            observed[train],
            rcond=None,
        )[0]
        predicted[test] = design_from_directions(
            usable.loc[test, "direction"].to_numpy()
        ) @ beta
    valid = np.isfinite(predicted)
    denominator = np.sum((observed[valid] - observed[valid].mean()) ** 2)
    return float(1.0 - np.sum((observed[valid] - predicted[valid]) ** 2) / denominator) if denominator > 0 else np.nan


def neuron_metrics(rates: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (monkey, global_neuron), neuron in rates.groupby(["monkey", "global_neuron"], sort=True):
        c1 = neuron[neuron["condition"] == 1]
        metadata = neuron.iloc[0]
        for window in WINDOWS:
            train = c1[["direction", window]].dropna()
            if len(train) < 6:
                continue
            beta = np.linalg.lstsq(
                design_from_directions(train["direction"].to_numpy()),
                train[window].to_numpy(float),
                rcond=None,
            )[0]
            reliability = cv_r2(c1, window)
            for evaluation in EVALUATIONS:
                conditions = [int(evaluation)] if evaluation != "pooled" else [2, 3]
                test = neuron[neuron["condition"].isin(conditions)][
                    ["condition", "direction", window]
                ].dropna()
                if len(test) < 3:
                    continue
                chosen_design = design_from_directions(test["direction"].to_numpy())
                expected = np.vstack(
                    [set_vector(int(c), int(d)) for c, d in zip(test["condition"], test["direction"])]
                )
                wrong = np.vstack(
                    [set_vector(int(c), int(d), rotate=3) for c, d in zip(test["condition"], test["direction"])]
                )
                set_design = np.column_stack((np.ones(len(test)), expected))
                wrong_design = np.column_stack((np.ones(len(test)), wrong))
                observed = test[window].to_numpy(float)
                variance = float(np.var(observed))
                if variance <= 1e-12:
                    continue
                mse_chosen = float(np.mean((observed - chosen_design @ beta) ** 2))
                mse_set = float(np.mean((observed - set_design @ beta) ** 2))
                mse_wrong = float(np.mean((observed - wrong_design @ beta) ** 2))
                records.append(
                    {
                        "monkey": monkey,
                        "global_neuron": int(global_neuron),
                        "recording_triplet": metadata["recording_triplet"],
                        "recording_component": metadata["recording_component"],
                        "window": window,
                        "evaluation": evaluation,
                        "n_trials": len(test),
                        "cv_r2": reliability,
                        "mse_chosen": mse_chosen,
                        "mse_set": mse_set,
                        "mse_wrong": mse_wrong,
                        "delta": (mse_chosen - mse_set) / variance,
                        "wrong_control": (mse_wrong - mse_set) / variance,
                    }
                )
    return pd.DataFrame(records)


def aggregate_metrics(neurons: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["delta", "wrong_control", "cv_r2", "n_trials"]
    triplets = (
        neurons.groupby(["monkey", "recording_triplet", "window", "evaluation"], as_index=False)[columns]
        .median()
    )
    component_map = neurons[["monkey", "recording_triplet", "recording_component"]].drop_duplicates()
    triplets = triplets.merge(component_map, on=["monkey", "recording_triplet"], validate="many_to_one")
    components = (
        triplets.groupby(["monkey", "recording_component", "window", "evaluation"], as_index=False)[columns]
        .median()
    )
    return triplets, components


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def one_group_summary(values: np.ndarray, label: str) -> dict[str, object]:
    values = values[np.isfinite(values)]
    rng = stable_rng(label)
    observed = float(values.mean())
    null = (rng.choice((-1.0, 1.0), size=(N_SIGN_FLIPS, len(values))) * values).mean(axis=1)
    nonzero = values[values != 0]
    sign_p = float(stats.binomtest(int((nonzero > 0).sum()), len(nonzero), 0.5, alternative="greater").pvalue) if len(nonzero) else np.nan
    try:
        wilcoxon_p = float(stats.wilcoxon(values, alternative="greater", zero_method="wilcox").pvalue)
    except ValueError:
        wilcoxon_p = np.nan
    boot = rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1)
    return {
        "n_components": len(values),
        "mean": observed,
        "median": float(np.median(values)),
        "positive": int((values > 0).sum()),
        "negative": int((values < 0).sum()),
        "sign_test_p_one_sided": sign_p,
        "wilcoxon_p_one_sided": wilcoxon_p,
        "sign_flip_p_one_sided": float((1 + np.sum(null >= observed)) / (N_SIGN_FLIPS + 1)),
        "mean_bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


def pooled_summary(groups: dict[str, np.ndarray], label: str) -> dict[str, object]:
    rng = stable_rng(label)
    clean = {key: value[np.isfinite(value)] for key, value in groups.items()}
    monkey_means = np.asarray([values.mean() for values in clean.values()])
    observed = float(monkey_means.mean())
    null_means = []
    boot_means = []
    for values in clean.values():
        signs = rng.choice((-1.0, 1.0), size=(N_SIGN_FLIPS, len(values)))
        null_means.append((signs * values).mean(axis=1))
        boot_means.append(rng.choice(values, size=(5000, len(values)), replace=True).mean(axis=1))
    null = np.vstack(null_means).mean(axis=0)
    boot = np.vstack(boot_means).mean(axis=0)
    return {
        "n_monkeys": len(clean),
        "n_components": {key: len(value) for key, value in clean.items()},
        "equal_weight_monkey_mean": observed,
        "equal_weight_monkey_median": float(np.mean([np.median(value) for value in clean.values()])),
        "sign_flip_p_one_sided": float((1 + np.sum(null >= observed)) / (N_SIGN_FLIPS + 1)),
        "mean_bootstrap_95ci": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
    }


def summarize(components: pd.DataFrame, label: str) -> dict[str, object]:
    output: dict[str, object] = {}
    for window in WINDOWS:
        output[window] = {}
        for evaluation in EVALUATIONS:
            subset = components[(components["window"] == window) & (components["evaluation"] == evaluation)]
            groups = {
                monkey: group["delta"].to_numpy(float)
                for monkey, group in subset.groupby("monkey", sort=True)
            }
            if not groups:
                continue
            result = {monkey: one_group_summary(values, f"{label}-{window}-{evaluation}-{monkey}") for monkey, values in groups.items()}
            result["pooled_equal_monkey"] = pooled_summary(groups, f"{label}-{window}-{evaluation}-pooled")
            output[window][evaluation] = result
    return output


def paired_contrast_summary(
    components: pd.DataFrame, minuend: str, subtrahend: str, evaluation: str, label: str
) -> dict[str, object]:
    subset = components[components["evaluation"] == evaluation]
    wide = subset.pivot(
        index=["monkey", "recording_component"], columns="window", values="delta"
    ).dropna(subset=[minuend, subtrahend])
    wide["contrast"] = wide[minuend] - wide[subtrahend]
    groups = {
        monkey: group["contrast"].to_numpy(float)
        for monkey, group in wide.reset_index().groupby("monkey", sort=True)
    }
    result = {
        monkey: one_group_summary(values, f"{label}-{monkey}")
        for monkey, values in groups.items()
    }
    result["pooled_equal_monkey"] = pooled_summary(groups, f"{label}-pooled")
    return result


def measure_summary(
    components: pd.DataFrame, window: str, evaluation: str, measure: str, label: str, sign: float = 1.0
) -> dict[str, object]:
    subset = components[(components["window"] == window) & (components["evaluation"] == evaluation)]
    groups = {
        monkey: sign * group[measure].to_numpy(float)
        for monkey, group in subset.groupby("monkey", sort=True)
    }
    result = {
        monkey: one_group_summary(values, f"{label}-{monkey}")
        for monkey, values in groups.items()
    }
    result["pooled_equal_monkey"] = pooled_summary(groups, f"{label}-pooled")
    return result


def plot_results(components: pd.DataFrame, output_dir: Path) -> None:
    colors = {"2": "#0072B2", "3": "#D55E00", "pooled": "#222222"}
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(WINDOWS))
    for evaluation in EVALUATIONS:
        means, lower, upper = [], [], []
        for window in WINDOWS:
            subset = components[(components.window == window) & (components.evaluation == evaluation)]
            monkey_stats = subset.groupby("monkey")["delta"].mean()
            means.append(monkey_stats.mean())
            rng = stable_rng(f"figure-{evaluation}-{window}")
            groups = [group.delta.to_numpy() for _, group in subset.groupby("monkey")]
            boot = np.vstack([rng.choice(values, (3000, len(values)), replace=True).mean(axis=1) for values in groups]).mean(axis=0)
            lo, hi = np.quantile(boot, [0.025, 0.975])
            lower.append(means[-1] - lo)
            upper.append(hi - means[-1])
        ax.errorbar(x, means, yerr=[lower, upper], marker="o", linewidth=2, capsize=3, color=colors[evaluation], label=f"Condition {evaluation}" if evaluation != "pooled" else "Conditions 2+3")
    ax.axhline(0, color="#777777", linewidth=1)
    ax.set_xticks(x, WINDOW_LABELS)
    ax.set_ylabel("Normalized MSE improvement (set vs chosen)")
    ax.set_title("Target-set model advantage across task epochs")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_dir / "effect_timecourse.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 4.2), sharey=True)
    panel_windows = ["prep_late", "post_reveal", "movement"]
    panel_titles = ["Late preparation", "Post-revelation", "Movement (exploratory)"]
    for ax, window, title in zip(axes, panel_windows, panel_titles):
        subset = components[(components.window == window) & (components.evaluation == "pooled")]
        for index, (monkey, group) in enumerate(subset.groupby("monkey", sort=True)):
            rng = stable_rng(f"jitter-{window}-{monkey}")
            ax.scatter(index + rng.uniform(-0.12, 0.12, len(group)), group.delta, s=25, alpha=0.7, label=monkey)
            ax.plot([index - 0.18, index + 0.18], [group.delta.median()] * 2, color="black", linewidth=2)
        ax.axhline(0, color="#777777", linewidth=1)
        ax.set_xticks([0, 1], ["Joe", "Lili"])
        ax.set_title(title)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Component-level normalized improvement")
    fig.tight_layout()
    fig.savefig(output_dir / "component_effects.png", dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rates", type=Path, default=ROOT / "data" / "processed" / "gnode_uncertainty_rates.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "gnode_target_set")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rates = pd.read_csv(args.rates)
    neurons = neuron_metrics(rates)
    triplets, components = aggregate_metrics(neurons)
    neurons.to_csv(args.output_dir / "neuron_metrics.csv", index=False)
    triplets.to_csv(args.output_dir / "triplet_metrics.csv", index=False)
    components.to_csv(args.output_dir / "component_metrics.csv", index=False)

    all_summary = summarize(components, "all")
    reliable_neurons = neurons[neurons["cv_r2"] > 0].copy()
    _, reliable_components = aggregate_metrics(reliable_neurons)
    sensitivity_summary = summarize(reliable_components, "positive-cv")
    primary = all_summary["prep_late"]["pooled"]
    post = all_summary["post_reveal"]["pooled"]
    movement = all_summary["movement"]["pooled"]
    late_minus_baseline = paired_contrast_summary(
        components, "prep_late", "baseline", "pooled", "late-minus-baseline"
    )
    wrong_control = measure_summary(
        components, "prep_late", "pooled", "wrong_control", "wrong-control"
    )
    post_reversal_test = measure_summary(
        components, "post_reveal", "pooled", "delta", "post-reversal", sign=-1.0
    )
    movement_reversal_test = measure_summary(
        components, "movement", "pooled", "delta", "movement-reversal", sign=-1.0
    )
    passed = (
        primary["joe"]["median"] > 0
        and primary["lili"]["median"] > 0
        and primary["pooled_equal_monkey"]["sign_flip_p_one_sided"] < 0.05
    )
    reversal = (
        post["joe"]["median"] < 0
        and post["lili"]["median"] < 0
        and post["pooled_equal_monkey"]["equal_weight_monkey_mean"] < 0
    )
    summary = {
        "protocol": "second_dataset_protocol.md",
        "n_sign_flips": N_SIGN_FLIPS,
        "neuron_metric_rows": len(neurons),
        "triplet_metric_rows": len(triplets),
        "component_metric_rows": len(components),
        "primary_gate_pass": bool(passed),
        "post_revelation_direction_reversal": bool(reversal),
        "primary": primary,
        "post_revelation": post,
        "exploratory_movement": movement,
        "late_minus_baseline": late_minus_baseline,
        "wrong_set_control": wrong_control,
        "post_revelation_reversal_test_on_negative_delta": post_reversal_test,
        "exploratory_movement_reversal_test_on_negative_delta": movement_reversal_test,
        "all_results": all_summary,
        "positive_cv_sensitivity": sensitivity_summary,
        "positive_cv_neuron_metric_rows": len(reliable_neurons),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot_results(components, args.output_dir)
    print(json.dumps({key: summary[key] for key in ["primary_gate_pass", "post_revelation_direction_reversal", "primary", "post_revelation", "exploratory_movement", "late_minus_baseline", "wrong_set_control"]}, indent=2))


if __name__ == "__main__":
    main()
