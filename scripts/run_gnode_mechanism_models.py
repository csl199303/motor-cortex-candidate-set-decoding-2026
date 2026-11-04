"""Compare latency, accumulation-rate, and variable-threshold evidence models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit


ROOT = Path(__file__).resolve().parents[1]
MODELS = ("latency", "rate", "threshold")
N_BOOTSTRAPS = 5_000
N_SIGN_FLIPS = 10_000

BOUNDS = {
    "latency": (
        np.asarray([-0.5, 0.01, 5.0, -100.0, -100.0, 0.01]),
        np.asarray([0.2, 1.0, 300.0, 300.0, 300.0, 0.99]),
    ),
    "rate": (
        np.asarray([-0.5, 0.01, -100.0, 5.0, 5.0, 0.01]),
        np.asarray([0.2, 1.0, 300.0, 300.0, 300.0, 0.99]),
    ),
    "threshold": (
        np.asarray([-0.5, 0.01, -100.0, 5.0, 0.01, 0.01]),
        np.asarray([0.2, 1.0, 300.0, 300.0, 0.99, 0.99]),
    ),
}


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def predict(model: str, parameters: np.ndarray, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if model == "latency":
        baseline, amplitude, tau, latency_fast, latency_slow, threshold = parameters
        latencies = np.asarray([latency_fast, latency_slow])
        taus = np.asarray([tau, tau])
        thresholds = np.asarray([threshold, threshold])
    elif model == "rate":
        baseline, amplitude, latency, tau_fast, tau_slow, threshold = parameters
        latencies = np.asarray([latency, latency])
        taus = np.asarray([tau_fast, tau_slow])
        thresholds = np.asarray([threshold, threshold])
    else:
        baseline, amplitude, latency, tau, threshold_fast, threshold_slow = parameters
        latencies = np.asarray([latency, latency])
        taus = np.asarray([tau, tau])
        thresholds = np.asarray([threshold_fast, threshold_slow])
    curves = baseline + amplitude * expit(
        (time[None, :] - latencies[:, None]) / taus[:, None]
    )
    reaction_times = latencies + taus * logit(thresholds)
    return curves, reaction_times


def robust_scales(curves: np.ndarray, reaction_times: np.ndarray) -> tuple[float, float]:
    curve_center = np.median(curves, axis=0, keepdims=True)
    curve_mad = np.median(np.abs(curves - curve_center))
    rt_center = np.median(reaction_times, axis=0, keepdims=True)
    rt_mad = np.median(np.abs(reaction_times - rt_center))
    return max(1.4826 * curve_mad, 1e-3), max(1.4826 * rt_mad, 1.0)


def residuals(
    parameters: np.ndarray,
    model: str,
    time: np.ndarray,
    curves: np.ndarray,
    reaction_times: np.ndarray,
    evidence_scale: float,
    rt_scale: float,
) -> np.ndarray:
    predicted_curves, predicted_rt = predict(model, parameters, time)
    evidence_residual = (curves - predicted_curves[None, :, :]) / evidence_scale
    rt_residual = (reaction_times - predicted_rt[None, :]) / rt_scale
    evidence_residual /= np.sqrt(curves.shape[1] * curves.shape[2])
    rt_residual /= np.sqrt(reaction_times.shape[1])
    return np.concatenate((evidence_residual.ravel(), rt_residual.ravel()))


def initial_guess(model: str, curves: np.ndarray, reaction_times: np.ndarray) -> np.ndarray:
    baseline = float(np.clip(np.median(curves[:, :, :4]), -0.4, 0.1))
    amplitude = float(np.clip(np.median(np.max(curves, axis=2)) - baseline, 0.03, 0.8))
    median_rt = np.median(reaction_times, axis=0)
    if model == "latency":
        return np.asarray([baseline, amplitude, 45.0, 90.0, 100.0, 0.85])
    if model == "rate":
        return np.asarray([baseline, amplitude, 95.0, 35.0, 50.0, 0.85])
    common_rt = float(np.mean(median_rt))
    tau = 45.0
    latency = 95.0
    thresholds = expit((median_rt - latency) / tau)
    return np.asarray([baseline, amplitude, latency, tau, thresholds[0], thresholds[1]])


def fit_model(
    model: str,
    time: np.ndarray,
    curves: np.ndarray,
    reaction_times: np.ndarray,
    evidence_scale: float,
    rt_scale: float,
    label: str,
    warm_start: np.ndarray | None = None,
    multistart: bool = False,
) -> tuple[np.ndarray, float]:
    lower, upper = BOUNDS[model]
    starts = []
    if warm_start is not None:
        starts.append(np.clip(warm_start, lower + 1e-6, upper - 1e-6))
    starts.append(initial_guess(model, curves, reaction_times))
    if multistart:
        rng = stable_rng(label)
        for _ in range(14):
            starts.append(lower + (upper - lower) * rng.uniform(size=len(lower)))
    best_parameters, best_cost = None, np.inf
    for start in starts:
        result = least_squares(
            residuals,
            np.clip(start, lower + 1e-6, upper - 1e-6),
            bounds=(lower, upper),
            args=(model, time, curves, reaction_times, evidence_scale, rt_scale),
            max_nfev=5000,
            xtol=1e-10,
            ftol=1e-10,
            gtol=1e-10,
        )
        if result.cost < best_cost:
            best_parameters, best_cost = result.x, float(result.cost)
    if best_parameters is None:
        raise RuntimeError(f"No fit completed for {model}, {label}")
    return best_parameters, best_cost


def held_out_score(
    model: str,
    parameters: np.ndarray,
    time: np.ndarray,
    curves: np.ndarray,
    reaction_times: np.ndarray,
    evidence_scale: float,
    rt_scale: float,
) -> tuple[float, float, float]:
    predicted_curves, predicted_rt = predict(model, parameters, time)
    evidence_mse = float(np.mean(((curves - predicted_curves) / evidence_scale) ** 2))
    rt_mse = float(np.mean(((reaction_times - predicted_rt) / rt_scale) ** 2))
    return 0.5 * evidence_mse + 0.5 * rt_mse, evidence_mse, rt_mse


def paired_comparison(
    winner_scores: np.ndarray, competitor_scores: np.ndarray, label: str
) -> dict[str, object]:
    improvement = competitor_scores - winner_scores
    rng = stable_rng(label)
    bootstrap = rng.choice(
        improvement, size=(N_BOOTSTRAPS, len(improvement)), replace=True
    ).mean(axis=1)
    observed = float(np.mean(improvement))
    null = (
        rng.choice((-1.0, 1.0), size=(N_SIGN_FLIPS, len(improvement)))
        * improvement[None, :]
    ).mean(axis=1)
    return {
        "mean_score_improvement": observed,
        "median_score_improvement": float(np.median(improvement)),
        "bootstrap_95ci": [float(value) for value in np.quantile(bootstrap, [0.025, 0.975])],
        "paired_sign_flip_p_two_sided": float(
            (1 + np.sum(np.abs(null) >= abs(observed))) / (N_SIGN_FLIPS + 1)
        ),
        "components_favoring_winner": int(np.sum(improvement > 0)),
        "n_components": len(improvement),
    }


def plot_fits(
    data: dict[str, dict[str, object]], time: np.ndarray, output: Path
) -> None:
    model_colors = {"latency": "#0072B2", "rate": "#D55E00", "threshold": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2), sharex=True, sharey=True)
    for row, monkey in enumerate(("joe", "lili")):
        observed = data[monkey]["curves"]
        for group_index, group_name in enumerate(("Fast RT", "Slow RT")):
            ax = axes[row, group_index]
            mean = observed[:, group_index].mean(axis=0)
            sem = observed[:, group_index].std(axis=0, ddof=1) / np.sqrt(len(observed))
            ax.fill_between(time, mean - 1.96 * sem, mean + 1.96 * sem, color="#999999", alpha=0.25)
            ax.plot(time, mean, color="black", linewidth=2.4, label="Observed")
            for model in MODELS:
                prediction, _ = predict(model, data[monkey]["parameters"][model], time)
                ax.plot(time, prediction[group_index], color=model_colors[model], linewidth=2, label=model.capitalize())
            ax.axvline(0, color="#777777", linestyle="--", linewidth=1)
            ax.set_title(f"{monkey.capitalize()} - {group_name}")
            ax.set_xlabel("Time from response signal (ms)")
            ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("Correct-direction evidence")
    axes[1, 0].set_ylabel("Correct-direction evidence")
    axes[0, 1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_scores(scores: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    colors = {"latency": "#0072B2", "rate": "#D55E00", "threshold": "#009E73"}
    for ax, monkey in zip(axes, ("joe", "lili")):
        subset = scores[scores["monkey"] == monkey]
        for index, model in enumerate(MODELS):
            values = subset[subset["model"] == model]["joint_score"].to_numpy()
            jitter = stable_rng(f"score-jitter-{monkey}-{model}").uniform(-0.08, 0.08, len(values))
            ax.scatter(index + jitter, values, s=20, alpha=0.45, color=colors[model])
            ax.plot([index - 0.18, index + 0.18], [np.mean(values)] * 2, color="black", linewidth=2)
        ax.set_xticks(range(3), ["Latency", "Rate", "Threshold"])
        ax.set_title(monkey.capitalize())
        ax.set_ylabel("Held-out standardized joint MSE")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--curves", type=Path, default=ROOT / "results" / "gnode_rt_evidence" / "component_curves.npz"
    )
    parser.add_argument(
        "--session-rt",
        type=Path,
        default=ROOT / "results" / "gnode_rt_evidence" / "session_rt_inventory.csv",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "results" / "gnode_mechanism_models"
    )
    parser.add_argument("--q-upper", type=float, default=0.99)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not 0.5 < args.q_upper < 1.0:
        raise ValueError("--q-upper must be between 0.5 and 1.0")
    BOUNDS["latency"][1][-1] = args.q_upper
    BOUNDS["rate"][1][-1] = args.q_upper
    BOUNDS["threshold"][1][-2:] = args.q_upper

    archive = np.load(args.curves, allow_pickle=True)
    centers = archive["centers_ms"].astype(float)
    interval = (centers >= -50) & (centers <= 220)
    time = centers[interval]
    monkey_values = archive["monkey"].astype(str)
    component_values = archive["component"].astype(str)
    fast_curves = archive["fast_rs"][:, 2, :][:, interval].astype(float)
    slow_curves = archive["slow_rs"][:, 2, :][:, interval].astype(float)

    session_rt = pd.read_csv(args.session_rt)
    condition_rt = (
        session_rt.groupby(
            ["monkey", "condition", "recording_component"], as_index=False
        )[["fast_median_rt_ms", "slow_median_rt_ms"]]
        .median()
    )
    component_rt = (
        condition_rt.groupby(["monkey", "recording_component"], as_index=False)[
            ["fast_median_rt_ms", "slow_median_rt_ms"]
        ]
        .mean()
    )
    rt_lookup = component_rt.set_index(["monkey", "recording_component"])

    rows, full_results, analysis_data = [], {}, {}
    for monkey in ("joe", "lili"):
        selected = np.flatnonzero(monkey_values == monkey)
        curves = np.stack((fast_curves[selected], slow_curves[selected]), axis=1)
        components = component_values[selected]
        reaction_times = np.asarray(
            [
                rt_lookup.loc[(monkey, component), ["fast_median_rt_ms", "slow_median_rt_ms"]].to_numpy(float)
                for component in components
            ]
        )
        evidence_scale, rt_scale = robust_scales(curves, reaction_times)
        full_results[monkey] = {}
        analysis_data[monkey] = {
            "curves": curves,
            "reaction_times": reaction_times,
            "components": components,
            "parameters": {},
        }
        for model in MODELS:
            full_parameters, full_cost = fit_model(
                model,
                time,
                curves,
                reaction_times,
                evidence_scale,
                rt_scale,
                f"full-{monkey}-{model}",
                multistart=True,
            )
            analysis_data[monkey]["parameters"][model] = full_parameters
            held_out_scores = []
            for held_out in range(len(components)):
                train = np.arange(len(components)) != held_out
                fold_evidence_scale, fold_rt_scale = robust_scales(
                    curves[train], reaction_times[train]
                )
                parameters, _ = fit_model(
                    model,
                    time,
                    curves[train],
                    reaction_times[train],
                    fold_evidence_scale,
                    fold_rt_scale,
                    f"loco-{monkey}-{model}-{held_out}",
                    multistart=True,
                )
                joint, evidence_mse, rt_mse = held_out_score(
                    model,
                    parameters,
                    time,
                    curves[held_out],
                    reaction_times[held_out],
                    fold_evidence_scale,
                    fold_rt_scale,
                )
                rows.append(
                    {
                        "monkey": monkey,
                        "recording_component": components[held_out],
                        "model": model,
                        "joint_score": joint,
                        "evidence_score": evidence_mse,
                        "rt_score": rt_mse,
                        "training_evidence_scale": fold_evidence_scale,
                        "training_rt_scale": fold_rt_scale,
                    }
                )
                held_out_scores.append(joint)
            predicted_curves, predicted_rt = predict(model, full_parameters, time)
            full_results[monkey][model] = {
                "parameters": [float(value) for value in full_parameters],
                "full_fit_cost": full_cost,
                "mean_loco_score": float(np.mean(held_out_scores)),
                "median_loco_score": float(np.median(held_out_scores)),
                "predicted_rt_ms": [float(value) for value in predicted_rt],
                "observed_component_median_rt_ms": [
                    float(value) for value in np.median(reaction_times, axis=0)
                ],
            }
        full_results[monkey]["scales"] = {
            "evidence": evidence_scale,
            "rt_ms": rt_scale,
        }

    scores = pd.DataFrame(rows)
    scores.to_csv(args.output_dir / "loco_scores.csv", index=False)
    comparisons, winners = {}, {}
    for monkey in ("joe", "lili"):
        means = (
            scores[scores["monkey"] == monkey]
            .groupby("model")["joint_score"]
            .mean()
            .sort_values()
        )
        winner, second = means.index[:2]
        winners[monkey] = winner
        pivot = scores[scores["monkey"] == monkey].pivot(
            index="recording_component", columns="model", values="joint_score"
        )
        comparisons[monkey] = {
            "ranking": {model: float(value) for model, value in means.items()},
            "winner": winner,
            "second_best": second,
            "winner_vs_second": paired_comparison(
                pivot[winner].to_numpy(),
                pivot[second].to_numpy(),
                f"comparison-{monkey}-{winner}-{second}",
            ),
        }
    same_winner = len(set(winners.values())) == 1
    unique_winner = bool(
        same_winner
        and all(
            comparisons[monkey]["winner_vs_second"]["bootstrap_95ci"][0] > 0
            for monkey in ("joe", "lili")
        )
    )
    summary = {
        "protocol": "mechanism_model_protocol.md",
        "fit_interval_ms": [-50, 220],
        "models": list(MODELS),
        "parameters_per_model": 6,
        "threshold_fraction_upper_bound": args.q_upper,
        "n_bootstraps": N_BOOTSTRAPS,
        "n_sign_flips": N_SIGN_FLIPS,
        "full_fits": full_results,
        "comparisons": comparisons,
        "winner_by_monkey": winners,
        "unique_mechanism_winner": unique_winner,
        "decision": winners["joe"] if unique_winner else "inconclusive_or_mixed",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_fits(analysis_data, time, args.output_dir / "mechanism_curve_fits.png")
    plot_scores(scores, args.output_dir / "mechanism_loco_scores.png")
    print(
        json.dumps(
            {
                "winner_by_monkey": winners,
                "unique_mechanism_winner": unique_winner,
                "decision": summary["decision"],
                "comparisons": comparisons,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
