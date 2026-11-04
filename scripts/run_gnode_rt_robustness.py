"""Post hoc robustness checks for the held-out evidence--RT association."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_gnode_rt_evidence import held_out_evidence, target_sets


ROOT = Path(__file__).resolve().parents[1]
N_FLIPS = 10_000
N_BOOTSTRAPS = 5_000


def stable_rng(label: str) -> np.random.Generator:
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def rank_residual(values: np.ndarray, trial_order: np.ndarray | None) -> np.ndarray:
    ranked = stats.rankdata(values).astype(float)
    if trial_order is None:
        return ranked - ranked.mean()
    design = np.column_stack((np.ones(len(values)), stats.rankdata(trial_order)))
    return ranked - design @ np.linalg.lstsq(design, ranked, rcond=None)[0]


def direction_correlation(
    rt: np.ndarray,
    evidence: np.ndarray,
    trial_order: np.ndarray,
    control_trial_order: bool,
) -> float:
    x = rank_residual(rt, trial_order if control_trial_order else None)
    y = rank_residual(evidence, trial_order if control_trial_order else None)
    denominator = np.sqrt(np.sum(x * x) * np.sum(y * y))
    return float(np.sum(x * y) / denominator) if denominator > 0 else np.nan


def session_correlation(
    rt: np.ndarray,
    evidence: np.ndarray,
    labels: np.ndarray,
    trial_order: np.ndarray,
    condition: int,
    min_trials: int,
    weighting: str,
    control_trial_order: bool,
) -> tuple[float, int, int]:
    set_values, set_weights, valid_directions = [], [], 0
    for directions in target_sets(condition):
        values, weights = [], []
        for direction in directions:
            selected = labels == direction
            n = int(np.sum(selected))
            if n < min_trials or np.ptp(rt[selected]) == 0:
                continue
            r = direction_correlation(
                rt[selected], evidence[selected], trial_order[selected], control_trial_order
            )
            if np.isfinite(r):
                values.append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
                weights.append(max(n - 3, 1) if weighting == "n_minus_3" else 1)
                valid_directions += 1
        if values:
            set_values.append(float(np.average(values, weights=weights)))
            set_weights.append(float(np.sum(weights)) if weighting == "n_minus_3" else 1.0)
    if not set_values:
        return np.nan, valid_directions, 0
    return float(np.tanh(np.average(set_values, weights=set_weights))), valid_directions, len(set_values)


def summarize(values: np.ndarray, label: str) -> dict[str, object]:
    values = values[np.isfinite(values)]
    if not len(values):
        return {
            "n_components": 0,
            "status": "insufficient_data",
            "mean_r": None,
            "median_r": None,
            "negative_components": 0,
            "monte_carlo_p_negative": None,
            "bootstrap_95ci": [None, None],
        }
    rng = stable_rng(label)
    observed = float(np.mean(values))
    null = np.mean(
        rng.choice((-1.0, 1.0), size=(N_FLIPS, len(values))) * values[None, :], axis=1
    )
    bootstrap = np.mean(
        rng.choice(values, size=(N_BOOTSTRAPS, len(values)), replace=True), axis=1
    )
    return {
        "n_components": int(len(values)),
        "mean_r": observed,
        "median_r": float(np.median(values)),
        "negative_components": int(np.sum(values < 0)),
        "monte_carlo_p_negative": float((1 + np.sum(null <= observed)) / (N_FLIPS + 1)),
        "bootstrap_95ci": [float(x) for x in np.quantile(bootstrap, [0.025, 0.975])],
    }


def main() -> None:
    output_dir = ROOT / "results" / "gnode_rt_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    archive = np.load(ROOT / "data" / "processed" / "gnode_rs_sliding_rates.npz")
    rates = archive["rates_hz"].astype(float)
    centers = archive["centers_ms"].astype(float)
    movement_onset = archive["movement_onset_ms"].astype(float)
    metadata = pd.read_csv(
        ROOT / "data" / "processed" / "gnode_rs_sliding_metadata.csv"
    ).reset_index(names="rate_row")
    metadata = metadata[metadata["condition"].isin([2, 3])].copy()
    fixed_window = (centers >= 100) & (centers <= 200)

    scenarios = {
        "reproduction_min5": dict(min_trials=5, weighting="equal", control=False, min_rt=50),
        "min10": dict(min_trials=10, weighting="equal", control=False, min_rt=50),
        "min15": dict(min_trials=15, weighting="equal", control=False, min_rt=50),
        "precision_weighted": dict(min_trials=5, weighting="n_minus_3", control=False, min_rt=50),
        "trial_order_adjusted": dict(min_trials=5, weighting="equal", control=True, min_rt=50),
        "premovement_225ms": dict(min_trials=5, weighting="equal", control=False, min_rt=225),
        "premovement_250ms": dict(min_trials=5, weighting="equal", control=False, min_rt=250),
        "premovement_275ms": dict(min_trials=5, weighting="equal", control=False, min_rt=275),
        "early_80_120_buffer25": dict(min_trials=5, weighting="equal", control=False,
                                      min_rt=170, window=[80,120], require_all_directions=True),
        "early_100_150_buffer25": dict(min_trials=5, weighting="equal", control=False,
                                       min_rt=200, window=[100,150], require_all_directions=True),
    }
    session_rows = []
    for (monkey, condition, session), group in metadata.groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        lookup = group.pivot(index="trial", columns="global_neuron", values="rate_row")
        row_indices = lookup.to_numpy(int)
        rt_all = movement_onset[row_indices[:, 0]]
        labels_all = (
            group[["trial", "direction"]]
            .drop_duplicates()
            .set_index("trial")
            .loc[lookup.index, "direction"]
            .to_numpy(int)
        )
        base_valid = (rt_all >= 50) & (rt_all <= 600)
        base_rows = row_indices[base_valid]
        base_rt = rt_all[base_valid]
        base_labels = labels_all[base_valid]
        base_trials = lookup.index.to_numpy()[base_valid]
        base_evidence = held_out_evidence(rates[base_rows], base_labels, int(condition))
        fixed_evidence = base_evidence[:, fixed_window].mean(axis=1)
        for name, spec in scenarios.items():
            selected = base_rt >= spec["min_rt"]
            start, stop = spec.get("window", [100,200])
            scenario_evidence = base_evidence[:, (centers >= start) & (centers <= stop)].mean(axis=1)
            r, n_directions, n_sets = session_correlation(
                base_rt[selected],
                scenario_evidence[selected],
                base_labels[selected],
                base_trials[selected],
                int(condition),
                spec["min_trials"],
                spec["weighting"],
                spec["control"],
            )
            if spec.get("require_all_directions", False) and n_directions != 6:
                r = np.nan
            session_rows.append(
                dict(
                    scenario=name,
                    monkey=monkey,
                    condition=int(condition),
                    session=int(session),
                    recording_component=group["recording_component"].iloc[0],
                    r=r,
                    n_trials=int(np.sum(selected)),
                    valid_directions=n_directions,
                    valid_sets=n_sets,
                    window_start_ms=start,
                    window_stop_ms=stop,
                    sample_support_stop_ms=stop+25,
                )
            )

    sessions = pd.DataFrame(session_rows)
    sessions.to_csv(output_dir / "session_sensitivity_correlations.csv", index=False)
    component_rows = []
    for keys, group in sessions.groupby(
        ["scenario", "monkey", "condition", "recording_component"], sort=True
    ):
        values = group["r"].dropna().to_numpy()
        component_rows.append((*keys, float(np.median(values)) if len(values) else np.nan))
    components = pd.DataFrame(
        component_rows, columns=["scenario", "monkey", "condition", "recording_component", "r"]
    )
    pooled_rows = []
    for keys, group in components.groupby(["scenario", "monkey", "recording_component"]):
        by_condition = group.set_index("condition")["r"]
        if {2, 3}.issubset(by_condition.index) and np.isfinite(by_condition.loc[[2, 3]]).all():
            pooled_rows.append((*keys, float(np.mean(by_condition.loc[[2, 3]]))))
    pooled = pd.DataFrame(pooled_rows, columns=["scenario", "monkey", "recording_component", "r"])
    pooled.to_csv(output_dir / "component_sensitivity_correlations.csv", index=False)

    summary = {"post_hoc": True, "fixed_window_centers_ms": [100, 200], "window_support_ms": [75, 225]}
    summary["scenarios"] = {}
    for scenario, spec in scenarios.items():
        summary["scenarios"][scenario] = {"settings": spec, "animals": {}}
        for monkey in ("joe", "lili"):
            values = pooled.loc[(pooled.scenario == scenario) & (pooled.monkey == monkey), "r"].to_numpy()
            summary["scenarios"][scenario]["animals"][monkey] = summarize(
                values, f"rt-robustness-{scenario}-{monkey}"
            )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["scenarios"], indent=2))


if __name__ == "__main__":
    main()
