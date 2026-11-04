"""Validate mechanism-model LOCO scores, rankings, and decision logic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir", type=Path, default=ROOT / "results" / "gnode_mechanism_models"
    )
    args = parser.parse_args()
    summary = json.loads((args.result_dir / "summary.json").read_text(encoding="utf-8"))
    scores = pd.read_csv(args.result_dir / "loco_scores.csv")

    checks: dict[str, bool] = {}
    checks["frozen_model_set"] = (
        summary["models"] == ["latency", "rate", "threshold"]
        and summary["parameters_per_model"] == 6
        and summary["fit_interval_ms"] == [-50, 220]
    )
    checks["budgets"] = (
        summary["n_bootstraps"] == 5_000 and summary["n_sign_flips"] == 10_000
    )
    checks["score_rows"] = len(scores) == (30 + 48) * 3
    checks["score_uniqueness"] = not scores.duplicated(
        ["monkey", "recording_component", "model"]
    ).any()
    checks["scores_finite_nonnegative"] = bool(
        np.isfinite(scores[["joint_score", "evidence_score", "rt_score"]]).all(axis=None)
        and (scores[["joint_score", "evidence_score", "rt_score"]] >= 0).all(axis=None)
    )
    checks["component_counts"] = (
        scores[["monkey", "recording_component"]]
        .drop_duplicates()["monkey"]
        .value_counts()
        .to_dict()
        == {"lili": 48, "joe": 30}
    )
    recomputed_winners = {}
    for monkey in ("joe", "lili"):
        means = (
            scores[scores["monkey"] == monkey]
            .groupby("model")["joint_score"]
            .mean()
            .sort_values()
        )
        recomputed_winners[monkey] = means.index[0]
        stored = summary["comparisons"][monkey]["ranking"]
        checks[f"{monkey}_ranking_reproduced"] = all(
            np.isclose(means[model], stored[model], rtol=0, atol=1e-12)
            for model in means.index
        )
        for model in ("latency", "rate", "threshold"):
            parameters = np.asarray(summary["full_fits"][monkey][model]["parameters"])
            predicted_rt = np.asarray(
                summary["full_fits"][monkey][model]["predicted_rt_ms"]
            )
            checks[f"{monkey}_{model}_fit_finite"] = bool(
                len(parameters) == 6
                and np.isfinite(parameters).all()
                and np.isfinite(predicted_rt).all()
            )
    checks["winner_labels_reproduced"] = recomputed_winners == summary["winner_by_monkey"]
    expected_unique = bool(
        len(set(recomputed_winners.values())) == 1
        and all(
            summary["comparisons"][monkey]["winner_vs_second"]["bootstrap_95ci"][0]
            > 0
            for monkey in ("joe", "lili")
        )
    )
    checks["decision_gate_reproduced"] = (
        summary["unique_mechanism_winner"] == expected_unique
        and summary["decision"]
        == (recomputed_winners["joe"] if expected_unique else "inconclusive_or_mixed")
    )
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"checks": checks, "all_pass": bool(all(checks.values()))}
    (args.result_dir / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
