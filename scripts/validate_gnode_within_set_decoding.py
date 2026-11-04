"""Validate invariants of the simultaneous-session within-set decoder."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_root = ROOT / "results" / "gnode_within_set_decoding"
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    timecourses = pd.read_csv(result_root / "component_timecourses.csv")
    sessions = pd.read_csv(result_root / "session_timecourses_wide.csv")
    cross = np.load(result_root / "cross_temporal_decoding.npz")
    centers = cross["centers_ms"].astype(float)

    checks: dict[str, bool] = {}
    checks["frozen_budgets"] = (
        summary["n_folds"] == 5
        and summary["n_sign_flips"] == 10_000
        and summary["n_bootstraps"] == 5_000
        and summary["sesoi"] == 0.05
    )
    checks["session_count"] = len(sessions) == 160
    checks["session_component_uniqueness"] = bool(
        sessions.groupby(["monkey", "condition", "session"])[
            "recording_component"
        ].nunique().max()
        == 1
    )
    counts = (
        timecourses[["monkey", "recording_component"]]
        .drop_duplicates()["monkey"]
        .value_counts()
        .to_dict()
    )
    checks["component_counts"] = counts == {"lili": 48, "joe": 30}
    checks["unique_component_time_rows"] = not timecourses.duplicated(
        ["monkey", "recording_component", "evaluation", "center_ms"]
    ).any()
    checks["finite_component_scores"] = bool(
        np.isfinite(timecourses["accuracy_above_chance"]).all()
    )
    checks["score_bounds"] = bool(
        timecourses["accuracy_above_chance"].between(-0.5, 2.0 / 3.0).all()
    )

    diagonal_errors = {}
    for monkey in ("joe", "lili"):
        curve = (
            timecourses[
                (timecourses["monkey"] == monkey)
                & (timecourses["evaluation"].astype(str) == "pooled")
            ]
            .groupby("center_ms")["accuracy_above_chance"]
            .mean()
            .reindex(centers)
            .to_numpy()
        )
        diagonal_errors[monkey] = float(
            np.max(np.abs(curve - np.diag(cross[monkey])))
        )
    checks["cross_temporal_diagonal_identity"] = max(diagonal_errors.values()) < 1e-7

    for monkey in ("joe", "lili"):
        pooled = summary["inference"][monkey]["pooled"]
        checks[f"{monkey}_no_pre_cluster"] = not any(
            cluster["p_corrected"] < 0.05 for cluster in pooled["pre"]["clusters"]
        )
        checks[f"{monkey}_pre_equivalent"] = pooled["pre_equivalence"][
            "upper_below_sesoi"
        ]
        checks[f"{monkey}_post_cluster"] = any(
            cluster["p_corrected"] < 0.05 for cluster in pooled["post"]["clusters"]
        )
    checks["claim_gate_consistent"] = summary["emergence_claim_pass"] == all(
        summary["success_by_monkey"].values()
    )
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "checks": checks,
        "cross_temporal_diagonal_max_abs_error": diagonal_errors,
        "all_pass": bool(all(checks.values())),
    }
    (result_root / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
