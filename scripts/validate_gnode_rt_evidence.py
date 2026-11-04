"""Validate the reaction-time evidence outputs and inferential criteria."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_root = ROOT / "results" / "gnode_rt_evidence"
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    primary = pd.read_csv(result_root / "component_primary_correlations.csv")
    sessions = pd.read_csv(result_root / "session_rt_inventory.csv")
    curves = np.load(result_root / "component_curves.npz", allow_pickle=True)

    checks: dict[str, bool] = {}
    checks["frozen_budgets"] = (
        summary["n_sign_flips"] == 10_000
        and summary["n_bootstraps"] == 5_000
        and summary["rt_bounds_ms"] == [50, 600]
        and summary["fixed_evidence_window_ms"] == [100, 200]
    )
    checks["trial_accounting"] = (
        summary["retained_trials_across_sessions"] == int(sessions["n_trials"].sum())
        and summary["excluded_rt_trials_across_sessions"] == 219
    )
    checks["session_count"] = len(sessions) == 160
    checks["component_counts"] = (
        primary[["monkey", "recording_component"]]
        .drop_duplicates()["monkey"]
        .value_counts()
        .to_dict()
        == {"lili": 48, "joe": 30}
    )
    checks["primary_finite_and_bounded"] = bool(
        np.isfinite(primary["fixed_window_r"]).all()
        and primary["fixed_window_r"].between(-1, 1).all()
    )
    centers = curves["centers_ms"]
    confirmatory = (centers >= 80) & (centers <= 300)
    checks["confirmatory_curves_finite"] = bool(
        np.isfinite(curves["correlation"][:, :, confirmatory]).all()
    )
    checks["alignment_shapes"] = (
        curves["fast_rs"].shape == curves["slow_rs"].shape
        and curves["fast_mo"].shape == curves["slow_mo"].shape
        and curves["fast_mo"].shape[-1] == 41
    )
    for monkey in ("joe", "lili"):
        pooled = summary["primary"][monkey]["pooled"]
        checks[f"{monkey}_negative_primary"] = pooled["median_r"] < 0
        checks[f"{monkey}_significant_primary"] = pooled["sign_flip_p_negative"] < 0.05
    checks["claim_gate_consistent"] = summary["rt_association_claim_pass"] == all(
        summary["success_by_monkey"].values()
    )
    checks = {key: bool(value) for key, value in checks.items()}
    report = {"checks": checks, "all_pass": bool(all(checks.values()))}
    (result_root / "validation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
