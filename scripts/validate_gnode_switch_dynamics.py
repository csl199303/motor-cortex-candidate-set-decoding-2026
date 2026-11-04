"""Validate time-resolved switching and cross-temporal output invariants."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    preprocess = json.loads(
        (ROOT / "data" / "processed" / "gnode_rs_sliding_rates.json").read_text(
            encoding="utf-8"
        )
    )
    result_root = ROOT / "results" / "gnode_switch_dynamics"
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    timecourses = pd.read_csv(result_root / "component_timecourses.csv")
    cross = np.load(result_root / "cross_temporal_matrices.npz")
    centers = cross["centers_ms"].astype(float)

    checks: dict[str, bool] = {}
    checks["sliding_preprocess_qa"] = bool(preprocess["qa_pass"])
    checks["frozen_time_axis"] = bool(
        len(centers) == 161
        and centers[0] == -950
        and centers[-1] == 650
        and np.allclose(np.diff(centers), 10)
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
    checks["permutation_budget"] = summary["n_sign_flips"] == 10_000
    checks["bootstrap_budget"] = summary["n_bootstraps"] == 5_000

    diagonal_errors = {}
    for monkey in ("joe", "lili"):
        raw = (
            timecourses[
                (timecourses["monkey"] == monkey)
                & (timecourses["evaluation"].astype(str) == "pooled")
            ]
            .groupby("center_ms")["delta_raw"]
            .mean()
            .reindex(centers)
            .to_numpy()
        )
        diagonal_errors[monkey] = float(np.nanmax(np.abs(raw - np.diag(cross[monkey]))))
    checks["cross_temporal_diagonal_identity"] = max(diagonal_errors.values()) < 1e-7
    checks["finite_switches"] = all(
        np.isfinite(summary["latencies"][monkey]["pooled"]["observed_ms"])
        for monkey in ("joe", "lili")
    )
    checks["claim_gate_consistent"] = summary["switching_claim_pass"] == all(
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
