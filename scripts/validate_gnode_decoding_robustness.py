"""Validate decoding robustness outputs and internal consistency."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result_root = ROOT / "results" / "gnode_decoding_robustness"
    summary = json.loads((result_root / "summary.json").read_text(encoding="utf-8"))
    table = pd.read_csv(result_root / "robustness_summary.csv")
    sessions = pd.read_csv(result_root / "gaussian_nb_session_timecourses.csv")

    expected = {
        "nearest_centroid",
        "gaussian_naive_bayes",
        "nearest_centroid_min_2_units",
        "nearest_centroid_min_3_units",
    }
    checks: dict[str, bool] = {}
    checks["analysis_set"] = set(summary["results"]) == expected
    checks["table_rows"] = len(table) == 2 * len(expected)
    checks["animals_complete"] = bool(
        table.groupby("analysis")["monkey"].apply(set).eq({"joe", "lili"}).all()
    )
    checks["finite_pre_results"] = bool(
        np.isfinite(table[["pre_mean", "pre_ci_low", "pre_ci_high"]]).all().all()
    )
    checks["ordered_pre_intervals"] = bool(
        (table["pre_ci_low"] <= table["pre_mean"]).all()
        and (table["pre_mean"] <= table["pre_ci_high"]).all()
    )
    checks["component_counts_positive"] = bool((table["n_components"] > 0).all())
    checks["gnb_session_count"] = len(sessions) == 160
    time_columns = [column for column in sessions if column.startswith("t_")]
    checks["gnb_scores_finite"] = bool(
        np.isfinite(sessions[time_columns].to_numpy(float)).all()
    )
    checks["sesoi_consistency"] = True
    checks["post_start_consistency"] = True
    for analysis, animal_results in summary["results"].items():
        for monkey, values in animal_results.items():
            upper = values["pre_equivalence"]["bootstrap_95ci"][1]
            for bound, decision in values["sesoi_grid"].items():
                checks["sesoi_consistency"] &= bool(decision) == (upper < float(bound))
            significant = [
                cluster
                for cluster in values["post"]["clusters"]
                if cluster["p_corrected"] < 0.05
            ]
            expected_start = significant[0]["start_ms"] if significant else None
            checks["post_start_consistency"] &= (
                values["first_significant_post_center_ms"] == expected_start
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
