"""Validate revised estimands, empirical decoder calibration and buffered RT rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def read(name: str) -> dict:
    return json.loads((ROOT / "results" / name / "summary.json").read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-root", type=Path)
    args = parser.parse_args()
    checks = {}
    encoding = read("gnode_encoding_controls")
    checks["prediction_decomposition"] = encoding["decomposition_max_abs_error"] < 1e-12
    controls = pd.read_csv(ROOT / "results/gnode_encoding_controls/component_controls.csv")
    checks["finite_encoding_controls"] = bool(np.isfinite(controls.select_dtypes("number")).all().all())
    checks["encoding_component_counts"] = controls.groupby("monkey").recording_component.nunique().to_dict() == {"joe":30,"lili":48}
    calibration = read("gnode_decoder_calibration")
    checks["decoder_implementation_identity"] = calibration["decoder_implementation_max_error"] < 1e-12
    arrays = np.load(ROOT / "results/gnode_decoder_calibration/calibration_arrays.npz")
    checks["finite_calibration"] = all(np.isfinite(arrays[key]).all() for key in ("observed","permuted","simulated"))
    checks["permutation_shape"] = arrays["permuted"].shape == (78,3,calibration["n_permutations"],84)
    main_curves = pd.read_csv(ROOT / "results/gnode_within_set_decoding/component_timecourses.csv")
    maximum_error = 0.0
    for monkey in ("joe","lili"):
        original = main_curves[(main_curves.monkey==monkey) & (main_curves.evaluation.astype(str)=="pooled")]
        original = original.groupby("center_ms").accuracy_above_chance.mean().reindex(arrays["centers_ms"]).to_numpy()
        observed = arrays["observed"][arrays["monkey"]==monkey,2].mean(axis=0)
        maximum_error = max(maximum_error,float(np.max(np.abs(original-observed))))
    checks["calibration_main_decoder_identity"] = maximum_error < 1e-12
    robust = read("gnode_decoding_robustness")
    checks["bound_semantics"] = True
    for animals in robust["results"].values():
        for result in animals.values():
            lower,upper = result["pre_equivalence"]["bootstrap_95ci"]
            for bound,decision in result["sesoi_grid"].items():
                checks["bound_semantics"] &= bool(decision) == (upper < float(bound))
                checks["bound_semantics"] &= result["two_sided_95ci_grid"][bound] == (lower > -float(bound) and upper < float(bound))
    sessions = pd.read_csv(ROOT / "results/gnode_rt_robustness/session_sensitivity_correlations.csv")
    rt = read("gnode_rt_robustness")
    early = sessions[sessions.scenario.str.startswith("early_")].dropna(subset=["r"])
    checks["buffered_complete_direction_strata"] = bool(early.valid_directions.eq(6).all())
    checks["buffered_window_support"] = all(
        rt["scenarios"][name]["settings"]["min_rt"] - stop >= 25
        for name,stop in (("early_80_120_buffer25",145),("early_100_150_buffer25",175))
    )
    crossing = read("gnode_switch_dynamics")
    checks["separate_raw_crossings"] = all("pooled" in crossing["raw_crossings"][m] for m in ("joe","lili"))
    if args.clean_root is not None:
        cleaned = args.clean_root / "data/processed"
        checks["cleanroom_inventory_generated"] = (args.clean_root / "results/gnode_uncertainty_inventory.csv").exists()
        name = "gnode_uncertainty_rates.csv"
        checks["cleanroom_trial_table_identity"] = hashlib.sha256((cleaned/name).read_bytes()).digest() == hashlib.sha256((ROOT/"data/processed"/name).read_bytes()).digest()
        fresh = np.load(cleaned / "gnode_rs_sliding_rates.npz")
        original = np.load(ROOT / "data/processed/gnode_rs_sliding_rates.npz")
        checks["cleanroom_array_identity"] = set(fresh.files) == set(original.files) and all(np.array_equal(fresh[k],original[k],equal_nan=True) for k in fresh.files)
    report = {"checks":{k:bool(v) for k,v in checks.items()},
              "calibration_main_decoder_max_error":maximum_error,"all_pass":bool(all(checks.values()))}
    (ROOT / "results/gnode_decoder_calibration/revision_validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
