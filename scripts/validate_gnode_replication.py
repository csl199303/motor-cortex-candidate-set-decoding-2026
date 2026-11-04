"""Independent consistency checks for the G-Node replication artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    inventory_summary = json.loads(
        (ROOT / "results" / "gnode_uncertainty_inventory.json").read_text(encoding="utf-8")
    )
    rates = pd.read_csv(ROOT / "data" / "processed" / "gnode_uncertainty_rates.csv")
    neurons = pd.read_csv(ROOT / "results" / "gnode_target_set" / "neuron_metrics.csv")
    components = pd.read_csv(ROOT / "results" / "gnode_target_set" / "component_metrics.csv")
    summary = json.loads(
        (ROOT / "results" / "gnode_target_set" / "summary.json").read_text(encoding="utf-8")
    )

    checks: dict[str, bool] = {}
    checks["inventory_integrity"] = bool(inventory_summary["integrity_pass"])
    checks["matched_neuron_count"] = (
        rates[["monkey", "global_neuron"]].drop_duplicates().shape[0] == 221
    )
    checks["no_duplicate_unit_trials"] = not rates.duplicated(
        ["monkey", "global_neuron", "condition", "session", "trial"]
    ).any()
    checks["directions_valid"] = rates["direction"].isin(range(1, 7)).all()
    checks["fixed_rates_finite"] = bool(
        np.isfinite(
            rates[["baseline", "prep_early", "prep_middle", "prep_late", "movement"]]
        ).all(axis=None)
    )
    checks["component_counts"] = (
        components[["monkey", "recording_component"]]
        .drop_duplicates()["monkey"]
        .value_counts()
        .to_dict()
        == {"lili": 48, "joe": 30}
    )

    primary_neurons = neurons[
        (neurons["window"] == "prep_late") & (neurons["evaluation"] == "pooled")
    ]
    recomputed = (
        primary_neurons.groupby(["monkey", "recording_triplet"], as_index=False)["delta"]
        .median()
        .merge(
            primary_neurons[["monkey", "recording_triplet", "recording_component"]]
            .drop_duplicates(),
            on=["monkey", "recording_triplet"],
            validate="one_to_one",
        )
        .groupby(["monkey", "recording_component"], as_index=False)["delta"]
        .median()
    )
    stored = components[
        (components["window"] == "prep_late") & (components["evaluation"] == "pooled")
    ][["monkey", "recording_component", "delta"]]
    comparison = recomputed.merge(
        stored, on=["monkey", "recording_component"], suffixes=("_new", "_stored"), validate="one_to_one"
    )
    checks["primary_aggregation_reproduced"] = np.allclose(
        comparison["delta_new"], comparison["delta_stored"], rtol=0, atol=1e-12
    )
    medians = stored.groupby("monkey")["delta"].median()
    checks["primary_direction_both_monkeys"] = bool((medians > 0).all())
    checks["primary_gate_matches"] = bool(
        summary["primary_gate_pass"]
        and summary["primary"]["pooled_equal_monkey"]["sign_flip_p_one_sided"] < 0.05
    )
    movement = components[
        (components["window"] == "movement") & (components["evaluation"] == "pooled")
    ]
    checks["movement_reversal_both_monkeys"] = bool(
        (movement.groupby("monkey")["delta"].median() < 0).all()
    )

    checks = {key: bool(value) for key, value in checks.items()}
    report = {"checks": checks, "all_pass": bool(all(checks.values()))}
    output = ROOT / "results" / "gnode_target_set" / "validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
