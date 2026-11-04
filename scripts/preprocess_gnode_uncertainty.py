"""Create trial-level firing rates for the frozen G-Node replication cohort."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MISSING_EVENT = np.iinfo(np.int64).min
FIXED_WINDOWS = {
    "baseline": (0.0, 400.0),
    "prep_early": (550.0, 800.0),
    "prep_middle": (850.0, 1100.0),
    "prep_late": (1150.0, 1400.0),
}


def load_hdf5(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return handle["events/eventtimes"][:], handle["spiketimes/spiketimes"][:]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inventory", type=Path, default=ROOT / "results" / "gnode_uncertainty_inventory.csv"
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "raw" / "g-node_rz77m8"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "processed" / "gnode_uncertainty_rates.csv"
    )
    parser.add_argument(
        "--metadata", type=Path, default=ROOT / "data" / "processed" / "gnode_uncertainty_rates.json"
    )
    args = parser.parse_args()

    inventory = pd.read_csv(args.inventory)
    hdf5_root = args.data_root / "derived_files" / "HDF5" / "Data"
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    exclusion_counts: Counter[str] = Counter()
    session_summaries = []
    for (monkey, condition, session), units in inventory.groupby(
        ["monkey", "condition", "session"], sort=True
    ):
        session_files = sorted(hdf5_root.glob(f"{monkey}{int(session):03d}_*.hdf5"))
        if not session_files:
            raise FileNotFoundError(f"No HDF5 files for {monkey} session {session}")

        events, _ = load_hdf5(session_files[0])
        trial_directions: dict[int, int] = {}
        for path in session_files:
            other_events, spikes = load_hdf5(path)
            if not np.array_equal(events, other_events):
                raise ValueError(f"Event mismatch within session: {path}")
            for trial, direction in zip(spikes[1].astype(int), spikes[2].astype(int)):
                previous = trial_directions.setdefault(int(trial), int(direction))
                if previous != int(direction):
                    raise ValueError(f"Direction conflict in {path}, trial {trial}")

        valid_trials = []
        for trial, event_row in enumerate(events):
            if np.any(event_row == MISSING_EVENT):
                exclusion_counts["missing_event"] += 1
            elif np.any(np.diff(event_row) <= 0):
                exclusion_counts["nonordered_event"] += 1
            elif trial not in trial_directions:
                exclusion_counts["missing_direction"] += 1
            else:
                valid_trials.append(trial)

        for unit in units.itertuples(index=False):
            unit_events, spikes = load_hdf5(hdf5_root / unit.hdf5_file)
            if not np.array_equal(events, unit_events):
                raise ValueError(f"Inventory file event mismatch: {unit.hdf5_file}")
            spike_times = spikes[0]
            spike_trials = spikes[1].astype(int)
            for trial in valid_trials:
                trial_spikes = spike_times[spike_trials == trial]
                record: dict[str, object] = {
                    "monkey": monkey,
                    "condition": int(condition),
                    "session": int(session),
                    "trial": int(trial),
                    "direction": trial_directions[trial],
                    "global_neuron": int(unit.global_neuron),
                    "neuron": int(unit.neuron),
                    "hdf5_file": unit.hdf5_file,
                    "recording_triplet": unit.recording_triplet,
                    "recording_component": unit.recording_component,
                }
                for name, (start, stop) in FIXED_WINDOWS.items():
                    count = np.count_nonzero((trial_spikes >= start) & (trial_spikes < stop))
                    record[name] = count / ((stop - start) / 1000.0)
                rs, movement_onset = float(events[trial, 2]), float(events[trial, 3])
                post_start = rs + 25.0
                post_duration = movement_onset - post_start
                if post_duration >= 50.0:
                    count = np.count_nonzero(
                        (trial_spikes >= post_start) & (trial_spikes < movement_onset)
                    )
                    record["post_reveal"] = count / (post_duration / 1000.0)
                    record["post_duration_ms"] = post_duration
                else:
                    record["post_reveal"] = np.nan
                    record["post_duration_ms"] = post_duration
                    exclusion_counts["short_post_window_unit_rows"] += 1
                movement_start, movement_end = float(events[trial, 3]), float(events[trial, 4])
                movement_duration = movement_end - movement_start
                count = np.count_nonzero(
                    (trial_spikes >= movement_start) & (trial_spikes < movement_end)
                )
                record["movement"] = count / (movement_duration / 1000.0)
                record["movement_duration_ms"] = movement_duration
                rows.append(record)

        session_summaries.append(
            {
                "monkey": monkey,
                "condition": int(condition),
                "session": int(session),
                "units": len(units),
                "all_trials": len(events),
                "valid_trials": len(valid_trials),
            }
        )

    rates = pd.DataFrame(rows).sort_values(
        ["monkey", "global_neuron", "condition", "trial"]
    )
    rates.to_csv(args.output, index=False)
    metadata = {
        "source_inventory": str(args.inventory),
        "rows": len(rates),
        "matched_neurons": int(rates[["monkey", "global_neuron"]].drop_duplicates().shape[0]),
        "sessions": len(session_summaries),
        "rows_by_condition": {
            str(key): int(value) for key, value in rates["condition"].value_counts().sort_index().items()
        },
        "unique_valid_session_trials": int(
            rates[["monkey", "condition", "session", "trial"]].drop_duplicates().shape[0]
        ),
        "exclusions": dict(exclusion_counts),
        "fixed_windows_ms": FIXED_WINDOWS,
        "post_window": "RS+25 ms to MO; minimum duration 50 ms",
        "exploratory_movement_window": "MO to ME; descriptive movement-aligned interval",
        "session_summaries": session_summaries,
    }
    args.metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "session_summaries"}, indent=2))


if __name__ == "__main__":
    main()
