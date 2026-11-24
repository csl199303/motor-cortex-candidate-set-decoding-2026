"""Build RS-aligned sliding-window rates for the frozen dynamics protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CENTERS_MS = np.arange(-950.0, 651.0, 10.0)
WINDOW_MS = 50.0
META_COLUMNS = [
    "monkey",
    "condition",
    "session",
    "trial",
    "direction",
    "global_neuron",
    "neuron",
    "hdf5_file",
    "recording_triplet",
    "recording_component",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trial-rates",
        type=Path,
        default=ROOT / "data" / "processed" / "gnode_uncertainty_rates.csv",
    )
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "raw" / "g-node_rz77m8"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data" / "processed" / "gnode_rs_sliding_rates.npz"
    )
    parser.add_argument(
        "--metadata", type=Path, default=ROOT / "data" / "processed" / "gnode_rs_sliding_metadata.csv"
    )
    parser.add_argument(
        "--summary", type=Path, default=ROOT / "data" / "processed" / "gnode_rs_sliding_rates.json"
    )
    args = parser.parse_args()

    trial_rows = pd.read_csv(args.trial_rates, usecols=META_COLUMNS)
    hdf5_root = args.data_root / "derived_files" / "HDF5" / "Data"
    rates = np.empty((len(trial_rows), len(CENTERS_MS)), dtype=np.float32)
    movement_onset = np.empty(len(trial_rows), dtype=np.float32)
    movement_end = np.empty(len(trial_rows), dtype=np.float32)
    starts = CENTERS_MS - WINDOW_MS / 2.0
    stops = CENTERS_MS + WINDOW_MS / 2.0

    for hdf5_file, indices in trial_rows.groupby("hdf5_file", sort=True).groups.items():
        path = hdf5_root / hdf5_file
        with h5py.File(path, "r") as handle:
            events = handle["events/eventtimes"][:]
            spikes = handle["spiketimes/spiketimes"][:]
        spike_times = spikes[0].astype(float)
        spike_trials = spikes[1].astype(int)
        for index in indices:
            trial = int(trial_rows.at[index, "trial"])
            rs = float(events[trial, 2])
            aligned = np.sort(spike_times[spike_trials == trial] - rs)
            counts = np.searchsorted(aligned, stops, side="left") - np.searchsorted(
                aligned, starts, side="left"
            )
            rates[index] = counts / (WINDOW_MS / 1000.0)
            movement_onset[index] = float(events[trial, 3] - rs)
            movement_end[index] = float(events[trial, 4] - rs)

    sample_indices = np.linspace(0, len(trial_rows) - 1, 20, dtype=int)
    sample_centers = np.linspace(0, len(CENTERS_MS) - 1, 15, dtype=int)
    qa_failures = []
    for index in sample_indices:
        row = trial_rows.iloc[index]
        with h5py.File(hdf5_root / row.hdf5_file, "r") as handle:
            events = handle["events/eventtimes"][:]
            spikes = handle["spiketimes/spiketimes"][:]
        trial = int(row.trial)
        aligned = spikes[0, spikes[1].astype(int) == trial] - events[trial, 2]
        for center_index in sample_centers:
            center = CENTERS_MS[center_index]
            direct = np.count_nonzero(
                (aligned >= center - WINDOW_MS / 2.0)
                & (aligned < center + WINDOW_MS / 2.0)
            ) / (WINDOW_MS / 1000.0)
            if not np.isclose(direct, rates[index, center_index]):
                qa_failures.append({"row": int(index), "center_ms": float(center)})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        rates_hz=rates,
        centers_ms=CENTERS_MS.astype(np.float32),
        movement_onset_ms=movement_onset,
        movement_end_ms=movement_end,
    )
    trial_rows.to_csv(args.metadata, index=False)
    summary = {
        "protocol": "switch_dynamics_protocol.md",
        "rows": len(trial_rows),
        "time_points": len(CENTERS_MS),
        "rate_shape": list(rates.shape),
        "window_ms": WINDOW_MS,
        "step_ms": float(np.diff(CENTERS_MS).min()),
        "center_range_ms": [float(CENTERS_MS[0]), float(CENTERS_MS[-1])],
        "rate_min_hz": float(rates.min()),
        "rate_max_hz": float(rates.max()),
        "movement_onset_quantiles_ms": [
            float(value) for value in np.quantile(movement_onset, [0.05, 0.5, 0.95])
        ],
        "movement_end_quantiles_ms": [
            float(value) for value in np.quantile(movement_end, [0.05, 0.5, 0.95])
        ],
        "qa_comparisons": int(len(sample_indices) * len(sample_centers)),
        "qa_failures": qa_failures,
        "qa_pass": not qa_failures,
        "source_trial_rates_sha256": sha256(args.trial_rates),
    }
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if qa_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
