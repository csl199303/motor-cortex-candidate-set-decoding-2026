"""Audit the G-Node ambiguous-target dataset and freeze its matched cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MISSING_EVENT = np.iinfo(np.int64).min
EXPECTED_ARCHIVE_SHA256 = "c644f98fa9f9d178e7b494a573521d0261e68430774cd68692851052416c52a1"


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values]
    )


def connected_components(triplets: set[tuple[int, int, int]]) -> list[set[tuple[int, int, int]]]:
    remaining = set(triplets)
    components: list[set[tuple[int, int, int]]] = []
    while remaining:
        component = {remaining.pop()}
        changed = True
        while changed:
            sessions = set().union(*(set(triplet) for triplet in component))
            linked = {triplet for triplet in remaining if sessions.intersection(triplet)}
            changed = bool(linked)
            component.update(linked)
            remaining.difference_update(linked)
        components.append(component)
    return components


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root", type=Path, default=ROOT / "data" / "raw" / "g-node_rz77m8"
    )
    parser.add_argument(
        "--archive", type=Path, default=ROOT / "data" / "raw" / "g-node_rz77m8.zip"
    )
    parser.add_argument(
        "--output-csv", type=Path, default=ROOT / "results" / "gnode_uncertainty_inventory.csv"
    )
    parser.add_argument(
        "--summary-json", type=Path, default=ROOT / "results" / "gnode_uncertainty_inventory.json"
    )
    args = parser.parse_args()
    archive_digest = file_sha256(args.archive) if args.archive.exists() else None
    if archive_digest is not None and archive_digest != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Source archive checksum differs from the analyzed release")

    with (args.data_root / "pickle" / "toc").open("rb") as handle:
        toc = pickle.load(handle, encoding="latin1")
    fields = ("monkey", "condition", "session", "neuron", "global_neuron", "file", "all_conditions")
    frame = pd.DataFrame({field: np.asarray(toc[field]) for field in fields})
    frame["monkey"] = decode(frame["monkey"].to_numpy())
    frame["file"] = decode(frame["file"].to_numpy())
    frame = frame[frame["all_conditions"].astype(bool)].copy()
    frame["hdf5_file"] = frame["file"].str.replace(".pickle", ".hdf5", regex=False)

    pivot = frame.pivot(index=["monkey", "global_neuron"], columns="condition", values="session")
    complete = pivot.dropna(subset=[1, 2, 3]).copy()
    triplet_lookup = {
        (monkey, int(global_neuron)): tuple(int(value) for value in row[[1, 2, 3]])
        for (monkey, global_neuron), row in complete.iterrows()
    }
    frame = frame[
        frame.apply(lambda row: (row["monkey"], int(row["global_neuron"])) in triplet_lookup, axis=1)
    ].copy()
    frame["recording_triplet"] = frame.apply(
        lambda row: "-".join(map(str, triplet_lookup[(row["monkey"], int(row["global_neuron"]))])),
        axis=1,
    )

    component_lookup: dict[tuple[str, tuple[int, int, int]], str] = {}
    component_counts: dict[str, int] = {}
    for monkey in sorted(frame["monkey"].unique()):
        triplets = {
            triplet for (animal, _), triplet in triplet_lookup.items() if animal == monkey
        }
        components = connected_components(triplets)
        component_counts[monkey] = len(components)
        for index, component in enumerate(sorted(components, key=lambda values: min(values)), start=1):
            for triplet in component:
                component_lookup[(monkey, triplet)] = f"{monkey}_{index:03d}"
    frame["recording_component"] = frame.apply(
        lambda row: component_lookup[
            (row["monkey"], triplet_lookup[(row["monkey"], int(row["global_neuron"]))])
        ],
        axis=1,
    )

    hdf5_root = args.data_root / "derived_files" / "HDF5" / "Data"
    missing_files = []
    malformed_files = []
    invalid_event_trials = 0
    incomplete_event_trials = 0
    incomplete_session_trials: set[tuple[str, int, int, int]] = set()
    invalid_session_trials: set[tuple[str, int, int, int]] = set()
    invalid_spike_references = 0
    trial_counts = []
    direction_counts: defaultdict[int, int] = defaultdict(int)
    for row in frame.itertuples():
        path = hdf5_root / row.hdf5_file
        if not path.exists():
            missing_files.append(str(path))
            continue
        try:
            with h5py.File(path, "r") as handle:
                events = handle["events/eventtimes"][:]
                event_names = decode(handle["events/event_names"][:]).tolist()
                spikes = handle["spiketimes/spiketimes"][:]
            if event_names != ["TS", "PS", "RS", "MO", "ME", "RW", "TE"]:
                malformed_files.append(f"{path.name}: event names")
            complete_events = np.all(events != MISSING_EVENT, axis=1)
            incomplete_event_trials += int((~complete_events).sum())
            invalid_order = complete_events & np.any(np.diff(events, axis=1) <= 0, axis=1)
            invalid_event_trials += int(invalid_order.sum())
            incomplete_session_trials.update(
                (row.monkey, int(row.condition), int(row.session), int(trial))
                for trial in np.flatnonzero(~complete_events)
            )
            invalid_session_trials.update(
                (row.monkey, int(row.condition), int(row.session), int(trial))
                for trial in np.flatnonzero(invalid_order)
            )
            trials = spikes[1].astype(int)
            directions = spikes[2].astype(int)
            invalid_spike_references += int(((trials < 0) | (trials >= len(events))).sum())
            invalid_spike_references += int(((directions < 1) | (directions > 6)).sum())
            trial_counts.append(len(events))
            unique_trials = np.unique(trials)
            for direction in range(1, 7):
                direction_counts[direction] += int(
                    np.intersect1d(unique_trials, trials[directions == direction]).size
                )
        except (OSError, KeyError, ValueError) as error:
            malformed_files.append(f"{path.name}: {error}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    csv_root = args.data_root / "derived_files" / "CSV" / "Data"
    sampled_rows = frame.sort_values("hdf5_file").iloc[
        np.linspace(0, len(frame) - 1, num=10, dtype=int)
    ]
    csv_crosscheck_failures = []
    for row in sampled_rows.itertuples():
        stem = Path(row.hdf5_file).stem
        with h5py.File(hdf5_root / row.hdf5_file, "r") as handle:
            hdf_events = handle["events/eventtimes"][:].astype(float)
            hdf_spikes = handle["spiketimes/spiketimes"][:].T.astype(float)
        hdf_events[hdf_events == MISSING_EVENT] = np.nan
        csv_events = pd.read_csv(csv_root / f"{stem}_events.csv").iloc[:, 1:].to_numpy(float)
        csv_spikes = pd.read_csv(csv_root / f"{stem}_spikes.csv").to_numpy(float)
        if not np.allclose(hdf_events, csv_events, equal_nan=True):
            csv_crosscheck_failures.append(f"{stem}: events")
        if not np.allclose(hdf_spikes, csv_spikes, equal_nan=True):
            csv_crosscheck_failures.append(f"{stem}: spikes")

    all_hdf5 = list(hdf5_root.glob("*.hdf5"))
    included_names = set(frame["hdf5_file"])
    excluded_hdf5 = sorted(path.name for path in all_hdf5 if path.name not in included_names)
    frame.sort_values(["monkey", "global_neuron", "condition"]).to_csv(args.output_csv, index=False)
    summary = {
        "dataset": "doi:10.12751/g-node.rz77m8",
        "archive_bytes": args.archive.stat().st_size if args.archive.exists() else None,
        "archive_sha256": archive_digest,
        "archive_verification": "checksum verified" if archive_digest is not None else "archive unavailable; extracted files audited, archive checksum not verified",
        "all_hdf5_files": len(all_hdf5),
        "toc_rows": len(np.asarray(toc["file"])),
        "included_rows": len(frame),
        "matched_global_neurons": int(frame[["monkey", "global_neuron"]].drop_duplicates().shape[0]),
        "neurons_by_monkey": frame[["monkey", "global_neuron"]].drop_duplicates()["monkey"].value_counts().to_dict(),
        "recording_triplets_by_monkey": frame[["monkey", "recording_triplet"]].drop_duplicates()["monkey"].value_counts().to_dict(),
        "recording_components_by_monkey": component_counts,
        "condition_rows": {str(key): int(value) for key, value in frame["condition"].value_counts().sort_index().items()},
        "trial_count": {
            "sum_across_unit_files": int(np.sum(trial_counts)),
            "min": int(np.min(trial_counts)),
            "median": float(np.median(trial_counts)),
            "max": int(np.max(trial_counts)),
        },
        "direction_trial_references": {str(key): value for key, value in sorted(direction_counts.items())},
        "excluded_hdf5_count": len(excluded_hdf5),
        "excluded_hdf5": excluded_hdf5,
        "missing_files": missing_files,
        "malformed_files": malformed_files,
        "nonordered_event_trials_across_unit_files": invalid_event_trials,
        "nonordered_unique_session_trials": len(invalid_session_trials),
        "incomplete_event_trials_across_unit_files": incomplete_event_trials,
        "incomplete_unique_session_trials": len(incomplete_session_trials),
        "invalid_spike_references": invalid_spike_references,
        "csv_crosscheck_sample_size": len(sampled_rows),
        "csv_crosscheck_failures": csv_crosscheck_failures,
        "integrity_pass": not (
            missing_files
            or malformed_files
            or invalid_spike_references
            or csv_crosscheck_failures
        ),
    }
    args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not summary["integrity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
