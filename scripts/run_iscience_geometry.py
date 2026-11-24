"""Compare transferred cue geometry to two training-fold prediction baselines."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from run_gnode_target_set_replication import design_from_directions, set_vector
from run_gnode_within_set_decoding import fixed_folds, target_sets
from run_iscience_decoding import ROOT, DEFAULT_OUT, bootstrap

WINDOWS = ("baseline", "prep_late")
METRICS = ("geometry_over_constant", "category_over_constant", "geometry_over_category",
           "selected_minus_geometry", "dispersion", "cross_term")


def evaluate(rates):
    audit_rows, drops = [], []
    for (animal, unit), rows in rates.groupby(["monkey", "global_neuron"]):
        c1 = rows[rows.condition.eq(1)].sort_values("trial")
        x_train = design_from_directions(c1.direction.to_numpy())
        metadata = c1.iloc[0]
        for condition, trial in rows[rows.condition.isin([2, 3])].groupby("condition"):
            trial = trial.sort_values("trial")
            y = trial.direction.to_numpy()
            cue = np.full(len(y), -1)
            for i, cand in enumerate(target_sets(condition)):
                cue[np.isin(y, cand)] = i
            assert (cue >= 0).all()
            if np.bincount(y, minlength=7)[1:].min() < 5:
                drops.append(dict(monkey=animal, unit=unit, condition=condition, fold=-1,
                                  window="all", reason="fewer than five trials per direction"))
                continue
            folds = fixed_folds(y)
            selected_design = design_from_directions(y)
            geometry_design = np.column_stack([
                np.ones(len(y)), [set_vector(condition, int(d)) for d in y],
            ])
            trial_ids = trial.trial.to_numpy()
            for window in WINDOWS:
                beta = np.linalg.lstsq(x_train, c1[window].to_numpy(), rcond=None)[0]
                selected = selected_design @ beta
                geometry = geometry_design @ beta
                observed = trial[window].to_numpy(float)
                for fold in range(5):
                    test = folds == fold
                    train = ~test
                    variance = observed[train].var()
                    if variance <= 1e-12:
                        drops.append(dict(monkey=animal, unit=unit, condition=condition,
                                          fold=fold, window=window, reason="degenerate training variance"))
                        continue
                    constant = observed[train].mean()
                    category_means = np.array([
                        observed[train & (cue == c)].mean() for c in np.unique(cue)
                    ])
                    category = category_means[cue[test]]
                    R, P, G = observed[test], selected[test], geometry[test]
                    audit_rows.append(pd.DataFrame(dict(
                        monkey=animal, unit=unit, condition=condition, window=window,
                        triplet=metadata.recording_triplet, component=metadata.recording_component,
                        session=int(trial.session.iloc[0]), single_session=int(c1.session.iloc[0]),
                        trial=trial_ids[test], direction=y[test], cue=cue[test], fold=fold,
                        observed=R, selected=P, geometry=G, constant=constant, category=category,
                        train_variance=variance,
                    )))
    data = pd.concat(audit_rows, ignore_index=True)
    r, p, g, c, k, v = (data[name].to_numpy() for name in
                         ("observed", "selected", "geometry", "constant", "category", "train_variance"))
    data[list(METRICS)] = np.column_stack([
        ((r - c) ** 2 - (r - g) ** 2) / v,
        ((r - c) ** 2 - (r - k) ** 2) / v,
        ((r - k) ** 2 - (r - g) ** 2) / v,
        ((r - p) ** 2 - (r - g) ** 2) / v,
        (p - g) ** 2 / v,
        -2 * (r - g) * (p - g) / v,
    ])
    return data, pd.DataFrame(drops)


def summarize(data):
    base = ["monkey", "condition", "window", "component"]
    unit = data.groupby(base + ["triplet", "unit"])[list(METRICS)].mean().reset_index()
    triplet = unit.groupby(base + ["triplet"])[list(METRICS)].median().reset_index()
    comp = triplet.groupby(base)[list(METRICS)].median().reset_index()
    rows = []
    for (animal, condition, window), g in comp.groupby(["monkey", "condition", "window"]):
        trials = data.query("monkey == @animal and condition == @condition and window == @window")
        for metric in METRICS:
            v = g[metric].to_numpy()
            lo, hi = bootstrap(v, f"iscience-geometry-{animal}-{condition}-{window}-{metric}")
            rows.append(dict(monkey=animal, condition=condition, window=window, metric=metric,
                             mean=v.mean(), low=lo, high=hi, n_components=len(v),
                             n_units=trials.unit.nunique(), n_sessions=trials.session.nunique(),
                             n_trial_unit_rows=len(trials)))
    return comp, pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--root", type=Path, default=ROOT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = args.root / "data/processed/gnode_uncertainty_rates.csv"
    data, drops = evaluate(pd.read_csv(source))
    error = np.max(np.abs(data.selected_minus_geometry - data.dispersion - data.cross_term))
    assert error < 1e-9, error
    data.to_csv(args.output / "geometry_holdout.csv.gz", index=False, compression="gzip")
    drops.to_csv(args.output / "geometry_exclusions.csv", index=False)
    comp, summary = summarize(data)
    comp.to_csv(args.output / "geometry_components.csv", index=False)
    summary.to_csv(args.output / "geometry_summary.csv", index=False)
    audit = {"max_decomposition_error": error, "n_holdout_rows": len(data),
             "n_excluded_unit_windows_or_folds": len(drops),
             "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "input_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    (args.output / "geometry_manifest.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(summary.query("window == 'prep_late' and metric == 'geometry_over_constant'").to_string(index=False))


if __name__ == "__main__":
    main()
