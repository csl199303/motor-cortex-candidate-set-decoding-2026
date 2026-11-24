"""Run paired cue-conditioned decoding and the matched single-target control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from run_gnode_within_set_decoding import fixed_folds, stable_rng, target_sets

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "results/iscience_revision"
BETAS = np.array([0, .125, .25, .5, 1, 2, 4, 8])
DIRECTIONS = np.arange(1, 7)
WINDOWS = {"pre": (-350, -30), "early": (80, 120), "late": (150, 300)}
METRICS = (
    "neural_only_bits", "cue_only_bits", "conditional_bits",
    "gain_bits", "gain_empirical_bits", "neural_only_ba", "cue_only_ba",
    "conditional_ba",
)
LN2 = np.log(2)


def logits(x_train, y_train, x_test, classes):
    """Mean squared standardized distance to each training class centroid."""
    mean = x_train.mean(axis=0)
    sd = x_train.std(axis=0, ddof=1)
    sd = np.where(sd > 1e-12, sd, 1)
    centers = np.array([(x_train[y_train == c].mean(axis=0) - mean) / sd for c in classes])
    return -np.mean(((x_test - mean)[:, None] / sd - centers[None, :]) ** 2, axis=2)


def fit_temperature(x, y, classes):
    n_splits = min(4, min(np.sum(y == c) for c in classes))
    if n_splits < 2:
        raise ValueError("Insufficient training data for inner calibration")
    inner = np.empty(len(y), dtype=int)
    for c in classes:
        idx = np.flatnonzero(y == c)
        inner[idx] = np.arange(len(idx)) % n_splits
    val = np.empty((len(x), len(classes), x.shape[-1]))
    for f in range(n_splits):
        mask = inner == f
        val[mask] = logits(x[~mask], y[~mask], x[mask], classes)
    correct = val[np.arange(len(y)), np.searchsorted(classes, y)]
    loss = np.array([(logsumexp(val * b, axis=1) - correct * b).mean(axis=0) for b in BETAS])
    return BETAS[np.argmin(loss, axis=0)]


def decode(x, y, folds, classes):
    """Cross-fitted log probabilities; every prediction is out of fold."""
    classes = np.sort(classes)
    out = np.empty((len(x), len(classes), x.shape[-1]))
    temperatures = np.empty((5, x.shape[-1]))
    for f in range(5):
        test = folds == f
        train = ~test
        assert np.all(np.isin(classes, y[train])) and test.any()
        beta = fit_temperature(x[train], y[train], classes)
        z = logits(x[train], y[train], x[test], classes) * beta
        out[test] = z - logsumexp(z, axis=1, keepdims=True)
        temperatures[f] = beta
    assert np.allclose(logsumexp(out, axis=1), 0, atol=1e-10)
    return out, temperatures


def session_data(root=ROOT):
    data = root / "data/processed"
    raw = pd.read_csv(data / "gnode_rs_sliding_metadata.csv")
    with np.load(data / "gnode_rs_sliding_rates.npz") as arrays:
        centers = arrays["centers_ms"]
        rates = arrays["rates_hz"].astype(float)
        rt = arrays["movement_onset_ms"]
    if rates.shape != (len(raw), len(centers)):
        raise ValueError("Metadata and rate axes disagree")
    keep = (centers >= -350) & (centers <= 500)
    centers, rates = centers[keep], rates[:, keep]
    sessions = {}
    inventory = []
    assert np.allclose(rates * .05, np.rint(rates * .05))
    for key, rows in raw.groupby(["monkey", "condition", "session"], sort=True):
        lookup = rows.assign(rate_row=rows.index).pivot(
            index="trial", columns="global_neuron", values="rate_row",
        ).dropna().astype(int)
        idx = lookup.to_numpy()
        if not len(idx):
            inventory.append((*key, str(rows.recording_component.iloc[0]), len(lookup.columns),
                              rows.trial.nunique(), 0, 0))
            continue
        comp = rows.recording_component.unique()
        y = raw.direction.to_numpy()[idx]
        reaction = rt[idx]
        assert len(comp) == 1 and (y == y[:, :1]).all()
        assert np.all(reaction == reaction[:, :1]) and np.all(np.isfinite(reaction))
        sessions[key] = (str(comp[0]), lookup.index.to_numpy(), lookup.columns.to_numpy(),
                         y[:, 0], reaction[:, 0], rates[idx])
        inventory.append((*key, str(comp[0]), len(lookup.columns),
                          rows.trial.nunique(), len(idx), np.sum(reaction[:, 0] >= 170)))
    return centers, sessions, pd.DataFrame(
        inventory, columns=["monkey", "condition", "session", "component", "units",
                            "input_trials", "complete_trials", "rt170_trials"],
    )


def balanced_accuracy(p, y):
    win = np.isclose(p, p.max(axis=1, keepdims=True), rtol=0, atol=1e-12)
    per_trial = win[np.arange(len(y)), y - 1] / win.sum(axis=1)
    return np.stack([per_trial[y == c].mean(axis=0) for c in DIRECTIONS]).mean(axis=0)


def evaluate_session(x, y, condition):
    folds = fixed_folds(y)
    six, temp6 = decode(x, y, folds, DIRECTIONS)
    conditional = np.full_like(six, -np.inf)
    cue = np.zeros_like(six[:, :, 0])
    empirical = np.zeros_like(cue)
    temps = []
    groups = np.full(len(y), -1)
    for i, cand in enumerate(target_sets(condition)):
        classes = np.sort(cand)
        idx = np.flatnonzero(np.isin(y, classes))
        groups[idx] = i
        conditional[np.ix_(idx, classes - 1)] , temp = decode(x[idx], y[idx], folds[idx], classes)
        temps.append(temp)
        cue[np.ix_(idx, classes - 1)] = 1 / len(classes)
        for f in range(5):
            train = idx[folds[idx] != f]
            test = idx[folds[idx] == f]
            counts = np.array([np.count_nonzero(y[train] == c) for c in classes])
            empirical[np.ix_(test, classes - 1)] = (counts + 1) / (len(train) + len(classes))
    assert (groups >= 0).all()
    assert np.allclose(cue.sum(axis=1), 1) and np.allclose(empirical.sum(axis=1), 1)
    assert np.array_equal(np.isfinite(conditional[:, :, 0]), cue > 0)
    assert np.allclose(logsumexp(conditional, axis=1), 0, atol=1e-10)
    true = np.arange(len(y)), y - 1
    score6 = six[true] / LN2
    scorec = conditional[true] / LN2
    scorecue = np.log2(cue[true])[:, None]
    gain = scorec - scorecue
    metrics = np.stack([
        score6.mean(axis=0) + np.log2(6),
        np.full(x.shape[-1], (scorecue + np.log2(6)).mean()),
        scorec.mean(axis=0) + np.log2(6),
        gain.mean(axis=0),
        (scorec - np.log2(empirical[true])[:, None]).mean(axis=0),
        balanced_accuracy(six, y),
        balanced_accuracy(np.broadcast_to(cue[:, :, None], six.shape), y),
        balanced_accuracy(conditional, y),
    ], axis=-1)
    return metrics, (folds, groups, six, conditional, cue, empirical, temp6, np.stack(temps))


def matched_controls(sessions, centers, eligible, out):
    details, curves = [], []
    by_component = {}
    for key, entry in sessions.items():
        if key[1] == 1:
            by_component.setdefault((key[0], entry[0]), []).append((key[2], entry))
    for key in eligible:
        monkey, condition, session = key
        component, _, units, y, _, x = sessions[key]
        choices = by_component.get((monkey, component), [])
        for cand in target_sets(condition):
            classes = np.sort(cand)
            cue = "-".join(map(str, classes))
            values = dict(monkey=monkey, condition=condition, session=session, component=component,
                          cue=cue, uncertain_units=len(units))
            if not choices:
                details.append(dict(values, reason="no single-target session in component"))
                continue
            single_session, single = min(
                choices, key=lambda pair: (-len(np.intersect1d(units, pair[1][2])), pair[0]),
            )
            _, _, one_units, one_y, _, one_x = single
            common, ui, oi = np.intersect1d(units, one_units, return_indices=True)
            n_uncertain = [int(np.sum(y == c)) for c in classes]
            n_single = [int(np.sum(one_y == c)) for c in classes]
            counts = np.minimum(n_uncertain, n_single)
            values.update(single_session=single_session, matched_units=len(common),
                          uncertain_counts="/".join(map(str, n_uncertain)),
                          single_counts="/".join(map(str, n_single)),
                          matched_counts="/".join(map(str, counts)), matched_trials=counts.sum())
            if not len(common) or min(counts) < 5:
                details.append(dict(values, reason="no shared units" if not len(common) else "fewer than five trials per class"))
                continue
            details.append(dict(values, reason="matched"))
            for cohort, xx, yy, ii in (("single", one_x, one_y, oi), ("uncertain", x, y, ui)):
                rng = stable_rng(f"iscience-match-{key}-{cue}-{cohort}")
                chosen = np.sort(np.concatenate([
                    rng.choice(np.flatnonzero(yy == c), n, replace=False) for c, n in zip(classes, counts)
                ]))
                yy = yy[chosen]
                pp, _ = decode(xx[chosen][:, ii], yy, fixed_folds(yy), classes)
                gain = pp[np.arange(len(yy)), np.searchsorted(classes, yy)].mean(axis=0) / LN2 + np.log2(len(classes))
                curves.append(pd.DataFrame(dict(monkey=monkey, condition=condition, session=session,
                                                 component=component, cue=cue, cohort=cohort,
                                                 center_ms=centers, gain_bits=gain)))
    matches = pd.DataFrame(details)
    matches.to_csv(out / "positive_control_matches.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(
        out / "positive_control_set_curves.csv.gz", index=False, compression="gzip",
    )
    return matches, pd.concat(curves, ignore_index=True)


def bootstrap(values, label):
    if len(values) < 2:
        return np.full_like(values[0], np.nan), np.full_like(values[0], np.nan)
    draws = stable_rng(label).integers(len(values), size=(5000, len(values)))
    means = np.array([values[part].mean(axis=1) for part in np.array_split(draws, 20)]).reshape(
        (5000,) + values.shape[1:],
    )
    return np.quantile(means, [.025, .975], axis=0)


def component_curves(curves, metrics):
    keys = ["monkey", "condition", "component", "cohort", "center_ms"]
    comp = curves.groupby(keys, sort=True)[list(metrics)].mean().reset_index()
    pooled_keys = [key for key in keys if key != "condition"]
    grouped = comp.groupby(pooled_keys, sort=True)
    pooled = grouped[list(metrics)].mean().loc[grouped.size().eq(2)].reset_index()
    return pd.concat([comp, pooled.assign(condition=0)], ignore_index=True)


def summaries(comp, metrics, name):
    points, windows = [], []
    for (monkey, condition, cohort), group in comp.groupby(["monkey", "condition", "cohort"], sort=True):
        keys = dict(monkey=monkey, condition=condition, cohort=cohort)
        for metric in metrics:
            wide = group.pivot(index="component", columns="center_ms", values=metric)
            t = wide.columns.to_numpy()
            values = wide.to_numpy()
            label = f"{name}-{monkey}-{condition}-{cohort}-{metric}"
            low, high = bootstrap(values, label)
            points.append(pd.DataFrame(dict(**keys, center_ms=t, metric=metric,
                                            n_components=len(wide), mean=values.mean(axis=0),
                                            low=low, high=high)))
            for win, (a, b) in WINDOWS.items():
                centers = (t >= a) & (t <= b)
                if not np.any(centers):
                    continue
                v = values[:, centers].mean(axis=1)
                low, high = bootstrap(v, f"{label}-{win}")
                windows.append(dict(**keys, metric=metric, window=win, n_components=len(v),
                                    mean=v.mean(), low=low, high=high))
    return pd.concat(points, ignore_index=True), pd.DataFrame(windows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--root", type=Path, default=ROOT, help="Root containing the existing data/processed inputs")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    centers, sessions, inventory = session_data(args.root)
    inventory.to_csv(out / "input_sessions.csv", index=False)
    curve_rows, trial_rows, sample_rows = [], [], []
    eligible = []
    archive = out / "holdout"
    archive.mkdir(exist_ok=True)
    for key, entry in sessions.items():
        monkey, condition, session = key
        if condition == 1:
            continue
        component, trial, units, y, rt, x = entry
        full_eligible = False
        for cohort, mask, time in (
            ("full", np.ones(len(y), dtype=bool), np.ones(len(centers), dtype=bool)),
            ("rt170", rt >= 170, centers <= 120),
        ):
            yy = y[mask]
            cc = centers[time]
            counts = np.bincount(yy, minlength=7)[1:]
            accepted = bool(min(counts) >= 5)
            row = dict(monkey=monkey, condition=condition, session=session, component=component,
                       units=len(units), cohort=cohort, n_trials=len(yy), accepted=accepted,
                       direction_counts="/".join(map(str, counts)),
                       reason="" if accepted else "fewer than five trials per direction")
            sample_rows.append(row)
            if not accepted:
                continue
            if cohort == "full":
                full_eligible = True
                eligible.append(key)
            else:
                assert full_eligible and np.all(rt[mask] - (cc.max() + 25) >= 25)
            metrics, data = evaluate_session(x[mask][..., time], yy, condition)
            folds, groups, six, conditional, cue, empirical, temp6, temp_cond = data
            assert np.all(np.bincount(folds, minlength=5) > 0)
            stem = f"{monkey}_c{condition}_s{session}_{cohort}"
            np.savez_compressed(
                archive / f"{stem}.npz", centers_ms=cc, direction=yy, cue_id=groups,
                trial=trial[mask], rt_ms=rt[mask], fold=folds, units=units,
                logp_neural_only=six.astype(np.float32),
                logp_conditional=conditional.astype(np.float32),
                p_cue=cue, p_cue_empirical=empirical, temperature_neural_only=temp6,
                temperature_conditional=temp_cond,
            )
            curve_rows.append(pd.DataFrame(dict(monkey=monkey, condition=condition, session=session,
                                                 component=component, cohort=cohort, center_ms=cc,
                                                 **dict(zip(METRICS, metrics.T)))))
            trial_rows.append(pd.DataFrame(dict(monkey=monkey, condition=condition, session=session,
                                                 component=component, cohort=cohort, trial=trial[mask],
                                                 direction=yy, cue=groups, rt_ms=rt[mask], fold=folds)))
    samples = pd.DataFrame(sample_rows)
    samples.to_csv(out / "cohort_sessions.csv", index=False)
    trials = pd.concat(trial_rows, ignore_index=True)
    trials.to_csv(out / "evaluated_trials.csv.gz", index=False, compression="gzip")
    rows = []
    for (monkey, condition, cohort), q in trials.groupby(["monkey", "condition", "cohort"]):
        s = samples.query("monkey == @monkey and condition == @condition and cohort == @cohort and accepted")
        n = len(trials.query("monkey == @monkey and condition == @condition and cohort == 'full'"))
        rows.append(dict(monkey=monkey, condition=condition, cohort=cohort, n_trials=len(q),
                         n_sessions=q.session.nunique(), n_components=q.component.nunique(),
                         fraction_of_full=len(q) / n, unit_min=s.units.min(), unit_median=s.units.median(),
                         unit_max=s.units.max(), rt_min=q.rt_ms.min(), rt_p05=q.rt_ms.quantile(.05),
                         rt_median=q.rt_ms.median(), rt_p95=q.rt_ms.quantile(.95), rt_max=q.rt_ms.max()))
    pd.DataFrame(rows).to_csv(out / "cohort_summary.csv", index=False)
    curves = pd.concat(curve_rows, ignore_index=True)
    curves.to_csv(out / "session_curves.csv.gz", index=False, compression="gzip")
    comp = component_curves(curves, METRICS)
    comp.to_csv(out / "component_curves.csv.gz", index=False, compression="gzip")
    points, windows = summaries(comp, METRICS, "main")
    points.to_csv(out / "decoding_curves.csv", index=False)
    windows.to_csv(out / "decoding_windows.csv", index=False)

    matches, controls = matched_controls(sessions, centers, eligible, out)
    control_sessions = controls.groupby(
        ["monkey", "condition", "session", "component", "cohort", "center_ms"], as_index=False,
    ).gain_bits.mean()
    control_comp = component_curves(control_sessions, ("gain_bits",))
    p, w = summaries(control_comp, ("gain_bits",), "positive")
    p.to_csv(out / "positive_control_curves.csv", index=False)
    w.to_csv(out / "positive_control_windows.csv", index=False)

    paths = [args.root / "data/processed" / name for name in
             ("gnode_rs_sliding_metadata.csv", "gnode_rs_sliding_rates.npz",
              "gnode_uncertainty_rates.csv", "gnode_rs_sliding_rates.json")]
    manifest = {
        "dataset_doi": "10.12751/g-node.rz77m8",
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "n_model_archives": len(trial_rows), "n_matched_set_pairs": int(matches.reason.eq("matched").sum()),
        "bootstrap_draws": 5000, "temperature_grid": BETAS.tolist(),
        "python_version": __import__("sys").version,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
    }
    (out / "decoding_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(windows.query("metric == 'gain_bits' and condition == 0").to_string(index=False))
    print(w.query("window == 'pre' and condition == 0").to_string(index=False))


if __name__ == "__main__":
    main()
