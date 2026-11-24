"""Verify trial pairing, held-out scores, priors, and geometry output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logsumexp

from run_gnode_within_set_decoding import target_sets, fixed_folds
from run_iscience_decoding import DEFAULT_OUT, LN2, component_curves, summaries
from run_iscience_geometry import METRICS as GEOMETRY_METRICS, summarize


def check_decoding(out):
    trials = pd.read_csv(out / "evaluated_trials.csv.gz")
    curves = pd.read_csv(out / "session_curves.csv.gz")
    assert not trials.duplicated(["monkey", "condition", "session", "cohort", "trial"]).any()
    group_cols = ["monkey", "condition", "session", "cohort"]
    by_curve = curves.groupby(group_cols)
    max_error = 0.
    min_buffer = np.inf
    files = list((out / "holdout").glob("*.npz"))
    groups = trials.groupby(group_cols)
    assert groups.ngroups == len(files)
    for key, t in groups:
        animal, condition, session, cohort = key
        t = t.sort_values("trial")
        name = f"{animal}_c{condition}_s{session}_{cohort}.npz"
        with np.load(out / "holdout" / name) as h:
            centers = h["centers_ms"]
            lp6, lpc, p0, p1 = (h[k] for k in
                                 ("logp_neural_only", "logp_conditional", "p_cue", "p_cue_empirical"))
            y = h["direction"]
            fold = h["fold"]
            for k, col in (("trial", "trial"), ("direction", "direction"), ("rt_ms", "rt_ms"), ("fold", "fold")):
                assert np.array_equal(h[k], t[col].to_numpy()), (name, k)
            assert np.array_equal(fold, fixed_folds(y)), name
            assert lp6.shape == lpc.shape == (len(t), 6, len(centers)), name
            assert np.isfinite(lp6).all()
            assert np.allclose(logsumexp(lp6, axis=1), 0, atol=3e-6)
            assert np.allclose(logsumexp(lpc, axis=1), 0, atol=3e-6)
            assert np.array_equal(np.isfinite(lpc), np.broadcast_to(p0[:, :, None] > 0, lpc.shape)), name
            for i, candidates in enumerate(target_sets(condition)):
                classes = np.sort(candidates)
                mask = np.isin(y, classes)
                assert np.array_equal(h["cue_id"] == i, mask)
                assert np.allclose(p0[np.ix_(mask, classes - 1)], 1 / len(classes))
                for f in range(5):
                    train = mask & (fold != f)
                    test = mask & (fold == f)
                    counts = np.array([np.sum(train & (y == cls)) for cls in classes])
                    expected = (counts + 1) / (train.sum() + len(classes))
                    assert np.allclose(p1[np.ix_(test, classes - 1)], expected)
            obs = by_curve.get_group(key).sort_values("center_ms")
            assert np.array_equal(centers, obs.center_ms.to_numpy())
            idx = np.arange(len(y)), y - 1
            score6 = lp6[idx].astype(float) / LN2
            scorec = lpc[idx].astype(float) / LN2
            cue = np.log2(p0[idx])[:, None]
            metrics = {
                "neural_only_bits": score6.mean(axis=0) + np.log2(6),
                "cue_only_bits": cue.mean() + np.log2(6),
                "conditional_bits": scorec.mean(axis=0) + np.log2(6),
                "gain_bits": (scorec - cue).mean(axis=0),
                "gain_empirical_bits": (scorec - np.log2(p1[idx])[:, None]).mean(axis=0),
            }
            for k, v in metrics.items():
                max_error = max(max_error, float(np.max(np.abs(v - obs[k]))))
            if cohort == "rt170":
                full = groups.get_group((animal, condition, session, "full"))
                assert set(t.trial) == set(full.loc[full.rt_ms >= 170, "trial"])
                min_buffer = min(min_buffer, float(h["rt_ms"].min() - (centers.max() + 25)))
    assert min_buffer >= 25 and max_error < 1e-6, (min_buffer, max_error)
    for col in ("gain_bits", "gain_empirical_bits"):
        assert (curves[col] < 0).any(), f"{col}: negative scores were lost"
    # Regenerate the main gain summaries from archived session-level scores.
    comp = component_curves(curves, ("gain_bits",))
    _, w = summaries(comp, ("gain_bits",), "main")
    reported = pd.read_csv(out / "decoding_windows.csv").query("metric == 'gain_bits'")
    for col in ("mean", "low", "high"):
        np.testing.assert_allclose(w[col].to_numpy(), reported[col].to_numpy(), atol=1e-12)
    return {"archives_checked": len(files), "max_rescoring_error_bits": max_error,
            "minimum_rt170_buffer_ms": min_buffer}


def check_controls(out):
    m = pd.read_csv(out / "positive_control_matches.csv")
    matched = m[m.reason.eq("matched")]
    cur = pd.read_csv(out / "positive_control_set_curves.csv.gz")
    keys = ["monkey", "condition", "session", "component", "cue"]
    matched_key = set(matched[keys].itertuples(index=False, name=None))
    assert matched_key == set(cur[keys].itertuples(index=False, name=None))
    assert cur.groupby(keys + ["center_ms"]).cohort.nunique().eq(2).all()
    for r in matched.itertuples():
        a = np.array([int(v) for v in r.uncertain_counts.split("/")])
        b = np.array([int(v) for v in r.single_counts.split("/")])
        n = np.array([int(v) for v in r.matched_counts.split("/")])
        assert (n == np.minimum(a, b)).all() and n.min() >= 5
        assert n.sum() == r.matched_trials and 0 < r.matched_units <= r.uncertain_units
    return {"matched_set_pairs": len(matched), "unmatched_set_pairs": len(m) - len(matched)}


def check_geometry(out):
    data = pd.read_csv(out / "geometry_holdout.csv.gz")
    assert not data.duplicated(["monkey", "unit", "condition", "window", "trial"]).any()
    r, p, g, c, k, v = (data[name].to_numpy() for name in
                        ("observed", "selected", "geometry", "constant", "category", "train_variance"))
    expected = np.column_stack([
        ((r - c) ** 2 - (r - g) ** 2) / v,
        ((r - c) ** 2 - (r - k) ** 2) / v,
        ((r - k) ** 2 - (r - g) ** 2) / v,
        ((r - p) ** 2 - (r - g) ** 2) / v,
        (p - g) ** 2 / v,
        -2 * (r - g) * (p - g) / v,
    ])
    assert np.all(v > 0)
    np.testing.assert_allclose(expected, data[list(GEOMETRY_METRICS)], atol=1e-10)
    max_error = np.max(np.abs(expected[:, 3] - expected[:, 4] - expected[:, 5]))
    assert max_error < 1e-9
    _, actual = summarize(data)
    summary = pd.read_csv(out / "geometry_summary.csv")
    np.testing.assert_allclose(actual[["mean", "low", "high"]], summary[["mean", "low", "high"]], atol=1e-12)
    return {"geometry_holdout_rows": len(data), "geometry_decomposition_error": float(max_error)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.output
    verification = {**check_decoding(out), **check_controls(out), **check_geometry(out)}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
