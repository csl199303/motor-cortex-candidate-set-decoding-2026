"""Render empirical figures and assemble auditable tables for the iScience draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd

from run_iscience_decoding import DEFAULT_OUT

BLUE, ORANGE, GREEN, GRAY = "#0072B2", "#D55E00", "#00875F", "#555555"
ANIMALS = ("joe", "lili")


def save(fig, out, stem):
    fig.savefig(out / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{stem}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw(ax, data, color, label, style="-", bands=True):
    data = data.sort_values("center_ms")
    ax.plot(data.center_ms, data["mean"], color=color, label=label, linestyle=style)
    if bands:
        ax.fill_between(data.center_ms, data.low, data.high, color=color, alpha=.14, linewidth=0)


def time_axis(ax, shade=True):
    ax.axvline(0, color=GRAY, lw=.8, ls=":")
    ax.axhline(0, color="#a0a0a0", lw=.7)
    if shade:
        ax.axvspan(80, 120, color="#dddddd", alpha=.45, zorder=0)


def main_figure(data, out):
    fig, axes = plt.subplots(2, 2, figsize=(9, 5.8), sharex=True, sharey="row", layout="constrained")
    for i, animal in enumerate(ANIMALS):
        d = data.query("condition == 0 and monkey == @animal")
        a, b = axes[:, i]
        a.set_title(animal.capitalize())
        for metric, color, label in (
            ("neural_only_bits", ORANGE, "Neural only"),
            ("cue_only_bits", GRAY, "Cue only"),
            ("conditional_bits", BLUE, "Cue + neural"),
        ):
            draw(a, d.query("cohort == 'full' and metric == @metric"), color, label, bands=False)
        for cohort, color, label in (("full", GRAY, "Full cohort"), ("rt170", BLUE, "RT >=170 ms")):
            draw(b, d.query("cohort == @cohort and metric == 'gain_bits'"), color, label)
        for j, ax in enumerate((a, b)):
            time_axis(ax)
            ax.text(.01, .97, "ABCD"[j * 2 + i], transform=ax.transAxes, va="top", fontweight="bold")
        b.set_xlabel("Time from response signal (ms)")
    axes[0, 0].set_ylabel("Score vs. 1/6 prior (bits/trial)")
    axes[1, 0].set_ylabel("Gain vs. cue only (bits/trial)")
    axes[0, 1].legend(loc="center right")
    axes[1, 1].legend(loc="upper left", bbox_to_anchor=(.08, 1))
    save(fig, out, "fig1_decoding")


def positive_figure(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.05), sharey=True, layout="constrained")
    for i, (animal, ax) in enumerate(zip(ANIMALS, axes)):
        d = data.query("condition == 0 and monkey == @animal")
        for cohort, color, label in (("single", GREEN, "Single target"),
                                     ("uncertain", GRAY, "Matched uncertain trials")):
            draw(ax, d.query("cohort == @cohort"), color, label)
        time_axis(ax, shade=False)
        ax.axvspan(-350, -30, color="#dddddd", alpha=.38, zorder=0)
        ax.set(title=animal.capitalize(), xlabel="Time from response signal (ms)", xlim=(-350, 500))
        ax.text(.02, .97, "AB"[i], transform=ax.transAxes, va="top", fontweight="bold")
    axes[0].set_ylabel("Within-set gain (bits/trial)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(.09, 1))
    save(fig, out, "fig2_positive_control")


def geometry_figure(data, out):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    keys = [("joe", 2), ("joe", 3), ("lili", 2), ("lili", 3)]
    for i, (ax, specs, label) in enumerate(zip(
        axes,
        [(("geometry_over_constant", BLUE, "Transferred geometry"), ("category_over_constant", GREEN, "Set category")),
         (("selected_minus_geometry", BLUE, "Error contrast"), ("dispersion", ORANGE, "Dispersion"),
          ("cross_term", GRAY, "Cross term"))],
        ["Gain over constant", "Error decomposition terms"],
    )):
        ax.axhline(0, color="#a0a0a0", lw=.8)
        for offset, (metric, color, series) in zip(np.linspace(-.15, .15, len(specs)), specs):
            d = data.query("window == 'prep_late' and metric == @metric").set_index(["monkey", "condition"]).loc[keys]
            y = d["mean"].to_numpy()
            ax.errorbar(np.arange(len(keys)) + offset, y, yerr=[y - d.low, d.high - y],
                        fmt="o", capsize=2, color=color, label=series, ms=4)
        ax.set_xticks(range(len(keys)), [f"{m.capitalize()}\n{k} targets" for m, k in keys])
        ax.set_ylabel(label + "\n(in training-variance units)")
        ax.text(.02, .97, "AB"[i], transform=ax.transAxes, va="top", fontweight="bold")
        ax.legend(loc="best", fontsize=8)
    save(fig, out, "fig3_geometry")


def abstract_figure(data, out):
    fig, (task, score) = plt.subplots(1, 2, figsize=(8.0, 3.15),
                                     gridspec_kw={"width_ratios": [1, 1.8]}, layout="constrained")
    for i, x in enumerate((-.04, 2.35)):
        for j, angle in enumerate(np.arange(6) * np.pi / 3):
            col = (GREEN if j in (1, 2) else "white") if i == 0 else (ORANGE if j == 1 else "white")
            task.add_patch(Circle((x + .85 * np.cos(angle), .85 * np.sin(angle)), .13,
                                  ec=GRAY, fc=col, lw=.8))
        task.add_patch(Circle((x, 0), .105, fc=GRAY, ec="none"))
    task.annotate("", xy=(1.32, 0), xytext=(.96, 0), arrowprops={"arrowstyle": "->", "color": GRAY})
    task.text(-.04, -1.05, "Two candidates", ha="center")
    task.text(2.35, -1.05, "Response signal", ha="center")
    task.set(xlim=(-1.12, 3.45), ylim=(-1.35, 1.07), aspect="equal")
    task.axis("off")
    for animal, color in zip(ANIMALS, (BLUE, ORANGE)):
        draw(score, data.query("monkey == @animal and condition == 0 and cohort == 'rt170' and metric == 'gain_bits'"),
             color, animal.capitalize())
    time_axis(score)
    score.set(xlabel="Time from response signal (ms)", ylabel="Gain vs. cue only (bits/trial)", xlim=(-350, 123))
    score.annotate("80-120 ms", (120, 1.015), xycoords=("data", "axes fraction"),
                   ha="right", va="bottom", fontsize=8)
    score.legend(loc="upper left")
    save(fig, out, "graphical_abstract")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    out = args.results
    fig = out / "figures"
    fig.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "legend.frameon": False, "pdf.fonttype": 42, "savefig.pad_inches": .12})
    curves = pd.read_csv(out / "decoding_curves.csv")
    main_figure(curves, fig)
    positive_figure(pd.read_csv(out / "positive_control_curves.csv"), fig)
    geometry_figure(pd.read_csv(out / "geometry_summary.csv"), fig)
    abstract_figure(curves, fig)

    samples = pd.read_csv(out / "cohort_summary.csv")
    windows = pd.read_csv(out / "decoding_windows.csv")
    keys = ["monkey", "condition", "cohort"]
    trials = pd.read_csv(out / "evaluated_trials.csv.gz")
    cohort_sessions = pd.read_csv(out / "cohort_sessions.csv").query("accepted")
    pooled = {}
    for (animal, cohort), t in trials.groupby(["monkey", "cohort"]):
        components = t.groupby("component").condition.nunique()
        complete = components[components.eq(2)].index
        t = t[t.component.isin(complete)]
        units = cohort_sessions.query("monkey == @animal and cohort == @cohort")
        units = units.loc[units.component.isin(complete), "units"]
        pooled[animal, cohort] = dict(
            monkey=animal, condition=0, cohort=cohort, n_trials=len(t),
            n_sessions=len(t[["condition", "session"]].drop_duplicates()),
            n_components=t.component.nunique(), unit_min=units.min(), unit_median=units.median(),
            unit_max=units.max(), rt_min=t.rt_ms.min(), rt_p05=t.rt_ms.quantile(.05),
            rt_median=t.rt_ms.median(), rt_p95=t.rt_ms.quantile(.95), rt_max=t.rt_ms.max(),
        )
    for (animal, cohort), row in pooled.items():
        row["fraction_of_full"] = row["n_trials"] / pooled[animal, "full"]["n_trials"]
    samples = pd.concat([samples, pd.DataFrame(pooled.values())], ignore_index=True)
    combined = windows.merge(samples, on=keys + ["n_components"], validate="many_to_one")
    assert len(combined) == len(windows)
    combined.to_csv(out / "main_effect_sizes_with_samples.csv", index=False)
    m = pd.read_csv(out / "positive_control_matches.csv")
    m = m.assign(matched=m.reason.eq("matched"), all_units=m.matched_units.eq(m.uncertain_units))
    s = m.groupby(["monkey", "condition"]).agg(
        n_sets=("matched", "size"), n_matched_sets=("matched", "sum"),
        n_full_unit_sets=("all_units", "sum"), median_matched_units=("matched_units", "median"),
        median_matched_trials=("matched_trials", "median"),
    )
    s.to_csv(out / "positive_control_match_summary.csv")


if __name__ == "__main__":
    main()
