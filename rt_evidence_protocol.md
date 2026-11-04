# Analysis specification: selected-direction evidence and reaction time

## Question

Does trial-by-trial selected-direction evidence emerge earlier or more strongly on
trials with shorter reaction time (RT), and does the apparent timing difference
collapse when neural evidence is aligned to movement onset (MO)?

This specification was defined before the neural evidence-versus-RT results were examined.
It reuses the simultaneous-session cohort, target-set definitions, deterministic
five folds, 50 ms windows, and fixed nearest-centroid decoder from
`within_set_decoding_protocol.md`.

## Behavioral inclusion

RT is `MO - RS`. Include successful behaviorally complete trials with
`50 <= RT <= 600 ms`. These fixed plausibility bounds were selected from the
behavioral distribution before neural evidence was computed. No neural value is
used for trial exclusion.

## Held-out direction evidence

Within each session, condition, and preparatory target set, train the decoder on
four folds and evaluate the held-out fold. For each held-out trial and time point,
define normalized correct-direction evidence as:

`(nearest incorrect distance - correct distance) /
 (nearest incorrect distance + correct distance + epsilon)`

Distances are squared Euclidean distances in training-standardized simultaneous
unit space. Evidence is bounded approximately by -1 and +1; positive values favor
the correct selected direction. Exact distance ties use the same fixed class-order
rule as the validated decoding analysis.

## Main RT association

Average each trial's evidence over the fixed `RS +100...+200 ms` interval. Within
every session and selected direction, compute Spearman correlation between evidence
and RT. Fisher-transform valid direction correlations, average them equally
within each preparatory set and then equally across sets, and transform back to a
session correlation. Require at least five trials in a direction.

Aggregate session correlations by median within connected recording components.
Pool Conditions 2 and 3 by averaging component correlations only when both are
available. Analyze Joe and Lili separately.

Use 10,000 component sign flips for the one-sided negative-correlation test and
5,000 component bootstraps for a 95% interval. Evidence for the pooled association requires
the pooled component median is negative and one-sided sign-flip `P < 0.05` in
both monkeys.

## Time-resolved association

Repeat the direction-controlled correlation at every RS-aligned time point.
Within the confirmatory `+80...+300 ms` interval, form negative clusters using a
one-sided 0.05 t threshold and 10,000 maximum-mass sign flips. This analysis
characterizes the timing of the association and is interpreted separately from
the fixed-window test.

## Fast/slow and movement alignment

Within each session and selected direction, split trials at the median RT; ties are
assigned deterministically to keep group sizes as even as possible. Compute the
held-out evidence mean for fast and slow trials.

Report curves in two alignments:

- RS aligned: existing centers from -350 through +500 ms.
- MO aligned: interpolate each trial's RS-aligned evidence onto -200 through
  +200 ms relative to MO, at 10 ms steps. Values outside the observed RS-aligned
  range are missing and are not extrapolated.

Fast/slow curves are descriptive. Convergence under MO alignment supports a
shared movement-linked state, while persistent separation suggests amplitude or
strategy differences beyond a simple latency shift.

## Interpretation limits

The analysis establishes association, not causal control of RT. Evidence is
cross-validated at the trial level, but RT and movement-related firing may share
common causes. RT bounds, evidence window, correlation method, fold assignment,
and component rule were specified in advance.
