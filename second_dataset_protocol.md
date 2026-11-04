# Analysis specification: candidate-set prediction contrasts in ambiguous reaching

## Dataset and scope

This reanalysis uses the public G-Node dataset
`10.12751/g-node.rz77m8`, *Monkey behavioral neurophysiology during a delayed
center-out reach task with ambiguous target information*. The archive is CC BY
4.0 and contains trials from two rhesus macaques (Joe and Lili).

The preparatory signal (PS, 500 ms) displays one, two, or three possible reach
targets. The response signal (RS, 1500 ms) reveals the final target. Condition 1
therefore supplies complete target information, while Conditions 2 and 3 require
preparation under a uniform set of two or three possible targets.

The fixed-window contrast compares predictions transferred from Condition 1.
It does not by itself identify a neural mechanism or the variance explained by
candidate-set geometry. Prediction-dispersion, intercept and shuffled-label
controls are specified in `revision_analysis_specification.md`.

## Cohort

Include every `toc` entry with `all_conditions=True` and a complete matched
Condition 1/2/3 triplet for the same `global_neuron`. No unit is selected using
firing rate, tuning strength, or the analysis outcome.

The neuron is not the inferential unit. A recording triplet is the ordered tuple
of Condition 1, 2, and 3 session identifiers for a matched neuron. Triplets sharing
any session are merged into a connected recording component. Components are the
independent units for inference and are analyzed separately for Joe and Lili.

Anatomical labels are not available per neuron in the released files. Results are
therefore reported as motor-cortical and by monkey, not as M1-versus-PMd effects.

## Encoding model

For each neuron and analysis window, fit the following model using Condition 1
trials only:

`rate = beta0 + beta_cos cos(theta) + beta_sin sin(theta)`

Direction identifiers 1--6 are mapped uniformly around the unit circle. The fitted
model is then evaluated without refitting on Conditions 2 and 3 with two competing inputs:

1. **Chosen-target model:** use the direction revealed later at RS.
2. **Target-set expectation model:** average `cos(theta)` and `sin(theta)` over
   every target allowed by the preparatory cue.

Condition 2 target sets are `{1,2}`, `{3,4}`, and `{5,6}`. Condition 3 target sets
are `{6,1,2}` and `{3,4,5}`. A set rotated by three directions is a negative
control.

The neuron-level effect is normalized paired loss improvement:

`delta = (MSE_chosen - MSE_set) / variance(observed_rate)`

Positive values indicate lower error for the averaged set prediction; the
contrast includes a nonnegative averaging term. Effects are
first median-aggregated within recording triplets and then within connected
recording components.

## Time windows

- Baseline: 0--400 ms after trial start.
- Early preparation: 550--800 ms after trial start.
- Middle preparation: 850--1100 ms.
- **Late preparation:** 1150--1400 ms.
- Post-revelation: 25 ms after RS until movement onset; trials with less than
  50 ms of usable duration are excluded.

The main endpoint pools Conditions 2 and 3 in late preparation. Conditions 2
and 3 separately and the other preparation windows provide time- and ambiguity-specific
sensitivity analyses.

## Inference

For each monkey, report component count, mean, median, positive count, sign test,
and one-sided Wilcoxon test. The group null uses 10,000 random sign flips
of component effects. A pooled sign-flip statistic averages monkey-level means so
that the monkey with more components does not dominate.

The original software success criteria for the fixed-window contrast were:

1. the late-preparation median effect is positive in both monkeys; and
2. the pooled one-sided sign-flip P value is below 0.05.

These criteria are retained to reproduce the original contrast, but are not
sufficient evidence for set encoding: averaging predictions can itself produce
a positive contrast. The current manuscript uses animal-specific prediction
controls from `revision_analysis_specification.md` to qualify this interpretation.

The post-revelation falsification check should reverse direction: the chosen-target
model should outperform the target-set model after RS. Failure of this reversal is
reported and limits the interpretation of the preparatory effect.

Condition 1 grouped cross-validation and an analysis restricted to neurons with
positive Condition 1 tuning reliability assess sensitivity to model reliability;
neither changes the prespecified cohort.

## Data integrity contract

- Verify the downloaded archive byte count and ZIP CRC before extraction.
- Confirm every included HDF5 file exists and matches its `toc` metadata.
- Treat the integer sentinel `-9223372036854775808` as a missing event. Exclude
  trials unless all seven events are present and ordered
  `TS < PS < RS < MO < ME < RW < TE`; retain exclusion counts in the audit.
- Confirm selected directions are integers 1--6 and trial references in spike tables
  are valid.
- Cross-check a deterministic sample of HDF5 event/spike arrays against CSV.
- Preserve all excluded or malformed files in the inventory report.
