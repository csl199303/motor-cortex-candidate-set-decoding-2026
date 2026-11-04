# Analysis specification: within-set selected-direction decoding

## Question

Can the final reach direction be decoded before the response signal when trials
share the same preparatory target set, and when does that information become
decodable after the response signal?

This specification was defined before the within-set decoding results were examined. It uses
the cohort and RS-aligned 50 ms windows from `switch_dynamics_protocol.md` but a
new outcome and a new model.

## Simultaneous-session decoder

Analyze Conditions 2 and 3. For every monkey, condition, and recording session,
construct a trial-by-simultaneously-recorded-unit feature matrix at each time.
Use only matched cohort units, with no tuning or firing-rate selection.

Decode selected direction separately inside each preparatory target set:

- Condition 2: `{1,2}`, `{3,4}`, and `{5,6}`; chance balanced accuracy is 1/2.
- Condition 3: `{6,1,2}` and `{3,4,5}`; chance balanced accuracy is 1/3.

Use deterministic five-fold stratification within each selected direction. In every
fold and time window, standardize units using training trials only, estimate one
nearest centroid per direction, and classify held-out trials by Euclidean
distance. The session effect is the equal-set mean balanced accuracy minus the
set-specific theoretical chance level. Cross-validation folds are fixed across
time.

Aggregate session effects by median within connected recording components. A
session maps to exactly one component. Pool Conditions 2 and 3 by averaging their
component effects only when both are available. Analyze Joe and Lili separately.

## Confirmatory inference

Use two disjoint intervals so an RS-spanning cluster cannot satisfy both tests:

- Pre-RS absence interval: -350 through -30 ms.
- Post-RS emergence interval: +80 through +500 ms.

At each time, compute a one-sample t statistic across component effects. Form only
positive clusters at an uncorrected one-sided 0.05 t threshold. Use 10,000
component-level sign flips and the maximum positive cluster mass within each
interval for family-wise correction.

Non-significance alone does not establish pre-RS absence. Also bootstrap the mean
component effect averaged across the full pre-RS interval with 5,000 component
resamples. The original operational upper-bound criterion requires the upper
95% bootstrap bound to be below 0.05 absolute balanced-accuracy points above chance. The 0.05 SESOI is
specified before result inspection.

## Inferential criteria

Evidence for selected-direction emergence requires both monkeys to show, for the
pooled Condition 2+3 component effect:

1. no corrected positive cluster in the pre-RS interval;
2. a pre-RS 95% bootstrap upper bound below +0.05; and
3. at least one corrected positive cluster in the post-RS interval.

Conditions 2 and 3 are also reported separately. A difference between conditions
is descriptive unless independently significant in both monkeys.

## Cross-temporal decoding

Repeat the same held-out nearest-centroid decoder on a 50 ms train/test grid
(every fifth 10 ms center). Standardization and centroids are learned at the train
time and applied unchanged at every test time. Aggregate matrices using the same
session-to-component hierarchy.

Cross-temporal matrices are interpreted as stability of simultaneously recorded
direction information. No inference about across-session noise correlations is
made. Display choices may clip colors, but numerical matrices remain unchanged.

## Sensitivity analyses

Repeat time-resolved decoding with diagonal Gaussian naive Bayes while retaining
the same deterministic folds, target sets, and component aggregation. Estimate
class-specific means and variances from the training fold only after training-fold
standardization; apply a variance floor of `1e-3` in standardized units.

Evaluate practical equivalence at bounds of 0.005, 0.01, 0.02, 0.03, 0.04,
and 0.05 balanced-accuracy points. Separately repeat nearest-centroid inference
after restricting sessions to at least two and at least three simultaneously
recorded units. These analyses assess classifier dependence, equivalence-bound
dependence, and sensitivity to sparse sessions without changing the main cohort.

The main-analysis intervals, nearest-centroid decoder, fold assignment, component
rule, and 0.05 SESOI were specified before the main result was examined. The
additional analyses assess robustness and do not redefine the main test.

## Interpretation of the interval criteria

The original upper-bound criterion is one-sided. It bounds the component-mean
advantage averaged across the pre-RS interval and does not establish two-sided
equivalence or absence at every time. The revised analysis reports symmetric
95% interval containment separately; it does not label that containment as a
TOST p-value. Label-permutation and injection specifications are provided in
`revision_analysis_specification.md`.
