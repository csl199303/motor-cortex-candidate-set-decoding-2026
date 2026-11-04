# Analysis specification: time-resolved prediction contrasts and crossings

## Status and scope

This specification was defined before the sliding-window results were examined.
It extends the fixed-window replication in
`second_dataset_protocol.md`. The cohort, direction mapping, set definitions,
recording-triplet aggregation, and recording-component inferential units remain
unchanged.

## Time-resolved rates

- Align spikes to the response signal (RS).
- Use 50 ms half-open windows stepped every 10 ms.
- Window centers span -950 through +650 ms relative to RS, inclusive.
- The confirmatory cluster-search interval is -350 through +650 ms. The earlier
  interval is retained for visualization and cross-temporal generalization only.
- Use all behaviorally valid trials. No trial is censored at movement onset or
  movement end because censoring would make the tested population change with
  time; event markers are retained for descriptive overlays.

At every time center, fit the Condition 1 cosine encoding model independently,
apply its coefficients without refitting, and compute the normalized set-versus-chosen loss
improvement in Conditions 2, 3, and their pooled trials. Aggregate neuron effects
by median within recording triplets and then by median within connected recording
components.

## Baseline correction

For each component and ambiguity condition, subtract the component's fixed
TS-aligned 0--400 ms effect from every time point. This paired correction was
specified before the sliding analysis and prevents the small block-level baseline
offset found in the fixed-window experiment from defining a switch.

## Cluster inference

Analyze Joe and Lili separately. At each center in the confirmatory interval,
compute a one-sample t statistic across component effects. Form positive and
negative clusters from adjacent centers exceeding a two-sided uncorrected 0.05
t threshold. Cluster mass is the sum of absolute t statistics.

Use 10,000 component-level random sign flips and the maximum cluster mass across
both signs and all confirmatory times to control family-wise error. Report every
cluster with corrected P, but mechanistic success requires a significant positive
pre-RS cluster and a significant negative post-RS cluster in each monkey.

## Switching latency

The group switch time is the first center at or after RS where the component mean
is negative for at least 100 ms (11 consecutive 10 ms centers). Estimate
uncertainty with 5,000 component bootstraps. A bootstrap replicate without a switch
inside the observed range is retained as censored and excluded from percentile
latency intervals, with the censored proportion reported.

Estimate Condition 2 and Condition 3 separately. The ambiguity-delay contrast is
`latency_C3 - latency_C2`; positive values mean that the three-target condition
switches later. The contrast is supportive and is not declared different unless
its two-sided bootstrap interval excludes zero in both monkeys.

## Cross-temporal generalization

For each neuron, train the Condition 1 cosine model at every train time and apply
it to ambiguous-condition firing rates at every test time. Compute the same
set-versus-chosen normalized loss difference for each train-test pair, then use
the specified triplet/component median aggregation. Plot the mean component matrix
for each monkey and their equal-weight mean.

Cross-temporal matrices are descriptive representational geometry. Because the
dataset contains at most seven simultaneously recorded neurons and the matrices
use matched pseudo-population structure across sessions, they are not interpreted
as single-trial population dynamics or noise-correlation estimates.

## Inferential criteria

The original software success criteria for the baseline-adjusted contrast required:

1. a cluster-corrected positive effect before RS;
2. a cluster-corrected negative effect after RS; and
3. a finite sustained group switch within +650 ms.

These operational criteria do not establish complete representational replacement.
The current analysis reports raw and baseline-adjusted crossings separately, as
specified in `revision_analysis_specification.md`. A negative adjusted contrast
tests a baseline-relative change, and even a negative raw contrast does not imply
the disappearance of set-related activity. Cluster boundaries are descriptive and
do not provide pointwise physiological onset inference.

The Condition 3 versus Condition 2 latency contrast is evaluated separately from
the overall switching analysis. Time range, smoothing width, persistence duration,
neuron filter, and component definition were specified in advance.
