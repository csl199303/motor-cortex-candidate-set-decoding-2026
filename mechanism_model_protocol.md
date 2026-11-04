# Analysis specification: latency, rate, and threshold model comparison

## Scope

This is a phenomenological model comparison of the relationship between
cross-validated selected-direction evidence and reaction time. Candidate models and
evaluation criteria were defined before fitting. Decoder evidence is not assumed to be a literal
biophysical accumulator state.

## Data

Use the pooled Condition 2+3 component-level fast-RT and slow-RT evidence curves
from `rt_evidence_protocol.md`. Analyze Joe and Lili separately. Fit the emergence
interval from -50 through +220 ms relative to RS, inclusive. For each component,
use the median fast and slow RT obtained from its contributing sessions and
conditions.

The model predicts two evidence curves and two RT values jointly. Evidence and RT
domains receive equal total weight. Within each monkey, scale residuals using a
robust, model-independent measurement scale calculated before fitting. For each
LOCO fold, recalculate both scales from the training components only and use
those training scales for held-out scoring:

- evidence scale: 1.4826 times the MAD of component curves around the across-
  component median curve, pooled over group and time;
- RT scale: 1.4826 times the MAD of component RT values around the across-
  component group median.

Scales are fixed across all candidate models.

## Shared observation function

Evidence follows a logistic emergence curve:

`e(t) = b + A / (1 + exp(-(t - L) / tau))`

where `b` is baseline, `A > 0` is amplitude, `L` is midpoint, and `tau > 0`
controls rise speed. Movement occurs when the curve reaches a threshold fraction
`q` of its dynamic range. The predicted RT is:

`RT = L + tau * log(q / (1-q))`

## Equal-complexity candidate models

Each model has six fitted parameters.

1. **Latency-shift model**:
   common `b, A, tau, q`; group-specific `L_fast, L_slow`.
2. **Accumulation-rate model**:
   common `b, A, L, q`; group-specific `tau_fast, tau_slow`.
3. **Variable-threshold model**:
   common `b, A, L, tau`; group-specific `q_fast, q_slow`.

Bounds are `b in [-0.5, 0.2]`, `A in [0.01, 1.0]`,
`L in [-100, 300] ms`, `tau in [5, 300] ms`, and `q in [0.01, 0.99]`.

## Fitting and validation

For each monkey and model:

1. fit all components with deterministic multistart bounded least squares;
2. fit every leave-one-component-out training set with an independent,
   deterministic multistart search that never uses the held-out component;
3. evaluate the held-out component with
   `0.5 * evidence standardized MSE + 0.5 * RT standardized MSE`;
4. retain one held-out score per component.

Compare candidate scores using paired component differences. Report mean,
median, 5,000-component-bootstrap 95% intervals, and two-sided paired sign-flip
P values. The best-fitting model is the model with the lowest mean held-out score.

## Inferential criteria

A unique mechanistic winner is declared only if the same model has the lowest
mean held-out score in both monkeys and its paired score improvement over the
second-best model has a bootstrap 95% interval entirely above zero in both
monkeys. Otherwise the mechanism comparison is inconclusive or mixed.

Parameter estimates and predicted curves are descriptive. Models, time range,
residual weighting, bounds, and selection rule were specified in advance. The
model comparison is interpreted independently of the decoding results.
