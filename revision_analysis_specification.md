# Prediction Controls and Decoder Calibration

These secondary analyses were specified after inspection of the original results.
They are sensitivity analyses on the same dataset, not preregistered confirmation.
No alternative specification is selected according to its statistical significance.

## Candidate-Set Prediction

Fit the same cosine tuning function in Condition 1 and apply it without refitting
to uncertain trials in the baseline and late-preparation windows. Retain the
original matched cohort, triplet medians and component medians. For each unit,
decompose the selected-minus-set normalized loss into prediction dispersion and
a residual cross term. Verify the exact decomposition before aggregation.
Separately report gain over the Condition 1 intercept and held-out R-squared.
Because medians are nonlinear, aggregated terms are not an additive decomposition.

Generate 999 candidate-label shuffles within each session and condition, sharing
trial permutations across simultaneously recorded units. Keep predictions and
label frequencies fixed and aggregate each permutation using the original
hierarchy. Compare observed component-mean gain with the shuffled distribution.
The late-preparation comparison is the substantive control; baseline is a
negative control. Bootstrap components 5,000 times within each animal.

## Crossings and Upper Bounds

Estimate sustained negative raw-error crossings separately from crossings after
baseline subtraction. Both require at least 100 ms of negative mean contrast.
Bootstrap intervals condition on a crossing and report censoring fractions.
These estimates do not test disappearance of candidate-set information.

Report a one-sided upper-confidence-limit criterion separately from symmetric
95-percent interval containment. Both apply to component-mean pre-RS decoding
averaged from -350 to -30 ms. A 95-percent interval within symmetric bounds is
a conservative containment assessment; no TOST p-value is inferred from it.

## Decoder Calibration

Shuffle selected-direction labels 499 times within session and candidate set.
Repeat stratified five-fold assignment, training-only scaling, centroid fitting,
held-out balanced accuracy and hierarchical aggregation for every permutation.
Evaluate pre-RS interval means and a positive maximum-cluster statistic over
0-500 ms. The expanded post-RS interval is secondary; neither its boundaries nor
the original cluster boundaries are localized physiological onset estimates.

For independent Poisson rate-code injections, preserve empirical session, set,
unit and trial counts and set-average pre-RS mean rates (floor 0.2 Hz). Generate
37 ten-millisecond bins and form 33 overlapping fifty-millisecond counts. Inject
direction-specific gains exp(a*cos(theta-phi)), normalized within each set, for
a in 0, 0.1, 0.25, 0.5 and 1. Preferred angles are uniform per unit and set.
Use 200 simulated datasets per amplitude. The first 100 zero-signal datasets
calibrate a 95th-percentile threshold on the animal's component-mean decoding
advantage; the other 100 assess false detections. Nonzero amplitudes use all 200
runs. Report detection fractions and Wilson intervals. These simulations concern
this rate-code family and do not establish sensitivity to arbitrary neural codes.

## Exploratory RT Windows

Use centers 80-120 ms with RT >=170 ms, and centers 100-150 ms with RT >=200 ms.
Their spike-count support ends at 145 and 175 ms, respectively, leaving 25 ms
before overt movement on every retained test trial. Obtain evidence from the
original held-out classifier trained on the RT-filtered base cohort (50-600 ms).
Require at least five retained trials in each of all six directions per session
and both uncertainty conditions in a pooled component. Report retained trials,
sessions, components and intervals. P-values across these exploratory window
specifications are unadjusted; no causal preparation claim is made.

## Reproduction

Seeds derive from SHA-256 labels in the code. The source-to-result map is in
README.md. The data stage generates the inventory before
preprocessing. The original archive checksum is verified when that archive is
available; extracted-file auditing is reported separately when it is not.
