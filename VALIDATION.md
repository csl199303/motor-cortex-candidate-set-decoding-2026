# Release preparation checks

The following checks were performed on Windows with Python 3.12.10 and the
versions in `requirements.txt` during preparation of this analysis-only release.

- All included Python files compiled; the release entry point checked all ten
  supplied reference-summary SHA-256 hashes.
- In an isolated test copy, the data stage was run on an extracted copy of the
  original dataset. Its inventory reported 221 matched neurons and 30 and 48
  recording components. The trial-rate and sliding-metadata CSV files were
  byte-identical to the existing analysis inputs, and all arrays in the
  sliding-rate NPZ were element-identical. The original archive was not present
  in the isolated test directory, so that run did not verify its ZIP checksum.
- The candidate-set prediction script was rerun on those generated inputs. Its
  primary means were 0.05933201478805052 and 0.05592468880080934 for Joe and
  Lili, respectively, agreeing with the existing results to floating-point
  precision.
- All eight validation commands completed using the rerun prediction outputs and
  previously generated outputs copied into the isolated test directory. They
  check internal consistency; passing them is not equivalent to rerunning every
  underlying analysis.

The complete analysis, including label permutations, injection simulations,
reaction-time analyses and both curve-model fits, was not rerun from scratch as
part of this packaging check. No claim is made that the current release exactly
reproduces the manuscript's final figure styling or schematic illustrations.
