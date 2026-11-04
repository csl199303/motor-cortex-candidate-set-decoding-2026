# Candidate-set geometry and direction information in motor cortex

Analysis code for a reanalysis of a delayed center-out reaching task. The code
compares candidate-set and selected-direction predictions, estimates their
time-varying error contrast, decodes direction within fixed candidate sets, and
examines exploratory reaction-time associations and descriptive curve models.

## Data

The input data are the public dataset by Riehle (2024),
[G-Node DOI: 10.12751/g-node.rz77m8](https://doi.org/10.12751/g-node.rz77m8)
(CC BY 4.0). Download the original ZIP from that record and extract it into
`data/raw/g-node_rz77m8/`. The expected layout is:

```text
data/raw/g-node_rz77m8/
  pickle/toc
  derived_files/HDF5/Data/
```

The analyzed ZIP had SHA-256
`c644f98fa9f9d178e7b494a573521d0261e68430774cd68692851052416c52a1`.
If it is also saved as `data/raw/g-node_rz77m8.zip`, the inventory script checks
that checksum. An extracted dataset without the ZIP can be processed, but its
original archive checksum is not verified. Verify the source before loading:
the upstream dataset contains pickle files, which must not be loaded from an
untrusted source. Raw and processed data are not included in this repository.

## Environment

The original analyses used Python 3.13.9; subsequent control analyses used
Python 3.12.10, on Windows. Pinned packages are in `requirements.txt`. Set up a
virtual environment with Python 3.12 or 3.13, then:

```bash
python -m pip install -r requirements.txt
python run_pipeline.py --stage verify-reference
```

`verify-reference` checks that the supplied reference summaries are intact; it
does not run the scientific analyses. Reproduction requires the source data,
substantial CPU time and disk space. Different numerical libraries or platforms
may produce small differences. A GPU is not required.

## Run

```bash
python run_pipeline.py --stage data
python run_pipeline.py --stage analysis
python run_pipeline.py --stage validate
```

Alternatively, `python run_pipeline.py --stage all` executes these stages in
order. The data stage creates `data/processed/` and an inventory in `results/`;
the analysis stage writes new result directories under `results/`; validation
checks those outputs. To avoid accidental overwrites, data and analysis stages
stop if their output paths already exist. `--overwrite` explicitly permits
replacement; the validation stage also writes validation reports in `results/`.
Use a fresh clone when comparing runs. Execution stops when a step fails.

## Files and outputs

- `scripts/`: preprocessing, analysis and validation scripts.
- `*_protocol.md` and `revision_analysis_specification.md`: historical analysis
  specifications. The study was not preregistered; control and sensitivity
  analyses include work specified after inspection of earlier results.
- `reference_results/`: archived summary JSON files and their SHA-256 manifest.
  These are retained separately from regenerated `results/`; some summaries
  contain historical field names and should be interpreted with the manuscript.
- `results/`: generated locally; not included in the repository.

| Analysis | Output directory |
| --- | --- |
| Candidate-set prediction | `results/gnode_target_set/` |
| Prediction controls | `results/gnode_encoding_controls/` |
| Sliding-window prediction and crossings | `results/gnode_switch_dynamics/` |
| Within-set decoding and cross-temporal decoding | `results/gnode_within_set_decoding/` |
| Decoder robustness | `results/gnode_decoding_robustness/` |
| Label permutations and simulations | `results/gnode_decoder_calibration/` |
| Direction evidence and reaction time | `results/gnode_rt_evidence/` and `results/gnode_rt_robustness/` |
| Descriptive curve-model comparisons | `results/gnode_mechanism_models/` and `results/gnode_mechanism_models_q9999/` |

This release provides the analysis and original plotting code, not a pipeline
for compiling the final typeset manuscript or regenerating its revised
schematics and publication-styled figures. Graphical illustrations are
schematic. Generated plots may differ in styling from the manuscript.

## Citation and license

Please cite the G-Node dataset above. A citation for the reanalysis and a code
license will be added once the authors have selected and confirmed them. Until
then, no reuse license for this repository's code is stated here; the dataset's
CC BY 4.0 license applies to the dataset, not automatically to this code.
