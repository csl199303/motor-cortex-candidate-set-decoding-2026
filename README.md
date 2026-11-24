# Motor cortex direction decoding: iscience-r1

This version provides the code and small reference summaries for the
cue-conditioned decoding and geometric baseline analyses. No source recordings,
processed inputs, or trial-level predictions are distributed here.

## Data

Download the ZIP from [G-Node Open Data](https://doi.gin.g-node.org/10.12751/g-node.rz77m8/),
DOI [10.12751/g-node.rz77m8](https://doi.org/10.12751/g-node.rz77m8) (Riehle, 2024;
CC BY 4.0). Save it locally as `data/raw/g-node_rz77m8.zip` and verify its
SHA-256 before extracting or loading any data:

~~~powershell
Get-FileHash data/raw/g-node_rz77m8.zip -Algorithm SHA256
~~~

Expected SHA-256: `c644f98fa9f9d178e7b494a573521d0261e68430774cd68692851052416c52a1`.
Extract it so that `data/raw/g-node_rz77m8/` directly contains `pickle/toc`
and `derived_files/HDF5/Data/`. The upstream dataset includes pickle files;
only load the verified archive from the official source.

## Run

Run from the repository root in PowerShell on Windows. The reference run used
Python 3.12.10, with dependencies pinned below. A CPU is sufficient.

~~~powershell
py -3.12 -m venv .venv
$python = (Resolve-Path .venv/Scripts/python.exe).Path
& $python -m pip install -r requirements-gnode-paper.txt
& $python scripts/test_iscience_decoding.py
& $python scripts/inventory_gnode_uncertainty.py
& $python scripts/preprocess_gnode_uncertainty.py
& $python scripts/preprocess_gnode_sliding_rates.py
& ./scripts/run_iscience_analysis.ps1 -Python $python -Output results/reproduced
~~~

The three preparation scripts create the local inputs used by the analysis.
The analysis writes summary tables, verification JSON, and local caches and
plots. Keep them separate from the nine checked-in reference files under
[`results/iscience_revision/`](results/iscience_revision/).

Compare the six CSVs to the files of the same name in `results/reproduced/`:

~~~powershell
Get-ChildItem results/iscience_revision -Filter '*.csv' | ForEach-Object {
    $ref = (Get-FileHash $_.FullName).Hash
    $new = (Get-FileHash (Join-Path results/reproduced $_.Name)).Hash
    [pscustomobject]@{ File = $_.Name; Identical = ($ref -eq $new) }
}
Get-Content results/reproduced/verification.json
~~~

Use `main_effect_sizes_with_samples.csv` for the principal effects and sample
counts. The other CSVs separate decoding, matching, cohort, and geometry
summaries. The two manifests record the analysis source and processed input
checksums. Minor floating-point or file encoding differences are possible in
other environments. Do not commit the downloaded data or local outputs.

## License and citation

The code is [MIT-licensed](LICENSE). The G-Node dataset is separately
licensed under CC BY 4.0. Cite the dataset DOI and the tag
[`v2.0-iscience-r1`](https://github.com/csl199303/motor-cortex-candidate-set-decoding-2026/tree/v2.0-iscience-r1).
