"""Run the analysis in the order used for the manuscript (Python 3.12/3.13)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / 'scripts'
DATA = [
    ('inventory_gnode_uncertainty.py',),
    ('preprocess_gnode_uncertainty.py',),
    ('preprocess_gnode_sliding_rates.py',),
]
ANALYSIS = [
    ('run_gnode_target_set_replication.py',),
    ('run_gnode_encoding_controls.py',),
    ('run_gnode_switch_dynamics.py',),
    ('run_gnode_within_set_decoding.py',),
    ('run_gnode_decoding_robustness.py',),
    ('run_gnode_decoder_calibration.py',),
    ('run_gnode_rt_evidence.py',),
    ('run_gnode_mechanism_models.py',),
    ('run_gnode_mechanism_models.py', '--q-upper', '0.9999', '--output-dir', 'results/gnode_mechanism_models_q9999'),
    ('run_gnode_rt_robustness.py',),
]
VALIDATE = [
    ('validate_gnode_replication.py',),
    ('validate_gnode_switch_dynamics.py',),
    ('validate_gnode_within_set_decoding.py',),
    ('validate_gnode_decoding_robustness.py',),
    ('validate_gnode_rt_evidence.py',),
    ('validate_gnode_mechanism_models.py',),
    ('validate_gnode_mechanism_models.py', '--result-dir', 'results/gnode_mechanism_models_q9999'),
    ('validate_gnode_review_revision.py',),
]
OUTPUT_NAMES = ('gnode_target_set', 'gnode_encoding_controls', 'gnode_switch_dynamics',
                'gnode_within_set_decoding', 'gnode_decoding_robustness',
                'gnode_decoder_calibration', 'gnode_rt_evidence', 'gnode_rt_robustness',
                'gnode_mechanism_models', 'gnode_mechanism_models_q9999')


def verify_reference() -> None:
    manifest = json.loads((ROOT / 'reference_results/manifest.json').read_text(encoding='utf-8'))
    for name, expected in manifest['reference_summaries_sha256'].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f'Reference summary checksum mismatch: {name}')
    print(f"Verified {len(manifest['reference_summaries_sha256'])} reference summaries.", flush=True)


def execute(steps: list[tuple[str, ...]]) -> None:
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    for step in steps:
        command = [sys.executable, str(SCRIPTS / step[0]), *step[1:]]
        print('\nRunning: ' + ' '.join(command), flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'data', 'analysis', 'validate', 'verify-reference'), default='all')
    parser.add_argument('--overwrite', action='store_true', help='Allow scripts to overwrite previously generated data or results.')
    args = parser.parse_args()
    if args.stage == 'verify-reference':
        verify_reference()
        return
    if args.stage in ('all', 'data'):
        raw = ROOT / 'data/raw/g-node_rz77m8'
        if not (raw / 'pickle/toc').is_file() or not (raw / 'derived_files/HDF5/Data').is_dir():
            parser.error(f'Extract the official dataset into {raw}; see README.md.')
        outputs = [ROOT / 'data/processed', ROOT / 'results/gnode_uncertainty_inventory.csv', ROOT / 'results/gnode_uncertainty_inventory.json']
        if not args.overwrite and any(path.exists() for path in outputs):
            parser.error('Preprocessing outputs already exist; use a fresh directory or --overwrite.')
    if args.stage in ('all', 'analysis'):
        existing = [ROOT / 'results' / name for name in OUTPUT_NAMES if (ROOT / 'results' / name).exists()]
        if existing and not args.overwrite:
            parser.error('Analysis output directories already exist; use a fresh directory or --overwrite.')
        if args.stage == 'analysis':
            required = ('gnode_uncertainty_rates.csv', 'gnode_rs_sliding_rates.npz', 'gnode_rs_sliding_metadata.csv')
            missing = [name for name in required if not (ROOT / 'data/processed' / name).is_file()]
            if missing:
                parser.error(f'Run the data stage first; missing: {missing}')
    if args.stage in ('all', 'data'):
        execute(DATA)
    if args.stage in ('all', 'analysis'):
        execute(ANALYSIS)
    if args.stage in ('all', 'validate'):
        execute(VALIDATE)


if __name__ == '__main__':
    main()
