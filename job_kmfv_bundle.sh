#!/bin/bash
#SBATCH --time=00:15:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=slurm_out/D5_bundle_%j.out
#SBATCH --error=slurm_err/D5_bundle_%j.err

set -euo pipefail

mkdir -p $SLURM_SUBMIT_DIR/slurm_out $SLURM_SUBMIT_DIR/slurm_err
cd "$SLURM_SUBMIT_DIR"

echo "Step D5: completeness check + tar bundle"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Completeness gate: every required state file must exist with the expected
# entry count. If anything is missing, skip the tar (no bundle produced).
python - <<'PY' && \
    tar -cf km-feature-viz.tar -C results km-feature-viz && \
    echo "bundle: km-feature-viz.tar ($(du -h km-feature-viz.tar | cut -f1))" \
  || echo "bundle: skipped (pipeline incomplete — see message above)"
import json, sys
base = 'results/km-feature-viz'
checks = {
    'state/01_compute_kms_alexnet.json':   500,
    'state/01_compute_kms_resnet18.json':  500,
    'state/01_compute_kms_vgg11.json':     500,
    'state/02_gradcam.json':              1500,
    'state/02_ig.json':                   1500,
    'state/02_smoothgrad.json':           1500,
    'state/02_feature_maps.json':         1500,
    'state/02_pgd.json':                  1500,
    'state/03_deepdream_alexnet.json':      15,
    'state/03_deepdream_resnet18.json':     15,
    'state/03_deepdream_vgg11.json':        15,
}
missing = []
for rel, expected in checks.items():
    p = f'{base}/{rel}'
    try:
        data = json.load(open(p))
    except FileNotFoundError:
        missing.append(f'{rel}: MISSING'); continue
    if expected is not None and len(data) != expected:
        missing.append(f'{rel}: {len(data)}/{expected}')
if missing:
    print('INCOMPLETE (skipping bundle):')
    for m in missing: print('  -', m)
    sys.exit(1)
print('All state files complete — ready to bundle.')
PY

echo "D5 completed"
