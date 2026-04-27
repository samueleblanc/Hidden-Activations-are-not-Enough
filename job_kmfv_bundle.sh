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
# Pillar 3 launch: 3 archs (resnet152, densenet121, googlenet), 3 classes
# (207, 282, 340), 7+7+6 = 20 images per arch.
#
# Baselines run all archs in one process via the manifest, so per-method
# counts = (#archs × #images) = 3 × 20 = 60.
# DeepDream is per-arch, 5 neurons × 3 layers = 15 entries per arch.
# Neuron-selection (03a) is a single-document state file per arch (not
# entry-counted) — verified via existence + JSON parse + presence of a
# 'method' key.
#
# IMPORTANT: D_MODELS in run_pipeline.sh is the source of truth for the
# arch list. Keep this gate in sync if D_MODELS changes.
counted = {
    'state/01_compute_kms_resnet152.json':    20,
    'state/01_compute_kms_densenet121.json':  20,
    'state/01_compute_kms_googlenet.json':    20,
    'state/02_gradcam.json':                  60,
    'state/02_ig.json':                       60,
    'state/02_smoothgrad.json':               60,
    'state/02_feature_maps.json':             60,
    'state/02_pgd.json':                      60,
    'state/03_deepdream_resnet152.json':      15,
    'state/03_deepdream_densenet121.json':    15,
    'state/03_deepdream_googlenet.json':      15,
}
# Existence-only state files: single-document JSON (not a dict-of-entries).
# Validity = file exists, parses as JSON, and has a 'method' key.
existence_only = [
    'state/03a_neuron_selection_resnet152.json',
    'state/03a_neuron_selection_densenet121.json',
    'state/03a_neuron_selection_googlenet.json',
]
missing = []
for rel, expected in counted.items():
    p = f'{base}/{rel}'
    try:
        data = json.load(open(p))
    except FileNotFoundError:
        missing.append(f'{rel}: MISSING'); continue
    except json.JSONDecodeError as e:
        missing.append(f'{rel}: INVALID JSON ({e})'); continue
    if expected is not None and len(data) != expected:
        missing.append(f'{rel}: {len(data)}/{expected}')
for rel in existence_only:
    p = f'{base}/{rel}'
    try:
        data = json.load(open(p))
    except FileNotFoundError:
        missing.append(f'{rel}: MISSING'); continue
    except json.JSONDecodeError as e:
        missing.append(f'{rel}: INVALID JSON ({e})'); continue
    if not (isinstance(data, dict) and 'method' in data):
        missing.append(f"{rel}: missing 'method' key")
if missing:
    print('INCOMPLETE (skipping bundle):')
    for m in missing: print('  -', m)
    sys.exit(1)
print('All state files complete — ready to bundle.')
PY

echo "D5 completed"
