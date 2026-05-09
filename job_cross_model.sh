#!/bin/bash
#SBATCH --account=def-amorales
#SBATCH --array=0-10
#SBATCH --time=1-12:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --output=slurm_out/E_xmodel_%A_%a.out
#SBATCH --error=slurm_err/E_xmodel_%A_%a.err

set -euo pipefail

# Pillar 3 (May 2026 reframing): Cross-model representation comparison.
# Tests the conjecture that knowledge matrices admit basis-free per-sample
# comparison across DIFFERENTLY-TRAINED same-arch checkpoints, while
# penultimate features require learned alignment (CKA / Re-Basin) to be
# even defined.
#
# Scope: resnet152 (k=5) -> 10 pairs; densenet121 (k=2) -> 1 pair = 11 total.
# GoogLeNet excluded (only k=1 public; declined to self-train).
#
# All checkpoints are differently-trained (different recipes, not same-recipe
# seed variants). This is a deliberate framing choice — see km-notes.md
# 2026-05-02 entry and paper-plan.md §1.1.
#
# Per-pair execution: each array task processes ONE (i, j) pair and writes
# results/cross_model/<arch>/per_pair/<i>__<j>.json. Mirrors job_theorem45.sh
# per-attack pattern.

mkdir -p $SLURM_SUBMIT_DIR/slurm_out
mkdir -p $SLURM_SUBMIT_DIR/slurm_err

# Map array task ID to (arch, ckpt_i, ckpt_j)
# 11 pairs = 10 resnet152 + 1 densenet121.
ALL_PAIRS=(
    # resnet152 — k=5 -> C(5,2) = 10 pairs (tv_v1, tv_v2, timm_a1, timm_a2, timm_a3)
    "resnet152:tv_v1:tv_v2"
    "resnet152:tv_v1:timm_a1"
    "resnet152:tv_v1:timm_a2"
    "resnet152:tv_v1:timm_a3"
    "resnet152:tv_v2:timm_a1"
    "resnet152:tv_v2:timm_a2"
    "resnet152:tv_v2:timm_a3"
    "resnet152:timm_a1:timm_a2"
    "resnet152:timm_a1:timm_a3"
    "resnet152:timm_a2:timm_a3"
    # densenet121 — k=2 -> 1 pair (tv_v1, timm_ra)
    "densenet121:tv_v1:timm_ra"
)

PAIR=${ALL_PAIRS[$SLURM_ARRAY_TASK_ID]}
IFS=':' read -r ARCH CKPT_I CKPT_J <<< "$PAIR"

echo "Step E: Cross-model — $ARCH / $CKPT_I vs $CKPT_J (task $SLURM_ARRAY_TASK_ID)"

module load StdEnv/2023 python/3.11.5 scipy-stack/2025a
source env/bin/activate

# Self-skip on verify failure. The orchestrator no longer filters E_NEEDED
# (verify runs as its own SLURM CPU job, after which we get scheduled here
# via --dependency=afterany). Conservative semantics: skip ONLY if the file
# explicitly marks a pair member "failed". Missing entries / missing file
# default to "passed" so manual sbatch (outside run_pipeline.sh) still works.
VERIFY_FILE="results/cross_model/verify_results.json"
if [ -f "$VERIFY_FILE" ]; then
    STATUS_I=$(python3 -c "
import json, sys
try:
    with open('$VERIFY_FILE') as f: d = json.load(f)
    print(d.get('checkpoints', {}).get('${ARCH}:${CKPT_I}', 'missing'))
except Exception:
    print('missing')
" 2>/dev/null || echo missing)
    STATUS_J=$(python3 -c "
import json, sys
try:
    with open('$VERIFY_FILE') as f: d = json.load(f)
    print(d.get('checkpoints', {}).get('${ARCH}:${CKPT_J}', 'missing'))
except Exception:
    print('missing')
" 2>/dev/null || echo missing)
    if [ "$STATUS_I" = "failed" ] || [ "$STATUS_J" = "failed" ]; then
        echo "SKIP: ${ARCH}/${CKPT_I}=${STATUS_I}, ${ARCH}/${CKPT_J}=${STATUS_J} — verify failed for at least one pair member; not running pair compute."
        echo "      Fix the remap in cross_model_experiment.py:build_km_model_with_alt_weights() and re-run."
        exit 0
    fi
    echo "Verify gate: ${ARCH}/${CKPT_I}=${STATUS_I}, ${ARCH}/${CKPT_J}=${STATUS_J} → proceeding."
else
    echo "Verify gate: $VERIFY_FILE not found — proceeding (manual sbatch path)."
fi

# Apply patches (idempotent) — needed for the knowledgematrix wrappers to
# load alternate state-dicts from torchvision V2 / timm RSB checkpoints.
bash patches/apply_neuralteleportation_patches.sh ./env || {
    echo "ERROR: failed to apply neuralteleportation patches"; exit 1;
}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python cross_model_experiment.py \
    --arch $ARCH \
    --ckpt-i $CKPT_I \
    --ckpt-j $CKPT_J \
    --num-samples 1000 \
    --matrix-batch-size 1024 \
    --imagenet-root /datashare/imagenet/ILSVRC2012

echo "Task $SLURM_ARRAY_TASK_ID ($ARCH / $CKPT_I vs $CKPT_J) completed"
