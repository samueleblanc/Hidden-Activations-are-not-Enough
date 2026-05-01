#!/bin/bash
# Apply compatibility patches to neuralteleportation for PyTorch 2.x and NumPy 2.x.
#
# These patches fix the following issues:
#
#   1. PyTorch 2.x JIT graph ordering: The inlined_graph reorders downsample
#      block nodes (BatchNorm before Conv), breaking the positional module
#      assignment. Fixed by reordering graph keys to match forward-hook order.
#      (patch: neuralteleportation_pytorch2_compat.patch)
#
#   2. NumPy 2.x deprecation: np.float was removed. Replaced with np.float64.
#      (patch: neuralteleportation_pytorch2_compat.patch)
#
#   3. Parallel-branch cob bleed in Inception/GoogLeNet: when a parallel branch
#      contains a non-neuron layer (MaxPool / ReLU / etc.) before joining at a
#      Concat, the linear `current_cob` walker bleeds the previous branch's
#      output cob into that non-neuron layer, causing a teleport-time size
#      mismatch. Fix: scale-invariant non-neuron layers always read their cob
#      from the actual input layer's cob (max(layer['in'])).
#      (patch: neuralteleportation_parallel_branch.patch)
#
#   4. New COB module — GoogLeNetCOB: copies googlenetcob.py into the package's
#      models/model_zoo/ directory. Required because the upstream library does
#      not ship a GoogLeNet COB factory but Step B of the Hidden-Activations-
#      are-not-Enough Pillar-1 experiment needs to teleport torchvision's
#      pretrained GoogLeNet.
#
# Usage:
#   bash patches/apply_neuralteleportation_patches.sh [venv_path]
#   Default venv_path: ./env

set -euo pipefail

VENV_PATH_RAW="${1:-./env}"
if [ ! -d "$VENV_PATH_RAW" ]; then
    echo "ERROR: venv directory not found: $VENV_PATH_RAW"
    echo "Install with: python -m venv $VENV_PATH_RAW && pip install -r requirements-slurm.txt"
    exit 1
fi
# Canonicalize to absolute path. The patch loop below does `cd $SITE_PACKAGES/..`
# once per iteration; keeping VENV_PATH absolute makes that resolve correctly
# regardless of the loop's cumulative cwd state. (Previous bug: relative
# VENV_PATH → second iteration's cd resolved against the post-first-iter cwd
# → "No such file or directory" → set -e killed the script.)
VENV_PATH="$(cd "$VENV_PATH_RAW" && pwd)"
SITE_PACKAGES="$VENV_PATH/lib/python3.11/site-packages/neuralteleportation"

if [ ! -d "$SITE_PACKAGES" ]; then
    echo "ERROR: neuralteleportation not found at $SITE_PACKAGES"
    echo "Install it first: pip install git+https://github.com/vitalab/neuralteleportation.git"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_CWD="$(pwd)"

# Apply text patches. Restore cwd at the end so the caller's relative paths
# (e.g., job_teleportation.sh's downstream commands) still resolve.
for PATCH_NAME in neuralteleportation_pytorch2_compat.patch neuralteleportation_parallel_branch.patch; do
    PATCH_FILE="$SCRIPT_DIR/$PATCH_NAME"
    if [ ! -f "$PATCH_FILE" ]; then
        echo "ERROR: Patch file not found: $PATCH_FILE"
        exit 1
    fi
    echo "Applying $PATCH_NAME to: $SITE_PACKAGES"
    cd "$SITE_PACKAGES/.."
    patch -p0 --forward < "$PATCH_FILE" || {
        echo "Patch may have already been applied. Continuing."
    }
done
cd "$ORIGINAL_CWD"

# Copy GoogLeNetCOB module
GOOGLENET_SRC="$SCRIPT_DIR/googlenetcob.py"
GOOGLENET_DST="$SITE_PACKAGES/models/model_zoo/googlenetcob.py"
if [ ! -f "$GOOGLENET_SRC" ]; then
    echo "ERROR: googlenetcob.py not found at $GOOGLENET_SRC"
    exit 1
fi
echo "Installing googlenetcob.py at: $GOOGLENET_DST"
cp "$GOOGLENET_SRC" "$GOOGLENET_DST"

echo "Done."
