#!/bin/bash
# Apply compatibility patches to neuralteleportation for PyTorch 2.x and NumPy 2.x.
#
# These patches fix two issues:
#   1. PyTorch 2.x JIT graph ordering: The inlined_graph reorders downsample
#      block nodes (BatchNorm before Conv), breaking the positional module
#      assignment. Fixed by reordering graph keys to match forward-hook order.
#   2. NumPy 2.x deprecation: np.float was removed. Replaced with np.float64.
#
# Usage:
#   bash patches/apply_neuralteleportation_patches.sh [venv_path]
#   Default venv_path: ./env

set -euo pipefail

VENV_PATH="${1:-./env}"
SITE_PACKAGES="$VENV_PATH/lib/python3.11/site-packages/neuralteleportation"

if [ ! -d "$SITE_PACKAGES" ]; then
    echo "ERROR: neuralteleportation not found at $SITE_PACKAGES"
    echo "Install it first: pip install git+https://github.com/vitalab/neuralteleportation.git"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/neuralteleportation_pytorch2_compat.patch"

if [ ! -f "$PATCH_FILE" ]; then
    echo "ERROR: Patch file not found: $PATCH_FILE"
    exit 1
fi

echo "Applying neuralteleportation patches to: $SITE_PACKAGES"
cd "$SITE_PACKAGES/.."
patch -p0 --forward < "$PATCH_FILE" || {
    echo "Patch may have already been applied (or failed). Check output above."
}
echo "Done."
