#!/bin/bash
# bin/tar_artifacts.sh -- Step D: produce final tarball, gated on sanity_report.json.

set -euo pipefail

OUT_DIR="${OUT_DIR:-results/phase1/artifact}"
SANITY_FILE="${SANITY_FILE:-results/phase1/aggregated/sanity_report.json}"
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"

mkdir -p "$OUT_DIR"

if [ ! -f "$SANITY_FILE" ]; then
    echo "ERROR: $SANITY_FILE not found; cannot determine pipeline health" >&2
    exit 1
fi

ALL_PASS=$(python3 -c "import json; print(json.load(open('$SANITY_FILE'))['all_pass'])")

if [ "$ALL_PASS" = "True" ]; then
    OUT_PATH="$OUT_DIR/phase1-results-${GIT_SHA}-${TIMESTAMP}.tar.gz"
    echo "Sanity all-pass — building $OUT_PATH"

    # Wire computed numbers into paper TODOs (added in Phase 7). Only when the
    # paper sections tree is present — it is absent on a results-only cluster
    # checkout (the paper sources live in the writing copy, not on the cluster).
    if [ -f wire_paper_results.py ] && [ -d docs/Final-twist/paper/sections ]; then
        python wire_paper_results.py \
            --results_dir results/phase1/aggregated \
            --paper_dir docs/Final-twist/paper/sections || \
            echo "WARN: wire_paper_results.py failed; tarball will use unsubstituted TODOs"
    fi

    # Compile paper PDF (best effort; only if the paper sources are present).
    if [ -f docs/Final-twist/paper/main.tex ]; then
        ( cd docs/Final-twist/paper && latexmk -pdf -interaction=nonstopmode main.tex ) || \
            echo "WARN: latexmk failed; tarball will not include main.pdf"
    fi

    # Bundle only the inputs that exist. results/phase1/aggregated/ is the one
    # REQUIRED artifact (the gate output); the paper tree, calibration, and the
    # pipeline_state.json log are optional and may be absent on a cluster
    # checkout — tar runs under `set -e`, so a missing literal input would abort
    # the whole step (the bug that failed job 14474456: sections/ +
    # pipeline_state.json absent on Rorqual).
    if [ ! -d results/phase1/aggregated ]; then
        echo "ERROR: results/phase1/aggregated/ missing — nothing to bundle" >&2
        exit 1
    fi
    INPUTS=("results/phase1/aggregated/")
    [ -d docs/Final-twist/paper/sections ]   && INPUTS+=("docs/Final-twist/paper/sections/")
    [ -d docs/Final-twist/paper/tables ]     && INPUTS+=("docs/Final-twist/paper/tables/")
    [ -f docs/Final-twist/paper/main.pdf ]   && INPUTS+=("docs/Final-twist/paper/main.pdf")
    [ -d experiments/calibration ]           && INPUTS+=("experiments/calibration/")
    [ -f pipeline_state.json ]               && INPUTS+=("pipeline_state.json")
    echo "Bundling: ${INPUTS[*]}"
    tar czf "$OUT_PATH" "${INPUTS[@]}"

    echo "SUCCESS: $OUT_PATH"
    ls -lh "$OUT_PATH"
    exit 0
else
    OUT_PATH="$OUT_DIR/phase1-FAILED-${GIT_SHA}-${TIMESTAMP}.tar.gz"
    echo "Sanity FAILED — building diagnostic tarball $OUT_PATH" >&2
    tar czf "$OUT_PATH" \
        "$SANITY_FILE" \
        results/phase1/aggregated/controls.json \
        $( [ -f overall_errors.json ] && echo "overall_errors.json" ) \
        slurm_out/ slurm_err/
    echo "FAILED: see $OUT_PATH"
    exit 1
fi
