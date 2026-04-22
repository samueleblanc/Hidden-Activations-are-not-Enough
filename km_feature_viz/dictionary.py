"""Step 04: PCA / NMF on stacked per-class KM rows.

For each (model, class) pair, stack the corresponding row across all
sample images, then run PCA and NMF to extract top-k components.
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from sklearn.decomposition import NMF, PCA

from km_feature_viz import paths
from km_feature_viz.compute_kms import load_km_slice
from km_feature_viz.manifest import TIER_A_CLASSES, read_manifest

logger = logging.getLogger(__name__)


def stack_class_rows(km_paths: List[Path], target_class_idx_in_slice: int) -> torch.Tensor:
    """Load each KM file, pull the row at target_class_idx_in_slice, and drop
    the trailing bias column so the row corresponds to pixel contributions only.

    Returns (N_images, C*H*W) — e.g., (N, 150528) for (3, 224, 224) inputs.
    """
    rows = []
    for p in km_paths:
        km, _ = load_km_slice(p)
        row = km[target_class_idx_in_slice].to(torch.float32)
        # Drop the bias column (last entry) so the row is C*H*W and reshape-able to the input.
        if row.shape[0] % 2 == 1:  # odd length → has bias column
            row = row[:-1]
        rows.append(row)
    return torch.stack(rows)


def pca_top_k(X: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """PCA via randomized SVD. Returns (components, explained_variance_ratio)."""
    pca = PCA(n_components=k, svd_solver="randomized", random_state=0)
    pca.fit(X.numpy())
    return (
        torch.from_numpy(pca.components_),
        torch.from_numpy(pca.explained_variance_ratio_),
    )


def nmf_top_k(X: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """NMF on |X|. Returns (components, reconstruction_err) — components >= 0."""
    X_pos = X.abs().numpy()
    model = NMF(n_components=k, init="nndsvd", random_state=0, max_iter=400)
    W = model.fit_transform(X_pos)
    return torch.from_numpy(model.components_), torch.tensor(model.reconstruction_err_)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    entries = read_manifest(args.manifest)
    by_model_class = {}
    for e in entries:
        by_model_class.setdefault((e.model, e.class_id), []).append(e)

    for (model_name, class_id), samples in by_model_class.items():
        km_files = [paths.km_path(model_name, class_id, s.image_id) for s in samples]
        km_files = [p for p in km_files if p.exists()]
        if len(km_files) < args.top_k:
            logger.warning(
                "Skipping %s/%s: only %d KMs (need >= top_k=%d)",
                model_name, class_id, len(km_files), args.top_k,
            )
            continue

        # The slice index: position of class_id in TIER_A_CLASSES.
        slice_idx = TIER_A_CLASSES.index(class_id)
        X = stack_class_rows(km_files, target_class_idx_in_slice=slice_idx)

        # PCA
        pca_comp, pca_var = pca_top_k(X, k=args.top_k)
        out_components = paths.dictionary_path(model_name, class_id, "components")
        out_var = paths.dictionary_path(model_name, class_id, "explained_variance")
        out_components.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"pca": pca_comp.to(torch.float16)}, out_components)
        torch.save({"pca": pca_var.to(torch.float32)}, out_var)

        # NMF
        nmf_comp, nmf_err = nmf_top_k(X, k=args.top_k)
        existing = torch.load(out_components, weights_only=False)
        existing["nmf"] = nmf_comp.to(torch.float16)
        torch.save(existing, out_components)
        existing_var = torch.load(out_var, weights_only=False)
        existing_var["nmf_err"] = nmf_err
        torch.save(existing_var, out_var)

        logger.info("done %s/%s (n=%d)", model_name, class_id, len(km_files))

    return 0


if __name__ == "__main__":
    sys.exit(main())
