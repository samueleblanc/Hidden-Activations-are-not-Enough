"""Figure rendering for km-feature-viz Tier A.

Each `cell_*` function loads the relevant cache and writes a PDF figure
to docs/km-feature-viz/figures/. Patch-blocked cells (G, H) are stubs
that print a message and skip.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from km_feature_viz import paths
from km_feature_viz.compute_kms import load_image, load_km_slice
from km_feature_viz.manifest import TIER_A_CLASSES, read_manifest

logger = logging.getLogger(__name__)


def _norm_for_display(t: torch.Tensor) -> np.ndarray:
    a = t.detach().cpu().to(torch.float32).numpy()
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a


def cell_a_attribution_vs_feature_maps(manifest_entries, model_name, class_id, image_id):
    from knowledgematrix.visualization import attribution_map

    km_path = paths.km_path(model_name, class_id, image_id)
    fm_path = paths.baseline_path("feature_maps", model_name, class_id, image_id)
    if not (km_path.exists() and fm_path.exists()):
        logger.warning("cell A skip — missing %s or %s", km_path, fm_path)
        return
    km, classes = load_km_slice(km_path)
    slice_idx = TIER_A_CLASSES.index(class_id)
    attr = attribution_map(
        km.to(torch.float32), output_class=slice_idx, input_shape=(3, 224, 224)
    )
    feature_maps = torch.load(fm_path, weights_only=False)  # (1, C, h, w)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(_norm_for_display(attr), cmap="seismic")
    axes[0].set_title("KM attribution_map")
    for i in range(4):
        axes[i + 1].imshow(_norm_for_display(feature_maps[0, i]), cmap="viridis")
        axes[i + 1].set_title(f"Feature map ch{i}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Cell A — {model_name}, class {class_id}, {image_id}")
    out = paths.figure_path(f"cell_a_{model_name}_{class_id}_{image_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def cell_b_attribution_vs_gradcam(manifest_entries, model_name, class_id, image_id):
    from knowledgematrix.visualization import attribution_map

    km_path = paths.km_path(model_name, class_id, image_id)
    gc_path = paths.baseline_path("gradcam", model_name, class_id, image_id)
    if not (km_path.exists() and gc_path.exists()):
        logger.warning("cell B skip — missing")
        return
    km, _ = load_km_slice(km_path)
    slice_idx = TIER_A_CLASSES.index(class_id)
    attr = attribution_map(km.to(torch.float32), output_class=slice_idx, input_shape=(3, 224, 224))
    gradcam = torch.load(gc_path, weights_only=False)[0, 0]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(_norm_for_display(attr), cmap="seismic")
    axes[0].set_title("KM attribution (exact)")
    axes[1].imshow(_norm_for_display(gradcam), cmap="seismic")
    axes[1].set_title("Grad-CAM (approximate)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Cell B — {model_name}, class {class_id}, {image_id}")
    out = paths.figure_path(f"cell_b_{model_name}_{class_id}_{image_id}")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def cell_c_attribution_vs_ig_smoothgrad(manifest_entries, model_name, class_id, image_id):
    from knowledgematrix.visualization import attribution_map

    km_path = paths.km_path(model_name, class_id, image_id)
    ig_path = paths.baseline_path("ig", model_name, class_id, image_id)
    sg_path = paths.baseline_path("smoothgrad", model_name, class_id, image_id)
    if not (km_path.exists() and ig_path.exists() and sg_path.exists()):
        logger.warning("cell C skip — missing")
        return
    km, _ = load_km_slice(km_path)
    slice_idx = TIER_A_CLASSES.index(class_id)
    attr = attribution_map(km.to(torch.float32), output_class=slice_idx, input_shape=(3, 224, 224))
    ig = torch.load(ig_path, weights_only=False)[0].sum(0)
    sg = torch.load(sg_path, weights_only=False)[0].sum(0)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_norm_for_display(attr), cmap="seismic"); axes[0].set_title("KM (exact)")
    axes[1].imshow(_norm_for_display(ig), cmap="seismic"); axes[1].set_title("IG")
    axes[2].imshow(_norm_for_display(sg), cmap="seismic"); axes[2].set_title("SmoothGrad")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"Cell C — {model_name}, class {class_id}, {image_id}")
    out = paths.figure_path(f"cell_c_{model_name}_{class_id}_{image_id}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def cell_d_dictionary_vs_max_activating(manifest_entries, model_name, class_id):
    """KM dictionary top-3 PCA components vs. top-3 max-activating images for class_id.

    Max-activating computed from cached KMs: image's "class-c neuron activation"
    equals the sum of the KM row for class c (since A.sum(1) == out).
    """
    dict_components_path = paths.dictionary_path(model_name, class_id, "components")
    if not dict_components_path.exists():
        logger.warning("cell D skip — missing %s", dict_components_path)
        return
    components = torch.load(dict_components_path, weights_only=False)
    pca = components["pca"][:3]  # top 3 components: (3, 150528)

    # Max-activating images for class_id: sort by KM row sum.
    sample_entries = [e for e in manifest_entries if e.model == model_name and e.class_id == class_id]
    slice_idx = TIER_A_CLASSES.index(class_id)
    scored = []
    for s in sample_entries:
        km_path = paths.km_path(s.model, s.class_id, s.image_id)
        if not km_path.exists():
            continue
        km, _ = load_km_slice(km_path)
        score = km[slice_idx].to(torch.float32).sum().item()
        scored.append((score, s))
    scored.sort(reverse=True, key=lambda t: t[0])
    top3 = [s for _, s in scored[:3]]
    if len(top3) < 3:
        logger.warning("cell D skip — fewer than 3 cached KMs for %s/%s", model_name, class_id)
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        comp = pca[i].reshape(3, 224, 224).sum(0)
        axes[0, i].imshow(_norm_for_display(comp), cmap="seismic")
        axes[0, i].set_title(f"KM-PCA comp {i}")
        img = load_image(top3[i].image_path)
        axes[1, i].imshow(img.permute(1, 2, 0).clamp(0, 1).cpu().numpy())
        axes[1, i].set_title(f"Max-activating {i} ({top3[i].image_id[:12]})")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Cell D — {model_name}, class {class_id}")
    out = paths.figure_path(f"cell_d_{model_name}_{class_id}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def _compute_penultimate(model_name, entries, device="cpu"):
    """Forward each entry through the torchvision model, capture the activation
    just before the final Linear classifier. Returns (N, D) tensor."""
    from km_feature_viz.compute_baselines import build_torchvision_model

    model = build_torchvision_model(model_name, device)
    if model_name == "alexnet":
        target = model.classifier[5]   # ReLU before final Linear
    elif model_name == "resnet18":
        target = model.avgpool         # output of avgpool before fc
    elif model_name == "vgg11":
        target = model.classifier[4]   # ReLU before final Linear
    else:
        raise ValueError(model_name)

    captured = []

    def hook(module, inp, out):
        captured.append(out.detach().flatten(1).cpu())

    handle = target.register_forward_hook(hook)
    try:
        for e in entries:
            x = load_image(e.image_path).unsqueeze(0).to(device)
            with torch.no_grad():
                model(x)
    finally:
        handle.remove()
    return torch.cat(captured, dim=0)  # (N, D)


def cell_e_dictionary_vs_penultimate_pca(manifest_entries, model_name, class_id):
    """KM-PCA top-3 components vs. penultimate-PCA top-3 components for class_id.

    Penultimate features are extracted on-the-fly via a forward hook; cheap on
    a 50-image subsample.
    """
    from sklearn.decomposition import PCA

    dict_components_path = paths.dictionary_path(model_name, class_id, "components")
    if not dict_components_path.exists():
        logger.warning("cell E skip — missing %s", dict_components_path)
        return
    components = torch.load(dict_components_path, weights_only=False)
    km_pca = components["pca"][:3]  # (3, 150528)

    sample_entries = [e for e in manifest_entries if e.model == model_name and e.class_id == class_id]
    if len(sample_entries) < 5:
        logger.warning("cell E skip — too few samples for %s/%s", model_name, class_id)
        return
    feats = _compute_penultimate(model_name, sample_entries).numpy()
    pen_pca = PCA(n_components=3, svd_solver="randomized", random_state=0).fit(feats).components_

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        km_comp = km_pca[i].reshape(3, 224, 224).sum(0)
        axes[0, i].imshow(_norm_for_display(km_comp), cmap="seismic")
        axes[0, i].set_title(f"KM-PCA comp {i}")
        axes[1, i].bar(range(pen_pca.shape[1]), pen_pca[i])
        axes[1, i].set_title(f"Penultimate-PCA comp {i}\n(D={pen_pca.shape[1]})")
        axes[1, i].set_xlabel("penultimate dim")
    for ax in axes[0]:
        ax.axis("off")
    fig.suptitle(f"Cell E — {model_name}, class {class_id}")
    out = paths.figure_path(f"cell_e_{model_name}_{class_id}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def cell_f_dictionary_vs_deepdream(manifest_entries, model_name, class_id):
    """KM dictionary components vs. DeepDream renderings for this model."""
    dict_components_path = paths.dictionary_path(model_name, class_id, "components")
    if not dict_components_path.exists():
        logger.warning("cell F skip — missing %s", dict_components_path)
        return
    components = torch.load(dict_components_path, weights_only=False)
    pca = components["pca"][:3]

    # Pull 3 DeepDream renderings for this model (any layer/neuron)
    dd_dir = Path(paths.RESULTS_ROOT / "deepdream" / model_name)
    if not dd_dir.exists():
        logger.warning("cell F skip — no deepdream cache")
        return
    dd_files = sorted(dd_dir.rglob("*.pt"))[:3]
    if not dd_files:
        logger.warning("cell F skip — no deepdream files")
        return

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i in range(3):
        comp = pca[i].reshape(3, 224, 224).sum(0)
        axes[0, i].imshow(_norm_for_display(comp), cmap="seismic")
        axes[0, i].set_title(f"KM-PCA comp {i}")
        dd_img = torch.load(dd_files[i], weights_only=False)
        axes[1, i].imshow(dd_img.permute(1, 2, 0).clamp(0, 1).to(torch.float32).cpu().numpy())
        axes[1, i].set_title(f"DeepDream {dd_files[i].stem}")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"Cell F — {model_name}, class {class_id}")
    out = paths.figure_path(f"cell_f_{model_name}_{class_id}")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out)


def cell_g_lp_vs_pgd(*args, **kwargs):
    logger.info("cell G PATCH-BLOCKED — implement in Task 19 after extract_weff lands")


def cell_h_jacobian_vs_smoothgrad(*args, **kwargs):
    logger.info("cell H PATCH-BLOCKED — implement in Task 19 after extract_weff lands")


def cell_i_summary_table(manifest_entries):
    """Print a summary table of method timings, exactness, parameters."""
    rows = [
        ("Method", "Exact?", "Tunable params", "Notes"),
        ("KM attribution_map", "yes", "0", "exact: A.sum(1) == out"),
        ("KM dictionary (PCA)", "yes (lossy below k)", "k", "isomorphism-invariant"),
        ("KM top-k contributors", "yes", "k", "static rank by |A|"),
        ("KM LP counterfactual", "exact in-region", "margin", "PATCH-BLOCKED"),
        ("KM Jacobian sensitivity", "exact in-region", "0", "PATCH-BLOCKED"),
        ("Feature maps", "n/a (raw activation)", "0", "qualitative only"),
        ("Grad-CAM", "no (gradient approximation)", "target_layer", "iconic baseline"),
        ("Integrated Gradients", "no (path integral)", "n_steps", "diffuse"),
        ("SmoothGrad", "no (gaussian average)", "n_samples, stdev", "stochastic"),
        ("Max-activating images", "n/a (retrieval)", "k", "dataset-dependent"),
        ("DeepDream", "no (gradient ascent)", "lr, steps, jitter", "iconic but expensive"),
        ("Penultimate PCA", "yes (linear projection)", "k", "permutation-NOT invariant"),
        ("PGD", "no (constrained SGD)", "eps, steps, alpha", "adversarial, not interpretive"),
    ]
    out = paths.figure_path("cell_i_summary_table").with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for row in rows:
            f.write("| " + " | ".join(row) + " |\n")
            if row is rows[0]:
                f.write("|" + "|".join("---" for _ in row) + "|\n")
    logger.info("wrote %s", out)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    entries = read_manifest(args.manifest)

    # Render one panel per model/class for the static cells; one (model, class, image)
    # exemplar for the per-image cells.
    exemplar = entries[0]
    cell_a_attribution_vs_feature_maps(entries, exemplar.model, exemplar.class_id, exemplar.image_id)
    cell_b_attribution_vs_gradcam(entries, exemplar.model, exemplar.class_id, exemplar.image_id)
    cell_c_attribution_vs_ig_smoothgrad(entries, exemplar.model, exemplar.class_id, exemplar.image_id)
    cell_d_dictionary_vs_max_activating(entries, exemplar.model, exemplar.class_id)
    cell_e_dictionary_vs_penultimate_pca(entries, exemplar.model, exemplar.class_id)
    cell_f_dictionary_vs_deepdream(entries, exemplar.model, exemplar.class_id)
    cell_g_lp_vs_pgd()
    cell_h_jacobian_vs_smoothgrad()
    cell_i_summary_table(entries)
    return 0


if __name__ == "__main__":
    sys.exit(main())
