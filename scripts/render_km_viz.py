"""Render the contents of km-feature-viz cluster output as viewable PNGs.

Input dir : results/km-feature-viz-cluster/
Output dir: results/km-feature-viz-cluster/_viewable/
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
from PIL import Image


ROOT = Path("results/km-feature-viz-cluster")
OUT = ROOT / "_viewable"

# Architectures rendered. Pillar 3 launches with three CNN-family
# representatives (residual / dense / inception) — all 224×224 input, all
# 1000-class ImageNet logits, so no per-arch shape branching is needed in
# the renderer. Order mirrors `km_feature_viz/manifest.py:TIER_A_MODELS`.
ARCHS = ["resnet152", "densenet121", "googlenet"]

# Display names for the rendered architectures. The "GoogLeNet (InceptionV1)"
# spelling is deliberate: the knowledgematrix wrapper now exposes both
# Inception_v3 (PR #11, 299×299, NOT used for Pillar 3) and a new GoogLeNet
# wrapper (224×224, used here). Calling it just "Inception" would be
# ambiguous to a reviewer.
ARCH_DISPLAY_NAMES = {
    "resnet152":   "ResNet-152",
    "densenet121": "DenseNet-121",
    "googlenet":   "GoogLeNet (InceptionV1)",
}

# ImageNet class indices in scope for the Pillar 3 figure (golden retriever,
# tiger cat, zebra). Matches `km_feature_viz/manifest.py:TIER_A_CLASSES`.
CLASS_IDS = [207, 282, 340]

# Pretty class names for the ImageNet classes in scope.
CLASS_NAMES = {
    207: "golden retriever",
    282: "tiger cat",
    340: "zebra",
}


def load_neuron_selection(arch: str) -> dict:
    """Read state/03a_neuron_selection_<arch>.json; return parsed dict.

    Returns empty dict if the file is missing (e.g., before the cluster
    pipeline has run for that arch) — caller should fall back to a
    bare 'channel <idx>' caption in that case.

    Expected schema (written by the neuron-selection sub-pipeline):
        {
          "method": "gradcam_class_conditional" | "catalogued_distill",
          "channels": {
              "<layer_name>": [
                  {"channel": int, "rank": int, "mean_alpha": float}        # gradcam
                  OR
                  {"channel": int, "rank": int, "label": str, "citation": str}  # catalogued
              ],
              ...
          }
        }
    """
    p = ROOT / "state" / f"03a_neuron_selection_{arch}.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


def deepdream_caption(selection: dict, layer: str, channel: int) -> tuple[str, str | None]:
    """Format a per-tile DeepDream caption based on the neuron-selection metadata.

    Returns (caption, footnote) where `footnote` is an optional citation string
    (only set for catalogued_distill entries) — the renderer attaches one
    citation footnote per panel.
    """
    if not selection:
        return f"channel {channel}", None
    method = selection.get("method", "")
    entries = selection.get("channels", {}).get(layer, [])
    meta = next((e for e in entries if int(e.get("channel", -1)) == channel), None)
    if meta is None:
        return f"{layer} ch{channel}", None
    if method == "gradcam_class_conditional":
        alpha = float(meta.get("mean_alpha", 0.0))
        return f"{layer} ch{channel} (mean α={alpha:.3f})", None
    if method == "catalogued_distill":
        label = str(meta.get("label", "?"))
        citation = meta.get("citation")
        return f"{layer} ch{channel} ({label})", citation
    return f"{layer} ch{channel}", None


def load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False)


def to_numpy(x: torch.Tensor) -> np.ndarray:
    if x.dtype in (torch.float16, torch.bfloat16):
        x = x.float()
    return x.detach().cpu().numpy()


def safe_clip(arr: np.ndarray, lo=1.0, hi=99.0) -> np.ndarray:
    """Replace inf/nan with finite extremes, then percentile-clip for viz.

    Note: under the fp32 pipeline (post-2026-04-25), inf entries should not
    appear — they were an fp16-saturation artifact on the ResNet18 residual
    path that the rewrite eliminated. The inf/nan branch below is retained
    as defensive handling for legacy fp16 outputs that may still be re-loaded
    from `results/km-feature-viz-cluster/`. The 1/99 percentile clip itself is
    a *visualization-dynamic-range* choice, not numerical safety: it
    suppresses the top/bottom 1% of magnitudes so the heatmap colormap
    saturates on the bulk of the distribution rather than on outlier pixels.
    """
    a = arr.copy()
    finite_mask = np.isfinite(a)
    if not finite_mask.all():
        finite_vals = a[finite_mask]
        if finite_vals.size == 0:
            return np.zeros_like(a)
        a[~finite_mask & (a > 0)] = finite_vals.max()
        a[~finite_mask & (a < 0)] = finite_vals.min()
        a[np.isnan(a)] = 0.0
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi <= p_lo:
        p_hi = p_lo + 1e-6
    return np.clip(a, p_lo, p_hi)


def denorm_image_uint8(t: torch.Tensor) -> np.ndarray:
    """images/* are stored as uint8 (3, H, W). Return HWC uint8."""
    arr = to_numpy(t)
    if arr.dtype != np.uint8:
        # Should already be uint8, but if normalized floats slip in, rescale
        a = arr.astype(np.float32)
        a = (a - a.min()) / max(a.max() - a.min(), 1e-9) * 255.0
        arr = a.astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def render_image(t: torch.Tensor, out_path: Path):
    arr = denorm_image_uint8(t)
    Image.fromarray(arr).save(out_path)


def saliency_to_heat(arr: np.ndarray) -> tuple[np.ndarray, tuple[float, float]]:
    """Reduce a (C,H,W) or (1,C,H,W) saliency to a 2D heatmap normalized to [0,1].

    Returns (heat01, (vmin, vmax)) where (vmin, vmax) are the 1/99-percentile-clipped
    values of the channel-summed |saliency| that map to 0/1 in the heatmap — these are
    used as colorbar tick labels so the dynamic range remains visible.
    """
    a = np.asarray(arr)
    if a.ndim == 4:
        a = a[0]
    if a.ndim == 3:
        a = np.abs(a).sum(axis=0)
    a = safe_clip(a, 1, 99)
    vmin, vmax = float(a.min()), float(a.max())
    h = a - vmin
    rng = vmax - vmin
    if rng > 0:
        h = h / rng
    return h, (vmin, vmax)


def upsample_to(arr: np.ndarray, hw=(224, 224)) -> np.ndarray:
    """Bilinear upsample a 2D array."""
    img = Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))
    img = img.resize((hw[1], hw[0]), Image.BILINEAR)
    return np.asarray(img).astype(np.float32) / 255.0


def overlay(image_uint8_hwc: np.ndarray, heat01: np.ndarray, alpha=0.45, cmap="jet") -> np.ndarray:
    """Return an RGB overlay (uint8 HWC) of heat on top of the input image."""
    cm = plt.get_cmap(cmap)
    rgba = cm(heat01)
    heat_rgb = (rgba[..., :3] * 255).astype(np.uint8)
    out = ((1 - alpha) * image_uint8_hwc + alpha * heat_rgb).astype(np.uint8)
    return out


def render_deepdream(t: torch.Tensor, out_path: Path):
    arr = to_numpy(t)
    if arr.ndim == 3 and arr.shape[0] == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = safe_clip(arr, 1, 99)
    a = arr - arr.min()
    rng = a.max() - a.min()
    if rng > 0:
        a = a / rng
    Image.fromarray((a * 255).astype(np.uint8)).save(out_path)


def render_feature_map_grid(t: torch.Tensor, out_path: Path, max_channels: int = 64):
    """Render a (1,C,H,W) feature map as a grid of channel heatmaps (first N channels)."""
    arr = to_numpy(t)
    if arr.ndim == 4:
        arr = arr[0]
    C, H, W = arr.shape
    n = min(C, max_channels)
    cols = 8
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.atleast_2d(axes)
    for i in range(rows * cols):
        ax = axes[i // cols, i % cols]
        ax.axis("off")
        if i < n:
            ch = safe_clip(arr[i], 1, 99)
            ax.imshow(ch, cmap="viridis")
    fig.suptitle(f"feature map: showing {n}/{C} channels (first N)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def km_class_row_to_heat(km: torch.Tensor, classes: list[int], target_class: int):
    """Reshape the KM row corresponding to `target_class` into a 2D heatmap.

    KM is (1000, 150529) for ImageNet — 1000 output classes × (3·224·224 + 1 bias).
    All three Pillar-3 archs (resnet152, densenet121, googlenet) share this shape.
    Channel sum + percentile clip → 2D heatmap in [0,1].

    Returns (heat01, (vmin, vmax)) — same convention as saliency_to_heat — or None
    if `target_class` is not in `classes`.
    """
    if target_class not in classes:
        return None
    idx = classes.index(target_class)
    row = to_numpy(km[idx])  # (150529,)
    pix = row[:150528].reshape(3, 224, 224)
    heat = np.abs(pix).sum(axis=0)
    heat = safe_clip(heat, 1, 99)
    vmin, vmax = float(heat.min()), float(heat.max())
    h = heat - vmin
    rng = vmax - vmin
    if rng > 0:
        h = h / rng
    return h, (vmin, vmax)


def select_panel_images(per_class: int = 5) -> dict[int, list[str]]:
    """Pick the first `per_class` image_ids alphabetically per class for full rendering."""
    chosen = {}
    for cid in CLASS_IDS:
        files = sorted((ROOT / "images" / str(cid)).glob("*.pt"))
        chosen[cid] = [p.stem for p in files[:per_class]]
    return chosen


def fig_to_png(fig, path: Path, dpi=120):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def render_comparison_panel(arch: str, class_id: int, image_id: str, out_path: Path):
    img_path = ROOT / "images" / str(class_id) / f"{image_id}.pt"
    if not img_path.exists():
        return False
    t_img = load(img_path)
    img_hwc = denorm_image_uint8(t_img)

    # Knowledge matrix for true class
    km_dict = load(ROOT / "kms" / arch / str(class_id) / f"{image_id}.pt")
    km_heat = km_class_row_to_heat(km_dict["km"], km_dict["classes"], class_id)

    # Baselines
    def maybe(p):
        return load(p) if p.exists() else None

    gc = maybe(ROOT / "baselines" / "gradcam" / arch / str(class_id) / f"{image_id}.pt")
    ig = maybe(ROOT / "baselines" / "ig" / arch / str(class_id) / f"{image_id}.pt")
    sg = maybe(ROOT / "baselines" / "smoothgrad" / arch / str(class_id) / f"{image_id}.pt")
    pgd = maybe(ROOT / "baselines" / "pgd" / arch / str(class_id) / f"{image_id}.pt")

    # Each panel is (title, rgb_image_to_show, vrange) where vrange is None for
    # raw RGB columns (input, PGD δ) and (vmin, vmax) for jet-overlay heatmaps —
    # signaling the renderer to attach a jet colorbar showing the actual
    # percentile-clipped numeric range that maps to the [low, high] of the colormap.
    panels = []
    panels.append(("input", img_hwc, None))

    if gc is not None:
        gc_arr = to_numpy(gc)[0, 0]  # (h, w) — Grad-CAM is signed; clamp to ≥0 then normalize
        gc_pos = np.maximum(gc_arr, 0)
        gc_max = float(gc_pos.max())
        if gc_max > 0:
            gc_heat = upsample_to(gc_pos / gc_max, (224, 224))
            panels.append(("Grad-CAM", overlay(img_hwc, gc_heat), (0.0, gc_max)))
        else:
            gc_heat = upsample_to(gc_pos, (224, 224))
            panels.append(("Grad-CAM (all ≤0)", overlay(img_hwc, gc_heat), (0.0, 0.0)))
    else:
        panels.append(("Grad-CAM (missing)", None, None))

    if ig is not None:
        ig_heat, ig_range = saliency_to_heat(to_numpy(ig))
        panels.append(("Integrated Gradients", overlay(img_hwc, ig_heat), ig_range))
    else:
        panels.append(("IG (missing)", None, None))

    if sg is not None:
        sg_heat, sg_range = saliency_to_heat(to_numpy(sg))
        panels.append(("SmoothGrad", overlay(img_hwc, sg_heat), sg_range))
    else:
        panels.append(("SmoothGrad (missing)", None, None))

    if pgd is not None:
        delta = to_numpy(pgd["delta"])[0]  # (3,224,224)
        target = pgd.get("target")
        d = np.transpose(delta, (1, 2, 0))
        d_vis = (d - d.min()) / max(d.max() - d.min(), 1e-9)
        panels.append((f"PGD δ → cls {target}", (d_vis * 255).astype(np.uint8), None))
    else:
        panels.append(("PGD (missing)", None, None))

    if km_heat is not None:
        km_heat_arr, km_range = km_heat
        panels.append(("Knowledge Matrix (true class row)",
                       overlay(img_hwc, km_heat_arr), km_range))
    else:
        panels.append(("KM (no row)", None, None))

    n = len(panels)
    # Wider per-cell to leave room for the colorbar; taller so the colorbar
    # tick labels remain readable at gallery thumbnail scale.
    fig, axes = plt.subplots(1, n, figsize=(n * 2.8, 3.4))
    cls_name = CLASS_NAMES.get(class_id, "?")
    arch_name = ARCH_DISPLAY_NAMES.get(arch, arch)
    fig.suptitle(f"{arch_name}  |  class {class_id} ({cls_name})  |  {image_id}", fontsize=10)
    norm_for_colorbar = matplotlib.colors.Normalize  # local alias
    for ax, (title, content, vrange) in zip(axes, panels):
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        if content is not None:
            ax.imshow(content)
        if vrange is not None:
            vmin, vmax = vrange
            sm = plt.cm.ScalarMappable(cmap="jet", norm=norm_for_colorbar(vmin=vmin, vmax=vmax))
            sm.set_array([])
            cax = ax.inset_axes([1.02, 0.0, 0.05, 1.0])
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=6, length=2, pad=1)
            cbar.outline.set_linewidth(0.4)
            # Use scientific notation for very large/small magnitudes
            mag = max(abs(vmin), abs(vmax))
            if mag >= 1e3 or (mag > 0 and mag < 1e-2):
                cbar.formatter = matplotlib.ticker.ScalarFormatter(useMathText=True)
                cbar.formatter.set_powerlimits((-2, 3))
                cbar.update_ticks()
                cbar.ax.yaxis.get_offset_text().set_fontsize(6)
    fig_to_png(fig, out_path, dpi=130)
    return True


def render_inputs(out_dir: Path):
    n = 0
    for cid in CLASS_IDS:
        cdir = out_dir / str(cid)
        cdir.mkdir(parents=True, exist_ok=True)
        for pt in sorted((ROOT / "images" / str(cid)).glob("*.pt")):
            t = load(pt)
            render_image(t, cdir / f"{pt.stem}.png")
            n += 1
    return n


_NEURON_RE = re.compile(r"neuron_0*(\d+)")


def _channel_from_filename(stem: str) -> int | None:
    """Extract the integer channel index from a 'neuron_####' filename stem."""
    m = _NEURON_RE.match(stem)
    return int(m.group(1)) if m else None


def render_deepdreams(out_dir: Path):
    """Render DeepDream tiles for every (arch, layer, channel).

    Returns a per-arch list of dicts:
        {"rel": "<layer>/<file>.png",
         "layer": <layer>, "channel": <int|None>,
         "caption": <str>, "citation": <str|None>}
    so the HTML index can attach methodology-aware captions and citation
    footnotes from `state/03a_neuron_selection_<arch>.json`.
    """
    rendered = {}
    for arch in ARCHS:
        selection = load_neuron_selection(arch)
        layers = sorted((ROOT / "deepdream" / arch).glob("*"))
        rendered[arch] = []
        for ldir in layers:
            if not ldir.is_dir():
                continue
            for npt in sorted(ldir.glob("neuron_*.pt")):
                t = load(npt)
                out_path = out_dir / arch / ldir.name / f"{npt.stem}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                render_deepdream(t, out_path)
                ch = _channel_from_filename(npt.stem)
                if ch is not None:
                    caption, citation = deepdream_caption(selection, ldir.name, ch)
                else:
                    caption, citation = npt.stem, None
                rendered[arch].append({
                    "rel": f"{ldir.name}/{npt.stem}.png",
                    "layer": ldir.name,
                    "channel": ch,
                    "caption": caption,
                    "citation": citation,
                })
    return rendered


def render_panels(out_dir: Path, per_class: int):
    chosen = select_panel_images(per_class)
    panels = []
    for arch in ARCHS:
        for cid, ids in chosen.items():
            for image_id in ids:
                out = out_dir / arch / str(cid) / f"{image_id}.png"
                ok = render_comparison_panel(arch, cid, image_id, out)
                if ok:
                    panels.append((arch, cid, image_id, str(out.relative_to(OUT))))
    return panels


def write_html_index(panels, deepdreams, n_inputs, include_deepdream: bool = False):
    """Write a simple browsable index.html under OUT.

    If `include_deepdream` is False, the DeepDream section is rendered as an
    *opt-in supplementary plate* (collapsed `<details>`) rather than the
    default top-level section — per the round-3 Interp/KM debate decision
    that DeepDream should not be in the headline figure.
    """
    by_arch = {a: {} for a in ARCHS}
    for arch, cid, image_id, rel in panels:
        by_arch[arch].setdefault(cid, []).append((image_id, rel))

    html = ["<!doctype html><html><head><meta charset='utf-8'>",
            "<title>km-feature-viz gallery</title>",
            "<style>",
            "body{font-family:-apple-system,sans-serif;background:#111;color:#eee;padding:20px;max-width:1400px;margin:auto}",
            "h1{margin-top:0}",
            "h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:32px}",
            "h3{color:#9cf;margin-top:24px}",
            ".panel{margin:8px 0;border:1px solid #333;background:#1a1a1a}",
            ".panel img{display:block;width:100%}",
            ".caption{padding:6px 10px;font-size:12px;color:#bbb}",
            ".dd-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px}",
            ".dd-cell{border:1px solid #333;background:#1a1a1a;padding:4px;text-align:center;font-size:11px;color:#aaa}",
            ".dd-cell img{width:100%;display:block}",
            "a{color:#9cf}",
            "nav a{margin-right:14px}",
            ".legend{background:#181818;border:1px solid #333;border-radius:6px;padding:14px 18px;margin:14px 0 26px;font-size:13.5px;line-height:1.5}",
            ".legend dl{margin:0;display:grid;grid-template-columns:max-content 1fr;column-gap:14px;row-gap:6px}",
            ".legend dt{color:#ffd06b;font-weight:600;white-space:nowrap}",
            ".legend dd{margin:0;color:#ccc}",
            ".legend .km dt{color:#9cf}",
            ".legend .note{margin-top:10px;color:#aaa;font-size:12.5px}",
            "</style></head><body>"]
    html.append("<h1>km-feature-viz — cluster run</h1>")
    arch_names_str = ", ".join(ARCH_DISPLAY_NAMES.get(a, a) for a in ARCHS)
    n_dd_total = sum(len(v) for v in deepdreams.values())
    dd_blurb = (f"{n_dd_total} deep-dream visualizations (supplementary) · "
                if include_deepdream
                else f"{n_dd_total} deep-dream visualizations (excluded from headline; supplementary only) · ")
    html.append("<p>"
                f"<b>Pillar 3</b> — exact image-space attribution: knowledge matrices as "
                f"canonical saliency. "
                f"{n_inputs} input images across {len(CLASS_IDS)} classes "
                f"({', '.join(CLASS_NAMES.get(c, str(c)) for c in CLASS_IDS)}) · "
                f"{len(panels)} comparison panels · "
                f"{dd_blurb}"
                f"architectures: {arch_names_str}."
                "</p>")
    html.append("<p>See <a href='README.md'>README.md</a> for what each visualization means.</p>")
    html.append("<nav><a href='#panels'>Comparison panels</a> "
                "<a href='#deepdream'>DeepDream</a> "
                "<a href='#inputs'>Input images</a></nav>")

    # DeepDream methodology blurb (always shown — explains the two-method
    # neuron-selection split even when the section itself is collapsed).
    html.append(
        "<div class='legend'>"
        "<div><b>DeepDream neuron-selection methodology</b></div>"
        "<dl>"
        "<dt>ResNet-152 / DenseNet-121</dt>"
        "<dd>Channels are picked <i>class-conditionally</i>: for each (target class, "
        "candidate layer), we compute the channel-wise mean of the Grad-CAM weight "
        "α<sub>k</sub> = GAP(∂y<sub>c</sub>/∂A<sup>k</sup>) over a held-out subset of "
        "the class's images, then take the top-ranked channels. Each tile's caption "
        "reports the mean α value for the selected channel. "
        "Cite: Selvaraju et&nbsp;al. 2017 (Grad-CAM, ch-wise variant).</dd>"
        "<dt>GoogLeNet (InceptionV1)</dt>"
        "<dd>Channels are <i>catalogued</i> from the Distill <i>Circuits Thread</i> "
        "and OpenAI Microscope (curve detectors, high-low frequency detectors, "
        "dog-head detectors, etc.) — picking documented, named features rather than "
        "Grad-CAM-ranked ones. Each tile's caption reports the published feature "
        "label and the citation appears in the panel footer. "
        "Cite: Olah et&nbsp;al. 2017 (Feature Visualization), "
        "Cammarata et&nbsp;al. 2020 (Thread: Circuits).</dd>"
        "</dl>"
        "<div class='note'>The two methods are <i>not</i> directly comparable as "
        "selection strategies — Grad-CAM ranking gives class-conditional saliency, "
        "the Distill catalogue gives canonical feature identity. Both are used here "
        "as inputs to the same DeepDream optimizer (noise + jitter), which is "
        "deliberately under-regularized; do not compare these tiles directly to the "
        "Distill-2017 polished visualizations.</div>"
        "</div>")

    html.append("<h2 id='panels'>Comparison panels</h2>")
    html.append(
        "<div class='legend'>"
        "<div><b>What each column shows</b> (left → right):</div>"
        "<dl>"
        "<dt>1. input</dt>"
        "<dd>The 224×224 RGB ImageNet validation image fed into all three networks.</dd>"
        "<dt>2. Grad-CAM</dt>"
        "<dd>Saliency at the last convolutional layer for the true class. The signed map is "
        "ReLU&#8217;d (negative contributions dropped), bilinearly upsampled to 224×224, and "
        "alpha-overlaid on the input. Coarse — resolution is the conv map (e.g. 13×13 for AlexNet).</dd>"
        "<dt>3. Integrated Gradients</dt>"
        "<dd>Pixel-level attribution from integrating <code>∂logit/∂x</code> along a straight path "
        "from a black baseline to the input. Channel-wise <code>|·|</code> summed → 2D heatmap, "
        "1/99-percentile clipped, overlaid.</dd>"
        "<dt>4. SmoothGrad</dt>"
        "<dd>Pixel-level attribution averaged over noisy copies of the input. Same channel-summed "
        "<code>|·|</code> + percentile-clip + overlay as IG. Less noisy than vanilla gradient but "
        "still a gradient-based method.</dd>"
        "<dt>5. PGD δ → cls&nbsp;<i>k</i></dt>"
        "<dd>The adversarial <i>perturbation</i> (not <code>x+δ</code>) computed by projected "
        "gradient descent that flips the prediction to the labeled target class <i>k</i>. "
        "Min-max-rescaled per-image so the structured attack noise is visible.</dd>"
        "<dt class='km'>6. Knowledge Matrix (true-class row)</dt>"
        "<dd class='km'>The row of the per-input KM corresponding to the image&#8217;s <b>true</b> "
        "class, reshaped from the flat <code>(150528,)</code> back to <code>(3, 224, 224)</code>, "
        "channel-<code>|·|</code>-summed, 1/99-percentile clipped, and overlaid. Each pixel value is "
        "the actual coefficient on that input pixel in the network&#8217;s input-dependent linear "
        "factorization <code>f(x) = W<sub>eff</sub>(x)·x + b(x)</code> for the true-class logit. "
        "Unlike the gradient-based columns, it&#8217;s the literal linear weight in pixel space — "
        "and it&#8217;s invariant under hidden-layer neuron permutations, which Grad-CAM and the "
        "saliency methods are not.</dd>"
        "</dl>"
        "<div class='note'>The jet colorbar to the right of each heatmap reports the actual numeric "
        "range (1/99-percentile clipped) that maps to the bottom and top of the colormap. KM "
        "magnitudes are stored at fp32, so the residual path no longer saturates and KM rows "
        "should be free of <code>inf</code>/<code>NaN</code> entries — the percentile clip is for "
        "visualization dynamic range only.</div>"
        "</div>")
    for arch in ARCHS:
        arch_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        html.append(f"<h3>{arch_name}</h3>")
        for cid in CLASS_IDS:
            entries = by_arch[arch].get(cid, [])
            if not entries:
                continue
            cname = CLASS_NAMES.get(cid, "?")
            html.append(f"<details><summary><b>class {cid}</b> — {cname} ({len(entries)} panels)</summary>")
            for image_id, rel in entries:
                html.append("<div class='panel'>")
                html.append(f"<img loading='lazy' src='{rel}'>")
                html.append(f"<div class='caption'>{arch_name} · class {cid} ({cname}) · {image_id}</div>")
                html.append("</div>")
            html.append("</details>")

    # DeepDream is supplementary by default. With --include-deepdream the
    # section is rendered as a top-level section; otherwise it lives inside
    # a collapsed <details> block and is explicitly labeled "supplementary".
    if include_deepdream:
        html.append("<h2 id='deepdream'>DeepDream — synthesized inputs that maximize a single neuron <small>(supplementary)</small></h2>")
        dd_open_tag = "<div>"
        dd_close_tag = "</div>"
    else:
        html.append("<h2 id='deepdream'>DeepDream <small>(supplementary — excluded from headline figure)</small></h2>")
        dd_open_tag = "<details><summary>Show DeepDream tiles (supplementary plate, not in main-text figure)</summary>"
        dd_close_tag = "</details>"
    html.append(dd_open_tag)
    html.append(
        "<p>Per (architecture, layer) we visualize the selected neurons "
        f"(architectures: {arch_names_str}). Each image is produced by gradient ascent on noise-"
        "initialized input pixels with respect to a single neuron's mean activation, "
        "regularized only by random pixel-space jitter — no decorrelated FFT parameterization "
        "or transformation robustness, so these are best read as neuron <i>prototypes</i> under "
        "a minimal regularizer rather than the polished Distill-2017 visualizations. "
        "Per-arch neuron-selection methodology is described in the legend at the top.</p>")
    for arch in ARCHS:
        arch_name = ARCH_DISPLAY_NAMES.get(arch, arch)
        html.append(f"<h3>{arch_name}</h3>")
        # group by layer; entries are dicts now (see render_deepdreams).
        by_layer: dict[str, list[dict]] = {}
        cited = []  # collect unique citations for the per-arch panel footer
        for entry in deepdreams[arch]:
            by_layer.setdefault(entry["layer"], []).append(entry)
            if entry.get("citation") and entry["citation"] not in cited:
                cited.append(entry["citation"])
        for layer in sorted(by_layer.keys()):
            html.append(f"<h4>layer <code>{layer}</code></h4>")
            html.append("<div class='dd-grid'>")
            for entry in by_layer[layer]:
                full = f"deepdream/{arch}/{entry['rel']}"
                cap = entry["caption"]
                html.append(f"<div class='dd-cell'><img loading='lazy' src='{full}'><div>{cap}</div></div>")
            html.append("</div>")
        if cited:
            html.append("<div class='caption' style='margin-top:8px'>")
            for c in cited:
                html.append(f"<div>Citation: {c}</div>")
            html.append("</div>")
    html.append(dd_close_tag)

    html.append("<h2 id='inputs'>Input images</h2>")
    html.append(
        f"<p>All {n_inputs} input images, organized by class "
        f"({len(CLASS_IDS)} classes — the curated Pillar-3 exemplar set "
        f"chosen for inter-class contrast and intra-class consistency, not "
        f"a thinned subsample of a larger pool). Click to expand.</p>")
    for cid in CLASS_IDS:
        cname = CLASS_NAMES.get(cid, "?")
        files = sorted((OUT / "inputs" / str(cid)).glob("*.png"))
        html.append(f"<details><summary><b>class {cid}</b> — {cname} ({len(files)} images)</summary>")
        html.append("<div class='dd-grid' style='grid-template-columns:repeat(10,1fr)'>")
        for f in files:
            rel = f.relative_to(OUT)
            html.append(f"<div class='dd-cell'><img loading='lazy' src='{rel}'><div>{f.stem}</div></div>")
        html.append("</div></details>")

    html.append("</body></html>")
    (OUT / "index.html").write_text("\n".join(html))


def write_readme(n_inputs, n_panels, n_dd, include_deepdream: bool = False):
    archs_str = ", ".join(ARCH_DISPLAY_NAMES.get(a, a) for a in ARCHS)
    classes_str = ", ".join(f"{CLASS_NAMES.get(c, '?')}({c})" for c in CLASS_IDS)
    dd_status = ("included as a supplementary plate (--include-deepdream)"
                 if include_deepdream
                 else "excluded from the headline panel and rendered as a "
                      "supplementary plate only (re-render with "
                      "`--include-deepdream` to surface them in the gallery)")
    md = f"""# km-feature-viz — cluster output gallery

This is a rendered, viewable gallery of the visualizations produced by the km-feature-viz
pipeline on the cluster. The raw `.pt` tensors live one directory up (in `..`); everything
here is PNG.

**Open `index.html` in a browser** for the interactive gallery.

## Paper context — Pillar 3

These panels are the empirical figure for paper Pillar 3:
**"Exact image-space attribution: knowledge matrices as canonical saliency."**

The thesis is that the per-class row of the per-input knowledge matrix `M(x) ∈ R^(C×(d+1))`
is itself a saliency map — one with properties no baseline saliency method achieves on
this repo's models:

- **Exact completeness.** `Σ_j M(x)[c, j] = f_c(x)` at fp32 epsilon — *equality*, not the
  baseline-dependent bound that Integrated Gradients gives. There is no baseline to choose,
  no rule to pick, no hyperparameter that changes the answer.
- **Implementation invariance via canonical structure.** KMs are invariant under hidden-
  layer neuron permutations; Grad-CAM, IG, and SmoothGrad are not.
- **Sanity-check faithfulness by construction.** Because `M(x).sum(1) == f(x)` holds at
  every weight configuration, randomizing the network weights necessarily changes
  `M(x)` — KM rows pass the Adebayo (2018) model-randomization sanity check by
  construction. Several baselines (notably Guided Backprop) do not.

The figure budget is deliberately small (a *qualitative* exemplar set chosen for inter-
class contrast and intra-class consistency — not a thinned subsample of a larger pool).
Quantitative validation lives in separate experiments.

## Architectures and per-arch DeepDream methodology

Three CNN-family representatives are run end-to-end via `run_pipeline.sh` (D1–D5):

| Arch | Display name | Family | Input | DeepDream neuron selection |
| ---- | ------------ | ------ | ----- | --------------------------- |
| `resnet152`   | ResNet-152            | residual           | 224×224 | class-conditional Grad-CAM channel ranking (Selvaraju et al. 2017, ch-wise variant) |
| `densenet121` | DenseNet-121          | dense connectivity | 224×224 | class-conditional Grad-CAM channel ranking (Selvaraju et al. 2017, ch-wise variant) |
| `googlenet`   | GoogLeNet (InceptionV1) | multi-branch inception | 224×224 | catalogued from Distill *Circuits Thread* + OpenAI Microscope (Olah et al. 2017; Cammarata et al. 2020) |

DeepDream is **{dd_status}** — the round-3 Interp/KM debate concluded that DeepDream is
not a class-conditional explanation and should not appear next to KM/Grad-CAM/IG in the
main figure where a reader could read the columns as parallel.

## What's in the source tar

The cluster pipeline produced `km-feature-viz.tar` with **{len(ARCHS)} architectures**
({archs_str}) evaluated on **{n_inputs} ImageNet validation images** spread across
**{len(CLASS_IDS)} classes**: {classes_str}.

All tensors are stored at **fp32** (the previous fp16 storage caused inf overflow on the
ResNet18 residual path; the rewrite eliminated that artifact):

| Path | Tensor shape | Meaning |
| ---- | ------------ | ------- |
| `images/<class>/*.pt` | `(3, 224, 224)` uint8 | Pre-processed RGB input fed to all models |
| `kms/<arch>/<class>/*.pt` | dict `{{km: (1000, 150529) fp32, classes: [1000 ints]}}` | **Knowledge matrix** — one row per ImageNet output class. Each row is a per-input-pixel "explanation" of that class's logit, in pixel space (`150528 = 3·224·224`) plus 1 bias entry. Row sums equal logits exactly. |
| `baselines/feature_maps/<arch>/<class>/*.pt` | `(1, C, H, W)` fp32 | Penultimate-layer feature maps (the standard "hidden activation" baseline). |
| `baselines/gradcam/<arch>/<class>/*.pt` | `(1, 1, h, w)` fp32 | Grad-CAM saliency at the last conv layer (coarse — typically 7×7 on standard CNNs at 224×224). |
| `baselines/ig/<arch>/<class>/*.pt` | `(1, 3, 224, 224)` fp32 | Integrated Gradients pixel attribution (zero/black baseline — see "Notes" below). |
| `baselines/smoothgrad/<arch>/<class>/*.pt` | `(1, 3, 224, 224)` fp32 | SmoothGrad pixel attribution. |
| `baselines/pgd/<arch>/<class>/*.pt` | dict `{{delta: (1, 3, 224, 224), target: int}}` | PGD adversarial perturbation that flips the model's prediction to `target`. |
| `deepdream/<arch>/<layer>/neuron_####.pt` | `(3, 224, 224)` fp32 | Image synthesized via gradient ascent to maximize one neuron at `<layer>`. Selected per (arch, layer) by `state/03a_neuron_selection_<arch>.json`. {n_dd} total. |
| `state/03a_neuron_selection_<arch>.json` | JSON | Per-arch DeepDream neuron-selection metadata: `method ∈ {{gradcam_class_conditional, catalogued_distill}}`, `channels[layer] = [{{channel, rank, mean_alpha?, label?, citation?}}, ...]`. Consumed by `scripts/render_km_viz.py` to format DeepDream tile captions. |

`manifest.json` lists all (model, class_id, image_id) triples; `errors.json` is the
per-step error log (empty on a clean run).

## What this gallery contains

Rendered into `_viewable/`:

- **`inputs/<class>/*.png`** — all {n_inputs} input images.
- **`deepdream/<arch>/<layer>/*.png`** — {n_dd} deep-dream PNGs (per-neuron synthesized
  images, noise-initialized + jitter only — see "Notes" below). DeepDream is
  {dd_status}.
- **`panels/<arch>/<class>/*.png`** — {n_panels} side-by-side comparison panels. Each
  panel shows, left→right:
  1. **input** (raw image, 224×224 RGB),
  2. **Grad-CAM** overlay (red = high relevance for the true class, upsampled from the
     last conv map — coarse by construction; see "Notes"),
  3. **Integrated Gradients** overlay (channel-summed |attribution|),
  4. **SmoothGrad** overlay (channel-summed |attribution| over noise samples),
  5. **PGD δ** — the adversarial perturbation visualized in pixel space, with the
     target class label,
  6. **Knowledge-matrix row for the true class** — the row of the KM corresponding to
     the image's true class, reshaped from `(150528,)` back to `(3, 224, 224)`,
     channel-abs-summed, percentile-clipped, and overlaid on the input.

  Per the round-3 Interp/KM debate, **DeepDream is intentionally NOT a column** in this
  panel: it's a neuron prototype, not a class-conditional explanation, and including it
  would invite reading the columns as parallel.

## Notes on the rendering and methodology

- **Heatmap colorbars** — each heatmap subplot has a jet colorbar to its right showing
  the actual numeric range (after the 1/99-percentile clip) that maps to the bottom
  and top of the colormap. The percentile clip is for *visualization dynamic range*
  only; under fp32 storage there should be no `inf`/`NaN` entries to suppress.
- **Grad-CAM resolution caveat** — the last conv block on standard CNNs at 224×224 input
  produces a feature map of roughly 7×7 spatial resolution. Bilinearly upsampling to
  224×224 gives an effective ~32-pixel grid. Fine-grained localization claims read off
  Grad-CAM are illusory at that resolution.
- **IG baseline caveat** — Integrated Gradients is computed with a zero baseline
  (the Captum default). On ImageNet-normalized inputs this corresponds to a gray, not
  black, reference; either way, attributions on dark pixels are systematically
  downweighted because `(x_i − x'_i)` is small there. Read IG maps with that bias in
  mind. See Sundararajan-Taly-Yan (2017) and Sturmfels et al. (Distill 2020).
- **PGD is not feature visualization** — the δ column shows the *adversarial
  perturbation* that flips the prediction. On standard (non-adversarially-trained)
  ImageNet models it visualizes a non-robust direction across the decision boundary,
  not "what class k looks like" (Ilyas et al. 2019, Santurkar et al. 2019).
- **DeepDream caveat** — the per-neuron images are produced by noise-initialized
  gradient ascent with random pixel-space jitter only; no decorrelated FFT
  parameterization, no transformation robustness (rotation/scale/multi-octave). Read
  these as neuron prototypes under a *minimal* regularizer rather than the polished
  Distill-2017 visualizations.
- **DeepDream neuron-selection methodology** — the *which neurons* question is answered
  per arch by `state/03a_neuron_selection_<arch>.json`:
  - **ResNet-152, DenseNet-121** — *class-conditional Grad-CAM* channel ranking. For
    each (target class, candidate layer), we compute the Grad-CAM ch-wise weight
    `α_k = GAP(∂y_c/∂A^k)` averaged over a held-out subset of the class's images, then
    take the top-ranked channels. Caption reports `mean α`. Cite: Selvaraju et al. 2017.
  - **GoogLeNet (InceptionV1)** — *catalogued* from the Distill *Circuits Thread* and
    OpenAI Microscope (curve detectors, dog-head detectors, high-low frequency
    detectors, etc.). Caption reports the published feature label and the citation.
    Cite: Olah et al. 2017 (Feature Visualization), Cammarata et al. 2020 (Thread:
    Circuits).
- **KM percentile clip** — saliencies and KM rows are visualized on a 1/99-percentile
  clipped scale and channel-summed in absolute value. The clip is a rendering choice;
  the underlying KM row carries the exact decomposition.

## Reproducing

The renderer is `scripts/render_km_viz.py`. To regenerate everything from the raw tar:

```bash
tar -xf km-feature-viz.tar -C results/km-feature-viz-cluster --strip-components=1

# Headline render: KM + Grad-CAM + IG + SmoothGrad + PGD panels (no DeepDream column)
python scripts/render_km_viz.py --per-class 7

# Supplementary render: same as above, plus expanded DeepDream tile section
python scripts/render_km_viz.py --per-class 7 --include-deepdream

open results/km-feature-viz-cluster/_viewable/index.html
```

Use `--per-class N` to control how many comparison panels are rendered per (class, arch).
With the current manifest of 7 / 7 / 6 images for classes 207 / 282 / 340, `N=7` covers
the full set; `N` larger than the per-class image count is a no-op.

Use `--include-deepdream` to render the DeepDream tile section as an open top-level
section (otherwise it lives inside a collapsed `<details>` block, labeled "supplementary
— excluded from headline figure"). DeepDream is always *computed* and *stored as PNG*
either way; the flag only changes how prominently it appears in the gallery.
"""
    (OUT / "README.md").write_text(md)


def main():
    ap = argparse.ArgumentParser(
        description="Render the km-feature-viz cluster output as a viewable PNG gallery. "
                    "By default the headline panel layout (per arch×class) is "
                    "input + Grad-CAM + IG + SmoothGrad + PGD + KM (true-class row) — "
                    "no DeepDream column. Pass --include-deepdream to also surface the "
                    "DeepDream tile section as a top-level (rather than collapsed/"
                    "supplementary) part of the gallery.")
    ap.add_argument("--per-class", type=int, default=7,
                    help="number of comparison panels per (class, arch). Default 7 "
                         "matches the manifest's max per-class image count "
                         "(see km_feature_viz/manifest.py:TIER_A_IMAGES_PER_CLASS).")
    ap.add_argument("--skip-inputs", action="store_true")
    ap.add_argument("--skip-deepdream", action="store_true",
                    help="skip DeepDream rendering; reuse existing PNGs from disk")
    ap.add_argument("--skip-panels", action="store_true")
    ap.add_argument("--include-deepdream", action="store_true",
                    help="render DeepDream tiles as an open top-level section in the "
                         "gallery (default: collapsed under a 'supplementary' details "
                         "block, with the 2-method neuron-selection methodology blurb "
                         "always visible). Excluding DeepDream from the headline panel "
                         "layout is the round-3 Interp/KM debate decision.")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    if args.skip_inputs:
        n_inputs = sum(1 for _ in (OUT / "inputs").rglob("*.png"))
    else:
        print("rendering inputs...")
        n_inputs = render_inputs(OUT / "inputs")
        print(f"  -> {n_inputs} PNGs")

    if args.skip_deepdream:
        # Reconstruct the same per-arch dict shape that render_deepdreams returns,
        # using neuron-selection JSON for captions when available.
        deepdreams = {}
        for a in ARCHS:
            selection = load_neuron_selection(a)
            deepdreams[a] = []
            for png in sorted((OUT / "deepdream" / a).rglob("*.png")):
                rel = png.relative_to(OUT / "deepdream" / a)
                parts = rel.parts
                if len(parts) < 2:
                    continue
                layer = parts[0]
                stem = png.stem
                ch = _channel_from_filename(stem)
                if ch is not None:
                    caption, citation = deepdream_caption(selection, layer, ch)
                else:
                    caption, citation = stem, None
                deepdreams[a].append({
                    "rel": str(rel),
                    "layer": layer,
                    "channel": ch,
                    "caption": caption,
                    "citation": citation,
                })
    else:
        print("rendering deepdream neurons...")
        deepdreams = render_deepdreams(OUT / "deepdream")
        print(f"  -> {sum(len(v) for v in deepdreams.values())} PNGs")

    if args.skip_panels:
        panels = []
        for arch in ARCHS:
            for cid in CLASS_IDS:
                for png in sorted((OUT / "panels" / arch / str(cid)).glob("*.png")):
                    panels.append((arch, cid, png.stem,
                                   str(png.relative_to(OUT))))
    else:
        print(f"rendering comparison panels ({args.per_class} per class per arch, "
              f"DeepDream column intentionally NOT included in the per-panel layout — "
              f"DeepDream is rendered as a separate {'open' if args.include_deepdream else 'supplementary'} "
              f"section)...")
        panels = render_panels(OUT / "panels", args.per_class)
        print(f"  -> {len(panels)} panels")

    n_dd = sum(len(v) for v in deepdreams.values())
    print("writing README + index.html...")
    write_readme(n_inputs, len(panels), n_dd, include_deepdream=args.include_deepdream)
    write_html_index(panels, deepdreams, n_inputs,
                     include_deepdream=args.include_deepdream)
    print("done. open:", (OUT / "index.html").resolve())


if __name__ == "__main__":
    main()
