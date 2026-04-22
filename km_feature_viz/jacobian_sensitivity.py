"""Step 06: Jacobian sensitivity heatmap from W_eff."""
import argparse
import inspect
import logging
import sys
from pathlib import Path

import torch
from knowledgematrix.matrix_computer import KnowledgeMatrixComputer

from km_feature_viz import paths, state
from km_feature_viz.compute_kms import build_model, load_image
from km_feature_viz.counterfactual_lp import extract_weff_and_beff
from km_feature_viz.manifest import read_manifest, sample_key

logger = logging.getLogger(__name__)


def patch_available() -> bool:
    sig = inspect.signature(KnowledgeMatrixComputer.forward)
    return "extract_weff" in sig.parameters


def sensitivity_heatmap(model, x: torch.Tensor, predicted_class: int) -> torch.Tensor:
    """Return (W_eff[j, k] * x_k)^2 reshaped to (C, H, W) for j=predicted_class."""
    W_eff, _ = extract_weff_and_beff(model, x)
    j = predicted_class
    s = (W_eff[j] * x.flatten()) ** 2
    return s.reshape(x.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--n-images-per-class", type=int, default=3)
    parser.add_argument("--n-classes", type=int, default=3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not patch_available():
        print("ERROR: knowledgematrix patch missing.", file=sys.stderr)
        return 2

    entries = read_manifest(args.manifest)
    by_model_class = {}
    for e in entries:
        by_model_class.setdefault((e.model, e.class_id), []).append(e)

    completed = state.load_completed(paths.state_path("06_jacobian"))

    for (model_name, class_id), samples in list(by_model_class.items())[: args.n_classes * 3]:
        model = build_model(model_name, args.device)
        for e in samples[: args.n_images_per_class]:
            key = sample_key(e)
            if key in completed:
                continue
            try:
                x = load_image(e.image_path).to(args.device)
                with torch.no_grad():
                    pred = model.forward(x).argmax().item()
                s = sensitivity_heatmap(model, x, predicted_class=pred)
                out_path = paths.jacobian_path(model_name, class_id, e.image_id)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(s.cpu().to(torch.float16), out_path)
                state.mark_completed(paths.state_path("06_jacobian"), key)
                logger.info("done %s", key)
            except Exception as exc:
                state.log_error(
                    paths.errors_path(), step="06_jacobian", sample_id=key,
                    error_type=type(exc).__name__, message=str(exc),
                    tb=state.capture_traceback(),
                )
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
