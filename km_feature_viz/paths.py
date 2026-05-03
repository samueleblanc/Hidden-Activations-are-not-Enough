"""Storage layout for the km-feature-viz experiment.

Single source of truth for where every artifact lives. Other scripts
import these helpers — never hand-construct a path.
"""
from pathlib import Path

RESULTS_ROOT = Path("results/km-feature-viz")
FIGURES_ROOT = Path("docs/km-feature-viz/figures")


def km_path(model: str, class_id: int, image_id: str) -> Path:
    return RESULTS_ROOT / "kms" / model / str(class_id) / f"{image_id}.pt"


def image_path(class_id: int, image_id: str) -> Path:
    """Unnormalized 224×224 uint8 RGB tensor — shared across all models."""
    return RESULTS_ROOT / "images" / str(class_id) / f"{image_id}.pt"


def baseline_path(method: str, model: str, class_id: int, image_id: str) -> Path:
    return RESULTS_ROOT / "baselines" / method / model / str(class_id) / f"{image_id}.pt"


def deepdream_path(model: str, layer_name: str, neuron: int) -> Path:
    return RESULTS_ROOT / "deepdream" / model / layer_name / f"neuron_{neuron:04d}.pt"


def dictionary_path(model: str, class_id: int, kind: str) -> Path:
    """kind in {'components', 'explained_variance', 'projections'}."""
    return RESULTS_ROOT / "formulations" / "dictionary" / model / str(class_id) / f"{kind}.pt"


def counterfactual_path(model: str, class_id: int, image_id: str, target: int) -> Path:
    return (
        RESULTS_ROOT
        / "formulations"
        / "counterfactual"
        / model
        / str(class_id)
        / f"{image_id}__to_{target}.json"
    )


def jacobian_path(model: str, class_id: int, image_id: str) -> Path:
    return RESULTS_ROOT / "formulations" / "jacobian" / model / str(class_id) / f"{image_id}.pt"


def state_path(step_name: str) -> Path:
    return RESULTS_ROOT / "state" / f"{step_name}.json"


def errors_path() -> Path:
    return RESULTS_ROOT / "errors.json"


def manifest_path() -> Path:
    return RESULTS_ROOT / "manifest.json"


def figure_path(name: str) -> Path:
    return FIGURES_ROOT / f"{name}.pdf"
