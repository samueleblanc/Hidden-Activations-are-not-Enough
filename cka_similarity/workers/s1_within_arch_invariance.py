"""S1 worker: within-arch invariance under teleportation, 9-measure panel.

Per chunk: load chunk slice of ImageNet val, for each (arch, teleport_id)
compute h_W and h_~W via forward passes; emit per-chunk accumulators for
each measure. KMs are NOT computed (zero by Theorem 4.1, verified by
unit tests; see design spec §2.3 and §6.1).

Output: results/phase1/s1/{arch}_teleport{j}_chunk{i}.pt — one tensor file
per (arch, teleport, chunk).
"""
import os
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

import torch

from utils.utils import get_device, get_imagenet_val_dataset
from utils.atomic_io import atomic_torch_save
from cka_similarity.workers.common import chunk_slice
from cka_similarity.measures.panel import PANEL


def load_pretrained(arch: str):
    """Load the torchvision-pretrained network for the given arch."""
    import torchvision.models as tvm
    model = {
        "resnet152": lambda: tvm.resnet152(weights="DEFAULT"),
        "densenet121": lambda: tvm.densenet121(weights="DEFAULT"),
        "googlenet": lambda: tvm.googlenet(aux_logits=False, weights="DEFAULT"),
    }[arch]()
    return model.eval()


def teleport(model, seed: int):
    """Apply the seed-th random neural teleportation to a copy of the model."""
    from neuralteleportation.changeofbasisutils import get_random_cob
    teleported = deepcopy(model)
    torch.manual_seed(seed)
    cob = get_random_cob(teleported, cob_range=1.0)
    teleported.apply_cob(cob)
    return teleported.eval()


def _get_classifier_module(model):
    """Return the final FC/classifier module — input to which is the penultimate."""
    if hasattr(model, "fc"):
        return model.fc
    if hasattr(model, "classifier"):
        return model.classifier
    raise ValueError(f"No fc or classifier found on {type(model).__name__}")


def forward_penultimate(model, x):
    """Return penultimate-layer activations (input to the final FC) via forward hook.

    Architecture-agnostic: works for ResNet (model.fc), GoogLeNet (model.fc),
    and DenseNet (model.classifier). Uses a forward hook to capture the input
    to the classifier layer — that input is, by construction, the penultimate
    activation regardless of arch-specific feature-extraction details.
    """
    captured = {}
    cls_layer = _get_classifier_module(model)
    handle = cls_layer.register_forward_hook(
        lambda mod, inp, out: captured.__setitem__("h", inp[0].detach())
    )
    try:
        with torch.no_grad():
            model(x)
    finally:
        handle.remove()
    return captured["h"]


def forward_logits(model, x):
    with torch.no_grad():
        return model(x)


def load_imagenet_val_chunk(start: int, end: int, data_dir: str):
    """Load ImageNet val samples [start, end) as a (n, 3, 224, 224) tensor.

    Uses utils.utils.get_imagenet_val_dataset which returns the full 50K val
    set in alphabetical filename order — NOT get_dataset("imagenet", ...) which
    does a random_split with seed=42.
    """
    _, val_set = get_imagenet_val_dataset(data_path=data_dir)
    xs = []
    for i in range(start, end):
        x_i, _ = val_set[i]
        xs.append(x_i.unsqueeze(0))
    return torch.cat(xs, dim=0)


def run_chunk(chunk_id, num_chunks, num_samples_total, archs, num_teleports,
              out_dir, data_dir, batch_size=32):
    """Compute per-chunk S1 accumulators for one chunk slice."""
    os.makedirs(out_dir, exist_ok=True)
    device = get_device()

    start, end = chunk_slice(chunk_id, num_samples_total, num_chunks)
    inputs = load_imagenet_val_chunk(start, end, data_dir).to(device)

    panel_cls = list(PANEL)

    for arch in archs:
        W = load_pretrained(arch).to(device)
        h_W = forward_penultimate(W, inputs).cpu()
        logits_W = forward_logits(W, inputs).cpu()

        for teleport_id in range(num_teleports):
            out_path = Path(out_dir) / f"{arch}_teleport{teleport_id}_chunk{chunk_id}.pt"
            if out_path.exists():
                continue   # idempotent — skip if already done

            W_tilde = teleport(W, seed=teleport_id).to(device)
            h_W_tilde = forward_penultimate(W_tilde, inputs).cpu()
            logits_W_tilde = forward_logits(W_tilde, inputs).cpu()

            # Compute accumulators for each measure
            accumulators = {}
            for cls in panel_cls:
                m = cls()
                if cls.__name__ == "OutputJSD":
                    # JSD takes logits not penultimate
                    accumulators[m.name] = m.accumulate(logits_W, logits_W_tilde)
                else:
                    accumulators[m.name] = m.accumulate(h_W, h_W_tilde)

            atomic_torch_save(str(out_path), {
                "chunk_id": chunk_id, "arch": arch, "teleport_id": teleport_id,
                "n_samples": h_W.shape[0],
                "accumulators": accumulators,
            })

            # Free GPU memory
            del W_tilde, h_W_tilde, logits_W_tilde
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    parser = ArgumentParser()
    parser.add_argument("--chunk_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_chunks", type=int, default=64)
    parser.add_argument("--num_samples", type=int, default=25000)
    parser.add_argument("--num_teleports", type=int, default=50)
    parser.add_argument("--archs", nargs="+", default=["resnet152", "densenet121", "googlenet"])
    parser.add_argument("--out_dir", default="results/phase1/s1")
    parser.add_argument("--data_dir", default="/datashare/imagenet/ILSVRC2012")
    args = parser.parse_args()

    run_chunk(
        chunk_id=args.chunk_id, num_chunks=args.num_chunks,
        num_samples_total=args.num_samples, archs=args.archs,
        num_teleports=args.num_teleports, out_dir=args.out_dir, data_dir=args.data_dir,
    )


if __name__ == "__main__":
    main()
