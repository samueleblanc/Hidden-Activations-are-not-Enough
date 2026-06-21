"""Cui 2022 random-network control.

Compute each measure between trained h_W and an independently-initialized,
UNTRAINED h_R from the same architecture, on the same input subset.

If the trained-vs-teleported similarity values are not substantially above
this floor, the input-space population-structure confound (Cui 2022)
dominates and the headline measures are misleading.
"""
import torch
from typing import Dict, List

from cka_similarity.measures.panel import PANEL
from cka_similarity.workers.s1_within_arch_invariance import load_pretrained, forward_penultimate


def compute_cui_control(arch: str, inputs: torch.Tensor, device) -> Dict[str, float]:
    """Forward inputs through trained vs random-init same-arch; return measure dict."""
    import torchvision.models as tvm
    arch_loader = {
        "resnet152": lambda **kw: tvm.resnet152(**kw),
        "densenet121": lambda **kw: tvm.densenet121(**kw),
        "googlenet": lambda **kw: tvm.googlenet(aux_logits=False, **kw),
    }[arch]

    trained = load_pretrained(arch).to(device)
    # Seed BEFORE construction so torchvision's own default init is reproducible,
    # then take the network AS-INITIALISED. Do NOT re-randomise parameters by hand:
    # the previous loop zeroed every 1-D parameter -- including BatchNorm scale
    # (gamma) and shift (beta) -- which forces every BN output to 0, collapsing the
    # penultimate features to a constant (zero variance) and making CKA/Bures/RSA
    # NaN (0/0). The Cui control needs a NON-degenerate random network; torchvision's
    # default init (kaiming convs, BN gamma=1/beta=0) provides exactly that.
    torch.manual_seed(42)
    untrained = arch_loader(weights=None).to(device).eval()

    h_t = forward_penultimate(trained, inputs).cpu()
    h_r = forward_penultimate(untrained, inputs).cpu()

    out = {}
    for cls in PANEL:
        m = cls()
        if cls.__name__ == "OutputJSD":
            continue   # JSD on logits is in a different orbit; control on penultimate-only
        try:
            r = m.finalize([m.accumulate(h_t, h_r)])
            out[m.name] = r.value
        except Exception as e:
            out[m.name] = float("nan")
    return out
