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
    untrained = arch_loader(weights=None).to(device).eval()
    # Re-randomize all params with a fixed seed for reproducibility
    torch.manual_seed(42)
    for p in untrained.parameters():
        if p.dim() > 1:
            torch.nn.init.kaiming_normal_(p)
        else:
            torch.nn.init.zeros_(p)

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
