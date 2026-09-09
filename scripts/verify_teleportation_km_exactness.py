"""
Does the KNOWLEDGE-MATRIX drift under teleportation also scale with machine epsilon?

The logits are already shown exact (tp_exactness_final.py). The KM is
M(x) = [J diag(x) | f(x) - J x], so we test the Jacobian rows directly:
compute per-class VJP rows of J for the original and the teleported network, at
fp32 and at fp64, and compare.

Exact-isomorphism prediction: KM drift ratio fp32/fp64 ~ eps32/eps64 = 5.4e8.
"""
import sys, copy, numpy as np, torch
REPO = "/Users/marco/Desktop/Experiments - NNs/Hidden-Activations-are-not-Enough"
sys.path.insert(0, REPO)

import neuralteleportation.layers.neuralteleportation as ntl
import neuralteleportation.neuralteleportationmodel as ntm
import neuralteleportation.layers.merge as ntmerge
from neuralteleportation.network_graph import NetworkGrapher

def _fwd(self, input):
    if getattr(self, self.cob_field) is None:
        setattr(self, self.cob_field, torch.ones(input.shape[1], dtype=input.dtype))
    if self.reshape_cob:
        s = (input.shape[1],) + tuple([1]*(input.dim()-2))
        setattr(self, self.cob_field, getattr(self, self.cob_field).view(s).type_as(input).detach())
    return self._forward(input)
ntl.COBForwardMixin.forward = _fwd

def _add(self, i1, i2):
    if self.prev_cob is None: self.prev_cob = torch.ones(i2.shape[1], dtype=i1.dtype)
    if self.next_cob is None: self.next_cob = torch.ones(i2.shape[1], dtype=i1.dtype)
    s = (i2.shape[1],) + tuple([1]*(i2.dim()-2))
    self.prev_cob = self.prev_cob.view(s).type_as(i1)
    self.next_cob = self.next_cob.view(s).type_as(i1)
    return torch.add(i1, self.next_cob * i2 / self.prev_cob)
ntmerge.Add.forward = _add

def _init(self, network, input_shape):
    torch.nn.Module.__init__(self); self.network = network
    p = next(network.parameters()); self.eval()
    self.grapher = NetworkGrapher(network, torch.rand(input_shape).to(p.device).to(p.dtype))
    self.graph = self.grapher.get_graph(); self.initialize_cob()
ntm.NeuralTeleportationModel.__init__ = _init

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from teleportation_experiment import load_pretrained_cob

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
torch.manual_seed(7)
X32 = ((torch.rand(1,3,224,224)-mean)/std)

CLASSES = list(range(12))   # 12 KM rows is plenty to establish the scaling

def km_rows(model, x, classes):
    """Rows of [J diag(x) | c] for the given classes, via one VJP per class."""
    x = x.clone().requires_grad_(True)
    out = model(x)
    rows = []
    for c in classes:
        g, = torch.autograd.grad(out[0, c], x, retain_graph=True)
        Jrow = g.flatten()                        # dF_c/dx
        km_lin = Jrow * x.detach().flatten()      # J diag(x) row
        bias = out[0, c].detach() - km_lin.sum()  # c = f - Jx
        rows.append(torch.cat([km_lin, bias.reshape(1)]))
    return torch.stack(rows)

def teleport(m, seed, cr):
    mc = copy.deepcopy(m); torch.manual_seed(seed); np.random.seed(seed)
    NeuralTeleportationModel(mc, input_shape=(1,3,224,224)).random_teleport(cob_range=cr)
    return mc.eval()

print("KM drift under teleportation, by working precision")
print("(M = [J diag(x) | f - Jx]; 12 class rows; Frobenius over those rows)\n")
for arch, cr in [("resnet152",1.0), ("densenet121",1.0), ("googlenet",1.0)]:
    print(f"=== {arch}  cob_range={cr} ===", flush=True)
    for seed in range(2):
        res = {}
        for dt, tag in ((torch.float32,"fp32"), (torch.float64,"fp64")):
            m = load_pretrained_cob(arch, "cpu").to(dt).eval()
            t = teleport(m, seed, cr)
            x = X32.to(dt)
            A = km_rows(m, x, CLASSES); B = km_rows(t, x, CLASSES)
            d = (A-B).norm().item(); s = A.norm().item()
            res[tag] = (d, d/s)
            print(f"  seed {seed} {tag}: ||dM||_F = {d:.4e}   relative = {d/s:.3e}", flush=True)
        r = res["fp32"][0]/max(res["fp64"][0],1e-300)
        print(f"  seed {seed} --> fp32/fp64 KM-drift ratio = {r:.3e}"
              f"   (exact-isomorphism prediction ~5.4e8)\n", flush=True)
