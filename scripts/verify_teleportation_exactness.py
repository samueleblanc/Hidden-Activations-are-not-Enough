"""
Teleportation exactness: definitive multi-seed run, all three trio architectures.

Claim under test: neural teleportation (per-neuron change of basis) is an EXACT
function-preserving isomorphism on these BatchNorm networks in eval mode, and the
logit drift reported in the Step-B run is entirely floating-point roundoff.

Prediction if exact: drift is LINEAR in the working machine epsilon, so
  drift(fp32)/drift(fp64) ~ eps32/eps64 = 2^29 = 5.4e8.
Prediction if the function really changes: the ratio is ~1.

Three library fp32 downcasts are patched out so the fp64 arm is genuinely fp64.
"""
import sys, copy, json, numpy as np, torch
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
        shape = (input.shape[1],) + tuple([1]*(input.dim()-2))
        setattr(self, self.cob_field,
                getattr(self, self.cob_field).view(shape).type_as(input).detach())
    return self._forward(input)
ntl.COBForwardMixin.forward = _fwd

def _add_fwd(self, i1, i2):
    if self.prev_cob is None: self.prev_cob = torch.ones(i2.shape[1], dtype=i1.dtype)
    if self.next_cob is None: self.next_cob = torch.ones(i2.shape[1], dtype=i1.dtype)
    s = (i2.shape[1],) + tuple([1]*(i2.dim()-2))
    self.prev_cob = self.prev_cob.view(s).type_as(i1)
    self.next_cob = self.next_cob.view(s).type_as(i1)
    return torch.add(i1, self.next_cob * i2 / self.prev_cob)
ntmerge.Add.forward = _add_fwd

def _cat_fwd_patch():
    C = getattr(ntmerge, "Concat", None)
    if C is None or not hasattr(C, "forward"): return
    src = C.forward
    def wrapped(self, *a, **k):
        for f in ("prev_cob","next_cob","cob"):
            v = getattr(self, f, None)
            if isinstance(v, torch.Tensor) and a and isinstance(a[0], torch.Tensor):
                setattr(self, f, v.type_as(a[0]))
        return src(self, *a, **k)
    C.forward = wrapped
_cat_fwd_patch()

def _init(self, network, input_shape):
    torch.nn.Module.__init__(self)
    self.network = network
    p = next(network.parameters())
    self.eval()
    self.grapher = NetworkGrapher(network, torch.rand(input_shape).to(p.device).to(p.dtype))
    self.graph = self.grapher.get_graph()
    self.initialize_cob()
ntm.NeuralTeleportationModel.__init__ = _init

from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from teleportation_experiment import load_pretrained_cob

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
torch.manual_seed(7)
X32 = ((torch.rand(8,3,224,224)-mean)/std)

EPS32, EPS64, EPSTF32 = 2.0**-24, 2.0**-53, 2.0**-11

def teleport(m, seed, cr):
    mc = copy.deepcopy(m); torch.manual_seed(seed); np.random.seed(seed)
    NeuralTeleportationModel(mc, input_shape=(1,3,224,224)).random_teleport(cob_range=cr)
    return mc.eval()

results = {}
for arch, cr in [("resnet152",1.0), ("densenet121",1.0), ("googlenet",1.0)]:
    print(f"\n{'='*62}\n{arch}  (cob_range={cr})\n{'='*62}", flush=True)
    rows = []
    for seed in range(5):
        rec = {"seed": seed}
        for dt, tag in ((torch.float32,"fp32"), (torch.float64,"fp64")):
            try:
                m = load_pretrained_cob(arch, "cpu").to(dt).eval()
                t = teleport(m, seed, cr)
                x = X32.to(dt)
                with torch.no_grad(): a = m(x); b = t(x)
                d = (a-b).abs().max().item(); s = a.abs().max().item()
                rec[tag] = d; rec[tag+"_rel"] = d/s; rec["logit_scale"] = s
            except Exception as e:
                rec[tag] = None; rec[tag+"_err"] = f"{type(e).__name__}: {e}"
        if rec.get("fp32") and rec.get("fp64"):
            rec["ratio"] = rec["fp32"]/rec["fp64"]
            # C = drift / eps  -> extrapolate to TF32 (H100 cudnn default)
            rec["C_fp32"] = rec["fp32"]/EPS32
            rec["pred_tf32"] = rec["C_fp32"]*EPSTF32
            print(f"  seed {seed}: fp32={rec['fp32']:.3e}  fp64={rec['fp64']:.3e}  "
                  f"ratio={rec['ratio']:.2e}  ->predicted TF32 drift={rec['pred_tf32']:.3e}", flush=True)
        else:
            print(f"  seed {seed}: {rec}", flush=True)
        rows.append(rec)
    results[arch] = rows

print("\n\n================ SUMMARY ================")
print(f"eps32={EPS32:.3e}  eps64={EPS64:.3e}  epsTF32={EPSTF32:.3e}  "
      f"eps32/eps64={EPS32/EPS64:.2e}")
obs = {"resnet152":(3.1e-2,1.0e-1), "densenet121":(1.6e-2,3.9e-2), "googlenet":(1.8e-2,2.9e-2)}
for a, rows in results.items():
    ok = [r for r in rows if r.get("ratio")]
    if not ok:
        print(f"{a}: no complete pairs"); continue
    f32 = np.array([r["fp32"] for r in ok]); f64 = np.array([r["fp64"] for r in ok])
    rel64 = np.array([r["fp64_rel"] for r in ok]); pred = np.array([r["pred_tf32"] for r in ok])
    print(f"\n{a}:")
    print(f"   fp32 drift          {f32.min():.3e} .. {f32.max():.3e}")
    print(f"   fp64 drift          {f64.min():.3e} .. {f64.max():.3e}   "
          f"(relative {rel64.min():.2e} .. {rel64.max():.2e}  <- machine epsilon)")
    print(f"   fp32/fp64 ratio     {(f32/f64).min():.2e} .. {(f32/f64).max():.2e}   "
          f"(exact-isomorphism prediction {EPS32/EPS64:.2e})")
    print(f"   predicted TF32      {pred.min():.3e} .. {pred.max():.3e}")
    print(f"   OBSERVED on H100    {obs[a][0]:.3e} .. {obs[a][1]:.3e}")
json.dump(results, open("/private/tmp/claude-501/-Users-marco-Desktop-Experiments---NNs-Hidden-Activations-are-not-Enough/12375aa9-f293-40de-a1ad-fa184af2b615/scratchpad/tp_exactness.json","w"), indent=2, default=str)
print("\nwrote tp_exactness.json")
