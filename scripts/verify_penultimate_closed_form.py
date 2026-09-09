"""
Is the penultimate drift under teleportation exactly h -> tau (x) h, per neuron?
Robust test: fit tau_c by least squares per channel, then measure the residual.
  tau_c = <h_tp[:,c], h[:,c]> / <h[:,c], h[:,c]>
If Marco's claim holds, residual ||h_tp - tau*h|| / ||h_tp|| ~ float32 epsilon.
"""
import sys, copy, numpy as np, torch
REPO = "/Users/marco/Desktop/Experiments - NNs/Hidden-Activations-are-not-Enough"
sys.path.insert(0, REPO)
from neuralteleportation.neuralteleportationmodel import NeuralTeleportationModel
from teleportation_experiment import load_pretrained_cob, PenultimateExtractor

mean = torch.tensor([0.485,0.456,0.406]).view(1,3,1,1)
std  = torch.tensor([0.229,0.224,0.225]).view(1,3,1,1)
torch.manual_seed(7)
X = ((torch.rand(24,3,224,224)-mean)/std)

def teleport(m, seed, cr=1.0):
    mc = copy.deepcopy(m); torch.manual_seed(seed); np.random.seed(seed)
    NeuralTeleportationModel(mc, input_shape=(1,3,224,224)).random_teleport(cob_range=cr)
    return mc.eval()

for arch in ("resnet152", "densenet121", "googlenet"):
    m = load_pretrained_cob(arch, "cpu"); t = teleport(m, 0)
    eo = PenultimateExtractor(m, arch); h  = eo.extract(m, X).double(); eo.remove()
    et = PenultimateExtractor(t, arch); ht = et.extract(t, X).double(); et.remove()

    num = (ht*h).sum(0); den = (h*h).sum(0)
    live = den > 0
    tau = torch.where(live, num/den.clamp(min=1e-300), torch.ones_like(den))
    resid = (ht - tau*h).norm().item() / ht.norm().item()
    D = h.shape[1]
    measured  = ((ht-h).norm(dim=1)/np.sqrt(D)).mean().item()
    predicted = (((tau-1)*h).norm(dim=1)/np.sqrt(D)).mean().item()
    rmsh      = (h.norm(dim=1)/np.sqrt(D)).mean().item()
    print(f"\n=== {arch} (D={D}, {int(live.sum())} live channels) ===")
    print(f"  residual of the h -> tau*h model: ||h_tp - tau*h||/||h_tp|| = {resid:.3e}")
    print(f"  tau range over live channels: [{tau[live].min():.4f}, {tau[live].max():.4f}]")
    print(f"  measured RMS drift  = {measured:.6f}")
    print(f"  closed form ||(tau-1)*h||/sqrt(D) = {predicted:.6f}   "
          f"(rel. err {abs(measured-predicted)/measured:.2e})")
    print(f"  RMS(h) = {rmsh:.6f}   drift/RMS(h) = {measured/rmsh:.4f}"
          f"   [E = 1/sqrt(3) = 0.5774 for tau ~ U(0,2) indep. of h]")
