"""
How does the knowledge matrix M(x) = [ J(x) diag(x) | c(x) ] behave when the
network is EQUIVARIANT to a group acting on inputs?

Claim (derived):  Psi(pi(g)x) = rho(g) Psi(x)  implies
  (1) J(pi(g)x) = rho(g) J(x) pi(g)^{-1}
  (2) c(pi(g)x) = rho(g) c(x)
  (3) pi(g) = P a PERMUTATION  =>  M(Px) = rho(g) M(x) (P (+) 1)^T   [equivariant]
  (4) pi(g) = R a ROTATION     =>  M is NOT equivariant, but W_eff = [J|c] is.
"""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=3, suppress=True)

def jac_and_c(net, x):
    x = x.clone().requires_grad_(True)
    y = net(x)
    J = torch.stack([torch.autograd.grad(y[c], x, retain_graph=True)[0] for c in range(y.numel())])
    c = (y - J @ x.detach()).detach()
    return J.detach(), c, y.detach()

def KM(net, x):
    J, c, _ = jac_and_c(net, x)
    return torch.cat([J * x, c[:, None]], dim=1)

def Weff(net, x):
    J, c, _ = jac_and_c(net, x)
    return torch.cat([J, c[:, None]], dim=1)

# ---------------------------------------------------------------- permutation
# DeepSets-style: Psi(x) = W2 phi( sum_i psi(x_i) ) is permutation-INVARIANT.
# For a permutation-EQUIVARIANT map use  h_i = a*x_i + b*sum_j x_j  then a
# pointwise nonlinearity -- output permutes with the input (rho = pi = P).
d = 6
torch.manual_seed(0)
a, b = torch.randn(()), torch.randn(())
c1, c2 = torch.randn(()), torch.randn(())
def net_perm_equivariant(x):                      # R^d -> R^d, equivariant, rho = P
    h = a * x + b * x.sum()
    h = torch.tanh(h)
    return c1 * h + c2 * h.sum()

x = torch.randn(d)
perm = torch.randperm(d)
P = torch.eye(d)[perm]                            # (Px)_i = x_{perm[i]}

print("=== PERMUTATION-equivariant network ===")
print("  equivariance of Psi:      ", f"{(net_perm_equivariant(P@x) - P@net_perm_equivariant(x)).abs().max():.2e}")
Jx, cx, _ = jac_and_c(net_perm_equivariant, x)
Jpx, cpx, _ = jac_and_c(net_perm_equivariant, P@x)
print("  (1) J(Px) = P J(x) P^-1: ", f"{(Jpx - P@Jx@P.T).abs().max():.2e}")
print("  (2) c(Px) = P c(x):      ", f"{(cpx - P@cx).abs().max():.2e}")
Pd1 = torch.block_diag(P, torch.ones(1,1))        # P (+) 1  on the d+1 columns
lhs, rhs = KM(net_perm_equivariant, P@x), P @ KM(net_perm_equivariant, x) @ Pd1.T
print("  (3) M(Px) = P M(x)(P+1)^T:", f"{(lhs-rhs).abs().max():.2e}   <-- KM IS EQUIVARIANT")
print("      ||M(Px)||_F - ||M(x)||_F =",
      f"{(KM(net_perm_equivariant,P@x).norm() - KM(net_perm_equivariant,x).norm()).abs():.2e}  (norm invariant)")

# ---------------------------------------------------------------- rotation
# Psi(x) = g(||x||^2) * x  on R^2 is SO(2)-equivariant with rho = pi = R.
def net_rot_equivariant(x):
    r2 = (x*x).sum()
    return torch.tanh(r2) * x + 0.3 * torch.sin(r2) * x

th = torch.tensor(0.7)
R = torch.tensor([[torch.cos(th), -torch.sin(th)], [torch.sin(th), torch.cos(th)]])
x2 = torch.randn(2)
print("\n=== ROTATION-equivariant network (SO(2) on R^2) ===")
print("  equivariance of Psi:      ", f"{(net_rot_equivariant(R@x2) - R@net_rot_equivariant(x2)).abs().max():.2e}")
J2, c2_, _ = jac_and_c(net_rot_equivariant, x2)
J2r, c2r, _ = jac_and_c(net_rot_equivariant, R@x2)
print("  (1) J(Rx) = R J(x) R^-1: ", f"{(J2r - R@J2@R.T).abs().max():.2e}")
print("  (2) c(Rx) = R c(x):      ", f"{(c2r - R@c2_).abs().max():.2e}")
Rd1 = torch.block_diag(R, torch.ones(1,1))
lhsM, rhsM = KM(net_rot_equivariant, R@x2), R @ KM(net_rot_equivariant, x2) @ Rd1.T
lhsW, rhsW = Weff(net_rot_equivariant, R@x2), R @ Weff(net_rot_equivariant, x2) @ Rd1.T
print("  (4) M(Rx) vs R M(x)(R+1)^T:  ", f"{(lhsM-rhsM).abs().max():.3e}   <-- KM BREAKS")
print("      W_eff(Rx) vs R W_eff(R+1)^T:", f"{(lhsW-rhsW).abs().max():.2e}   <-- W_eff HOLDS")
print("      ||M(Rx)||_F =", f"{lhsM.norm():.6f}", "  ||M(x)||_F =", f"{KM(net_rot_equivariant,x2).norm():.6f}",
      " -> norm NOT invariant")
print("      why: diag(Rx) - R diag(x) R^-1 max =",
      f"{(torch.diag(R@x2) - R@torch.diag(x2)@R.T).abs().max():.3f}")

# ---------------------------------------------------------------- row sums
print("\n=== consistency: the row-sum identity respects equivariance ===")
one = torch.ones(d+1)
print("  M(Px).1 - Psi(Px):", f"{(KM(net_perm_equivariant,P@x)@one - net_perm_equivariant(P@x)).abs().max():.2e}")
print("  (P (+) 1)^T 1 = 1: ", f"{(Pd1.T@one - one).abs().max():.2e}  (so the identity is compatible)")
