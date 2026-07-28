"""CPU verification for FreqMIProduct / FreqMITopology and their shuffle controls.

The bar here is NOT "it runs". FreqCoherenceGate ran on five datasets while its gate
sat at its 0 init, so those runs were plain attention and the recorded conclusion was
void (AGENT.md sec. 13.10a). Every check below exists to make that class of silent
no-op impossible to miss:

  1  uniform coupling  -> output == weight-tied plain attention   (floor is correct)
  2  real coupling     -> output DIFFERS from the uniform case    (prior is ACTIVE)
  3  top-k             -> masked sources have provably zero influence (gradient test)
  4  shuffle           -> preserves the value multiset exactly
  5  shuffle           -> destroys the pairing, and redraws every forward
  6  dead channel      -> uniform fallback, no NaN
  7  gradients finite for every parameter, both mixers
  8  existing configs bit-identical after the mi_k plumbing change

Run: python scratchpad/smoke_mi_mixers.py
"""

import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from models.triaxial import (            # noqa: E402
    FreqAttention, FreqMIProduct, FreqMITopology, FREQ_MIXERS, _mi_prepare,
)

torch.manual_seed(0)
OK = "\033[92mPASS\033[0m"
BAD = "\033[91mFAIL\033[0m"
failures = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        failures.append(name)


M, L, D, H = 64, 8, 128, 4          # M = B*C*P, L = n_bands
x = torch.randn(M, L, D)
# Realistic coupling: non-negative, heavy-tailed, a few strong edges per row.
coupling = (torch.rand(M, L, L) ** 3) * 0.05


# --------------------------------------------------------------------------- #
print("\n1. FreqMIProduct: uniform coupling must reproduce plain attention")
# --------------------------------------------------------------------------- #
prod = FreqMIProduct(D, n_heads=H)
attn = FreqAttention(D, n_heads=H)
attn.mha.qkv.load_state_dict(prod.qkv.state_dict())
attn.mha.out.load_state_dict(prod.out.state_dict())

with torch.no_grad():
    out_uniform = prod(x, torch.ones(M, L, L))
    out_attn = attn(x)
gap = (out_uniform - out_attn).abs().max().item()
check("uniform coupling == tied plain attention", gap < 1e-5, f"max|diff| = {gap:.2e}")

# --------------------------------------------------------------------------- #
print("\n2. FreqMIProduct: REAL coupling must actually change the output")
#    (this is the check FreqCoherenceGate would have failed)
# --------------------------------------------------------------------------- #
with torch.no_grad():
    out_real = prod(x, coupling)
delta = (out_real - out_uniform).abs().max().item()
rel = delta / out_uniform.abs().max().item()
check("real coupling changes the output", delta > 1e-3, f"max|diff| = {delta:.4f} ({rel:.1%})")
check("mi_spread diagnostic is non-zero", prod.last_mi_spread > 1e-3,
      f"spread = {prod.last_mi_spread:.4f}")
check("uniform coupling gives spread 0",
      (prod(x, torch.ones(M, L, L)) is not None) and prod.last_mi_spread < 1e-6,
      f"spread = {prod.last_mi_spread:.2e}")

# --------------------------------------------------------------------------- #
print("\n3. FreqMITopology: masked sources must have ZERO influence")
# --------------------------------------------------------------------------- #
K = 3
topo = FreqMITopology(D, n_heads=H, mi_k=K)
c_t = _mi_prepare(coupling, False)
keep = torch.zeros_like(c_t, dtype=torch.bool).scatter_(
    -1, c_t.topk(K, dim=-1).indices, True)

per_row = keep.sum(-1)
check("exactly k sources kept per target band",
      bool((per_row == K).all()), f"k = {K}, rows all == {per_row.unique().tolist()}")

with torch.no_grad():
    out_topo = topo(x, coupling)
check("no NaN/Inf in top-k output", bool(torch.isfinite(out_topo).all()))
check("kept_frac diagnostic correct", abs(topo.last_kept_frac - K / L) < 1e-6,
      f"{topo.last_kept_frac:.4f} vs expected {K/L:.4f}")

# A masked source must be unable to reach the target through the key/value path.
# The target's OWN token is excluded from this claim: even when band j is absent from
# its own top-k, x[j] still forms q[j], which legitimately re-weights the sources that
# ARE kept. So "zero influence" means zero influence *as a source*, not "x[j] is inert".
tgt = 0
masked_as_source = ~keep[0, tgt].clone()
masked_as_source[tgt] = False                                 # exclude the query path

xg = x.clone().requires_grad_(True)
topo(xg, coupling)[0, tgt].sum().backward()
grad_per_band = xg.grad[0].abs().sum(-1)                      # (L,)
worst = grad_per_band[masked_as_source].max().item()
check(f"target {tgt}: masked sources {masked_as_source.nonzero().flatten().tolist()} "
      f"get zero gradient", worst < 1e-9, f"max grad = {worst:.2e}")
check("kept sources DO get gradient (mask is not blocking everything)",
      grad_per_band[keep[0, tgt]].min().item() > 1e-9)

# Direct causal test on a masked source that is not the target itself.
src = int(masked_as_source.nonzero().flatten()[0])
x2 = x.clone()
x2[0, src] += 10.0
with torch.no_grad():
    out2 = topo(x2, coupling)
moved = (out2[0, tgt] - out_topo[0, tgt]).abs().max().item()
check(f"perturbing masked source {src} does not move target {tgt}",
      moved < 1e-6, f"max|diff| = {moved:.2e}")

# Positive control for the test itself: a KEPT source must move the target, otherwise
# the check above would pass even if the mixer ignored its inputs entirely.
kept_src = int(keep[0, tgt].nonzero().flatten()[0])
x3 = x.clone()
x3[0, kept_src] += 10.0
with torch.no_grad():
    out3 = topo(x3, coupling)
moved_kept = (out3[0, tgt] - out_topo[0, tgt]).abs().max().item()
check(f"perturbing KEPT source {kept_src} DOES move target {tgt}",
      moved_kept > 1e-3, f"max|diff| = {moved_kept:.4f}")

# --------------------------------------------------------------------------- #
print("\n4-5. Shuffle control: matched distribution, destroyed pairing, fresh draw")
# --------------------------------------------------------------------------- #
torch.manual_seed(1)
s1 = _mi_prepare(coupling, True)
torch.manual_seed(2)
s2 = _mi_prepare(coupling, True)
plain = _mi_prepare(coupling, False)

sorted_gap = (s1.reshape(M, -1).sort(-1).values
              - plain.reshape(M, -1).sort(-1).values).abs().max().item()
check("shuffle preserves the value multiset exactly", sorted_gap < 1e-7,
      f"max|diff| of sorted values = {sorted_gap:.2e}")
check("shuffle destroys the pairing", (s1 - plain).abs().max().item() > 1e-6)
check("shuffle redraws every call (not a fixed, invertible permutation)",
      (s1 - s2).abs().max().item() > 1e-6)

for name in ("mi_product_shuffle", "mi_topk_shuffle"):
    mx = FREQ_MIXERS[name](D, n_heads=H)
    with torch.no_grad():
        a, b = mx(x, coupling), mx(x, coupling)
    check(f"{name}: two forwards differ (control is live)",
          (a - b).abs().max().item() > 1e-6)

# --------------------------------------------------------------------------- #
print("\n6. Dead channel (all-zero coupling row) -> uniform fallback, no NaN")
# --------------------------------------------------------------------------- #
c_dead = coupling.clone()
c_dead[0] = 0.0                                    # electrode/patch 0 fully flat
with torch.no_grad():
    o_dead = FreqMIProduct(D, n_heads=H)(x, c_dead)
check("no NaN with an all-zero coupling matrix", bool(torch.isfinite(o_dead).all()))

pd = FreqMIProduct(D, n_heads=H)
ad = FreqAttention(D, n_heads=H)
ad.mha.qkv.load_state_dict(pd.qkv.state_dict())
ad.mha.out.load_state_dict(pd.out.state_dict())
with torch.no_grad():
    fallback_gap = (pd(x, c_dead)[0] - ad(x)[0]).abs().max().item()
check("dead row falls back to exactly plain attention", fallback_gap < 1e-5,
      f"max|diff| = {fallback_gap:.2e}")

# --------------------------------------------------------------------------- #
print("\n7. Gradients finite for every parameter of all four variants")
# --------------------------------------------------------------------------- #
for name in ("mi_product", "mi_product_shuffle", "mi_topk", "mi_topk_shuffle"):
    mx = FREQ_MIXERS[name](D, n_heads=H)
    mx.zero_grad()
    mx(x, coupling).sum().backward()
    bad = [n for n, p in mx.named_parameters()
           if p.grad is None or not torch.isfinite(p.grad).all() or p.grad.abs().sum() == 0]
    check(f"{name}: all params get finite non-zero grad", not bad, str(bad) if bad else "")

print("\n" + ("=" * 60))
if failures:
    print(f"{BAD}  {len(failures)} check(s) failed: {failures}")
    sys.exit(1)
print(f"{OK}  all mixer-level checks green")
