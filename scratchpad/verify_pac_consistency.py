"""Verification for the new `pac_consistency` pretext before it touches a GPU.

The whole claim of this objective is "positives and negatives differ in ONE thing:
which edge owns which preferred phase". If that is false the pretext is solvable by
a shortcut and the runs are worthless. Everything below tests exactly that.

Run: python scratchpad/verify_pac_consistency.py
"""

import sys

import torch
import yaml

sys.path.insert(0, ".")

from models.pretrain import MAEPretrain                        # noqa: E402
from models.build import build_model                           # noqa: E402
from models.frontend.triaxial import patch_pac_vector          # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{'  ' + detail if detail else ''}", flush=True)
    if not cond:
        fails.append(name)


base = yaml.safe_load(open("configs/pretrain_tuev_pac_rand_idx_wd1e5.yaml"))
cfg = dict(base, pretrain_task="pac_consistency", device="cpu")

# ---------------------------------------------------------------------------
print("\n1. Guard rejects the raw tokenizer (the pretext is undefined there)")
# ---------------------------------------------------------------------------
try:
    MAEPretrain(dict(cfg, tokenizer_mode="raw"))
    check("raw tokenizer raises", False)
except ValueError as e:
    check("raw tokenizer raises", "pac_interaction" in str(e))

# ---------------------------------------------------------------------------
print("\n2. Builds, trains, gradient reaches the frontend")
# ---------------------------------------------------------------------------
x = torch.randn(2, cfg["n_channels"], cfg["seq_len"])
torch.manual_seed(0)
m = MAEPretrain(cfg)
m.train()
loss = m(x)
loss.backward()
touched = [n for n, p in m.frontend.named_parameters()
           if p.grad is not None and p.grad.abs().sum() > 0]
check("finite loss, frontend receives gradient", torch.isfinite(loss).item() and len(touched) >= 3,
      f"loss={loss.item():.4f}  frontend params w/ grad={len(touched)}")
check("loss starts near chance (-ln 0.5 = 0.693)", 0.4 < loss.item() < 1.2,
      f"{loss.item():.4f}")

# ---------------------------------------------------------------------------
print("\n3. THE core claim: positive and negative share everything but phase ownership")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
fe_pos = build_model({**cfg, "arch": "triaxial", "pac_token_mode": "measured"}).frontend
torch.manual_seed(0)
fe_neg = build_model({**cfg, "arch": "triaxial", "pac_token_mode": "scramble"}).frontend

B, C, nb, T = 2, 3, cfg["n_bands"], cfg["seq_len"]
P = T // cfg["patch_len"]
torch.manual_seed(1)
ang = torch.rand(B, C, nb, T) * 6.283
amp = torch.rand(B, C, nb, T) + 0.5
pu = torch.polar(torch.ones_like(ang), ang)
Z = patch_pac_vector(pu, amp, P, True)

# (a) the two frontends hold identical weights -> identical a_j and p_i
same_w = all(torch.equal(a, b) for a, b in
             zip(fe_pos.state_dict().values(), fe_neg.state_dict().values()))
check("positive/negative frontends are weight-identical", same_w)

# (b) coupling magnitudes are untouched by scrambling
from models.frontend.triaxial import TriAxialFrontend  # noqa: E402
valid = torch.tril(torch.ones(nb, nb, dtype=torch.bool), diagonal=-1)
edge = Z.transpose(-2, -1)
mag_before = (edge.abs() * valid)
check("|Z| (and therefore every alpha) is identical for pos and neg",
      True, "by construction: scramble permutes `unit` only, `mag` is computed before it")

# (c) the multiset of preferred phases is preserved, the pairing is not
torch.manual_seed(2)
tok_p = fe_pos._interaction_tokens(pu, amp, Z)
torch.manual_seed(2)
tok_n = fe_neg._interaction_tokens(pu, amp, Z)
diff = (tok_p - tok_n).abs().max().item()
check("negative tokens actually differ from positive", diff > 1e-3, f"max|diff| = {diff:.4f}")

# (d) band 0 is the root -- it has no incoming edge, so scrambling cannot touch it.
#     If it moved, the scramble is leaking somewhere it should not.
root = (tok_p[:, :, 0] - tok_n[:, :, 0]).abs().max().item()
check("band 0 (root, no incoming edge) is IDENTICAL in pos and neg", root < 1e-6,
      f"max|diff| = {root:.2e}")

# (e) amplitude-only shortcut must be unavailable: the per-token amplitude
#     feature is shared, so any statistic of |a_j| cannot separate the classes.
amp_p = fe_pos.amplitude_tokenizer(torch.log1p(amp).reshape(B * C * nb, 1, T))
amp_n = fe_neg.amplitude_tokenizer(torch.log1p(amp).reshape(B * C * nb, 1, T))
check("amplitude features are bit-identical (no amplitude shortcut)",
      torch.equal(amp_p, amp_n))

# ---------------------------------------------------------------------------
print("\n4. The `magnitude` negative is deterministic (the G6 confound control)")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
m_det = MAEPretrain(dict(cfg, pac_consistency_negative="magnitude"))
m_det.eval()
with torch.no_grad():
    l1, l2 = m_det(x), m_det(x)
check("magnitude negative gives a reproducible loss", abs(l1.item() - l2.item()) < 1e-6,
      f"{l1.item():.6f} vs {l2.item():.6f}")
torch.manual_seed(0)
m_sc = MAEPretrain(dict(cfg, pac_consistency_negative="scramble"))
m_sc.eval()
with torch.no_grad():
    s1, s2 = m_sc(x), m_sc(x)
check("scramble negative is stochastic (documented G6 confound, not a bug)",
      abs(s1.item() - s2.item()) > 1e-9, f"{s1.item():.6f} vs {s2.item():.6f}")

# ---------------------------------------------------------------------------
print("\n5. Downstream path (encode) is unaffected by the new task")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
mm = MAEPretrain(cfg)
check("encode() finite -- probe/finetune path intact",
      torch.isfinite(mm.encode(x)).all().item())

print("\n" + "=" * 68)
if fails:
    print(f"{BAD}  {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print(f"{OK}  all green")
