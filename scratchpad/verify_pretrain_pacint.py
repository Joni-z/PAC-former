"""BLOCKING pre-flight for the objective experiments (AGENT.md 13.43 -> objective 2x2).

Two things have never been tested and both can silently contaminate every run:

1. `MAEPretrain` has never been built with `tokenizer_mode=pac_interaction`.
   The plumbing was only exercised on the supervised path.

2. **Leakage.** pretrain.py's leakage control is now a NO-OP for this config:
   it zeroes the coupling handed to the encoder, but `freq_mixer=attention`
   never reads coupling (FreqAttention.forward ignores it), and the place that
   *does* use coupling has moved into the frontend, where tokens are built
   BEFORE masking happens. So the guard fires in the wrong place.

   The argument that it is nevertheless safe rests on two structural
   properties -- (a) edges are restricted to i<j, so a low-band token cannot
   depend on a higher band; (b) p_i is built from UNIT-modulus phase, which
   carries no amplitude. Both are load-bearing and neither is enforced by a
   test. This file is that test: for a given mask, no VISIBLE token may have
   non-zero gradient w.r.t. a MASKED band's amplitude -- amplitude being
   exactly the reconstruction target.

Run: python scratchpad/verify_pretrain_pacint.py
"""

import sys

import torch
import yaml

sys.path.insert(0, ".")

from models.pretrain import MAEPretrain                       # noqa: E402
from models.build import build_model                          # noqa: E402
from models.frontend.triaxial import patch_pac_vector         # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


base = yaml.safe_load(open("configs/pretrain_chbmit_q2_cfm_idx.yaml"))
base.update(seq_len=1000, device="cpu")   # shorter window; n_channels must stay
                                          # 16 -- xyz SpatialPE checks it against
                                          # the dataset's real montage.
pac = dict(base, tokenizer_mode="pac_interaction", pac_token_mode="measured")

# ---------------------------------------------------------------------------
print("\n1. MAEPretrain builds and trains with the PAC tokenizer")
# ---------------------------------------------------------------------------
x = torch.randn(2, base["n_channels"], base["seq_len"])
for name, cfg in [("raw", base), ("pac_interaction", pac)]:
    torch.manual_seed(0)
    m = MAEPretrain(cfg)
    m.train()
    loss = m(x)
    loss.backward()
    nofinite = [n for n, p in m.named_parameters()
                if p.grad is not None and not torch.isfinite(p.grad).all()]
    check(f"{name}: forward+backward finite", torch.isfinite(loss).item() and not nofinite,
          f"loss={loss.item():.4f}")

torch.manual_seed(0)
mp = MAEPretrain(pac)
check("pac_interaction: frontend really is the PAC tokenizer",
      mp.frontend.tokenizer_mode == "pac_interaction" and hasattr(mp.frontend, "phase_tokenizer"))
check("pac_interaction: encode() path (used by probe/finetune) is finite",
      torch.isfinite(mp.encode(x)).all().item())

# ---------------------------------------------------------------------------
print("\n2. LEAKAGE -- no visible token may depend on a MASKED band's amplitude")
#    Built directly on the frontend so the mask pattern is fully controlled.
# ---------------------------------------------------------------------------
torch.manual_seed(0)
fe = build_model({**pac, "arch": "triaxial"}).frontend
nb, P = pac["n_bands"], base["seq_len"] // pac["patch_len"]
B, C, T = 2, 3, base["seq_len"]

ang = torch.rand(B, C, nb, T) * 6.283
amp_raw = torch.rand(B, C, nb, T) + 0.5


def tokens_from(amp):
    pu = torch.polar(torch.ones_like(ang), ang)
    return fe._interaction_tokens(pu, amp, patch_pac_vector(pu, amp, P, True))


def leak_grad(masked_bands):
    """max |d(visible tokens) / d(amplitude of masked bands)|."""
    a = amp_raw.clone().requires_grad_(True)
    tok = tokens_from(a)                                   # (B,C,nb,P,D)
    visible = [j for j in range(nb) if j not in masked_bands]
    tok[:, :, visible].sum().backward()
    return a.grad[:, :, list(masked_bands)].abs().max().item()


# (a) crossfreq: hide the whole top half -- the objective of record
top_half = set(range(nb // 2, nb))
g = leak_grad(top_half)
check(f"crossfreq mask {sorted(top_half)}: visible tokens get ZERO grad from masked amplitude",
      g < 1e-9, f"max|grad| = {g:.2e}")

# (b) random-style: a mid band masked while HIGHER bands stay visible -- the
#     harder direction, since h[j'] for j'>j does contain p[j].
g = leak_grad({3})
check("band 3 masked, bands 4..7 visible: still ZERO grad from masked amplitude",
      g < 1e-9, f"max|grad| = {g:.2e}")

# (c) positive control for the test itself: an UNMASKED band must produce grad,
#     otherwise (a) and (b) would pass even if the gradient never flowed at all.
a = amp_raw.clone().requires_grad_(True)
tok = tokens_from(a)
tok[:, :, [5]].sum().backward()
own = a.grad[:, :, 5].abs().max().item()
check("positive control: band 5's own amplitude DOES reach its own token",
      own > 1e-6, f"max|grad| = {own:.4f}")

# (d) the structural property the whole argument rests on
low_dep = leak_grad(set(range(1, nb)))                     # everything but band 0 masked
check("band 0 (lowest) depends on NO higher band's amplitude (i<j holds)",
      low_dep < 1e-9, f"max|grad| = {low_dep:.2e}")

# ---------------------------------------------------------------------------
print("\n3. New `magnitude` arm: deterministic, measured-magnitude, no phase alignment")
# ---------------------------------------------------------------------------
sup = yaml.safe_load(open("configs/pacint_tuev_measured.yaml"))
counts = {}
for mode in ("measured", "uniform", "scramble", "magnitude"):
    torch.manual_seed(0)
    counts[mode] = sum(p.numel() for p in build_model({**sup, "pac_token_mode": mode}).parameters())
check("magnitude is parameter-matched to the other arms",
      len(set(counts.values())) == 1, f"{counts}")

xs = torch.randn(2, sup["n_channels"], sup["seq_len"])
outs = {}
for mode in ("measured", "magnitude"):
    torch.manual_seed(0)
    mm = build_model({**sup, "pac_token_mode": mode})
    mm.eval()
    with torch.no_grad():
        outs[mode] = mm(xs)
check("magnitude differs from measured (the arm is live)",
      (outs["measured"] - outs["magnitude"]).abs().max().item() > 1e-4,
      f"max|diff| = {(outs['measured'] - outs['magnitude']).abs().max().item():.4f}")

torch.manual_seed(0)
fm = build_model({**sup, "pac_token_mode": "magnitude"}).frontend
a1 = fm._interaction_tokens(torch.polar(torch.ones_like(ang), ang), amp_raw,
                            patch_pac_vector(torch.polar(torch.ones_like(ang), ang), amp_raw, P, True))
a2 = fm._interaction_tokens(torch.polar(torch.ones_like(ang), ang), amp_raw,
                            patch_pac_vector(torch.polar(torch.ones_like(ang), ang), amp_raw, P, True))
check("magnitude is DETERMINISTIC (unlike scramble, which redraws every forward)",
      (a1 - a2).abs().max().item() < 1e-9, f"max|diff| = {(a1 - a2).abs().max().item():.2e}")

print("\n" + "=" * 70)
if fails:
    print(f"{BAD}  {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print(f"{OK}  all green -- safe to launch the objective 2x2")
