"""Verification for two new pieces before they touch a GPU:

1. `interaction_mode="concat"` -- the SleepPACNet-style fusion control
   AGENT.md 13.43-G calls "the most important missing baseline". Same
   ingredients as `measured` (a_j, Re/Im of aligned_phase), combined by a
   learned Linear instead of the forced product.
2. The TUAB/TUSZ expansion configs (`pacint_{tuab,tusz}_measured.yaml`) --
   same discipline as scratchpad/verify_pacint_configs.py: exercise the real
   entry-point path, not just build_model, because config-schema errors are
   invisible to unit tests that construct their own config dicts.

Run: python scratchpad/verify_concat_and_expansion.py
"""

import sys

import torch
import yaml

sys.path.insert(0, ".")

from models.build import build_model            # noqa: E402
from models.frontend.triaxial import patch_pac_vector  # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


base_tuev = yaml.safe_load(open("configs/ours_scratch_tuev_bandidx.yaml"))
measured = yaml.safe_load(open("configs/pacint_tuev_measured.yaml"))
concat = yaml.safe_load(open("configs/pacint_tuev_concat.yaml"))
tuab_pacint = yaml.safe_load(open("configs/pacint_tuab_measured.yaml"))
tuab_base = yaml.safe_load(open("configs/ours_scratch_tuab_bandidx.yaml"))
tusz_pacint = yaml.safe_load(open("configs/pacint_tusz_measured.yaml"))
tusz_base = yaml.safe_load(open("configs/ours_scratch_tusz_bandidx_se.yaml"))

# ---------------------------------------------------------------------------
print("\n1. train.py startup path for every new config")
# ---------------------------------------------------------------------------
for name, cfg in [("concat", concat), ("tuab_measured", tuab_pacint),
                   ("tusz_measured", tusz_pacint)]:
    try:
        n = cfg.get("wandb_run_name", f"{cfg['dataset']}-{cfg['mixer']}")  # train.py:83
        _ = f"[{cfg['mixer']}]"                                            # train.py:101
        ok, detail = True, n
    except KeyError as e:
        ok, detail = False, f"KeyError {e}"
    check(f"{name}: wandb.init + param print", ok, detail)

# ---------------------------------------------------------------------------
print("\n2. Config deltas are exactly what's intended")
# ---------------------------------------------------------------------------
allowed_concat = {"tokenizer_mode", "pac_token_mode", "interaction_mode", "wandb_run_name"}
diff = {k for k in set(concat) | set(base_tuev) if concat.get(k) != base_tuev.get(k)}
check(f"concat: delta vs raw baseline ⊆ {sorted(allowed_concat)}",
      diff <= allowed_concat, str(sorted(diff)))

allowed_expand = {"tokenizer_mode", "pac_token_mode", "wandb_run_name"}
for name, cfg, base in [("tuab_measured", tuab_pacint, tuab_base),
                         ("tusz_measured", tusz_pacint, tusz_base)]:
    diff = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
    check(f"{name}: delta vs its raw baseline ⊆ {sorted(allowed_expand)}",
          diff <= allowed_expand, str(sorted(diff)))

# ---------------------------------------------------------------------------
print("\n3. Parameter counts -- concat is expected to have MORE (documented, not hidden)")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
n_raw = sum(p.numel() for p in build_model(base_tuev).parameters())
torch.manual_seed(0)
n_measured = sum(p.numel() for p in build_model(measured).parameters())
torch.manual_seed(0)
n_concat = sum(p.numel() for p in build_model(concat).parameters())
check("raw == measured (both 1,635,734)", n_raw == n_measured == 1_635_734,
      f"raw={n_raw:,} measured={n_measured:,}")
extra = n_concat - n_measured
# concat_proj = Linear(3*64 -> 128) = 192*128 + 128
expected_extra = 3 * 64 * 128 + 128
check("concat has EXACTLY the concat_proj's extra params, nothing more",
      extra == expected_extra,
      f"concat={n_concat:,} measured={n_measured:,} extra={extra:,} "
      f"(expected {expected_extra:,}, {extra/n_measured:.2%} of measured)")

for name, cfg, base_cfg in [("tuab_measured", tuab_pacint, tuab_base),
                            ("tusz_measured", tusz_pacint, tusz_base)]:
    torch.manual_seed(0)
    nb = sum(p.numel() for p in build_model(base_cfg).parameters())
    torch.manual_seed(0)
    nm = sum(p.numel() for p in build_model(cfg).parameters())
    check(f"{name}: params == its raw baseline ({nb:,})", nb == nm, f"got {nm:,}")

# ---------------------------------------------------------------------------
print("\n4. Forward/backward finite, gradient reaches concat_proj specifically")
# ---------------------------------------------------------------------------
x = torch.randn(2, base_tuev["n_channels"], base_tuev["seq_len"])
y = torch.randint(0, base_tuev["num_classes"], (2,))
torch.manual_seed(0)
m = build_model(concat)
m.train()
loss = torch.nn.functional.cross_entropy(m(x), y)
loss.backward()
proj = m.frontend.concat_proj
check("concat: finite loss", torch.isfinite(loss).item(), f"loss={loss.item():.4f}")
check("concat_proj.weight has finite non-zero grad",
      proj.weight.grad is not None and torch.isfinite(proj.weight.grad).all()
      and proj.weight.grad.abs().sum() > 0)

for name, cfg, ds_cfg in [("tuab_measured", tuab_pacint, tuab_base),
                          ("tusz_measured", tusz_pacint, tusz_base)]:
    xx = torch.randn(2, ds_cfg["n_channels"], ds_cfg["seq_len"])
    yy = torch.randint(0, ds_cfg["num_classes"], (2,))
    torch.manual_seed(0)
    mm = build_model(cfg)
    mm.train()
    ll = torch.nn.functional.cross_entropy(mm(xx), yy)
    ll.backward()
    touched = [n for n, p in mm.frontend.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    check(f"{name}: finite loss, frontend reached", torch.isfinite(ll).item()
          and len(touched) >= 3, f"loss={ll.item():.4f} frontend params w/ grad={len(touched)}")

# ---------------------------------------------------------------------------
print("\n5. Gauge invariance -- concat inherits it from aligned_phase (bands 1..n)")
#    Root band 0 is NOT claimed invariant (same as product mode).
# ---------------------------------------------------------------------------
B, C, nb, T, P = 2, 3, 8, 1000, 5
torch.manual_seed(0)
ang = torch.rand(B, C, nb, T) * 6.283
amp = torch.rand(B, C, nb, T) + 0.5
delta = torch.rand(nb).view(1, 1, nb, 1) * 6.283


def tokens_for(cfg, shift):
    torch.manual_seed(0)
    fe = build_model(cfg).frontend
    a = ang + (delta if shift else 0.0)
    pu = torch.polar(torch.ones_like(a), a)
    return fe._interaction_tokens(pu, amp, patch_pac_vector(pu, amp, P, True))


t0, t1 = tokens_for(concat, False), tokens_for(concat, True)
cross = (t0[:, :, 1:] - t1[:, :, 1:]).abs().max().item()
root = (t0[:, :, 0] - t1[:, :, 0]).abs().max().item()
scale = t0.abs().max().item()
check("concat: bands 1..n are phase-reference invariant", cross < 1e-4,
      f"max|diff| = {cross:.2e}")
check("concat: band 0 (root) DOES move -> shift really applied",
      root / scale > 1e-3, f"max|diff| = {root:.4f}")

print("\n" + "=" * 66)
if fails:
    print(f"{BAD}  {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print(f"{OK}  all checks green")
