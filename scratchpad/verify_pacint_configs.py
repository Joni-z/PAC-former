"""Independent verification of the three pacint_tuev_* arms before they get a GPU.

Covers what a config-only fix must not break, plus the two load-bearing claims in
AGENT.md 13.43 (parameter matching, gauge invariance), re-derived here rather than
trusting smoke_pac_interaction.py.

Run: python scratchpad/verify_pacint_configs.py
"""

import sys

import torch
import wandb
import yaml

sys.path.insert(0, ".")

from models.build import build_model            # noqa: E402

OK, BAD = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
fails = []


def check(name, cond, detail=""):
    print(f"  [{OK if cond else BAD}] {name}{'  ' + detail if detail else ''}")
    if not cond:
        fails.append(name)


ARMS = ["measured", "uniform", "scramble"]
cfgs = {a: yaml.safe_load(open(f"configs/pacint_tuev_{a}.yaml")) for a in ARMS}
base = yaml.safe_load(open("configs/ours_scratch_tuev_bandidx.yaml"))

# ---------------------------------------------------------------------------
print("\n1. train.py startup path (the KeyError that would have killed all 3 jobs)")
# ---------------------------------------------------------------------------
for a, cfg in cfgs.items():
    try:
        # verbatim train.py:83 -- the .get() default is evaluated EAGERLY
        name = cfg.get("wandb_run_name", f"{cfg['dataset']}-{cfg['mixer']}")
        _ = f"[{cfg['mixer']}]"                       # verbatim train.py:101
        ok, detail = True, name
    except KeyError as e:
        ok, detail = False, f"KeyError {e}"
    check(f"{a}: wandb.init + param print", ok, detail)

# ---------------------------------------------------------------------------
print("\n2. Every arm differs from the baseline ONLY in the intended keys")
# ---------------------------------------------------------------------------
allowed = {"tokenizer_mode", "pac_token_mode", "wandb_run_name"}
for a, cfg in cfgs.items():
    diff = {k for k in set(cfg) | set(base) if cfg.get(k) != base.get(k)}
    check(f"{a}: config delta ⊆ {sorted(allowed)}", diff <= allowed, str(sorted(diff)))

# ---------------------------------------------------------------------------
print("\n3. Strict parameter matching vs the raw-token baseline")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
n_raw = sum(p.numel() for p in build_model(base).parameters())
for a, cfg in cfgs.items():
    torch.manual_seed(0)
    n = sum(p.numel() for p in build_model(cfg).parameters())
    check(f"{a}: {n:,} params == raw {n_raw:,}", n == n_raw)

# ---------------------------------------------------------------------------
print("\n4. Forward/backward finite on real-shaped input")
# ---------------------------------------------------------------------------
x = torch.randn(2, base["n_channels"], base["seq_len"])
y = torch.randint(0, base["num_classes"], (2,))
for a, cfg in cfgs.items():
    torch.manual_seed(0)
    m = build_model(cfg)
    m.train()
    loss = torch.nn.functional.cross_entropy(m(x), y)
    loss.backward()
    fe = m.frontend
    touched = [n for n, p in fe.named_parameters()
               if p.grad is not None and p.grad.abs().sum() > 0]
    check(f"{a}: finite loss/grad, frontend reached",
          torch.isfinite(loss) and len(touched) >= 3, f"loss={loss.item():.4f} "
          f"frontend params w/ grad: {len(touched)}")

# ---------------------------------------------------------------------------
print("\n5. Gauge invariance: shift each band's phase reference by its own delta")
#    measured MUST be invariant; uniform/scramble must NOT be (they are controls).
# ---------------------------------------------------------------------------
from models.frontend.triaxial import patch_pac_vector      # noqa: E402

B, C, nb, T, P = 2, 3, 8, 1000, 5
torch.manual_seed(0)
phase_ang = torch.rand(B, C, nb, T) * 6.283
amp = torch.rand(B, C, nb, T) + 0.5
delta = torch.rand(nb).view(1, 1, nb, 1) * 6.283           # per-band gauge shift


def tokens_for(mode, shift):
    torch.manual_seed(0)                                    # identical weights
    fe = build_model({**cfgs[mode], "device": "cpu"}).frontend
    ang = phase_ang + (delta if shift else 0.0)
    pu = torch.polar(torch.ones_like(ang), ang)
    torch.manual_seed(123)                                  # fix scramble's draw
    return fe._interaction_tokens(pu, amp, patch_pac_vector(pu, amp, P, True))


#    Band 0 is EXCLUDED: it is the root of the directed hierarchy and gets its own
#    analytic token verbatim (_pac_interaction, the `aligned_phase[..., 0, :] = ...`
#    line), so it is gauge-covariant by construction and invariance is not claimed
#    for it. Checking it was my error; the repo's own smoke test slices [..., 1:, :].
for mode in ARMS:
    t0, t1 = tokens_for(mode, False), tokens_for(mode, True)
    cross = (t0[:, :, 1:] - t1[:, :, 1:]).abs().max().item()
    root = (t0[:, :, 0] - t1[:, :, 0]).abs().max().item()
    scale = t0.abs().max().item()
    if mode == "measured":
        check("measured: bands 1..n are phase-reference invariant", cross < 1e-4,
              f"max|diff| = {cross:.2e}")
        # Positive control for the test: if the gauge shift never reached the
        # tokens, the check above would pass vacuously. Band 0 must move.
        check("measured: band 0 (the root) DOES move -> shift really applied",
              root / scale > 1e-3, f"max|diff| = {root:.4f}")
    else:
        check(f"{mode}: bands 1..n are NOT invariant (control is a real control)",
              cross / scale > 1e-3, f"max|diff| = {cross:.4f} ({cross/scale:.1%})")

# ---------------------------------------------------------------------------
print("\n6. Band-order risk (AGENT.md 13.43-G1): are sinc bands still low->high?")
# ---------------------------------------------------------------------------
torch.manual_seed(0)
hz = build_model(cfgs["measured"]).frontend.band_hz()[:, 0].detach()
check("sinc centre frequencies are ascending at init",
      bool((hz[1:] > hz[:-1]).all()),
      " ".join(f"{v:.1f}" for v in hz.tolist()))
print("     (init only -- G1 says this MUST be re-checked on the trained model)")

print("\n" + "=" * 66)
if fails:
    print(f"{BAD}  {len(fails)} check(s) failed: {fails}")
    sys.exit(1)
print(f"{OK}  all checks green -- the three pending jobs will start cleanly")
