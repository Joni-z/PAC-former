"""Smoke the xyz SpatialPE (AGENT.md sec 13.23 A): backward compat + xyz path
on both the supervised backbone and the MAE pretrainer, forward+backward."""
import torch, torch.nn as nn, yaml, copy
from models.build import build_model, TriAxialPACFormer
from models.pretrain import MAEPretrain
from models.montage import coords_for

base = yaml.safe_load(open("configs/sleep_v2_coupling_nb32.yaml"))
base["device"] = "cpu"

# (1) backward compat: no spatial_pe key -> learned index embedding
m0 = build_model(base)
assert m0.spatial_pe.mlp is None and m0.spatial_pe.emb is not None, "compat broke"
print("(1) no spatial_pe key -> learned index embedding  OK")

# (2) xyz path builds with real coords, coords match n_channels
cfg = copy.deepcopy(base); cfg["spatial_pe"] = "xyz"       # sleepedf, 2 ch
m1 = build_model(cfg)
assert m1.spatial_pe.emb is None and m1.spatial_pe.mlp is not None, "xyz not active"
assert m1.spatial_pe.coords.shape == (2, 6), m1.spatial_pe.coords.shape
print(f"(2) spatial_pe=xyz active, coords {tuple(m1.spatial_pe.coords.shape)}  OK")

# forward/backward supervised
B, C, T = 2, cfg["n_channels"], cfg["seq_len"]
x = torch.randn(B, C, T); y = torch.randint(0, cfg["num_classes"], (B,))
loss = nn.CrossEntropyLoss()(m1(x), y); loss.backward()
g = m1.spatial_pe.mlp[0].weight.grad
assert g is not None and torch.isfinite(g).all(), "no/NaN grad into xyz MLP"
print(f"(3) supervised fwd/bwd, xyz-MLP finite grad, loss={loss.item():.4f}  OK")

# (4) MAE pretrainer with xyz + cf_mixed mask
pcfg = copy.deepcopy(cfg); pcfg["mask_mode"] = "mixed"; pcfg["mixed_p"] = 0.5
mae = MAEPretrain(pcfg)
assert mae.spatial_pe.mlp is not None, "pretrain xyz not active"
out = mae(x)
lo = out[0] if isinstance(out, tuple) else out
lo.backward()
print(f"(4) MAEPretrain xyz + cf_mixed fwd/bwd, loss={float(lo):.4f}  OK")

# (5) 16-ch montage (tuab) has the right shape + channel-count guard fires
assert coords_for("tuab").shape == (16, 6)
try:
    bad = copy.deepcopy(base); bad["spatial_pe"] = "xyz"; bad["dataset"] = "tuab"  # 2 vs 16
    build_model(bad); raise SystemExit("guard did NOT fire")
except ValueError as e:
    print(f"(5) 16-ch tuab montage OK; channel-mismatch guard fires: {str(e)[:60]}...")

print("ALL GREEN")
