"""Verify pretrain.py Phase-2 has a working full-finetune mode alongside the
default linear probe (AGENT.md sec 13.24 item 3). Checks grad flow: linear ->
only fc trains, encoder frozen; finetune -> encoder trains too."""
import torch, torch.nn as nn, yaml, copy
from models.pretrain import MAEPretrain
from pretrain import Probe, probe_epoch

cfg = yaml.safe_load(open("configs/pretrain_sleep_cf_mixed.yaml"))
cfg["device"] = "cpu"
mae = MAEPretrain(cfg)
B, C, T = 2, cfg["n_channels"], cfg["seq_len"]
X = torch.randn(B, C, T); y = torch.randint(0, cfg["num_classes"], (B,))
crit = nn.CrossEntropyLoss()

def enc_grad_norm(p):
    g = [q.grad.abs().sum() for q in p.mae.parameters() if q.grad is not None]
    return float(sum(g)) if g else 0.0

# linear probe (default): encoder must get NO grad
lp = Probe(copy.deepcopy(mae), cfg["num_classes"], finetune=False)
opt = torch.optim.Adam(lp.fc.parameters(), lr=1e-3)
probe_epoch(lp, [(X, y)], "cpu", crit, opt)
assert enc_grad_norm(lp) == 0.0, "linear probe leaked grad into encoder!"
assert lp.fc.weight.grad is not None, "fc got no grad"
print("(1) linear probe: encoder frozen (0 grad), fc trains  OK")

# finetune: encoder MUST get grad
ft = Probe(copy.deepcopy(mae), cfg["num_classes"], finetune=True)
opt = torch.optim.Adam(ft.parameters(), lr=1e-4)
probe_epoch(ft, [(X, y)], "cpu", crit, opt)
g = enc_grad_norm(ft)
assert g > 0.0, "finetune did NOT propagate grad into encoder!"
print(f"(2) finetune: encoder trains end-to-end (grad sum {g:.3f})  OK")

# val pass (opt=None) must not error and produce finite loss
loss, logits, yy = probe_epoch(ft, [(X, y)], "cpu", crit, None)
assert torch.isfinite(torch.tensor(loss)), "val loss not finite"
print(f"(3) eval pass loss={loss:.4f}, logits {logits.shape}  OK")
print("ALL GREEN")
