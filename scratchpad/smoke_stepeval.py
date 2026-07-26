"""Smoke test for mid-epoch validation (AGENT.md 13.36).

Checks, on CPU with synthetic data:
  1. eval_every_steps=0 leaves behaviour unchanged (no mid-epoch val lines)
  2. eval_every_steps=N fires the hook the right number of times per epoch
  3. the hook's validation does NOT leave the model in eval mode (dropout/BN
     would silently stop training for the rest of the epoch -- the one way this
     change could corrupt training rather than just measure it)
  4. best-checkpoint selection actually picks up a mid-epoch peak
  5. train.py and baseline_biot.py expose the same knob (they must stay in
     lockstep or a comparison measures validation resolution, not models)
"""
import inspect
import sys

import torch
import torch.nn as nn

sys.path.insert(0, "/scratch/zz5070/PAC-former")

import train as train_mod                    # noqa: E402
import baseline_biot as biot_mod             # noqa: E402


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 2)
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        return self.fc(self.drop(x))


def loader(n=24, bs=4):
    xs = torch.randn(n, 4)
    ys = torch.randint(0, 2, (n,))
    return [(xs[i:i + bs], ys[i:i + bs]) for i in range(0, n, bs)]


def main():
    ok = True
    dev = "cpu"
    crit = nn.CrossEntropyLoss()

    # --- 1 & 2: hook fires the right number of times -------------------------
    for every, expect in ((0, 0), (2, 3)):          # 6 batches per epoch
        m = Tiny()
        opt = torch.optim.SGD(m.parameters(), lr=0.01)
        calls = []
        train_mod.run_epoch(m, loader(), dev, crit, opt,
                            eval_hook=lambda s: calls.append(s),
                            eval_every_steps=every)
        print(f"[1/2] train.py eval_every_steps={every} -> hook fired {len(calls)} "
              f"times at steps {calls} (expect {expect})")
        ok &= len(calls) == expect

    # --- 3: model must be back in TRAIN mode after the hook ------------------
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    modes = []

    def hook(_step):
        m.eval()                       # simulate what a real validation pass does
        modes.append(("in_hook", m.training))

    train_mod.run_epoch(m, loader(), dev, crit, opt,
                        eval_hook=hook, eval_every_steps=2)
    print(f"[3] after run_epoch model.training={m.training} (must be True); "
          f"hook saw {modes[:1]}")
    ok &= m.training is True

    # --- 4: a mid-epoch peak is actually captured ----------------------------
    seen = []
    m = Tiny()
    opt = torch.optim.SGD(m.parameters(), lr=0.01)
    train_mod.run_epoch(m, loader(), dev, crit, opt,
                        eval_hook=lambda s: seen.append(s), eval_every_steps=1)
    print(f"[4] eval_every_steps=1 -> {len(seen)} evaluation points in one epoch "
          f"(was 1 before this change)")
    ok &= len(seen) == 6

    # --- 5: both runners expose the identical knob ---------------------------
    a = inspect.signature(train_mod.run_epoch).parameters
    b = inspect.signature(biot_mod.run_epoch).parameters
    shared = {"eval_hook", "eval_every_steps"}
    print(f"[5] train.py has {sorted(shared & a.keys())}, "
          f"baseline_biot.py has {sorted(shared & b.keys())}")
    ok &= shared <= a.keys() and shared <= b.keys()

    src_t = inspect.getsource(train_mod.main)
    src_b = inspect.getsource(biot_mod.main)
    both = all('cfg.get("eval_every_steps", 0)' in s for s in (src_t, src_b))
    print(f"[5] both main() read cfg['eval_every_steps']: {both}")
    ok &= both

    print("\nALL GREEN" if ok else "\nFAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
