"""Generate the substitution-hypothesis validation matrix (AGENT.md 13.42).

The claim under test (§13.39): a frequency prior injected from the OBJECTIVE
(cf_mixed) or from the ARCHITECTURE (hz BandPE) is a SUBSTITUTE — either alone
recovers the benefit, neither collapses, and both together interfere.

Evidence so far is ONE dataset at ONE seed, and the whole research direction rests
on it. This generates the replication: 2x2 (mask_mode x band_pe) across seeds and
datasets, with every other knob held identical to the original CHB-MIT cells so the
new numbers are directly comparable to the seed-0 ones already collected.

    python scripts/gen_substitution_matrix.py          # write configs, print plan
"""

import os

CFG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")

# Held identical to configs/pretrain_chbmit_q2_rand_idx.yaml (the seed-0 cells), so
# replications are comparable to what is already measured.
COMMON = """n_bands: 8
d_model: 128
depth: 6
dropout: 0.2
kernel_size: 201
patch_len: 200
n_heads: 4
batch_size: 32
num_workers: 8
device: cuda
arch: triaxial
freq_mixer: attention
spatial_pe: xyz
mask_ratio: 0.5
mixed_p: 0.5
pretrain_epochs: 15
probe_epochs: 10
probe_mode: finetune
finetune_lr: 0.0001
eval_every_steps: 1500
lr: 0.0001
weight_decay: 0.0001
probe_lr: 0.001
"""

DATASETS = {
    "chbmit": dict(root="/scratch/zz5070/PAC-former/chb_mit/processed",
                   seq_len=2000, num_classes=2),
    "tusz":   dict(root="/scratch/zz5070/PAC-former/tuh_eeg/tuh_eeg_seizure/v2.0.6/edf/processed",
                   seq_len=2000, num_classes=2),
    "tuev":   dict(root="/scratch/zz5070/PAC-former/tuh_eeg/v2.0.1/edf",
                   seq_len=1000, num_classes=6),
}

# the 2x2: objective route x architecture route
CELLS = [("rand", "random", "index"),   # neither prior  -> expect collapse
         ("cfm",  "mixed",  "index"),   # objective only -> expect recovery
         ("randhz", "random", "hz"),    # architecture only -> expect recovery
         ("cfmhz",  "mixed",  "hz")]    # both -> expect interference


def write(ds, seed, tag, mask_mode, band_pe):
    d = DATASETS[ds]
    name = f"sub_{ds}_{tag}_s{seed}"
    body = (f"dataset: {ds}\n"
            f"data_root: {d['root']}\n"
            f"n_channels: 16\n"
            f"seq_len: {d['seq_len']}\n"
            f"sample_rate: 200\n"
            f"sampling_rate: 200\n"
            f"num_classes: {d['num_classes']}\n"
            f"seed: {seed}\n"
            f"mask_mode: {mask_mode}\n"
            f"band_pe: {band_pe}\n"
            + COMMON
            + f"wandb_run_name: {name}\n")
    with open(os.path.join(CFG_DIR, name + ".yaml"), "w") as fh:
        fh.write(body)
    return name


def main():
    # Priority order: the decisive question first (is the CHB-MIT 2x2 noise?),
    # then cheap cross-dataset replication (TUEV), then the expensive third dataset.
    plan = []
    for seed in (1, 2):                       # seed 0 on chbmit already measured
        plan += [("chbmit", seed, c) for c in CELLS]
    for seed in (0, 1, 2):                    # TUEV is ~4x cheaper per run
        plan += [("tuev", seed, c) for c in CELLS]
    plan += [("tusz", 0, c) for c in CELLS]   # third dataset, direction check

    names = [write(ds, seed, *cell) for ds, seed, cell in plan]
    print(f"wrote {len(names)} configs")
    for n in names:
        print("  " + n)


if __name__ == "__main__":
    main()
