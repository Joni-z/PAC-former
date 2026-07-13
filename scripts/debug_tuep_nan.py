import torch, yaml, os, pickle, random, sys
import numpy as np
from models.build import build_model

cfg = yaml.safe_load(open('configs/tuep_mi.yaml'))
cfg['device'] = 'cpu'
torch.manual_seed(0)
model = build_model(cfg)
model.eval()

root = cfg['data_root']
val_dir = os.path.join(root, 'val')
files = sorted(os.listdir(val_dir))
random.seed(0)
random.shuffle(files)
files = files[:2000]  # sample, not the full 64746

log = open('/scratch/zz5070/PAC-former/logs/debug_tuep_nan.log', 'w')

bad_found = False
for i in range(0, len(files), 32):
    batch_files = files[i:i+32]
    Xs = []
    for f in batch_files:
        s = pickle.load(open(os.path.join(val_dir, f), 'rb'))
        X = s['X']
        X = X / (np.quantile(np.abs(X), q=0.95, axis=-1, keepdims=True) + 1e-8)
        Xs.append(X)
    X = torch.FloatTensor(np.stack(Xs))
    with torch.no_grad():
        out = model(X)
    if not torch.isfinite(out).all():
        log.write(f'NaN/Inf found at batch starting {i}\n'); log.flush()
        for j, f in enumerate(batch_files):
            single = X[j:j+1]
            with torch.no_grad():
                o = model(single)
            if not torch.isfinite(o).all():
                log.write(f'  bad file: {f}\n')
                s = pickle.load(open(os.path.join(val_dir, f), 'rb'))
                Xraw = s['X']
                log.write(f'  raw X min/max: {Xraw.min()} {Xraw.max()}\n')
                log.write(f'  raw X std per channel: {Xraw.std(axis=-1)}\n')
                log.flush()
        bad_found = True
        break
    log.write(f'scanned {i+len(batch_files)}/{len(files)} ok\n'); log.flush()

log.write(f'DONE. total scanned {len(files)} bad_found={bad_found}\n')
log.close()
