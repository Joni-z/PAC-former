import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch, yaml, os, pickle, numpy as np
from scipy.signal import resample
from models.build import build_model

cfg = yaml.safe_load(open('configs/tuev_mi.yaml')); cfg['device'] = 'cpu'
torch.manual_seed(0)
model = build_model(cfg); model.eval()

d = 'tuh_eeg/v2.0.1/edf/processed_train'
files = sorted(os.listdir(d))[:128]
Xs = []
for f in files:
    s = pickle.load(open(os.path.join(d, f), 'rb'))
    X = s['signal']                        # (16, T) @ 256 Hz
    X = resample(X, 5 * 200, axis=-1)      # match TUEVLoader: 5s @ 200 Hz -> 1000
    X = X / (np.quantile(np.abs(X), 0.95, axis=-1, keepdims=True) + 1e-8)
    Xs.append(X)
X = torch.FloatTensor(np.stack(Xs))

with torch.no_grad():
    out = model(X)
print('forward finite:', bool(torch.isfinite(out).all()), '| out', tuple(out.shape))

fr = model.frontend
with torch.no_grad():
    tok, phi, amp = fr(X)
    C = model.encoder.blocks[0].mixer.coupling_matrix(phi, amp)
print('coupling finite:', bool(torch.isfinite(C).all()),
      '| mean|C|=%.5f max|C|=%.5f' % (C.abs().mean(), C.abs().max()))
print('DONE')
