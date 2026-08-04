import sys, yaml, torch, numpy as np
sys.path.insert(0,"/scratch/zz5070/PACLock")
sys.path.insert(0,"/scratch/zz5070/PACLock/reference/CBraMod")
import baseline_cbramod as B
ok=True
for cfgp,init in [("configs/cbramod_tuab.yaml","pretrained"),("configs/cbramod_tuev.yaml","scratch")]:
    cfg=yaml.safe_load(open("/scratch/zz5070/PACLock/"+cfgp))
    ds=B.CBraModNpy(cfg["data_root"],"test",cfg["native_rate"],200,200,cfg.get("label_shift",0))
    X,y=ds[0]; npat=(200*cfg["duration_s"])//200
    print(f"{cfgp.split('/')[-1]}: X={tuple(X.shape)} exp(16,{npat},200) y={y} abs95%={np.quantile(np.abs(X.numpy()),0.95):.3f}")
    ok &= tuple(X.shape)==(16,npat,200)
    m=B.CBraModClassifier(cfg["num_classes"],cfg["n_channels"],npat,pretrained=(init=='pretrained'),dropout=0.1)
    xb=torch.stack([ds[i][0] for i in range(4)])
    out=m(xb); loss=torch.nn.functional.cross_entropy(out,torch.tensor([ds[i][1] for i in range(4)]))
    loss.backward()
    g=sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"   init={init} logits={tuple(out.shape)} finite={torch.isfinite(out).all().item()} loss={loss.item():.3f} grad={g:.0f}")
    ok &= tuple(out.shape)==(4,cfg["num_classes"]) and torch.isfinite(out).all().item() and g>0
print("SMOKE OK" if ok else "SMOKE FAILED")
