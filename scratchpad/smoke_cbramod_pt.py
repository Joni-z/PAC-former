import sys, torch, yaml
sys.path.insert(0,"/scratch/zz5070/PACLock")
sys.path.insert(0,"/scratch/zz5070/PACLock/reference/CBraMod")
import cbramod_pretrain as M
ok=True
x=torch.randn(2,16,10,200)
lp=M.lowpass_patches(x,0.5)
# 低通后高频谱应≈0
hi=torch.fft.rfft(lp,dim=-1).abs()[...,51:].mean().item()
print(f"lowpass: shape={tuple(lp.shape)} residual_high_freq={hi:.4f} (应≈0)")
ok &= lp.shape==x.shape and hi<1e-3
for obj in ("crossfreq","random"):
    m=M.CBraModPretrain(obj)
    loss=m(x); loss.backward()
    g=sum(p.grad.abs().sum().item() for p in m.parameters() if p.grad is not None)
    print(f"{obj:10s} recon_loss={loss.item():.4f} finite={torch.isfinite(loss).item()} grad={g:.0f}")
    ok &= torch.isfinite(loss).item() and g>0
# 分类器接backbone
c=M.Classifier(M.CBraModPretrain("random").backbone,2,16,10)
out=c(x); print(f"classifier out={tuple(out.shape)} finite={torch.isfinite(out).all().item()}")
ok &= out.shape==(2,2)
print("SMOKE OK" if ok else "SMOKE FAILED")
