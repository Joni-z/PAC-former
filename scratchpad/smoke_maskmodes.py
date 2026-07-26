import sys, torch
sys.path.insert(0,"/scratch/zz5070/PAC-former")
from models.pretrain import MAEPretrain
cfg=dict(dataset="chbmit",arch="triaxial",freq_mixer="attention",spatial_pe="index",band_pe="index",
  n_channels=4,seq_len=400,sample_rate=200,num_classes=2,n_bands=8,d_model=32,depth=2,dropout=0.0,
  kernel_size=51,patch_len=100,n_heads=4,mask_ratio=0.5)
ok=True
for mode in ("crossfreq","lowfreq","bandrand","random"):
    c=dict(cfg); c["mask_mode"]=mode
    m=MAEPretrain(c)
    # 直接查mask构造
    B,C,nb,P=2,4,8,4
    torch.manual_seed(0)
    msk=m._mask(B,C,nb,P,"cpu")
    # 每个频带是否整体被遮 (True/False per band, 取第一个c,p)
    perband=msk[0,:,:,0].any(0) & msk[0,:,:,0].all(0) if mode!="random" else None
    hidden_bands=[b for b in range(nb) if msk[0,0,b,0].item()] if mode!="random" else "scattered"
    x=torch.randn(2,4,400); loss=m(x); loss=loss[0] if isinstance(loss,tuple) else loss
    print(f"{mode:10s} hidden_bands={hidden_bands}  loss={loss.item():.4f} finite={torch.isfinite(loss).item()}")
    if mode=="crossfreq": ok&= hidden_bands==[4,5,6,7]
    if mode=="lowfreq":   ok&= hidden_bands==[0,1,2,3]
    if mode=="bandrand":  ok&= len(hidden_bands)==4
    ok&=torch.isfinite(loss).item()
print("SMOKE OK" if ok else "SMOKE FAILED")
