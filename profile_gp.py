# -*- coding: utf-8 -*-
"""精简剖析：分别测 GP 采样与模型训练步的耗时/显存，找出可行配置"""
import sys, os, time, gc
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "PCFM"))
import torch, yaml
from models import get_flow_model
from pcfm.pcfm_sampling import make_grid

cfg = yaml.safe_load(open(os.path.join(HERE, "PCFM", "configs", "heat.yml"), encoding="utf8"))
dev = "cuda"
nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

def reset():
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def mem():
    return torch.cuda.max_memory_allocated() / 2**30

print(f"GPU: {torch.cuda.get_device_name(0)}  {torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")

model = get_flow_model(cfg["model"], cfg["encoder"]).to(dev)
grid = make_grid((nx, nt), dev)

# ── A. GP 采样单独计时 ──
for B in (8, 32):
    reset()
    t = time.time()
    x0 = model.gp.sample(grid, (nx, nt), n_samples=B)
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"[GP ] B={B:3d}  {dt:6.2f}s  peak {mem():5.2f} GiB")
    del x0; reset()

# ── B. 训练步（GP + 前向 + 反向）──────────────────
for B in (16, 32, 64):
    reset()
    try:
        t = time.time()
        x = torch.randn(B, nx, nt, device=dev)
        loss = model.get_loss(x)
        loss.backward()
        torch.cuda.synchronize()
        dt = time.time() - t
        print(f"[full] B={B:3d}  {dt:6.2f}s  peak {mem():5.2f} GiB  loss {loss.item():.4f}")
        model.zero_grad(set_to_none=True)
        del x, loss
    except torch.OutOfMemoryError:
        print(f"[full] B={B:3d}  OOM")
    reset()

# ── C. 纯前向/反向（不含 GP），看模型本身的代价 ──
for B in (32, 64):
    reset()
    tau = torch.rand(B, device=dev)
    xt = torch.randn(B, nx, nt, device=dev)
    t = time.time()
    p = model(tau, xt); p.pow(2).mean().backward()
    torch.cuda.synchronize()
    dt = time.time() - t
    print(f"[net ] B={B:3d}  {dt:6.2f}s  peak {mem():5.2f} GiB")
    model.zero_grad(set_to_none=True)
    del tau, xt, p
    reset()
