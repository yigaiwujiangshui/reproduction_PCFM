# -*- coding: utf-8 -*-
"""量化 CE_IC 与 CE_CL 的剩余误差来源（输出写入 UTF-8 文件，避免控制台乱码）"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "PCFM"))
sys.path.insert(0, HERE)
import torch, numpy as np
import pcfm_compat  # noqa
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid
from run_heat_official import OUT, build, heat_family, load_cfg, make_residuals

L = []
def p(s=""):
    L.append(str(s))

cfg = load_cfg(os.path.join(HERE, "PCFM", "configs", "heat.yml"))
dev = "cuda"
nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

model = build(cfg, dev)
sd = {k: v for k, v in torch.load(os.path.join(OUT, "heat_20000.pt"),
                                 map_location=dev, weights_only=False)["model"].items()
      if k != "_metadata"}
model.load_state_dict(sd, strict=False)
model.eval()
sampler = FFM_sampler(model, model.gp)
grid = make_grid((nx, nt), dev)

torch.manual_seed(1234)
noise = model.gp.sample(grid, (nx, nt), n_samples=8).to(dev)

phi = 0.5128
fam = heat_family(phi, torch.linspace(1.0, 5.0, 16), nx, nt, dev)
h = make_residuals(cfg, fam[0].cpu(), dev)

p("=" * 74)
p("实测 CE 剩余误差的量级来源")
p("=" * 74)
p(f"网格 {nx}x{nt} | dx={2*np.pi/nx:.6f} | float32 eps={torch.finfo(torch.float32).eps:.3e}")
p(f"参考族: sin(x+{phi}) * exp(-nu*t), nu in [1,5], 16 条")
p("")

for tag, gen in (("PCFM n_step=20",
                  sampler.pcfm_sample(noise, 20, h, guided_interpolation=False)),
                 ("vanilla n_step=20",
                  sampler.vanilla_sample(noise, 20))):
    p("-" * 74)
    p(tag)
    p("-" * 74)
    p(f"{'i':>2} | {'||IC res||_2':>14} | {'||CL res||_2':>14} | "
      f"{'mass scale':>12} | {'CL rel.':>10}")
    icn, cln, ms = [], [], []
    for i in range(gen.shape[0]):
        u = gen[i].reshape(-1)
        ic = h.ic_residual(u)
        cl = h.mass_residual_heat(u)
        m0 = u.view(nx, nt)[:, 0].sum() * h.dx
        a, b = ic.pow(2).sum().sqrt(), cl.pow(2).sum().sqrt()
        icn.append(a.item()); cln.append(b.item()); ms.append(abs(m0.item()))
        p(f"{i:>2} | {a:14.4e} | {b:14.4e} | {m0:12.6f} | "
          f"{b/abs(m0):10.3e}" if abs(m0) > 0 else "")
    p("")
    p(f"  平均 ||IC||={np.mean(icn):.4e}  平均 ||CL||={np.mean(cln):.4e}  "
      f"|-*1e-2 后: IC={np.mean(icn)*1e2:.3e}, CL={np.mean(cln)*1e2:.3e}")
    p("")

p("=" * 74)
p("结论参考点")
p("=" * 74)
p("  * IC 残差 = u[:,0] - ic_target，是逐点恒等，投影后应到 float32 下限")
p("  * CL 残差 = sum_x u dx 在 t 上的差，含 100 项累加 + dx 乘子，误差按项数累积")
p(f"  * 若 CL 残差/质量 ≈ {3.9e-7/0.6:.1e}，与 float32 eps 同阶即说明已到数值极限")

with open(os.path.join(OUT, "ce_precision.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("written")
