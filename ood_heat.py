# -*- coding: utf-8 -*-
"""
(2) OOD 泛化试验（对应论文附录 L.4 / Table 8 的思路）
========================================================
论文在 Burgers 上做的 OOD：训练集 IC 位置 x0∈[0,0.6]，测试用 x0∈[0,0.2] 与 [0.8,1.0]。

Heat 的对应做法：训练时 φ ~ U(0, π)，因此「分布内」是 φ∈[0,π]。
本试验把留出 IC 的 φ 取到训练分布之外，检查：
  - 约束是否仍然精确满足（投影是代数的，理论上与分布无关）
  - 保真度如何退化

输出 out/ood_heat.txt
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "PCFM"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import pcfm_compat  # noqa: F401
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid
from run_heat_official import OUT, build, heat_family, load_cfg, make_residuals, metrics

CKPT = os.path.join(OUT, "heat_30000.pt")

# φ 的测试点：分布内 3 个 + 分布外 4 个
CASES = [
    (0.7853981633974483, "π/4",      True,  "分布内（论文留出点）"),
    (0.30,               "0.30",     True,  "分布内"),
    (2.80,               "2.80",     True,  "分布内"),
    (-1.20,              "−1.20",    False, "分布外（负相位）"),
    (3.50,               "3.50",     False, "分布外（略超 π）"),
    (4.80,               "4.80",     False, "分布外"),
    (6.00,               "6.00",     False, "分布外（接近 2π）"),
]


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(os.path.join(HERE, "PCFM", "configs", "heat.yml"))
    dev = "cuda"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    model = build(cfg, dev)
    sd = {k: v for k, v in torch.load(CKPT, map_location=dev, weights_only=False)["model"].items()
          if k != "_metadata"}
    model.load_state_dict(sd, strict=False)
    model.eval()
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    torch.manual_seed(4242)
    noise = model.gp.sample(grid, (nx, nt), n_samples=8).to(dev)

    L = []
    def p(s=""):
        L.append(str(s))

    p("(2) OOD 泛化试验 —— Heat 方程")
    p("=" * 104)
    p(f"checkpoint: {os.path.basename(CKPT)} | 共用噪声库 {tuple(noise.shape)} | "
      f"参考解族 n=32 (ν∈[1,5])")
    p("训练分布: φ ~ U(0, π)，故 φ∈[0,π] 为分布内，其余为分布外")
    p("")
    hdr = (f"{'φ':>8} | {'区间':>8} | {'vanilla MMSE':>12} | {'PCFM MMSE':>10} | "
           f"{'倍数':>7} | {'van CE_IC':>10} | {'PCFM CE_IC':>11} | {'PCFM CE_CL':>11}")
    p(hdr)
    p("-" * len(hdr))

    rows = []
    for phi, tag, in_dist, note in CASES:
        family = heat_family(phi, torch.linspace(1.0, 5.0, 32), nx, nt, dev)
        hfunc = make_residuals(cfg, family[0].cpu(), dev)

        u_v = sampler.vanilla_sample(noise, 20)
        u_p = sampler.pcfm_sample(noise, 20, hfunc, guided_interpolation=False)
        m_v = metrics(u_v, family, hfunc)
        m_p = metrics(u_p, family, hfunc)

        r = m_v["MMSE"] / m_p["MMSE"] if m_p["MMSE"] > 1e-12 else float("nan")
        rows.append({"phi": phi, "tag": tag, "in_dist": in_dist,
                     "vanilla": m_v, "pcfm": m_p, "ratio": r})

        zone = "分布内" if in_dist else "分布外"
        p(f"{tag:>8} | {zone:>8} | {m_v['MMSE']:12.4f} | {m_p['MMSE']:10.4f} | "
          f"{r:7.2f} | {m_v['CE_IC']:10.2e} | {m_p['CE_IC']:11.2e} | {m_p['CE_CL']:11.2e}")
        sys.stdout.flush()

    # ── 分区间汇总 ──
    p("")
    p("=" * 104)
    p("分区间汇总")
    p("=" * 104)
    for label, sel in (("分布内 (φ∈[0,π])", [r for r in rows if r["in_dist"]]),
                       ("分布外 (φ∉[0,π])", [r for r in rows if not r["in_dist"]])):
        if not sel:
            continue
        v_m = np.array([r["vanilla"]["MMSE"] for r in sel])
        p_m = np.array([r["pcfm"]["MMSE"] for r in sel])
        ce_ic = np.abs(np.array([r["pcfm"]["CE_IC"] for r in sel]))
        ce_cl = np.array([r["pcfm"]["CE_CL"] for r in sel])
        van_ce = np.array([r["vanilla"]["CE_IC"] for r in sel])
        p(f"{label}  ({len(sel)} 个点)")
        p(f"  vanilla MMSE 均值 {v_m.mean():.4f}   PCFM MMSE 均值 {p_m.mean():.4f}   "
          f"倍数均值 {(v_m/p_m).mean():.2f}")
        p(f"  PCFM CE_IC 最大 {ce_ic.max():.2e}（恒为零）   CE_CL ∈ [{ce_cl.min():.2e}, {ce_cl.max():.2e}]")
        p(f"  vanilla CE_IC ∈ [{van_ce.min():.2e}, {van_ce.max():.2e}]")
        p("")

    p("判读要点：")
    p("  · 若 PCFM 的 CE 在分布外仍为 0，说明约束满足是投影算子的代数性质，与数据分布无关。")
    p("  · 若 PCFM 的 MMSE 在分布外上升，说明保真度依赖训练覆盖，属预期。")
    p("  · 若 vanilla 的 CE 在分布外显著恶化，说明无约束模型对分布漂移敏感。")

    with open(os.path.join(OUT, "ood_heat.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n[saved]", os.path.join(OUT, "ood_heat.txt"))


if __name__ == "__main__":
    main()
