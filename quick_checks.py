# -*- coding: utf-8 -*-
"""
两项快速验证
=============
(1) 模式崩塌核查：逐 checkpoint 测量「样本间方差」，判断低 SMSE 是质量还是坍缩
(2) 中间投影必要性（论文附录 L.1 / Table 6）：每步投影 vs 仅末端投影 vs 完全不投影

输出 out/quick_checks.txt
"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "PCFM"))
sys.path.insert(0, HERE)

import numpy as np
import torch

import pcfm_compat  # noqa: F401
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import fast_project_batched, make_grid
from run_heat_official import OUT, build, heat_family, load_cfg, make_residuals, metrics

PHI = 0.7853981633974483


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(os.path.join(HERE, "PCFM", "configs", "heat.yml"))
    dev = "cuda"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    model = build(cfg, dev)
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    torch.manual_seed(2026)
    noise = model.gp.sample(grid, (nx, nt), n_samples=8).to(dev)
    B = noise.shape[0]

    family = heat_family(PHI, torch.linspace(1.0, 5.0, 32), nx, nt, dev)
    hfunc = make_residuals(cfg, family[0].cpu(), dev)
    truth_std = family.std(dim=0)          # 真解族的空间-时间标准差场
    truth_mean = family.mean(dim=0)

    files = glob.glob(os.path.join(OUT, "heat_*.pt"))

    def step_of(p):
        b = os.path.basename(p).replace("heat_", "").replace(".pt", "")
        try:
            return int(b)
        except ValueError:
            return 10 ** 9

    ckpts = sorted(files, key=step_of)

    L = []
    def p(s=""):
        L.append(str(s))

    # ══════════════════════════════════════════════════════════════
    # (1) 模式崩塌核查
    # ══════════════════════════════════════════════════════════════
    p("(1) 模式崩塌核查")
    p("=" * 96)
    p("样本间逐点标准差（跨 8 个样本求 std，再全空间平均）。")
    p("若某 checkpoint 该值骤降，说明 PCFM 把不同噪声映射到了几乎同一个解上 —— 即坍缩。")
    p("")
    p(f"{'ckpt':>7} | {'样本间 std':>11} | {'真解族 std':>10} | {'比值':>7} | "
      f"{'MMSE':>9} | {'SMSE':>8}")
    p("-" * 76)

    collapse = []
    for ck in ckpts:
        st = step_of(ck)
        sd = {k: v for k, v in torch.load(ck, map_location=dev, weights_only=False)["model"].items()
              if k != "_metadata"}
        model.load_state_dict(sd, strict=False)
        model.eval()

        u = sampler.pcfm_sample(noise, 20, hfunc, guided_interpolation=False)
        s_between = u.std(dim=0).mean().item()      # 样本间方差
        s_truth = truth_std.mean().item()
        m = metrics(u, family, hfunc)
        collapse.append((st, s_between, s_truth, m["MMSE"], m["SMSE"]))
        p(f"{st:>7} | {s_between:11.5f} | {s_truth:10.5f} | "
          f"{s_between/s_truth:7.3f} | {m['MMSE']:9.4f} | {m['SMSE']:8.4f}")
        sys.stdout.flush()

    p("")
    vals = np.array([c[1] for c in collapse])
    med = float(np.median(vals))
    p(f"样本间 std 中位数 = {med:.5f}；最小 = {vals.min():.5f}（ckpt {collapse[int(vals.argmin())][0]}）")
    low = [c for c in collapse if c[1] < med * 0.3]
    if low:
        p("疑似坍缩的 checkpoint：")
        for c in low:
            p(f"  ckpt {c[0]}: 样本间 std={c[1]:.5f}（为中位数的 {c[1]/med*100:.1f}%）, "
              f"MMSE={c[3]:.4f}, SMSE={c[4]:.4f}")
        p("  → 若这些点的 SMSE 也显著偏低，则「低 SMSE」确有坍缩成分，不能当作质量指标。")
    else:
        p("未发现样本间 std 骤降超过中位数 70% 的 checkpoint。")

    # ══════════════════════════════════════════════════════════════
    # (2) 中间投影必要性
    # ══════════════════════════════════════════════════════════════
    p("")
    p("(2) 中间投影必要性（论文附录 L.1，Table 6）")
    p("=" * 96)
    p("同一 checkpoint、同一噪声、同一留出 IC，只改「投影发生在哪里」：")
    p("  A 完全不投影   —— 纯 vanilla FFM")
    p("  B 仅末端投影   —— 先跑完 ODE，最后投一次")
    p("  C 每步投影     —— PCFM 本体")
    p("")
    p(f"{'ckpt':>7} | {'方式':<12} | {'MMSE':>9} | {'SMSE':>8} | {'CE_IC':>9} | {'CE_CL':>10}")
    p("-" * 76)

    for ck in ckpts:
        st = step_of(ck)
        sd = {k: v for k, v in torch.load(ck, map_location=dev, weights_only=False)["model"].items()
              if k != "_metadata"}
        model.load_state_dict(sd, strict=False)
        model.eval()

        u_a = sampler.vanilla_sample(noise, 20)
        u_b = fast_project_batched(u_a.reshape(B, -1), hfunc, max_iter=1).reshape(B, nx, nt)
        u_c = sampler.pcfm_sample(noise, 20, hfunc, guided_interpolation=False)

        for tag, u in (("A 不投影", u_a), ("B 仅末端", u_b), ("C 每步", u_c)):
            m = metrics(u, family, hfunc)
            p(f"{st:>7} | {tag:<12} | {m['MMSE']:9.4f} | {m['SMSE']:8.4f} | "
              f"{m['CE_IC']:9.2e} | {m['CE_CL']:10.2e}")
        p("-" * 76)
        sys.stdout.flush()

    with open(os.path.join(OUT, "quick_checks.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n[saved]", os.path.join(OUT, "quick_checks.txt"))


if __name__ == "__main__":
    main()
