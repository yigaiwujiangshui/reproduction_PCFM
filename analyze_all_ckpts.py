# -*- coding: utf-8 -*-
"""
全 checkpoint 分析：一次跑完所有已保存模型的 Heat 评测
=======================================================
对每个 checkpoint 做四组测量（配对：同一噪声库、同一留出 IC）：

    A. vanilla FFM（无约束基线）
    B. PCFM  N=20（基准）
    C. PCFM  N=5   （采样步数消融的弱端）
    D. PCFM  N=10  （中间）

再附一组「训练阶段 vs 收敛」的对照：PCFM 与其自身的 CE 稳定性。

输出
    out/analyze_all.json      逐 checkpoint 的完整指标
    out/analyze_all.txt       可直接粘进报告的表

用法
    python analyze_all_ckpts.py                # 自动收集 out/heat_*.pt
    python analyze_all_ckpts.py --n-sample 8   # 显卡紧张时降样本数
"""

import argparse
import glob
import json
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

PHI = 0.7853981633974483  # 论文附录 B.1: constrained sampling 固定 φ = π/4


def strip_meta(sd):
    return {k: v for k, v in sd.items() if k != "_metadata"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "PCFM", "configs", "heat.yml"))
    ap.add_argument("--n-ref", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--steps-list", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--out-json", default=os.path.join(OUT, "analyze_all.json"))
    ap.add_argument("--out-txt", default=os.path.join(OUT, "analyze_all.txt"))
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    files = glob.glob(os.path.join(OUT, "heat_*.pt"))

    def step_of(p):
        b = os.path.basename(p).replace("heat_", "").replace(".pt", "")
        try:
            return int(b)
        except ValueError:
            return 10 ** 9

    ckpts = sorted(files, key=step_of)

    model = build(cfg, dev)
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    torch.manual_seed(args.seed)
    noise_bank = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)
    family = heat_family(PHI, torch.linspace(1.0, 5.0, args.n_ref), nx, nt, dev)
    hfunc = make_residuals(cfg, family[0].cpu(), dev)

    L = []
    def p(s=""):
        L.append(str(s))

    p("PCFM 复现：全 checkpoint 分析")
    p("=" * 108)
    p(f"留出 IC: φ = π/4 | 参考解族 n={args.n_ref} (ν∈[1,5]) | "
      f"共用噪声库 {tuple(noise_bank.shape)} | 设备 {dev}")
    p("")
    hdr = (f"{'ckpt':>7} | {'vanilla MMSE':>12} | {'PCFM MMSE':>10} | {'倍数':>7} | "
           f"{'van SMSE':>9} | {'PCFM SMSE':>10} | {'SMSE倍数':>8} | "
           f"{'CE_IC':>9} | {'CE_CL':>10} | {'N=5':>8} | {'N=10':>8}")
    p(hdr)
    p("-" * len(hdr))

    rows = []
    for ck in ckpts:
        st = step_of(ck)
        sd = strip_meta(torch.load(ck, map_location=dev, weights_only=False)["model"])
        model.load_state_dict(sd, strict=False)
        model.eval()

        u_v = sampler.vanilla_sample(noise_bank, 20)
        m_v = metrics(u_v, family, hfunc)

        res = {"step": st, "vanilla": m_v}
        for ns in args.steps_list:
            u_p = sampler.pcfm_sample(noise_bank, ns, hfunc, guided_interpolation=False)
            res[f"pcfm_n{ns}"] = metrics(u_p, family, hfunc)

        base = res["pcfm_n20"]
        r_mmse = m_v["MMSE"] / base["MMSE"] if base["MMSE"] > 1e-12 else float("nan")
        r_smse = m_v["SMSE"] / base["SMSE"] if base["SMSE"] > 1e-12 else float("nan")
        res["ratio_MMSE"] = r_mmse
        res["ratio_SMSE"] = r_smse
        rows.append(res)

        p(f"{st:>7} | {m_v['MMSE']:12.3f} | {base['MMSE']:10.4f} | {r_mmse:7.2f} | "
          f"{m_v['SMSE']:9.3f} | {base['SMSE']:10.4f} | {r_smse:8.2f} | "
          f"{base['CE_IC']:9.2e} | {base['CE_CL']:10.2e} | "
          f"{res['pcfm_n5']['MMSE']:8.4f} | {res['pcfm_n10']['MMSE']:8.4f}")
        sys.stdout.flush()

    # ── 汇总 ──
    p("")
    p("=" * 108)
    p("汇总")
    p("=" * 108)

    def stat(key):
        v = np.array([r[key] for r in rows if np.isfinite(r.get(key, np.nan))])
        return v

    ratio = stat("ratio_MMSE")
    p(f"改善倍数 (MMSE)：均值 {ratio.mean():.2f}  中位 {np.median(ratio):.2f}  "
      f"范围 [{ratio.min():.2f}, {ratio.max():.2f}]  变异系数 {ratio.std()/ratio.mean()*100:.1f}%")

    for ns in args.steps_list:
        ce = np.array([r[f"pcfm_n{ns}"]["CE_CL"] for r in rows])
        ic = np.array([r[f"pcfm_n{ns}"]["CE_IC"] for r in rows])
        p(f"PCFM N={ns:<2}  CE_IC ∈ [{ic.min():.2e}, {ic.max():.2e}]   "
          f"CE_CL ∈ [{ce.min():.2e}, {ce.max():.2e}]")

    # 步数单调性：每个 checkpoint 内 N=5 > N=10 > N=20 是否成立
    ok = 0
    for r in rows:
        vals = [r[f"pcfm_n{ns}"]["MMSE"] for ns in args.steps_list]
        if all(a >= b - 1e-9 for a, b in zip(vals, vals[1:])):
            ok += 1
    p(f"步数单调性（MMSE 随 N 增大而下降）：{ok}/{len(rows)} 个 checkpoint 成立")

    # 基线是否随训练改善
    vv = np.array([r["vanilla"]["MMSE"] for r in rows])
    pp = np.array([r["pcfm_n20"]["MMSE"] for r in rows])
    p(f"vanilla MMSE 随训练：首 {vv[0]:.3f} → 末 {vv[-1]:.3f}  "
      f"（范围 [{vv.min():.3f}, {vv.max():.3f}]）")
    p(f"PCFM    MMSE 随训练：首 {pp[0]:.4f} → 末 {pp[-1]:.4f}  "
      f"（范围 [{pp.min():.4f}, {pp.max():.4f}]）")

    os.makedirs(OUT, exist_ok=True)
    with open(args.out_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"protocol": {"phi": PHI, "n_ref": args.n_ref,
                                "n_sample": args.n_sample, "seed": args.seed},
                   "rows": rows}, f, ensure_ascii=False, indent=2)
    print("\n[saved]", args.out_txt)
    print("[saved]", args.out_json)


if __name__ == "__main__":
    main()
