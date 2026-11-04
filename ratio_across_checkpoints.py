# -*- coding: utf-8 -*-
"""
跨 checkpoint 的一致性检验
==========================
目的：回答"改善倍数（vanilla MMSE / PCFM MMSE）是否只是单次抽样偶然，
      还是会随训练阶段在一个范围内浮动"。

设计（配对控制，只变一个变量）
    固定：噪声库（同一 seed 一次抽样，所有 ckpt 复用）
    固定：留出 IC 的 φ（同一 seed）
    固定：参考解族（同 IC、ν∈[1,5]）
    固定：采样步数 n_step、样本数 n_sample
    唯一变量：checkpoint（训练阶段）

用法
    python ratio_across_checkpoints.py
    python ratio_across_checkpoints.py --ckpts out/heat_2000.pt out/heat_20000.pt
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

import pcfm_compat  # noqa: F401  必须在 ffm_sampler 之前
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid

from run_heat_official import (
    OUT, build, heat_family, load_cfg, make_residuals, metrics,
)


def strip_meta(sd):
    return {k: v for k, v in sd.items() if k != "_metadata"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "PCFM", "configs", "heat.yml"))
    ap.add_argument("--ckpts", nargs="*", default=None,
                    help="留空则自动收集 out/heat_*.pt 并按步数排序")
    ap.add_argument("--n-ref", type=int, default=16)
    ap.add_argument("--n-sample", type=int, default=16)
    ap.add_argument("--n-step", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--phi", type=float, default=0.7853981633974483,
                    help="论文附录 B.1 固定 φ=π/4；默认即 π/4")
    ap.add_argument("--out", default=os.path.join(OUT, "ratio_across_ckpts.json"))
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    # ── 收集 checkpoint，按步数排序 ──
    if args.ckpts:
        ckpts = args.ckpts
    else:
        files = glob.glob(os.path.join(OUT, "heat_*.pt"))
        def steps(p):
            return int(os.path.basename(p).replace("heat_", "").replace(".pt", ""))
        ckpts = sorted(files, key=steps)
    print(f"[ckpt] 共 {len(ckpts)} 个: " +
          ", ".join(os.path.basename(c) for c in ckpts))

    # ── 固定噪声库 / IC / 参考解族（只做一次）──
    model = build(cfg, dev)
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    torch.manual_seed(args.seed)
    noise_bank = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)

    # 论文附录 B.1: constrained sampling 固定 φ = π/4
    if args.phi is not None:
        phi = float(args.phi)
    else:
        g = torch.Generator().manual_seed(args.seed + 1)
        phi = float(torch.empty(1).uniform_(0.0, np.pi, generator=g))
    family = heat_family(phi, torch.linspace(1.0, 5.0, args.n_ref), nx, nt, dev)
    hfunc = make_residuals(cfg, family[0].cpu(), dev)

    print(f"[fix ] 噪声库 {tuple(noise_bank.shape)} | IC φ={phi:.4f} | "
          f"参考族 n={args.n_ref} | n_step={args.n_step}")

    rows = []
    print(f"\n{'ckpt':>7} | {'vanilla MMSE':>13} | {'PCFM MMSE':>10} | "
          f"{'MMSE 倍数':>10} | {'vanilla SMSE':>13} | {'PCFM SMSE':>10} | "
          f"{'SMSE 倍数':>10} | {'PCFM CE_CL':>11}")
    print("-" * 116)

    for ck in ckpts:
        sd = strip_meta(torch.load(ck, map_location=dev, weights_only=False)["model"])
        model.load_state_dict(sd, strict=False)
        model.eval()

        u_v = sampler.vanilla_sample(noise_bank, args.n_step)
        m_v = metrics(u_v, family, hfunc)

        u_p = sampler.pcfm_sample(noise_bank, args.n_step, hfunc,
                                  guided_interpolation=False)
        m_p = metrics(u_p, family, hfunc)

        r_mmse = m_v["MMSE"] / m_p["MMSE"] if m_p["MMSE"] > 1e-12 else float("nan")
        r_smse = m_v["SMSE"] / m_p["SMSE"] if m_p["SMSE"] > 1e-12 else float("nan")
        st = os.path.basename(ck).replace("heat_", "").replace(".pt", "")

        rows.append({"step": int(st), "vanilla": m_v, "pcfm": m_p,
                     "ratio_MMSE": r_mmse, "ratio_SMSE": r_smse})
        print(f"{st:>7} | {m_v['MMSE']:13.3f} | {m_p['MMSE']:10.3f} | "
              f"{r_mmse:10.2f} | {m_v['SMSE']:13.3f} | {m_p['SMSE']:10.3f} | "
              f"{r_smse:10.2f} | {m_p['CE_CL']:11.2e}")

    # ── 稳定性汇总 ──
    def stat(key):
        v = np.array([r[key] for r in rows if np.isfinite(r[key])])
        return (float(v.mean()), float(v.std()), float(v.min()), float(v.max()),
                float(v.max() / max(v.min(), 1e-12)))

    print("\n" + "=" * 62)
    print("改善倍数的稳定性（跨训练阶段）")
    print("=" * 62)
    for key, name in (("ratio_MMSE", "MMSE 倍数"), ("ratio_SMSE", "SMSE 倍数")):
        m, s, lo, hi, spread = stat(key)
        cv = s / m * 100 if m else float("nan")
        print(f"  {name:<10} 均值 {m:8.2f}  标准差 {s:7.2f}  "
              f"范围 [{lo:.2f}, {hi:.2f}]  极差/最小 {spread:.2f}×  变异系数 {cv:.1f}%")

    ce = [r["pcfm"]["CE_CL"] for r in rows]
    print(f"\n  PCFM CE_CL 全程范围 [{min(ce):.2e}, {max(ce):.2e}]"
          f" —— 是否恒为机器精度：{'是' if max(ce) < 1e-3 else '否'}")

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"protocol": {"phi": phi, "n_ref": args.n_ref,
                                "n_sample": args.n_sample, "n_step": args.n_step,
                                "seed": args.seed}, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
