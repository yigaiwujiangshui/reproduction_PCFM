# -*- coding: utf-8 -*-
"""
IC 维度扫描：改善倍数对留出 IC 的依赖
=====================================
目的：把"checkpoint 维度"和"IC 维度"分开。
      前者由 ratio_across_checkpoints.py 回答，本脚本回答后者。

控制变量
    固定：噪声库（同一 seed，所有 φ 复用同一批初始场）
    固定：参考解族大小 n_ref、样本数 n_sample、步数 n_step
    变动：留出 IC 的相位 φ（Heat 的 IC = sin(x+φ)）
    可变：checkpoint（用 --ckpts 指定多个）

用法
    python sweep_phi.py                       # 默认 20000 与 8000 两个 ckpt，6 个 φ
    python sweep_phi.py --ckpts out/heat_20000.pt --n-phi 8
"""

import argparse
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

from run_heat_official import (
    OUT, build, heat_family, load_cfg, make_residuals, metrics,
)


def strip_meta(sd):
    return {k: v for k, v in sd.items() if k != "_metadata"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "PCFM", "configs", "heat.yml"))
    ap.add_argument("--ckpts", nargs="+",
                    default=[os.path.join(OUT, "heat_20000.pt"),
                             os.path.join(OUT, "heat_8000.pt")])
    ap.add_argument("--n-phi", type=int, default=6)
    ap.add_argument("--n-ref", type=int, default=16)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--n-step", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default=os.path.join(OUT, "sweep_phi.json"))
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(args.config)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    model = build(cfg, dev)
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    # 固定噪声库：所有 φ、所有 ckpt 共用同一批初始场
    torch.manual_seed(args.seed)
    noise_bank = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)

    # φ 在 [0,π] 上等距取 n_phi 个（覆盖相位全区间）
    phis = np.linspace(0.05, np.pi - 0.05, args.n_phi)
    print(f"[fix ] 噪声库 {tuple(noise_bank.shape)} | n_ref={args.n_ref} "
          f"| n_step={args.n_step} | φ 取 {args.n_phi} 个: "
          + ", ".join(f"{p:.3f}" for p in phis))

    all_rows = {}
    for ck in args.ckpts:
        if not os.path.exists(ck):
            print(f"[skip] 不存在: {ck}")
            continue
        sd = strip_meta(torch.load(ck, map_location=dev, weights_only=False)["model"])
        model.load_state_dict(sd, strict=False)
        model.eval()
        tag = os.path.basename(ck).replace("heat_", "").replace(".pt", "")

        print(f"\n{'='*96}")
        print(f"checkpoint {tag}")
        print(f"{'='*96}")
        print(f"{'phi':>6} | {'vanilla MMSE':>13} | {'PCFM MMSE':>10} | {'MMSE 倍数':>10} "
              f"| {'vanilla SMSE':>13} | {'PCFM SMSE':>10} | {'SMSE 倍数':>10} | {'CE_CL':>10}")
        print("-" * 104)

        rows = []
        for phi in phis:
            family = heat_family(float(phi), torch.linspace(1.0, 5.0, args.n_ref), nx, nt, dev)
            hfunc = make_residuals(cfg, family[0].cpu(), dev)

            m_v = metrics(sampler.vanilla_sample(noise_bank, args.n_step), family, hfunc)
            m_p = metrics(sampler.pcfm_sample(noise_bank, args.n_step, hfunc,
                                              guided_interpolation=False), family, hfunc)

            r_mmse = m_v["MMSE"] / m_p["MMSE"] if m_p["MMSE"] > 1e-12 else float("nan")
            r_smse = m_v["SMSE"] / m_p["SMSE"] if m_p["SMSE"] > 1e-12 else float("nan")
            rows.append({"phi": float(phi), "vanilla": m_v, "pcfm": m_p,
                         "ratio_MMSE": r_mmse, "ratio_SMSE": r_smse})
            print(f"{phi:6.3f} | {m_v['MMSE']:13.3f} | {m_p['MMSE']:10.4f} | "
                  f"{r_mmse:10.2f} | {m_v['SMSE']:13.3f} | {m_p['SMSE']:10.4f} | "
                  f"{r_smse:10.2f} | {m_p['CE_CL']:10.2e}")

        for key, name in (("ratio_MMSE", "MMSE 倍数"), ("ratio_SMSE", "SMSE 倍数")):
            v = np.array([r[key] for r in rows if np.isfinite(r[key])])
            print(f"  → {name:<10} 均值 {v.mean():7.2f}  中位 {np.median(v):7.2f}  "
                  f"范围 [{v.min():.2f}, {v.max():.2f}]  变异系数 {v.std()/v.mean()*100:5.1f}%")
        ce = [r["pcfm"]["CE_CL"] for r in rows]
        print(f"  → CE_CL 范围 [{min(ce):.2e}, {max(ce):.2e}]")
        all_rows[tag] = rows

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"protocol": {"n_phi": args.n_phi, "n_ref": args.n_ref,
                                "n_sample": args.n_sample, "n_step": args.n_step,
                                "seed": args.seed, "phis": phis.tolist()},
                   "by_ckpt": all_rows}, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}")

    # 跨 ckpt 总览
    if len(all_rows) > 1:
        print("\n" + "=" * 62)
        print("跨 checkpoint 的 IC 平均（同一组 φ）")
        print("=" * 62)
        for tag, rows in all_rows.items():
            v = np.array([r["ratio_MMSE"] for r in rows if np.isfinite(r["ratio_MMSE"])])
            p = np.array([r["pcfm"]["MMSE"] for r in rows])
            print(f"  ckpt {tag:>7}: MMSE 倍数均值 {v.mean():7.2f}  "
                  f"范围 [{v.min():6.2f}, {v.max():7.2f}]  |  PCFM MMSE 均值 {p.mean():.4f}")


if __name__ == "__main__":
    main()
