# -*- coding: utf-8 -*-
"""热方程上的多方法对照。

对象：同一 checkpoint 与同一批初始噪声。
算法：依次运行 vanilla FFM、PCFM（硬投影）、ECI（交替投影 + 混合）、DiffusionPDE（观测引导）、D-Flow（对初始噪声做梯度优化），统一计算 MMSE / SMSE / CE。
参考：Table 3。"""

import argparse
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, REPO, PCFM  # noqa: E402,F401

sys.path.insert(0, HERE)

import numpy as np  # noqa: E402
import torch  # noqa: E402

# 注意导入顺序：run_heat_official 会把 PCFM/ 插进 sys.path 并 import pcfm_compat
# （pcfm_compat 需要在 PCFM/ 已在路径上时导入，否则 CallableResiduals 拿不到基类）
from run_heat_official import REPO  # noqa: E402
import pcfm_compat  # noqa: E402,F401  （已由上一行导入，这里显式声明依赖）
from run_heat_official import (  # noqa: E402
    build,
    heat_family,
    load_cfg,
    make_residuals,
    metrics,
)

from pcfm.ffm_sampler import FFM_sampler  # noqa: E402
import pcfm.ffm_sampler as _ffm  # noqa: E402
from pcfm.pcfm_sampling import make_grid  # noqa: E402
from models.constraints import (  # noqa: E402
    ChainConstraint,
    InitialCondition,
    RegionConservationLaw,
)

# ── 只关掉进度条（纯显示层；算法与数值一律不动）──────────────────────────
_ffm.tqdm = lambda it=None, **kw: it

TWO_PI = 2.0 * math.pi
PHI = math.pi / 4.0          # 论文附录 B.1：固定留出 IC 的相位 φ = π/4
SEED = 2026                  # 共用噪声库的种子（题目指定）

# 论文 Table 3（Heat 行），单位 1e-2，与 run_heat_official.metrics 的 ×1e2 同量纲
PAPER = {
    "vanilla FFM": {"MMSE": 4.56, "CE_IC": 579.0},
    "PCFM": {"MMSE": 0.241, "CE_IC": 0.0},
    "ECI": {"MMSE": 0.697, "CE_IC": 0.0},
    "DiffusionPDE guided": {"MMSE": None, "CE_IC": None},   # Table 3 不含该基线
    "D-Flow": {"MMSE": 1.97, "CE_IC": 102.0},
}


# ══════════════════════════════════════════════════════════════════════
# 辅助
# ══════════════════════════════════════════════════════════════════════
def masked_mse(u_pred, u_true, mask):
    """基线用的 loss：逐元素 MSE（mean 归一）。

    官方 D-Flow 的 default_loss_fn 是 masked **sum**；导引基线用 sum 会让梯度
    量级随网格尺寸（1e4）放大，与 eta=2e2 不匹配，故统一用 mean 归一，
    并在报告里注明。
    """
    return ((u_pred - u_true) * mask).pow(2).mean()


def build_eci_constraint(family, nx, nt, dx, dev):
    """ECI 用的「IC + 质量守恒」约束。

    官方 `models/constraints.py` 的类层次：
        Constraint
          ├─ InitialCondition        (t=0 的 Dirichlet，即 IC)
          └─ RegionConservationLaw   (∫_Ω u dx = b(t)，守恒律)
    没有名字里直接带 Heat/Mass 的类——热方程的「质量守恒」就是
    `RegionConservationLaw` 配 `dims=(1,)`（沿 x 求均值），b(t) 取参考解的
    ∫u dx（对热方程它恒等于 ∫u(x,0)dx，也即 IC 的质量，**不含轨迹信息**）。

    顺序：先守恒律、后 IC。ChainConstraint 依次 apply，最后一步是 IC，
    才能保证 CE(IC) 精确为 0（与论文 Table 3 的 ECI CE(IC)=0 一致）。
    """
    ic_value = family[0][:, 0].detach().reshape(1, nx)          # (1,nx) 广播到 (B,nx)
    mass = (family[0].sum(dim=0) * dx).detach()                # (nt,) = ∫ u dx
    chain = ChainConstraint(
        RegionConservationLaw(mass.to(dev), dims=(1,), area=TWO_PI),
        InitialCondition(ic_value.to(dev)),
    )
    return chain.to(dev)


def finite(x):
    return isinstance(x, float) and math.isfinite(x)


def zero_tag(v):
    """CE(IC) 是否为「0」的三档判据。

    `1e-4` 是 1e-2 单位下的机器精度量级：低于它的值在论文表格里就是 0。
    （ECI 的 CE(IC) 之所以不是精确 0，是因为 eci_sample 末步做的是
    u = u1*(t+dt) + u0*(1-t-dt)，而 t+dt 有 ~1e-16 的浮点舍入。）
    """
    a = abs(v)
    if a < 1e-12:
        return True, "= 0（精确 0）"
    if a < 1e-4:
        return True, "≈ 0（机器精度级，论文口径下即 0）"
    return False, "≠ 0（未硬约束）"


def fmt_metric(v, width, spec):
    if v is None or not finite(v):
        return f"{'-':>{width}}"
    return f"{v:{width}{spec}}"


# ══════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════
def run(args):
    dev = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[env] torch {torch.__version__} | device {dev} | mode {args.mode}", flush=True)

    cfg = load_cfg(os.path.join(REPO, "configs", "heat.yml"))
    ds = cfg["datasets"]
    nx, nt = ds["nx"], ds["nt"]
    dx = TWO_PI / nx
    n_sample, n_step, n_ref = args.n_sample, args.n_step, args.n_ref

    # ── 模型 ────────────────────────────────────────────────────────
    model = build(cfg, dev)
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    # neuralop/tensorly 的谱卷积会把 "_metadata" 塞进 state_dict
    state_dict = {k: v for k, v in ck["model"].items() if k != "_metadata"}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.ckpt} | {n_par/1e6:.2f}M params "
          f"| missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing or unexpected:
        print(f"[warn] missing[:5]={list(missing)[:5]} unexpected[:5]={list(unexpected)[:5]}",
              flush=True)

    sampler = FFM_sampler(model, model.gp)

    # ── 协议：留出 IC + 参考解族 ────────────────────────────────────
    grid = make_grid((nx, nt), dev)
    nu_ref = torch.linspace(1.0, 5.0, n_ref, device=dev)
    family = heat_family(PHI, nu_ref, nx, nt, dev)              # (n_ref, nx, nt)
    hfunc = make_residuals(cfg, family[0].cpu(), dev)
    print(f"[protocol] φ = π/4 = {PHI:.16f} | 参考解族 n_ref={n_ref} (ν∈[1,5]) "
          f"| grid {nx}x{nt}", flush=True)

    # ── 共用噪声库（配对比较）──────────────────────────────────────
    torch.manual_seed(SEED)
    bank = model.gp.sample(grid, (nx, nt), n_samples=n_sample).to(dev)
    print(f"[protocol] 共用 GP 噪声库 {tuple(bank.shape)} (seed={SEED})", flush=True)

    # 探针：D-Flow 内部自己抽噪声，重新 seed 后必须复现同一批，否则配对性失效
    torch.manual_seed(SEED)
    probe = model.gp.sample(grid, (nx, nt), n_samples=n_sample).to(dev)
    replay_err = float((probe - bank).abs().max())
    print(f"[check] dflow noise replay: max|Δ| = {replay_err:.3e} "
          f"→ {'OK' if replay_err == 0.0 else '不一致（D-Flow 无法配对，需注明）'}",
          flush=True)

    # 两个基线需要的目标场 / 掩码（我方推测，见文件头）
    u1_true = family.mean(0, keepdim=True).expand(n_sample, nx, nt).contiguous()
    mask = torch.ones_like(u1_true)

    # ── ECI 约束 ───────────────────────────────────────────────────
    eci_constraint = build_eci_constraint(family, nx, nt, dx, dev)
    # eci_sample 内部：resample_step=None → n_step*n_mix+1，永不重采样
    eci_resample = n_step * args.eci_n_mix + 1
    print(f"[eci] n_mix={args.eci_n_mix} | resample_step=None → {eci_resample} "
          f"(循环上限 {n_step * args.eci_n_mix}，故全程不重采样)", flush=True)
    print(f"[baselines] u1_true = family.mean(0) 复制 {n_sample} 份；mask = 全 1；"
          f"loss = masked-MSE  ← 我方推测，论文 setting 未公开", flush=True)

    rows = []

    def call(name, params, fn):
        """跑一个方法并记录指标；异常也记录下来，不中断其它方法。"""
        t0 = time.time()
        print(f"\n[{name}] {params}", flush=True)
        try:
            gen = fn()
            if not torch.isfinite(gen).all():
                raise FloatingPointError("生成结果含 NaN/Inf")
            m = metrics(gen, family, hfunc)
            ok, err = True, ""
            del gen
            if dev == "cuda":
                torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            m = {"MMSE": float("nan"), "SMSE": float("nan"),
                 "CE_IC": float("nan"), "CE_CL": float("nan")}
            ok, err = False, f"{type(e).__name__}: {e}"
        dt = time.time() - t0
        rows.append({"method": name, "params": params, "ok": ok, "error": err,
                     "time_s": round(dt, 2), **m})
        print(f"  → MMSE {m['MMSE']:8.3f}  SMSE {m['SMSE']:8.3f}  "
              f"CE_IC {m['CE_IC']:.2e}  CE_CL {m['CE_CL']:.2e}  [{dt:.1f}s]"
              + ("" if ok else f"  !! {err}"), flush=True)
        return m

    # ── 1. vanilla FFM（无约束）─────────────────────────────────────
    call("vanilla FFM", f"n_step={n_step}（Euler，无任何投影）",
         lambda: sampler.vanilla_sample(bank.clone(), n_step))

    # ── 2. PCFM（硬投影 IC + 守恒律）────────────────────────────────
    call("PCFM", f"n_step={n_step}, newtonsteps=1, guided_interpolation=False"
                 f"（每步投影 + 末端投影）",
         lambda: sampler.pcfm_sample(bank.clone(), n_step, hfunc,
                                     guided_interpolation=False))

    # ── 3. ECI（Cheng et al.）──────────────────────────────────────
    call("ECI", f"n_step={n_step}, n_mix={args.eci_n_mix}, "
                f"resample_step=None(→{eci_resample}，不重采样), "
                f"constraint=Chain(RegionConservationLaw, InitialCondition)",
         lambda: sampler.eci_sample(bank.clone(), n_step, args.eci_n_mix,
                                    None, eci_constraint))

    # ── 4. DiffusionPDE 的 guided sampling ─────────────────────────
    call("DiffusionPDE guided", f"n_step={n_step}, eta=2e2, loss=masked-MSE, "
                                f"u1_true=family.mean, mask=ones  ← 近似",
         lambda: sampler.guided_sample(bank.clone(), u1_true, mask, n_step,
                                       masked_mse, eta=2e2))

    # ── 5. D-Flow ──────────────────────────────────────────────────
    def _dflow():
        # dflow_sample 内部自抽噪声；重新 seed 以复现共用噪声库（上面已校验）
        torch.manual_seed(SEED)
        return sampler.dflow_sample(u1_true, mask, n_sample, n_step,
                                    n_iter=20, lr=1e-1, loss_fn=masked_mse)

    call("D-Flow", f"n_step={n_step}, n_iter=20, lr=1e-1 (LBFGS), "
                   f"loss=masked-MSE, u1_true=family.mean, mask=ones  ← 近似",
         _dflow)

    # ── 报告 ───────────────────────────────────────────────────────
    meta = {
        "mode": args.mode, "ckpt": os.path.abspath(args.ckpt),
        "torch": torch.__version__, "device": dev,
        "phi": PHI, "n_sample": n_sample, "n_step": n_step, "n_ref": n_ref,
        "seed": SEED, "grid": [nx, nt],
        "eci_n_mix": args.eci_n_mix, "eci_resample_step": eci_resample,
        "dflow_noise_replay_max_abs_diff": replay_err,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    report = render_report(meta, rows)
    print("\n" + report, flush=True)

    os.makedirs(OUT, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"[saved] {os.path.abspath(args.out)}", flush=True)

    bad = [r["method"] for r in rows if not r["ok"]]
    if bad:
        print(f"[FAIL] 失败的方法：{bad}", flush=True)
        return 3
    return 0


# ══════════════════════════════════════════════════════════════════════
# 报告渲染
# ══════════════════════════════════════════════════════════════════════
def render_report(meta, rows):
    L = []
    L.append("=" * 108)
    L.append("PCFM 复现 · Heat 五方法对照（vanilla / PCFM / ECI / DiffusionPDE-guided / D-Flow）")
    L.append("=" * 108)
    L.append(f"checkpoint : {meta['ckpt']}")
    L.append(f"环境       : torch {meta['torch']} / device {meta['device']} / "
             f"grid {meta['grid'][0]}x{meta['grid'][1]}")
    L.append(f"协议       : φ = π/4 ({meta['phi']:.16f})，参考解族 n_ref={meta['n_ref']} (ν∈[1,5])")
    L.append(f"配对       : 共用 GP 噪声 seed={meta['seed']}，n_sample={meta['n_sample']}，"
             f"n_step={meta['n_step']}")
    L.append(f"时间       : {meta['created']}")
    L.append("")
    L.append("指标口径：run_heat_official.metrics()，均已 ×1e2，与论文 Table 3 的「1e-2 单位」同量纲。")
    L.append("  MMSE = ||E[gen]-E[ref]||²   SMSE = ||Std[gen]-Std[ref]||²")
    L.append("  CE_IC = IC 残差 L2           CE_CL = 守恒律（质量）残差 L2")
    L.append("")
    L.append("-" * 108)
    L.append(f"{'method':<22s}{'MMSE':>10s}{'SMSE':>10s}{'CE_IC':>11s}{'CE_CL':>11s}"
             f"{'time':>8s}   关键超参")
    L.append("-" * 108)
    for r in rows:
        L.append(f"{r['method']:<22s}"
                 f"{fmt_metric(r['MMSE'], 10, '.3f')}"
                 f"{fmt_metric(r['SMSE'], 10, '.3f')}"
                 f"{fmt_metric(r['CE_IC'], 11, '.2e')}"
                 f"{fmt_metric(r['CE_CL'], 11, '.2e')}"
                 f"{r['time_s']:>7.1f}s   {r['params']}")
        if not r["ok"]:
            L.append(f"{'':<22s}  ↑ 失败：{r['error']}")
    L.append("-" * 108)

    # ── 参考量级 ──────────────────────────────────────────────────
    L.append("")
    L.append("论文 Table 3（Heat 行，单位 1e-2）参考值：")
    L.append("  PCFM MMSE 0.241, ECI 0.697, D-Flow 1.97, vanilla FFM 4.56")
    L.append("  CE(IC)：PCFM 0, ECI 0, D-Flow 102, FFM 579")
    L.append("")

    return "\n".join(L)


# ══════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser(description="Heat 上的三个基线 vs vanilla / PCFM")
    ap.add_argument("--mode", choices=["smoke", "formal"], default="formal")
    ap.add_argument("--ckpt", default=os.path.join(OUT, "heat_30000.pt"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None, help="cuda / cpu；默认自动")
    ap.add_argument("--n-sample", type=int, default=None)
    ap.add_argument("--n-step", type=int, default=None)
    ap.add_argument("--n-ref", type=int, default=None)
    ap.add_argument("--eci-n-mix", type=int, default=10,
                    help="ECI 的 n_mix（官方推理脚本未公开，回落 10）")
    args = ap.parse_args()

    if args.mode == "smoke":
        args.n_sample = 2 if args.n_sample is None else args.n_sample
        args.n_step = 3 if args.n_step is None else args.n_step
        args.n_ref = 4 if args.n_ref is None else args.n_ref
        args.out = args.out or os.path.join(OUT, "baselines_smoke.txt")
    else:
        args.n_sample = 8 if args.n_sample is None else args.n_sample
        args.n_step = 20 if args.n_step is None else args.n_step
        args.n_ref = 32 if args.n_ref is None else args.n_ref
        args.out = args.out or os.path.join(OUT, "baselines_heat.txt")

    os.environ.setdefault("WANDB_MODE", "disabled")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
