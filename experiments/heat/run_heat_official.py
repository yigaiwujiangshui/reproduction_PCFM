# -*- coding: utf-8 -*-
"""热方程主实验：训练、评测与消融。

对象：`u_t = α u_xx`，`x ∈ [0,2π]`，`100×100` 网格；解析解 `u = e^{-αt} sin(x+φ)`，训练时 `α~U(1,5)`、`φ~U(0,π)`。
算法：FFM 训练 FNO；PCFM 采样在每步做高斯–牛顿投影，末端再投影一次；指标为 MMSE / SMSE / CE。
参考：附录 B.1（数据）、D.3 与式 (21)（约束 `H_Heat`）、Table 3。"""

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, REPO, PCFM  # noqa: E402,F401

REPO = PCFM

sys.path.insert(0, REPO)

import numpy as np
import torch
import yaml

from datasets import get_dataset
from models import get_flow_model

# 必须在 import pcfm.ffm_sampler 之前应用接口适配（见 pcfm_compat.py 顶部说明）
import pcfm_compat
from pcfm_compat import CallableResiduals

from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid


def seed_all(seed):
    """等价于 PCFM/scripts/training/utils.py::seed_all。

    官方版本还会 import easydict（用于 load_config），而 easydict 不在依赖里；
    这里只复刻种子设置，避免为一个 4 行函数引入新依赖。
    """
    import random as _random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)

TWO_PI = 2.0 * np.pi


# ══════════════════════════════════════════════════════════
def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build(cfg, dev):
    model = get_flow_model(cfg["model"], cfg["encoder"]).to(dev)
    return model


def make_residuals(cfg, ic_solution, dev):
    """ic_solution: (nx, nt) 参考解，用它第 0 列作为 IC 目标。

    返回 CallableResiduals：继承官方 Residuals 并补上 __call__，
    因为 pcfm_sampling.py:145 直接写 `hfunc(u_corr)`，要求可调用。
    """
    ds = cfg["datasets"]
    nx, nt = ds["nx"], ds["nt"]
    x = torch.linspace(0.0, TWO_PI, nx + 1, device=dev)[:-1]
    t = torch.linspace(ds["t_range"][0], ds["t_range"][1], nt, device=dev)
    dx = TWO_PI / nx
    data = ic_solution.unsqueeze(0).to(dev)          # (1, nx, nt) -> data[0] 为 (nx,nt)
    return CallableResiduals(data, x, t,
                             dx=torch.tensor(dx, device=dev), nx=nx, nt=nt)


def heat_family(phi, nu_list, nx, nt, dev):
    """同一 φ、不同 ν 的分析解族 —— 共享同一 IC。"""
    x = torch.linspace(0.0, TWO_PI, nx + 1, device=dev)[:-1].view(1, nx, 1)
    t = torch.linspace(0.0, 1.0, nt, device=dev).view(1, 1, nt)
    phi = torch.as_tensor(phi, device=dev).view(-1, 1, 1)
    nu = torch.as_tensor(nu_list, device=dev).view(-1, 1, 1)
    return torch.sin(x + phi) * torch.exp(-t * nu)


# ══════════════════════════════════════════════════════════
# 训练（与 main.py 同逻辑，去掉 tensorboard / tqdm 噪音）
# ══════════════════════════════════════════════════════════
def cmd_train(args, cfg, dev):
    # 数据集的 (ν, φ) 来自 Python random，t 与 GP 噪声来自 torch，
    # 三者都要种子，否则：只种 torch 时数据流每进程不同（不复现）；
    # 而若一律用同一个 seed，续训会把前 N 条数据原样重放（重复）。
    # 故：用 seed + resume_step 作为偏移，既复现又不重叠。
    resume_step = 0
    if getattr(args, "resume", None) and os.path.exists(args.resume):
        _ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        resume_step = int(_ck.get("step") or 0)
        if resume_step == 0:                       # 旧 checkpoint 没存 step，从文件名推断
            base = os.path.basename(args.resume)
            try:
                resume_step = int(base.replace("heat_", "").replace(".pt", ""))
            except ValueError:
                resume_step = 0
        del _ck

    seed_all(cfg["train"]["seed"] + resume_step)
    train_set, test_set = get_dataset(cfg["datasets"])
    loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    model = build(cfg, dev)
    # 断点续训：加载已有权重，优化器状态不恢复，用新的余弦调度
    if getattr(args, "resume", None):
        ck = torch.load(args.resume, map_location=dev, weights_only=False)
        state_dict = {k: v for k, v in ck["model"].items() if k != "_metadata"}
        model.load_state_dict(state_dict, strict=False)
        print(f"[resume] {args.resume}  (原训练步数 {resume_step}, "
              f"数据种子 {cfg['train']['seed'] + resume_step})")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_par/1e6:.2f}M params | train {len(train_set)} | test {len(test_set)}")

    base_lr = args.lr if getattr(args, "lr", None) else cfg["train"]["optimizer"]["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=base_lr)
    total = args.steps
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total)
    t0 = time.time()
    step = 0
    model.train()
    while step < total:
        for x in loader:
            x = x.to(dev)
            loss = model.get_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            opt.step()
            sched.step()
            step += 1
            if step % args.log_every == 0 or step == 1:
                print(f"  step {step:6d}/{total}  loss {loss.item():.6f}  "
                      f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s", flush=True)
            # 周期性存档，便于中途评估 / 断点续跑
            # 文件名用"累计步数"，避免续训时覆盖原有的 heat_2000 等同名文件
            cum = resume_step + step
            if args.save_every and step % args.save_every == 0 and step < total:
                os.makedirs(OUT, exist_ok=True)
                mid = os.path.join(OUT, f"heat_{cum}.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": cum}, mid)
                print(f"  [ckpt] {mid}", flush=True)
            if step >= total:
                break
    os.makedirs(OUT, exist_ok=True)
    cum_total = resume_step + total
    ck = os.path.join(OUT, f"heat_{cum_total}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": cum_total}, ck)
    print(f"[saved] {ck}  (累计 {cum_total} 步)")
    return model, ck


# ══════════════════════════════════════════════════════════
# 指标
# ══════════════════════════════════════════════════════════
@torch.no_grad()
def metrics(gen, ref, hfunc):
    """MMSE / SMSE（论文用 1e-2 单位）+ CE(IC) / CE(CL)"""
    mmse = ((gen.mean(0) - ref.mean(0)) ** 2).mean()
    smse = ((gen.std(0) - ref.std(0)) ** 2).mean()
    nx, nt = hfunc.nx, hfunc.nt
    ics, cls = [], []
    for i in range(gen.shape[0]):
        u = gen[i].reshape(-1)
        r = hfunc.full_residual_heat(u)
        ic = hfunc.ic_residual(u)
        cl = r[ic.shape[0]:]
        ics.append(ic.pow(2).sum().sqrt())
        cls.append(cl.pow(2).sum().sqrt() if cl.numel() else torch.tensor(0.0))
    k = 1e2
    return {
        "MMSE": float(mmse) * k,
        "SMSE": float(smse) * k,
        "CE_IC": float(torch.stack(ics).mean()) * k,
        "CE_CL": float(torch.stack(cls).mean()) * k,
    }


def cmd_ablate(args, cfg, dev):
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build(cfg, dev)
    # neuralop/tensorly 的谱卷积会把 "_metadata" 写进 state_dict，
    # torch>=2.x 严格模式下会被判为 Unexpected key —— 过滤掉
    state_dict = {k: v for k, v in ck["model"].items() if k != "_metadata"}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    model.eval()
    sampler = FFM_sampler(model, model.gp)

    ds = cfg["datasets"]
    nx, nt = ds["nx"], ds["nt"]
    grid = make_grid((nx, nt), dev)

    # 固定留出 IC：论文附录 B.1 明确 "fixing φ = π/4"
    if getattr(args, "phi", None) is not None:
        phi = float(args.phi)
    else:
        g = torch.Generator().manual_seed(args.seed)
        phi = float(torch.empty(1).uniform_(0.0, np.pi, generator=g))
    nu_ref = torch.linspace(1.0, 5.0, args.n_ref)
    family = heat_family(phi, nu_ref, nx, nt, dev)          # (n_ref, nx, nt)
    hfunc = make_residuals(cfg, family[0].cpu(), dev)
    print(f"[protocol] 固定 IC φ={phi:.4f} | 参考解族 n={args.n_ref} (ν∈[1,5])")

    # ★ 配对比较：所有配置共用同一批初始噪声
    #   否则每个配置抽到不同的 ξ，MMSE 差异里混杂了抽样噪声（实测同一配置可差 3 倍）
    #   同时固定 seed，使不同 ckpt / 不同 φ 的运行之间也可比
    torch.manual_seed(args.seed)
    noise_bank = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)
    print(f"[protocol] 共用噪声库 {tuple(noise_bank.shape)}（配对比较）")

    def noise(n):
        assert n <= noise_bank.shape[0], "n_sample 不得超过噪声库大小"
        return noise_bank[:n]

    rows = []

    def rec(name, gen):
        m = metrics(gen, family, hfunc)
        m["config"] = name
        rows.append(m)
        print(f"  {name:<38} MMSE {m['MMSE']:8.3f}  SMSE {m['SMSE']:8.3f}  "
              f"CE_IC {m['CE_IC']:.2e}  CE_CL {m['CE_CL']:.2e}")

    print("\n[a] vanilla FFM（无约束）")
    rec(f"vanilla n_step={args.n_step}",
        sampler.vanilla_sample(noise(args.n_sample), args.n_step))

    print("\n[b] PCFM（无罚项）")
    rec(f"PCFM n_step={args.n_step}",
        sampler.pcfm_sample(noise(args.n_sample), args.n_step, hfunc,
                            guided_interpolation=False))

    print("\n[c] 消融：采样步数")
    for ns in args.steps_list:
        rec(f"PCFM n_step={ns}",
            sampler.pcfm_sample(noise(args.n_sample), ns, hfunc,
                                guided_interpolation=False))

    print("\n[d] 消融：牛顿步数")
    for ns_newton in (1, 2):
        rec(f"PCFM newtonsteps={ns_newton}",
            sampler.pcfm_sample(noise(args.n_sample), args.n_step, hfunc,
                                newtonsteps=ns_newton, guided_interpolation=False))

    print("\n[e] 消融：松弛罚项 λ")
    for lam in args.lam_list:
        rec(f"PCFM guided lam={lam:.0e}",
            sampler.pcfm_sample(noise(args.n_sample), args.n_step, hfunc,
                                guided_interpolation=True,
                                interpolation_params={"custom_lam": lam,
                                                      "step_size": 1e-2,
                                                      "num_steps": 20}))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "heat_ablation.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"protocol": {"phi": phi, "n_ref": args.n_ref,
                                "n_sample": args.n_sample}, "rows": rows},
                  f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {p}")
    return rows


# ══════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "eval", "ablate"])
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "heat.yml"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-ref", type=int, default=32)
    ap.add_argument("--n-sample", type=int, default=16)
    ap.add_argument("--n-step", type=int, default=20)
    ap.add_argument("--steps-list", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--lam-list", type=float, nargs="+", default=[1e-2, 1e-1])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--phi", type=float, default=None,
                    help="留出 IC 的相位；论文附录 B.1 用 pi/4，传 0.7853981634")
    ap.add_argument("--resume", default=None, help="从已有 checkpoint 继续训练")
    ap.add_argument("--lr", type=float, default=None, help="续训时的初始学习率")
    args = ap.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(args.config)
    if args.batch:
        cfg["train"]["batch_size"] = args.batch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__} | cuda {torch.cuda.is_available()} | dev {dev}")

    if args.cmd == "train":
        _, ck = cmd_train(args, cfg, dev)
        print(f"\n下一步: python run_heat_official.py ablate --ckpt \"{ck}\"")
    else:
        if not args.ckpt:
            sys.exit("需要 --ckpt")
        cmd_ablate(args, cfg, dev)


if __name__ == "__main__":
    main()
