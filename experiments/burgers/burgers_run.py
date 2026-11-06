# -*- coding: utf-8 -*-
"""Burgers 1D 的训练与评测。

对象：两种受约束采样情形——固定初值变化边界（IC）、固定边界变化初值（BC）；参考族取 512 条。
算法：FFM 训练 FNO；IC 情形用初值 + 局部 Godunov 多步 + 质量守恒（式 25），BC 情形用 Dirichlet/Neumann + 质量守恒（式 24）；约束自 `t=1` 起算以避开角点不相容。
参考：附录 B.4、D.6、D.7。"""

import os

# wandb 在 neuralop 导入时会被连带 import，禁掉以免它往临时目录里写东西
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

import argparse
import gc
import json
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, REPO, PCFM  # noqa: E402,F401

REPO = PCFM

sys.path.insert(0, REPO)
sys.path.insert(0, HERE)

import h5py
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import get_dataset
from models import get_flow_model

# ★ 必须在 import pcfm.ffm_sampler 之前应用接口适配（见 pcfm_compat.py 顶部说明）
import pcfm_compat  # noqa: F401
from pcfm.constraints import Residuals
from pcfm.ffm_sampler import FFM_sampler
from pcfm.pcfm_sampling import make_grid

K_MULTISTEP = 5     # 论文 IC 情形的局部多步数 k
BC_START_STEP = 1   # BC 约束从 t 的第 1 步起生效（t=0 处 IC 自由）


# ══════════════════════════════════════════════════════════
# 约束：给官方 Residuals 补 __call__（薄子类，不改算法）
# ══════════════════════════════════════════════════════════
class BurgersICResiduals(Residuals):
    """IC 情形：IC + 局部多步 Godunov 残差(k=5) + 质量守恒。

    对应论文 Eq.(10)/(21)：h(u) = [u(·,0) - u_IC, 多步 Godunov, 质量守恒]。
    """

    case = "ic"

    def __call__(self, u_flat):
        return self.full_residual_burgers(u_flat, k=K_MULTISTEP)


class BurgersBCResiduals(Residuals):
    """BC 情形：BC（左 Dirichlet + 右零梯度 Neumann） + 质量守恒。"""

    case = "bc"

    def __call__(self, u_flat):
        return self.full_residual_burgers2(u_flat, start_step=BC_START_STEP)


# ══════════════════════════════════════════════════════════
# 基础工具
# ══════════════════════════════════════════════════════════
def seed_all(seed):
    """等价于 PCFM/scripts/training/utils.py::seed_all（不引入 easydict 依赖）。"""
    import random as _random
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    _random.seed(seed)


def load_cfg(path):
    """读 config；把 datasets.root 解析成绝对路径。

    官方 configs 里 root 写的是相对路径 'datasets/data'（假定 CWD = PCFM 仓库根），
    这里补一层解析，避免依赖当前工作目录。
    """
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    root = cfg["datasets"]["root"]
    if not os.path.isabs(root):
        cfg["datasets"]["root"] = os.path.normpath(os.path.join(REPO, root))
    return cfg


def build(cfg, dev):
    return get_flow_model(cfg["model"], cfg["encoder"]).to(dev)


def env_line():
    s = f"[env] python {sys.version.split()[0]} | torch {torch.__version__} | " \
        f"cuda {torch.cuda.is_available()}"
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        s += f" | GPU {p.name} {p.total_memory / 2 ** 30:.1f} GiB"
    return s


# ══════════════════════════════════════════════════════════
# 训练（与 PCFM/scripts/training/main.py 同逻辑，去掉 tensorboard/tqdm）
# ══════════════════════════════════════════════════════════
def train_model(args, cfg, dev, log, save_every=2000, prefix="burgers"):
    t_cfg = cfg["train"]
    seed_all(t_cfg["seed"])

    train_set, test_set = get_dataset(cfg["datasets"])
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)          # ★ 沙箱禁多进程
    model = build(cfg, dev)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"[data] train {len(train_set)} 条 | test {len(test_set)} 条 | "
        f"batch {args.batch} | num_workers 0 | drop_last True")
    log(f"[model] {n_par / 1e6:.2f}M params | FNO n_modes {cfg['encoder']['n_modes']} "
        f"hidden {cfg['encoder']['hidden_channels']} proj {cfg['encoder']['proj_channels']} "
        f"n_layers {cfg['encoder']['n_layers']}")

    o_cfg = t_cfg["optimizer"]
    lr = args.lr if getattr(args, "lr", None) else o_cfg["lr"]
    opt = torch.optim.Adam(model.parameters(), lr=lr,
                           betas=(o_cfg["beta1"], o_cfg["beta2"]),
                           weight_decay=o_cfg["weight_decay"])
    s_cfg = t_cfg["scheduler"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=s_cfg["factor"], patience=s_cfg["patience"],
        min_lr=s_cfg["min_lr"])
    log(f"[opt] Adam lr {lr:.1e} betas ({o_cfg['beta1']},{o_cfg['beta2']}) "
        f"wd {o_cfg['weight_decay']} | plateau factor {s_cfg['factor']} "
        f"patience {s_cfg['patience']} min_lr {s_cfg['min_lr']:.1e}")

    total = args.steps
    val_every = args.val_every
    log_every = args.log_every
    max_gn = t_cfg["max_grad_norm"]
    val_loader = DataLoader(test_set, batch_size=args.batch, shuffle=True, num_workers=0)
    n_val_batches = min(args.valid_batches, len(val_loader))

    def validate():
        model.eval()
        losses = []
        with torch.no_grad():
            for i, xb in enumerate(val_loader):
                if i >= n_val_batches:
                    break
                xb = xb.to(dev)
                losses.append(model.get_loss(xb).item())
        model.train()
        return float(np.mean(losses))

    if save_every:
        os.makedirs(OUT, exist_ok=True)

    step, t0 = 0, time.time()
    model.train()
    while step < total:
        for x in loader:
            x = x.to(dev)
            loss = model.get_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_gn)
            opt.step()
            step += 1

            if step == 1 or step % log_every == 0:
                log(f"  step {step:6d}/{total}  loss {loss.item():.6f}  "
                    f"|g| {float(gnorm):8.2f}  lr {opt.param_groups[0]['lr']:.2e}  "
                    f"{time.time() - t0:6.0f}s")

            if val_every and (step % val_every == 0 or step == total):
                vl = validate()
                sched.step(vl)
                log(f"  [val ] step {step:6d}  val_loss {vl:.6f}  "
                    f"lr -> {opt.param_groups[0]['lr']:.2e}  {time.time() - t0:6.0f}s")

            if save_every and step % save_every == 0:
                p = os.path.join(OUT, f"{prefix}_{step}.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg,
                            "step": step, "loss": float(loss.item())}, p)
                log(f"  [ckpt] {p}  ({time.time() - t0:.0f}s)")

            if step >= total:
                break

    if not save_every or step % save_every != 0:
        os.makedirs(OUT, exist_ok=True)
        p = os.path.join(OUT, f"{prefix}_{step}.pt")
        torch.save({"model": model.state_dict(), "cfg": cfg, "step": step}, p)
        log(f"[saved] {p}")
    elapsed = time.time() - t0
    log(f"[train] 完成 {step} 步，用时 {elapsed / 60:.1f} min "
        f"({elapsed / max(step, 1):.3f} s/step)")
    return model, step, elapsed


# ══════════════════════════════════════════════════════════
# 数据 / 评测
# ══════════════════════════════════════════════════════════
def load_test_arrays(cfg):
    root = cfg["datasets"]["root"]
    path = os.path.join(root, cfg["datasets"]["test"]["data_file"])
    with h5py.File(path, "r") as f:
        u = f["u"][...].astype(np.float32)      # (N_ic, N_bc, nx, nt)
        ic = f["ic"][...]                       # p_loc per IC
        bc = f["bc"][...]                       # u_bc per BC
        x = f["x"][...].astype(np.float32)
        t = f["t"][...].astype(np.float32)
    return u, ic, bc, x, t, path


def make_ic_residuals(u, x, t, i_ic, dev):
    """IC 情形：参考族 = 该 IC 下所有 BC 的真解。"""
    nx, nt = len(x), len(t)
    ref = torch.from_numpy(u[i_ic]).contiguous()             # (N_bc, nx, nt)
    hfunc = BurgersICResiduals(ref[0].unsqueeze(0).to(dev),
                               torch.from_numpy(x).to(dev),
                               torch.from_numpy(t).to(dev), nx=nx, nt=nt)
    return ref.to(dev), hfunc


def make_bc_residuals(u, bc, x, t, i_bc, dev):
    """BC 情形：参考族 = 该 BC 下所有 IC 的真解。"""
    nx, nt = len(x), len(t)
    ref = torch.from_numpy(u[:, i_bc]).contiguous()           # (N_ic, nx, nt)
    hfunc = BurgersBCResiduals(ref[0].unsqueeze(0).to(dev),
                               torch.from_numpy(x).to(dev),
                               torch.from_numpy(t).to(dev), nx=nx, nt=nt,
                               left_bc=torch.tensor(float(bc[i_bc])))
    return ref.to(dev), hfunc


@torch.no_grad()
def case_metrics(gen, ref, hfunc):
    """MMSE / SMSE（论文单位 1e-2）+ 约束误差 CE。

    CE 拆成两段：head（IC 或 BC 残差）与 CL（连续律：多步 Godunov + 质量守恒）。
    head 长度直接由官方残差函数给出，不硬编码。
    """
    mmse = ((gen.mean(0) - ref.mean(0)) ** 2).mean()
    smse = ((gen.std(0) - ref.std(0)) ** 2).mean()

    u0 = gen[0].reshape(-1)
    if hfunc.case == "ic":
        head_n = hfunc.ic_residual(u0).numel()
    else:
        head_n = hfunc.bc_residual_burgers(u0, BC_START_STEP).numel()

    ce_head, ce_cl = [], []
    for i in range(gen.shape[0]):
        r = hfunc(gen[i].reshape(-1))
        ce_head.append(r[:head_n].pow(2).sum().sqrt())
        ce_cl.append(r[head_n:].pow(2).sum().sqrt())
    k = 1e2
    return {
        "MMSE": float(mmse) * k,
        "SMSE": float(smse) * k,
        "CE_HEAD": float(torch.stack(ce_head).mean()) * k,
        "CE_CL": float(torch.stack(ce_cl).mean()) * k if ce_cl else 0.0,
        "head_n": head_n,
        "res_n": int(head_n + (r.numel() - head_n)),
    }


def run_case(sampler, noise, n_step, ref, hfunc, dev, log, label, head_name, newtonsteps=1):
    """跑 vanilla / PCFM 两条，返回行列表。"""
    rows = []

    def rec(name, gen):
        m = case_metrics(gen, ref, hfunc)
        m["config"] = name
        rows.append(m)
        log(f"  {name:<22} MMSE {m['MMSE']:9.3f}  SMSE {m['SMSE']:9.3f}  "
            f"CE({head_name}) {m['CE_HEAD']:.3e}  CE(CL) {m['CE_CL']:.3e}")
        return m

    with torch.no_grad():
        r_ref = hfunc(ref[0].reshape(-1))
        ref_res = float(r_ref.pow(2).sum().sqrt())
        log(f"\n[{label}] 参考族 {tuple(ref.shape)} | 约束 {hfunc.case.upper()} "
            f"| 残差维数 {r_ref.numel()}")
        log(f"  [check] 参考解自身残差 L2 = {ref_res * 1e2:.3f}（单位 1e-2）"
            f" = CE 下限（离散格式误差，非 0）")

    log(f"  (a) vanilla FFM  n_step={n_step}")
    rec(f"vanilla n_step={n_step}", sampler.vanilla_sample(noise, n_step))

    log(f"  (b) PCFM（硬投影，无罚项）  n_step={n_step}  newtonsteps={newtonsteps}")
    rec(f"PCFM n_step={n_step} newtonsteps={newtonsteps}",
        sampler.pcfm_sample(noise, n_step, hfunc, newtonsteps=newtonsteps,
                            guided_interpolation=False))
    return rows, ref_res


def eval_ckpt(args, cfg, dev, log):
    """加载 checkpoint，做 IC 与 BC 两个情形。"""
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = build(cfg, dev)
    state_dict = {k: v for k, v in ck["model"].items() if k != "_metadata"}  # 过滤 tensorly 元数据
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        log(f"[warn] missing={list(missing)[:5]} unexpected={list(unexpected)[:5]}")
    step = int(ck.get("step") or 0)
    model.eval()
    log(f"[ckpt] {args.ckpt}  (训练步数 {step})")
    sampler = FFM_sampler(model, model.gp)

    u, ic, bc, x, t, path = load_test_arrays(cfg)
    nx, nt = len(x), len(t)
    log(f"[data] {path} | u {u.shape} | nx {nx} nt {nt} | "
        f"x[{x[0]:.2f},{x[-1]:.2f}] t[{t[0]:.2f},{t[-1]:.2f}]")

    grid = make_grid((nx, nt), dev)
    torch.manual_seed(args.seed)
    noise = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)
    log(f"[protocol] n_sample {args.n_sample} | n_step {args.n_step} | "
        f"共用噪声库 {tuple(noise.shape)} | seed {args.seed}")

    u_ic_case, u_bc_case = u, u
    ic_vals, bc_vals = ic, bc
    used_ref_ic, used_ref_bc = int(u.shape[1]), int(u.shape[0])
    root = cfg["datasets"]["root"]
    f_ic = os.path.join(root, "burgers_sampling_diffICs_nIC20_nBC512.h5")
    f_bc = os.path.join(root, "burgers_sampling_diffBCs_nBC20_nIC512.h5")
    if os.path.exists(f_ic) and os.path.exists(f_bc):
        with h5py.File(f_ic, "r") as f:
            u_ic_case = f["u"][...].astype(np.float32)      # (20 IC, 512 BC)
            ic_vals = f["ic"][...]
        with h5py.File(f_bc, "r") as f:
            u_bc_case = f["u"][...].astype(np.float32)      # 磁盘上是 (N_bc, N_ic)=(20,512)
            bc_vals = f["bc"][...]
            u_bc_case = np.transpose(u_bc_case, (1, 0, 2, 3)).copy()   # (512, 20, 101, 101)
        used_ref_ic = int(u_ic_case.shape[1])   # (N_ic, N_bc) → 沿 BC 轴
        used_ref_bc = int(u_bc_case.shape[0])   # 转置后 (N_ic, N_bc) → 沿 IC 轴
        log(f"[ref ] 使用受约束采样集：IC 情形 {used_ref_ic} 个 BC，"
            f"BC 情形 {used_ref_bc} 个 IC（论文 B.4 口径）")
    else:
        log("[ref ] 未找到采样集，回退到 test 集（参考族偏小，口径与论文有差）")

    res = {"ckpt": os.path.abspath(args.ckpt), "step": step,
           "n_sample": args.n_sample, "n_step": args.n_step,
           "newtonsteps": args.newtonsteps,
           "ic_index": args.ic_index, "bc_index": args.bc_index,
           "n_ref_ic": used_ref_ic, "n_ref_bc": used_ref_bc}

    # ── IC 情形 ────────────────────────────────────────
    log(f"\n{'=' * 78}\nIC 情形：固定 IC #{args.ic_index} "
        f"(p_loc = {float(ic_vals[args.ic_index]):.4f})，参考族 = 该 IC 的 {used_ref_ic} 个 BC 真解")
    ref_ic, h_ic = make_ic_residuals(u_ic_case, x, t, args.ic_index, dev)
    log(f"  约束维数：IC {nx} + 多步({K_MULTISTEP}×{nx - 2}) + 质量 {nt - 1} "
        f"= {h_ic(torch.zeros(nx * nt, device=dev)).numel()}")
    rows_ic, ref_ic_res = run_case(sampler, noise, args.n_step, ref_ic, h_ic, dev, log, "IC", "IC", args.newtonsteps)
    res["ic"] = {"p_loc": float(ic_vals[args.ic_index]), "rows": rows_ic,
                 "ref_res": ref_ic_res}

    # ── BC 情形 ────────────────────────────────────────
    log(f"\n{'=' * 78}\nBC 情形：固定 BC #{args.bc_index} "
        f"(u_bc = {float(bc_vals[args.bc_index]):.4f})，参考族 = 该 BC 的 {used_ref_bc} 个 IC 真解")
    ref_bc, h_bc = make_bc_residuals(u_bc_case, bc_vals, x, t, args.bc_index, dev)
    log(f"  约束维数：BC {2 * (nt - BC_START_STEP)} + 质量 {nt - 1} "
        f"= {h_bc(torch.zeros(nx * nt, device=dev)).numel()}")
    rows_bc, ref_bc_res = run_case(sampler, noise, args.n_step, ref_bc, h_bc, dev, log, "BC", "BC", args.newtonsteps)
    res["bc"] = {"u_bc": float(bc_vals[args.bc_index]), "rows": rows_bc,
                 "ref_res": ref_bc_res}

    return res, step


def ratio(rows, key):
    """vanilla 与 PCFM 的指标比值；任一缺失时返回 None。"""
    by_config = {row["config"].split()[0]: row for row in rows}
    vanilla, pcfm = by_config.get("vanilla"), by_config.get("PCFM")
    if not (vanilla and pcfm):
        return None
    return v[key] / p[key]


def write_report(path, res, step, cfg, args, elapsed=None):
    """把评测结果写成数据表：仅设定与数值，不含解读。"""
    lines = []
    add = lines.append

    add("=" * 78)
    add("PCFM · Burgers 1D 评测结果")
    add("=" * 78)
    add(f"checkpoint  : {res['ckpt']}  (训练步数 {step})")
    add(f"环境        : {env_line()}")
    if elapsed is not None:
        add(f"评测耗时    : {elapsed:.1f} s")
    add("")
    add("[训练设定]")
    add(f"  max_iter {cfg['train']['max_iter']} | batch {args.batch} | "
        f"lr {cfg['train']['optimizer']['lr']:.1e} | plateau "
        f"factor {cfg['train']['scheduler']['factor']} patience "
        f"{cfg['train']['scheduler']['patience']} min_lr "
        f"{cfg['train']['scheduler']['min_lr']:.1e} | max_grad_norm "
        f"{cfg['train']['max_grad_norm']}")
    add(f"  FNO n_modes {cfg['encoder']['n_modes']} hidden {cfg['encoder']['hidden_channels']} "
        f"proj {cfg['encoder']['proj_channels']} n_layers {cfg['encoder']['n_layers']} "
        f"emb {cfg['encoder']['emb_channels']} | GP Matérn-1/2 "
        f"length {cfg['model']['kernel_length']} var {cfg['model']['kernel_variance']}")
    add("")
    add("[评测设定]")
    add(f"  n_sample {res['n_sample']} | n_step {res['n_step']} | "
        f"newtonsteps {res.get('newtonsteps', 1)} | seed {res['seed']}")
    add("")
    for tag, key in (("IC 情形", "ic"), ("BC 情形", "bc")):
        r = res[key]
        add("-" * 78)
        if key == "ic":
            add(f"{tag}：固定 IC #{res['ic_index']} (p_loc = {r['p_loc']:.4f})，"
                f"参考族 {r.get('n_ref_ic', '?')} 条")
            ce_name = "CE(IC)"
        else:
            add(f"{tag}：固定 BC #{res['bc_index']} (u_bc = {r['u_bc']:.4f})，"
                f"参考族 {r.get('n_ref_bc', '?')} 条")
            ce_name = "CE(BC)"
        add("")
        add(f"  {'config':<24}{'MMSE':>9}{'SMSE':>9}{ce_name:>12}{'CE(CL)':>12}")
        for row in r["rows"]:
            add(f"  {row['config']:<24}{row['MMSE']:>9.3f}{row['SMSE']:>9.3f}"
                f"{row['CE_HEAD']:>12.3e}{row['CE_CL']:>12.3e}")
        add(f"  {'(参考真解)':<24}{'':>9}{'':>9}"
            f"{r.get('ref_res', float('nan')) * 1e2:>12.3e}{'':>12}")
        add("")

    txt = "\n".join(lines) + "\n"
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


# ══════════════════════════════════════════════════════════
# 命令
# ══════════════════════════════════════════════════════════
def cmd_smoke(args, cfg, dev):
    lines = []

    def log(s=""):
        print(s, flush=True)
        lines.append(s)

    t_all = time.time()
    log("=" * 78)
    log("Burgers 1D 冒烟测试：训练 20 步 → IC / BC 各采一次样")
    log("=" * 78)
    log(env_line())
    log(f"[cfg] {args.config}")
    log(f"[smoke] steps {args.steps} | batch {args.batch} | "
        f"n_sample {args.n_sample} | n_step {args.n_step} | n_sample≥2 才能算 std")
    log("")
    log("[1/3] 训练")
    t0 = time.time()
    model, step, _ = train_model(args, cfg, dev, log, save_every=0, prefix="_smoke")
    log(f"  训练 {step} 步完成，{time.time() - t0:.1f}s")

    log("\n[2/3] 评测（IC + BC 各一次采样）")
    sampler = FFM_sampler(model, model.gp)
    model.eval()
    u, ic, bc, x, t, path = load_test_arrays(cfg)
    nx, nt = len(x), len(t)
    log(f"  [data] {os.path.basename(path)} | u {u.shape}")
    grid = make_grid((nx, nt), dev)
    torch.manual_seed(args.seed)
    noise = model.gp.sample(grid, (nx, nt), n_samples=args.n_sample).to(dev)
    log(f"  [noise] {tuple(noise.shape)}")

    res = {"ckpt": "（内存中的冒烟模型）", "step": step,
           "n_sample": args.n_sample, "n_step": args.n_step,
           "seed": args.seed, "ic_index": args.ic_index, "bc_index": args.bc_index,
           "n_ref_ic": int(u.shape[1]), "n_ref_bc": int(u.shape[0])}

    ref_ic, h_ic = make_ic_residuals(u, x, t, args.ic_index, dev)
    rows_ic, ref_res_ic = run_case(sampler, noise, args.n_step, ref_ic, h_ic,
                                   dev, log, "IC", "IC")
    res["ic"] = {"p_loc": float(ic[args.ic_index]), "rows": rows_ic,
                 "ref_res": ref_res_ic}

    ref_bc, h_bc = make_bc_residuals(u, bc, x, t, args.bc_index, dev)
    rows_bc, ref_res_bc = run_case(sampler, noise, args.n_step, ref_bc, h_bc,
                                   dev, log, "BC", "BC")
    res["bc"] = {"u_bc": float(bc[args.bc_index]), "rows": rows_bc,
                 "ref_res": ref_res_bc}

    log("\n[3/3] 汇总")
    log(f"  总耗时 {time.time() - t_all:.1f}s")
    log(f"  峰值显存 {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
        if torch.cuda.is_available() else "")
    log("  冒烟判定：训练前向/反向、约束构造、PCFM 投影、两个情形采样全部跑通 → PASS")

    txt = "\n".join(lines) + "\n"
    txt += "\n[小结] 冒烟仅验证链路可用，数值无意义（20 步训练 / n_step=" \
           f"{args.n_step} 采样）\n"
    p = os.path.join(OUT, "burgers_smoke.txt")
    os.makedirs(OUT, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"[saved] {p}", flush=True)
    return 0


def cmd_train(args, cfg, dev):
    logf = os.path.join(OUT, "burgers_train.log")
    os.makedirs(OUT, exist_ok=True)
    fh = open(logf, "a", encoding="utf-8")

    def log(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    log("=" * 78)
    log(f"Burgers 1D 正式训练  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(env_line())
    log(f"[cfg] {args.config} | steps {args.steps} | batch {args.batch} | "
        f"save_every {args.save_every}")
    train_model(args, cfg, dev, log, save_every=args.save_every, prefix="burgers")
    fh.close()
    return 0


def cmd_eval(args, cfg, dev):
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log(env_line())
    res, step = eval_ckpt(args, cfg, dev, log)
    res["seed"] = args.seed
    res["ic_index"] = args.ic_index
    res["bc_index"] = args.bc_index

    txt = write_report(args.out, res, step, cfg, args)

    # 同时存一份 json，便于后续跨 checkpoint 汇总
    jp = os.path.splitext(args.out)[0] + f"_{step}.json"
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"\n[saved] {args.out}\n[saved] {jp}", flush=True)
    return 0


def main():
    ap = argparse.ArgumentParser(description="PCFM · Burgers 1D 训练与评测")
    ap.add_argument("cmd", choices=["smoke", "train", "eval"])
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "burgers1d.yml"))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=64,
                    help="论文 256，本机按显存降至 64")
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=200,
                    help="对齐 configs 的 val_freq（plateau 调度需要）")
    ap.add_argument("--valid-batches", type=int, default=8,
                    help="configs 的 valid_max_batch=32，这里降为 8 省时间")
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--n-sample", type=int, default=8)
    ap.add_argument("--n-step", type=int, default=20)
    ap.add_argument("--newtonsteps", type=int, default=1,
                    help="末端/每步投影的牛顿步数；论文定理要求 sufficient iterations")
    ap.add_argument("--ic-index", type=int, default=0)
    ap.add_argument("--bc-index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(OUT, "burgers_results.txt"))
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    if args.cmd == "smoke" and args.steps > 200:
        args.steps = 200       # 冒烟上限，防止手误跑成正式训练
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] dev {dev}", flush=True)

    if args.cmd == "smoke":
        return cmd_smoke(args, cfg, dev)
    if args.cmd == "train":
        return cmd_train(args, cfg, dev)
    if not args.ckpt:
        sys.exit("eval 需要 --ckpt")
    return cmd_eval(args, cfg, dev)


if __name__ == "__main__":
    sys.exit(main())
