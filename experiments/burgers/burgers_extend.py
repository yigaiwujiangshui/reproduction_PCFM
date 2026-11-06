# -*- coding: utf-8 -*-
"""Burgers 模型的续训。

对象：已有 checkpoint 与固定的 6400 条训练解。
算法：载入权重后以新的 Adam + 平台调度继续训练，按累计步数命名 checkpoint，使数据遍历次数与论文对齐。"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, REPO, PCFM  # noqa: E402,F401

sys.path.insert(0, HERE)

import numpy as np
import torch
from torch.utils.data import DataLoader

from burgers_run import (OUT, REPO, build, get_dataset, seed_all)  # 复用同一套构造


def step_of(path):
    """从文件名推断累计步数；优先读 checkpoint 内的 step 字段。"""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ck, dict) and ck.get("step"):
        return int(ck["step"])
    b = os.path.basename(path).replace(".pt", "")
    for tok in b.split("_"):
        if tok.isdigit():
            return int(tok)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=os.path.join(OUT, "burgers_20000.pt"),
                    help="起点 checkpoint")
    ap.add_argument("--steps", type=int, default=60000, help="追加训练步数")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4, help="续训初始学习率")
    ap.add_argument("--save-every", type=int, default=2500,
                    help="存盘间隔。默认 2500：补训过程中会留下多个 checkpoint，"
                         "下游可对任意一个做评测，不必等训练结束")
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--valid-batches", type=int, default=8)
    ap.add_argument("--config", default=os.path.join(REPO, "configs", "burgers1d.yml"))
    args = ap.parse_args()

    if not os.path.exists(args.base):
        sys.exit(f"[FATAL] 起点 checkpoint 不存在: {args.base}")

    # 统一转绝对路径：后续可能从任意目录调用，避免相对路径解析歧义
    args.base = os.path.abspath(args.base)

    os.environ.setdefault("WANDB_MODE", "disabled")
    import yaml
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUT, exist_ok=True)
    logf = os.path.join(OUT, "burgers_extend.log")
    fh = open(logf, "a", encoding="utf-8")

    def log(s=""):
        print(s, flush=True)
        fh.write(s + "\n")
        fh.flush()

    base_step = step_of(args.base)
    # 数据顺序偏移：不与前半段共用同一条 shuffle 序列
    seed_all(cfg["train"]["seed"] + base_step)

    log("=" * 78)
    log(f"Burgers 补充训练  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"[base] {args.base}  (已训 {base_step} 步)")
    log(f"[plan] 追加 {args.steps} 步 × batch {args.batch} = "
        f"{args.steps*args.batch/1e6:.2f} M 次采样 | 数据集 6400 条 → "
        f"重复 {(base_step+args.steps)*args.batch/6400:.0f} 遍（论文约 800 遍）")
    log(f"[opt ] Adam lr {args.lr:.1e} → 0（plateau）, 优化器状态不恢复")

    # 配置里的 datasets.root 是相对 PCFM 根的路径，此处解析成绝对路径。
    # 不用 os.chdir：全局切换工作目录会影响调用方，且让后续相对路径难以推理。
    _root = cfg["datasets"]["root"]
    if not os.path.isabs(_root):
        cfg["datasets"]["root"] = os.path.join(PCFM, _root)

    train_set, test_set = get_dataset(cfg["datasets"])
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True,
                        num_workers=0, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=args.batch, shuffle=True, num_workers=0)
    n_val = min(args.valid_batches, len(val_loader))

    model = build(cfg, dev)
    ck = torch.load(args.base, map_location=dev, weights_only=False)
    state_dict = {k: v for k, v in ck["model"].items() if k != "_metadata"}
    model.load_state_dict(state_dict, strict=False)
    log(f"[load] 权重已载入 ({len(state_dict)} 个张量)")

    o = cfg["train"]["optimizer"]
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           betas=(o["beta1"], o["beta2"]),
                           weight_decay=o["weight_decay"])
    s = cfg["train"]["scheduler"]
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=s["factor"], patience=s["patience"], min_lr=s["min_lr"])

    def validate():
        model.eval()
        ls = []
        with torch.no_grad():
            for i, xb in enumerate(val_loader):
                if i >= n_val:
                    break
                ls.append(model.get_loss(xb.to(dev)).item())
        model.train()
        return float(np.mean(ls))

    model.train()
    t0 = time.time()
    step = 0
    best = float("inf")
    while step < args.steps:
        for x in loader:
            x = x.to(dev)
            loss = model.get_loss(x)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["max_grad_norm"])
            opt.step()
            step += 1

            if step % args.log_every == 0:
                log(f"  step {step:>7}/{args.steps}  loss {loss.item():.6f}  "
                    f"lr {opt.param_groups[0]['lr']:.2e}  {time.time()-t0:.0f}s")

            if args.val_every and step % args.val_every == 0:
                v = validate()
                best = min(best, v)
                sched.step(v)
                log(f"  [val ] step {step:>7}  val_loss {v:.6f}  "
                    f"lr -> {opt.param_groups[0]['lr']:.2e}")

            cum = base_step + step
            if args.save_every and step % args.save_every == 0 and step < args.steps:
                p = os.path.join(OUT, f"burgers_ext_{cum}.pt")
                torch.save({"model": model.state_dict(), "cfg": cfg, "step": cum}, p)
                log(f"  [ckpt] {p}")

            if step >= args.steps:
                break

    cum = base_step + args.steps
    p = os.path.join(OUT, f"burgers_ext_{cum}.pt")
    torch.save({"model": model.state_dict(), "cfg": cfg, "step": cum}, p)
    log(f"[saved] {p}  (累计 {cum} 步, 用时 {(time.time()-t0)/60:.1f} min)")
    log(f"[done] 重复遍数 = {cum*args.batch/6400:.0f}（论文约 800）")
    fh.close()


if __name__ == "__main__":
    main()
