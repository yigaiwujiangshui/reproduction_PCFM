# -*- coding: utf-8 -*-
"""复现实验的统一命令行入口。

对象：`experiments/` 下的全部实验脚本。
算法：注册表（名称 → 脚本 + 类别 + 说明）解析参数，以子进程转发调用，支持单实验与套件两种粒度。"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _bootstrap import OUT, PCFM, REPO  # noqa: E402

PY = sys.executable

# name: (脚本相对 experiments/ 的路径, 类别, 说明)
REGISTRY = {
    # ── 主实验 ───────────────────────────────────────────────
    "heat-train":       ("heat/run_heat_official.py",     "train", "热方程训练（可 --resume 续训）"),
    "heat-ablate":      ("heat/run_heat_official.py",     "eval",  "热方程消融：步数 / 牛顿步 / 罚项 λ"),
    "heat-ood":         ("heat/ood_heat.py",              "eval",  "分布外（φ 超出训练区间）泛化"),
    "heat-baselines":   ("heat/baselines_heat.py",        "eval",  "vanilla/PCFM/ECI/DiffusionPDE/D-Flow 对照"),

    # ── Burgers ─────────────────────────────────────────────
    "burgers-data":     ("core/gen_burgers_data.py",      "data",  "生成 Burgers 数据集（训练/测试/采样集）"),
    "burgers-train":    ("burgers/burgers_run.py",        "train", "Burgers 训练（smoke/train/eval 子命令）"),
    "burgers-eval":     ("burgers/burgers_run.py",        "eval",  "Burgers 评测（IC/BC 两情形，512 条参考族）"),
    "burgers-extend":   ("burgers/burgers_extend.py",     "train", "Burgers 续训（补齐与论文一致的遍历次数）"),

    # ── 补充与消融 ───────────────────────────────────────────
    "supp2":            ("ablation/supplementary2.py",    "ablation", "配点密度(修正版) / TVD / n_step"),

    
    # ── 流水线 ───────────────────────────────────────────────
    "pipeline":         ("pipeline/run_pipeline.py",      "train", "串行总控（burgers → eval → heat 续训）"),
}

SUITES = {
    "main": ["heat-ablate", "heat-ood", "heat-baselines"],
    "supp": ["supp2"],

}


def _path(name):
    rel, _, _ = REGISTRY[name]
    p = os.path.join(HERE, rel)
    if not os.path.exists(p):
        raise SystemExit(f"[FATAL] 脚本不存在: {p}")
    return p


def cmd_list():
    print(f"repro 根目录 : {REPO}")
    print(f"PCFM 上游    : {PCFM}")
    print(f"输出目录     : {OUT}\n")
    groups = [("主实验", "eval"), ("训练（耗时长）", "train"), ("数据", "data"),
              ("补充/消融", "ablation"), ("诊断论证", "diagnostic")]
    for title, cat in groups:
        rows = [(n, v) for n, v in REGISTRY.items() if v[1] == cat]
        if not rows:
            continue
        print(f"── {title} " + "─" * (60 - len(title)))
        width = max(len(n) for n, _ in rows)
        for n, (rel, _, desc) in rows:
            print(f"  {n:<{width}}  {desc}")
        print()
    print("套件：")
    for k, v in SUITES.items():
        print(f"  {k:<6} = {', '.join(v)}")


def cmd_env():
    import torch
    print(f"python      : {sys.version.split()[0]}  ({sys.executable})")
    print(f"torch       : {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu         : {torch.cuda.get_device_name(0)}  "
              f"{torch.cuda.get_device_properties(0).total_memory/2**30:.1f} GiB")
    for m in ("h5py", "yaml", "neuralop", "gpytorch", "torchdiffeq"):
        try:
            mod = __import__(m)
            print(f"{m:<12}: {getattr(mod, '__version__', 'ok')}")
        except Exception as e:
            print(f"{m:<12}: 缺失 ({type(e).__name__})")
    n_ckpt = len([f for f in os.listdir(OUT) if f.endswith(".pt")]) if os.path.isdir(OUT) else 0
    print(f"checkpoints : {n_ckpt} 个（{OUT}）")


def _run(name, extra):
    script = _path(name)
    cmd = [PY, script] + extra
    print(f"[run] {' '.join(cmd)}\n")
    r = subprocess.run(cmd, cwd=REPO)
    return r.returncode


def cmd_run(name, extra):
    if name not in REGISTRY:
        raise SystemExit(f"[FATAL] 未知实验: {name}\n用 `list` 查看全部")
    return _run(name, extra)


def cmd_suite(key, extra):
    if key not in SUITES:
        raise SystemExit(f"[FATAL] 未知套件: {key}\n可选: {list(SUITES)}")
    bad = 0
    for name in SUITES[key]:
        print(f"\n{'='*78}\n套件 {key} :: {name}\n{'='*78}")
        bad += _run(name, extra) != 0
    print(f"\n[suite {key}] 完成，失败 {bad} 项")
    return 1 if bad else 0


def main():
    argv = sys.argv[1:]
    # 分离 `--` 之后的透传参数
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    else:
        extra = []

    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    cmd = argv[0]
    if cmd == "list":
        cmd_list()
        return 0
    if cmd == "env":
        cmd_env()
        return 0
    if cmd == "run":
        if len(argv) < 2:
            raise SystemExit("[FATAL] 用法: run <name> [-- 透传参数]")
        return cmd_run(argv[1], extra)
    if cmd == "suite":
        if len(argv) < 2:
            raise SystemExit("[FATAL] 用法: suite <key> [-- 透传参数]")
        return cmd_suite(argv[1], extra)
    raise SystemExit(f"[FATAL] 未知命令: {cmd}（可用: list / env / run / suite）")


if __name__ == "__main__":
    sys.exit(main())
