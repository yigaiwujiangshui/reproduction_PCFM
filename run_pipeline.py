# -*- coding: utf-8 -*-
"""
实验流水线总控
==============
按顺序执行一串任务，前一个结束（成功）才启动下一个；已有产物则跳过，支持断点续跑。

设计要点
--------
* **串行**：单张 16 GiB 显卡，并行会互相抢显存并让计时失效，故严格串行。
* **幂等**：每个任务有一个 `marker`（产物文件）。marker 存在即跳过 —— 中断后重跑不会重复计算。
* **状态落盘**：`out/pipeline_state.json` 记录每任务的状态、开始/结束时间、耗时、退出码。
* **失败即停**：默认遇到非零退出码就停下并保留现场（`--keep-going` 可改为继续）。
* **日志分离**：每个任务的 stdout/stderr 写到 `out/logs/<name>.log`，便于事后核对。

用法
----
    python run_pipeline.py --list          # 只看任务清单与状态
    python run_pipeline.py                 # 从第一个未完成的任务开始跑到底
    python run_pipeline.py --only ood      # 只跑名字里含 ood 的任务
    python run_pipeline.py --from burgers  # 从 burgers 开始
    python run_pipeline.py --keep-going    # 某任务失败也继续
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PY = r"C:\Users\Jessewen_Wang\.conda\envs\MLGener\python.exe"
OUT = os.path.join(HERE, "out")
LOGDIR = os.path.join(OUT, "logs")
STATE = os.path.join(OUT, "pipeline_state.json")


def _p(*a):
    return os.path.join(HERE, *a)


# ── 任务表 ────────────────────────────────────────────────────────────────
# cmd: 传给 python 的参数列表；marker: 完成后应存在的产物
TASKS = [
    {
        "name": "ood",
        "desc": "OOD 泛化：φ 取到训练分布之外，检验约束是否仍精确",
        "cmd": ["ood_heat.py"],
        "marker": _p("out", "ood_heat.txt"),
        "eta_min": 5,
    },
    {
        "name": "baselines",
        "desc": "三基线对照：ECI / DiffusionPDE / D-Flow vs vanilla vs PCFM",
        "cmd": ["baselines_heat.py"],
        "marker": _p("out", "baselines_heat.txt"),
        "eta_min": 240,
    },
    {
        "name": "burgers",
        "desc": "Burgers 1D：训练 + IC/BC 两个约束情形评测",
        "cmd": ["burgers_run.py"],
        "marker": _p("out", "burgers_results.txt"),
        "eta_min": 600,
    },
    {
        "name": "heat_topup",
        "desc": "热方程续训：从 heat_30000 补到 80000 步（样本量与论文对齐）",
        "cmd": ["run_heat_official.py", "train",
                "--resume", _p("out", "heat_30000.pt"),
                "--steps", "50000", "--batch", "64", "--lr", "3e-5",
                "--save-every", "5000", "--log-every", "1000"],
        "marker": _p("out", "heat_80000.pt"),
        "eta_min": 720,
    },
]


def load_state():
    if os.path.exists(STATE):
        try:
            return json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass
    return {}


def save_state(st):
    os.makedirs(OUT, exist_ok=True)
    json.dump(st, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


def status_of(t, st):
    if os.path.exists(t["marker"]):
        return "done(产物已存在)"
    rec = st.get(t["name"])
    if rec and rec.get("status") == "failed":
        return "failed(exit=%s)" % rec.get("code")
    return "pending"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--only", default=None)
    ap.add_argument("--from", dest="from_name", default=None)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--force", action="store_true", help="忽略 marker，强制重跑")
    args = ap.parse_args()

    os.makedirs(LOGDIR, exist_ok=True)
    st = load_state()

    tasks = TASKS
    if args.only:
        tasks = [t for t in tasks if args.only in t["name"]]
        if not tasks:
            sys.exit("没有匹配的任务: %s" % args.only)
    if args.from_name:
        names = [t["name"] for t in tasks]
        hit = [i for i, n in enumerate(names) if args.from_name in n]
        if not hit:
            sys.exit("没有匹配的起点: %s" % args.from_name)
        tasks = tasks[hit[0]:]

    if args.list:
        print(f"{'任务':<12} {'状态':<22} {'预计':>6}  说明")
        print("-" * 92)
        for t in TASKS:
            print(f"{t['name']:<12} {status_of(t, st):<22} {t['eta_min']:>4}min  {t['desc']}")
        return

    # 预检
    if not os.path.exists(PY):
        sys.exit("[FATAL] 解释器不存在: %s" % PY)
    os.chdir(HERE)
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["WANDB_MODE"] = "disabled"

    t_all = time.time()
    for t in tasks:
        name = t["name"]
        if os.path.exists(t["marker"]) and not args.force:
            print(f"[skip] {name}: 产物已存在 -> {os.path.basename(t['marker'])}", flush=True)
            continue

        log = os.path.join(LOGDIR, f"{name}.log")
        print(f"\n{'='*78}\n[run ] {name}  {datetime.now():%H:%M:%S}\n"
              f"       {t['desc']}\n       日志 {log}\n{'='*78}", flush=True)

        t0 = time.time()
        rec = {"status": "running", "started": datetime.now().isoformat(timespec="seconds")}
        st[name] = rec
        save_state(st)

        with open(log, "w", encoding="utf-8") as f:
            f.write(f"# {name} :: {' '.join([PY] + t['cmd'])}\n")
            f.flush()
            proc = subprocess.run([PY] + t["cmd"], stdout=f, stderr=subprocess.STDOUT,
                                  cwd=HERE, env=os.environ.copy())
        dt = time.time() - t0

        ok = (proc.returncode == 0)
        rec.update({"status": "ok" if ok else "failed",
                    "code": proc.returncode,
                    "finished": datetime.now().isoformat(timespec="seconds"),
                    "elapsed_min": round(dt / 60, 1)})
        st[name] = rec
        save_state(st)

        print(f"[{'ok  ' if ok else 'FAIL'}] {name}  {dt/60:.1f} min  exit={proc.returncode}", flush=True)
        if not ok and not args.keep_going:
            print(f"\n[stop] {name} 失败，流水线中止。查看 {log}", flush=True)
            return 1
        if ok and not os.path.exists(t["marker"]):
            print(f"[warn] {name} 正常退出但未找到产物 {os.path.basename(t['marker'])}", flush=True)

    print(f"\n[全部完成] 总耗时 {(time.time()-t_all)/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
