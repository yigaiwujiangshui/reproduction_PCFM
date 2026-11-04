# -*- coding: utf-8 -*-
"""
Burgers 1D 数据集生成（Windows + DSH 沙箱安全版）
=================================================
两处必须绕开的坑：
  1. 上游脚本在**模块顶层**调用 mp.Pool，且没有 `if __name__ == "__main__":` 守卫；
     Windows 的 spawn 会让子进程重跑顶层代码 → 无限递归。
  2. DSH 沙箱禁止创建命名管道，`mp.Pool` 直接报 `PermissionError: [WinError 5]`。
     → 因此改为**单进程串行**生成。

数值逻辑完全用上游的 `solve_burgers`（从源码 exec 载入，不重写任何公式），
随机数的抽取顺序也与上游一致（np.random.seed(seed) → uniform(0.2,0.8) → uniform(0,1)），
因此生成的数据与上游脚本在分布上等价。

输出（文件名与 configs/burgers1d.yml 要求一致）：
  burgers_train_nIC80_nBC80.h5
  burgers_test_nIC30_nBC30.h5
"""

import io
import os
import time

import h5py
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "PCFM", "datasets", "generate_burgers1d_data.py")
DATA_DIR = os.path.join(HERE, "PCFM", "datasets", "data")


def load_upstream():
    """载入上游脚本的「函数定义部分」，跳过模块级生成调用。"""
    src = io.open(SRC, encoding="utf-8").read()
    cut = src.find("# training data:")
    if cut < 0:
        raise RuntimeError("找不到截断点，上游文件结构可能已变化")
    ns = {"__name__": "upstream_burgers"}
    exec(compile(src[:cut], SRC, "exec"), ns)
    if "solve_burgers" not in ns:
        raise RuntimeError("上游缺少 solve_burgers")
    return ns


def generate_serial(solve_burgers, path, N_ic, N_bc, Nx=100, Nt=100, T=1.0,
                    seed=42, filename="burgers_train"):
    """与上游 generate_burgers_dataset 完全同构，但串行填充。"""
    rng = np.random.seed(seed)                       # 与上游一致：用全局种子
    p_locs = np.random.uniform(0.2, 0.8, N_ic)
    u_bcs = np.random.uniform(0.0, 1.0, N_bc)
    x = np.linspace(0, 1.0, Nx + 1)
    t = np.linspace(0, T, Nt + 1)

    os.makedirs(path, exist_ok=True)
    full_path = os.path.join(path, f"{filename}_nIC{N_ic}_nBC{N_bc}.h5")

    with h5py.File(full_path, "w") as f:
        d = f.create_dataset("u", shape=(N_ic, N_bc, Nx + 1, Nt + 1), dtype=np.float32)
        f.create_dataset("ic", data=p_locs.astype(np.float32))
        f.create_dataset("bc", data=u_bcs.astype(np.float32))
        f.create_dataset("x", data=x.astype(np.float32))
        f.create_dataset("t", data=t.astype(np.float32))

        done = 0
        for i in range(N_ic):
            for j in range(N_bc):
                d[i, j] = solve_burgers(p_locs[i], u_bcs[j], Nx=Nx, Nt=Nt).astype(np.float32)
                done += 1
            if (i + 1) % 10 == 0 or i + 1 == N_ic:
                print(f"    {filename}: IC {i+1}/{N_ic} ({done} 条)", flush=True)
    return full_path


if __name__ == "__main__":
    up = load_upstream()
    solve = up["solve_burgers"]
    os.makedirs(DATA_DIR, exist_ok=True)

    # 先做一次正确性自检
    ref = solve(0.5, 0.3, Nx=100, Nt=100)
    print(f"[check] solve_burgers(0.5,0.3) shape={ref.shape} "
          f"min={ref.min():.4f} max={ref.max():.4f} finite={np.isfinite(ref).all()}", flush=True)
    assert ref.shape == (101, 101), ref.shape
    assert np.isfinite(ref).all()

    for n_ic, n_bc, seed, name in ((80, 80, 42, "burgers_train"),
                                   (30, 30, 0, "burgers_test")):
        t0 = time.time()
        p = generate_serial(solve, DATA_DIR + os.sep, n_ic, n_bc, seed=seed, filename=name)
        print(f"[done] {os.path.basename(p)}  {os.path.getsize(p)/1024/1024:.1f} MB  "
              f"{time.time()-t0:.1f}s", flush=True)
