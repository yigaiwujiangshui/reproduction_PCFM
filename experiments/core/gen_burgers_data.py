# -*- coding: utf-8 -*-
"""生成 Burgers 1D 数据集。

对象：无粘 Burgers 方程 `u_t + (u²/2)_x = 0`，`x,t ∈ [0,1]`，`101×101` 网格。
算法：Godunov 有限体积法。初值为平滑阶跃 `1/(1+exp((x-p)/ε))`，`ε=0.02`，`p ~ U(0.2,0.8)`；
      左端 Dirichlet `u(0,t)=u_bc`，`u_bc ~ U(0,1)`，右端零梯度 Neumann。
      生成四份数据：训练 80×80、测试 30×30、固定 IC 采样 20×512、固定 BC 采样 20×512。
参考：论文附录 B.4。

用法：
    python experiments/core/gen_burgers_data.py              # 全部
    python experiments/core/gen_burgers_data.py --what train # 只生成一份
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import PCFM  # noqa: E402,F401

import h5py  # noqa: E402
import numpy as np  # noqa: E402

DATA_DIR = os.path.join(PCFM, "datasets", "data")
NX, NT, T_END, EPS, SEED = 100, 100, 1.0, 0.02, 42


def godunov_flux(uL, uR):
    """Godunov 通量 f(u)=u²/2。稀疏波取两端最小值，激波按特征方向取迎风侧。"""
    flux = np.zeros_like(uL)
    rare = uL <= uR
    shock = ~rare
    flux[rare] = np.minimum(0.5 * uL[rare] ** 2, 0.5 * uR[rare] ** 2)
    flux[shock] = np.where((uL[shock] + uR[shock]) / 2 > 0,
                           0.5 * uL[shock] ** 2,
                           0.5 * uR[shock] ** 2)
    return flux


def solve_burgers(p_loc, u_bc, Nx=NX, Nt=NT, length=1.0, T=T_END, eps=EPS):
    """返回形状 (Nx+1, Nt+1) 的解，时间在最后一维。"""
    dx, dt = length / Nx, T / Nt
    x = np.linspace(0, length, Nx + 1)
    u = np.zeros((Nt + 1, Nx + 1))
    u[0] = 1.0 / (1 + np.exp((x - p_loc) / eps))

    for n in range(Nt):
        if n > 0:
            u[n, 0] = u_bc
        flux = godunov_flux(u[n, :-1], u[n, 1:])
        u[n + 1, 1:-1] = u[n, 1:-1] - (dt / dx) * (flux[1:] - flux[:-1])
        u[n + 1, 0] = u_bc
        u[n + 1, -1] = u[n + 1, -2]
    return u.T


def _write(path, u_bcs, p_locs, ic_first, Nx, Nt):
    """ic_first=True → 形状 (N_ic, N_bc)；False → (N_bc, N_ic)。"""
    n_ic, n_bc = len(p_locs), len(u_bcs)
    x = np.linspace(0, 1.0, Nx + 1).astype(np.float32)
    t = np.linspace(0, T_END, Nt + 1).astype(np.float32)
    shape = (n_ic, n_bc, Nx + 1, Nt + 1) if ic_first else (n_bc, n_ic, Nx + 1, Nt + 1)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset("u", shape=shape, dtype=np.float32)
        f.create_dataset("bc", data=np.asarray(u_bcs, dtype=np.float32))
        f.create_dataset("ic", data=np.asarray(p_locs, dtype=np.float32))
        f.create_dataset("x", data=x)
        f.create_dataset("t", data=t)
        if ic_first:
            for i, p in enumerate(p_locs):
                for j, b in enumerate(u_bcs):
                    dataset[i, j] = solve_burgers(p, b, Nx=Nx, Nt=Nt).astype(np.float32)
        else:
            for j, b in enumerate(u_bcs):
                for i, p in enumerate(p_locs):
                    dataset[j, i] = solve_burgers(p, b, Nx=Nx, Nt=Nt).astype(np.float32)
    return path


def make_train(seed=SEED, n_ic=80, n_bc=80):
    np.random.seed(seed)
    p_locs = np.random.uniform(0.2, 0.8, n_ic)
    u_bcs = np.random.uniform(0.0, 1.0, n_bc)
    p = os.path.join(DATA_DIR, f"burgers_train_nIC{n_ic}_nBC{n_bc}.h5")
    return _write(p, u_bcs, p_locs, True, NX, NT)


def make_test(seed=0, n_ic=30, n_bc=30):
    np.random.seed(seed)
    p_locs = np.random.uniform(0.2, 0.8, n_ic)
    u_bcs = np.random.uniform(0.0, 1.0, n_bc)
    p = os.path.join(DATA_DIR, f"burgers_test_nIC{n_ic}_nBC{n_bc}.h5")
    return _write(p, u_bcs, p_locs, True, NX, NT)


def make_diff_ics(seed=SEED, n_ic=20, n_bc=512):
    """固定 IC、变化 BC。"""
    np.random.seed(seed)
    p_locs = np.random.uniform(0.2, 0.8, n_ic)
    u_bcs = np.random.uniform(0.0, 1.0, n_bc)
    p = os.path.join(DATA_DIR, f"burgers_sampling_diffICs_nIC{n_ic}_nBC{n_bc}.h5")
    return _write(p, u_bcs, p_locs, True, NX, NT)


def make_diff_bcs(seed=SEED, n_bc=20, n_ic=512):
    """固定 BC、变化 IC。上游先抽 u_bcs 再抽 p_locs，此处保持同样顺序。"""
    np.random.seed(seed)
    u_bcs = np.random.uniform(0.0, 1.0, n_bc)
    p_locs = np.random.uniform(0.2, 0.8, n_ic)
    p = os.path.join(DATA_DIR, f"burgers_sampling_diffBCs_nBC{n_bc}_nIC{n_ic}.h5")
    return _write(p, u_bcs, p_locs, False, NX, NT)


BUILDERS = {"train": make_train, "test": make_test,
            "diffICs": make_diff_ics, "diffBCs": make_diff_bcs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all"] + list(BUILDERS))
    a = ap.parse_args()
    for k in (list(BUILDERS) if a.what == "all" else [a.what]):
        print(BUILDERS[k]())


if __name__ == "__main__":
    main()
