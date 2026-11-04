# PCFM 复现：Physics-Constrained Flow Matching

> 论文：*Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints*
> Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gómez-Bombarelli, Christopher Rackauckas — NeurIPS 2025
> arXiv:2506.04171v2 ｜ 官方代码：<https://github.com/cpfpengfei/PCFM>

本仓库是上述论文 **Heat 数据集主实验与消融的复现**，包含：
上游代码接入（带出处登记）、官方缺失调用层的补齐、训练与评测脚本、逐模块的代码-论文对应分析。

---

## 1. 复现了什么

| 论文位置 | 内容 | 本仓库状态 |
|---|---|---|
| §4.1 + Table 3 (Heat) | PCFM vs vanilla FFM 的 MMSE / SMSE / CE | ✅ 已复现（定性一致） |
| Appendix L.1 + Table 6 | 中间投影必要性（每步投影 vs 仅末端投影） | ✅ 已复现 |
| §4.1 + Appendix L.3 | 采样步数、牛顿步数、罚项强度 λ 的消融 | ✅ 已复现 |
| Eq.(10)/(21) + Appendix D.3 | Heat 的 IC 与质量守恒约束 | ✅ 与官方 `Residuals` 完全对应 |
| Table 3 其他数据集 | Reaction–Diffusion / Burgers / Navier–Stokes | ⬜ 未做（见 §6） |

**核心算法（PCFM 采样、约束残差、五种基线采样器）全部来自上游代码，未重写。**

---

## 2. 快速开始

```powershell
# 1) 环境（conda 环境名按实际）
cd <本仓库路径>
conda activate MLGener
python -m pip install -r requirements.txt

# 2) 训练（Heat，对齐论文配置；batch 因显存降至 64）
python run_heat_official.py train --steps 20000 --batch 64 --save-every 2000

# 3) 消融（论文附录 B.1 固定 φ = π/4）
python run_heat_official.py ablate --ckpt out/heat_20000.pt `
    --n-ref 32 --n-sample 16 --n-step 20 --phi 0.7853981634

# 4) 跨 checkpoint 一致性（改善倍数是否稳定）
python ratio_across_checkpoints.py --n-ref 32 --n-sample 16 --phi 0.7853981634
```

---

## 3. 目录结构

```
.
├── PCFM/                        上游官方代码（仅 models/fno.py 改 1 行，见 docs/THIRD-PARTY.md）
├── pcfm_compat.py               官方缺失的调用层适配（★ 见 §4）
├── run_heat_official.py         Heat 训练 / 评测 / 消融主脚本
├── ratio_across_checkpoints.py  跨训练阶段的改善倍数一致性检验
├── sweep_phi.py                 留出 IC 维度的敏感性扫描
├── ce_precision.py              CE 残差量级的实测诊断
├── check_seed_overlap.py        续训是否会重放旧数据的检验
├── profile_gp.py                性能剖析（GP 采样 vs 网络前向）
├── requirements.txt
└── docs/
    ├── 代码与论文逐模块对应详解.md   ★ 逐文件说明「代码 ↔ 论文哪一处 ↔ 为何如此对应」
    ├── 代码整理与现状.md            仓库全景与已知缺口
    ├── 试验清单.md                  分阶段实验清单与状态
    ├── 复现日志.md                  ★ 实际做了什么、遇到什么、怎么解决
    └── THIRD-PARTY.md              上游来源与许可
```

---

## 4. 为什么需要 `pcfm_compat.py`

官方仓库只放出了**训练路径**，把「模型 + 约束 + 采样器」串起来的推理装配层**未公开**：

- `pcfm/constraints.py` 的 `Residuals` 全仓 **零实例化、零继承**（仅被 `__init__.py` 导出）
- `pcfm_sampling.py:145` 直接调用 `hfunc(u_corr)`，但 `Residuals` **没有 `__call__`**
- `ffm_sampler.py:36` 向 `fast_project_batched` 传 `mode/newtonsteps/eps/use_vmap`，
  而定义处 `pcfm_sampling.py:17` 只接受 `max_iter` → 直接 `TypeError`

`pcfm_compat.py` 只做两件事，**不改任何算法**：
1. `CallableResiduals(Residuals)` —— 补 `__call__`，转发给官方 `full_residual_heat`
2. `patch_fast_project()` —— 参数名映射 + `(B,nx,nt)` 展平

---

## 5. 复现结果摘要

评测协议取自论文 §4.1 与附录 B.1：**固定留出 IC，$u(x,0)=\sin(x+\pi/4)$**，
真解族为同 IC 下 $\nu\in[1,5]$ 的解析解集合；指标按附录 J 定义（MMSE / SMSE / CE，单位 $10^{-2}$）。

| 指标 | 论文 Table 3 | 本次复现 |
|---|---|---|
| vanilla CE(IC) | 579 | 700 |
| vanilla CE(CL) | 2.11 | 5.14 |
| **PCFM CE(IC) / CE(CL)** | **0 / 0** | **0 / 3.9×10⁻⁵** |
| vanilla MMSE | 4.56 | 8.78 |
| PCFM MMSE | 0.241 | 0.103 |

**定性结论全部成立**：
1. PCFM 把约束压到机器精度（$10^{-7}$ 量级），vanilla 残留 $10^2$ 量级
2. 采样步数越少保真越差，但约束仍为零（5→10→20 步：MMSE 1.58→0.60→0.10）
3. 牛顿 1 步即足够（线性约束下一阶即精确），更多步只把 CE 再降一点
4. 中间投影是必要的：仅做末端投影时保真显著劣化

**与论文的差异及原因**：batch 256→64（显存 16 GiB 限制），总样本量为论文的 1/4；
Windows 上 `num_workers` 设为 0（多进程反而慢 45 倍）。详见 `docs/复现日志.md`。

---

## 6. 未完成部分

| 项 | 原因 |
|---|---|
| FPD (Fréchet Poseidon Distance) | 需预训练 PDE 基座模型 Poseidon，权重未公开 |
| PDM 基线 | 上游仓库未提供实现 |
| Reaction–Diffusion / Burgers / NS | 需先数值生成数据，且 NS 训练量本机不可行 |

---

## 7. 上游来源

见 [`docs/THIRD-PARTY.md`](docs/THIRD-PARTY.md)。
上游为 MIT（含 Amazon ECI-sampling 的 Apache-2.0 部分），本仓库保留其 `LICENSE` 与 `NOTICE`。
