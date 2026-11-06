# PCFM 复现代码

> 论文：*Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints*
> Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gómez-Bombarelli, Christopher Rackauckas
> NeurIPS 2025 ｜ arXiv:2506.04171v2 ｜ 官方代码：<https://github.com/cpfpengfei/PCFM>

本目录 = **上游代码 `PCFM/`** + **我们新增的实验代码 `experiments/`**，两者严格分离。

```
work/repro/
├── PCFM/                 # 上游官方仓库（MIT），仅 1 处必要改动，见下
├── experiments/          # 本复现新增的全部代码
│   ├── experiment.py     # ★ 统一入口：list / env / run / suite
│   ├── _bootstrap.py     # 统一路径引导（sys.path、OUT、PCFM、临时目录）
│   ├── core/             # 接口适配与数据生成
│   ├── heat/             # 热方程：训练、消融、基线、OOD
│   ├── burgers/          # Burgers：训练、评测、续训、诊断
│   ├── ablation/         # 补充消融（配点密度、TVD、中间投影）
│   ├── diagnostics/      # 论证性诊断（数值精度、指标口径、数据自检）
│   ├── pipeline/         # 串行总控
│   └── README.md         # ★ 每个实验的实现、调用方式与关键参数
├── out/                  # 全部产物：checkpoint、日志、结果 txt（.gitignore）
├── requirements.txt
└── README.md
```

## 快速开始

```powershell
# 1) 进入目录
cd C:\Resource\Code\DSH\paperResearch\work\repro

# 2) 启动环境（本复现使用 MLGener）
conda activate MLGener

# 3) 安装依赖
python -m pip install -r requirements.txt
```

```powershell
# 环境自检（torch / GPU / 依赖 / checkpoint 数量）
python experiments/experiment.py env

# 列出全部实验（含类别、说明、套件）
python experiments/experiment.py list

# 跑单个实验；`--` 之后的参数原样透传给目标脚本
python experiments/experiment.py run heat-ablate -- --ckpt out/heat_80000.pt
python experiments/experiment.py run burgers-eval -- --ckpt out/burgers_20000.pt --n-sample 8 --n-step 20

# 跑套件
python experiments/experiment.py suite main     # 主实验
python experiments/experiment.py suite supp     # 补充消融
python experiments/experiment.py suite diag     # 诊断论证
```

## 上游代码的改动

**只改了 1 处**，且是接口兼容必需：

| 文件 | 改动 | 原因 |
|---|---|---|
| `PCFM/models/fno.py` | 谱卷积的 `ratio` 参数 | `neuraloperator>=1.0` 更改了 FNO 构造签名，否则实例化失败 |

其余全部逻辑（约束残差、PCFM 投影、FFM 采样器、GP 先验）**未改动**。
我们的适配集中在 `experiments/core/pcfm_compat.py`：

- 给官方 `Residuals` 补 `__call__`（PCFM 采样器按 `hfunc(u)` 调用）
- 把 `newtonsteps` 映射到投影函数的 `max_iter`
- 展平 `(B, nx, nt) → (B, n)`

## 代码组织约定

* 所有脚本通过 `_bootstrap.py` 解析路径，因此**可在任意深度调用**，不依赖当前工作目录。
* `out/` 是唯一产物目录；数据集在 `PCFM/datasets/data/`。
* 每个实验的输出写入 `out/<name>.txt`，统一格式为「表头 + 数据 + 判读」。

## 实验与报告章节对应

| 实验名 | 报告位置 | 结论要点 |
|---|---|---|
| `heat-ablate` | §6.3 | 步数 5/10/20/50、牛顿步 1/2、罚项 λ |
| `heat-ckpts` | §6.3 | 12 个 checkpoint 的 CE / MMSE 单调性 |
| `heat-ood` | §6.9 | 分布外 CE 仍精确为 0（约束是代数性质） |
| `heat-baselines` | §6.10 | ECI / DiffusionPDE / D-Flow 与 vanilla、PCFM 对照 |
| `eci-sweep` | §6.10 | ECI 的 $n_{\rm mix}$ 敏感性；算力对齐后与 PCFM 等价 |
| `heat-cl` | §6.11 | 质量守恒约束的约束力量化 |
| `lambda-step` | §6.14 | $\lambda \times n_{\rm step}$ 网格：CE 与 λ 无关（末端投影保证） |
| `supp1` / `supp2` | §6.13 / §6.15 | BC 多索引、方差分解、配点密度、TVD |
| `ce-linear` | §6.12 | 四级精度阶梯证明 CE≈4e-7 是 float32 下限 |
| `mmse-def` | §6.16 | MMSE 三种口径对照（口径歧义是主要差距来源） |
| `burgers-eval` | §6.17 | Burgers IC/BC 主结果（512 条参考族） |

## 引用

```bibtex
@inproceedings{utkarsh2025pcfm,
  title     = {Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints},
  author    = {Utkarsh and Cai, Pengfei and Edelman, Alan and G{\'o}mez-Bombarelli, Rafael and Rackauckas, Christopher},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2025}
}
```
