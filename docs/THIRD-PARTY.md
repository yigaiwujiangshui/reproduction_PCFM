# 上游来源与许可登记

本仓库包含第三方代码。按其许可证要求登记如下。

## 1. PCFM（主上游）

| 项 | 内容 |
|---|---|
| 仓库 | <https://github.com/cpfpengfei/PCFM> |
| 引用 | Utkarsh, Pengfei Cai, Alan Edelman, Rafael Gómez-Bombarelli, Christopher Rackauckas. *Physics-Constrained Flow Matching: Sampling Generative Models with Hard Constraints*. NeurIPS 2025. arXiv:2506.04171 |
| 获取方式 | 通过 GitHub raw 逐文件下载（commit 分支 `main`，下载时间 2025-11-02） |
| 落地路径 | `./PCFM/`（保持原目录结构） |
| 许可 | MIT（见 `PCFM/LICENSE`） |
| 随附文件 | `PCFM/LICENSE`、`PCFM/LICENSE-APACHE-2.0`、`PCFM/NOTICE` 均原样保留 |

### 本仓库对其所做的改动

**唯一改动：`PCFM/models/fno.py`（第 38–48 行）**

原因：`neuraloperator` 在 1.0 之后移除了 `lifting_channels` 与 `proj_channels`，
改为相对 `hidden_channels` 的比例参数。上游代码写法在 1.0+ 上会 `TypeError`。

```python
# 上游原写法
_FNO(..., lifting_channels=proj_channels, proj_channels=proj_channels, n_layers=n_layers)

# 本仓库改为（保持“提升/投影层通道数 = proj_channels”的语义）
_lift_ratio = max(1, int(round(proj_channels / max(hidden_channels, 1))))
_FNO(..., lifting_channel_ratio=_lift_ratio, projection_channel_ratio=_lift_ratio, n_layers=n_layers)
```

未选用 `neuraloperator==0.3.0`（上游锁定版本）的原因：该版本无 `requires_python` 约束，
依赖链与本机 python 3.13 / torch 2.10 不兼容。

其余所有文件（`pcfm/`、`datasets/`、`models/` 的其他文件、`scripts/`、`configs/`）**未改动**。

## 2. PCFM 内部引用的 ECI 代码

| 项 | 内容 |
|---|---|
| 来源 | <https://github.com/amazon-science/ECI-sampling> |
| 许可 | Apache License 2.0 |
| 关系 | 上游 PCFM 声明其部分组件派生自该项目；本仓库间接包含，随 `PCFM/LICENSE-APACHE-2.0` 与 `PCFM/NOTICE` 一并保留 |

本仓库自行实现的 `models/constraints.py` 之外的约束接口（`pcfm/constraints.py`）由上游提供，
来源为 PCFM 作者，非本仓库新增。

## 3. 本仓库自行编写的文件

以下文件为本次复现新写，不含第三方代码：

| 文件 | 性质 |
|---|---|
| `pcfm_compat.py` | 接口适配（补 `__call__`、参数名映射、形状展平） |
| `run_heat_official.py` | 训练/评测/消融驱动。**注意**：训练循环与上游 `scripts/training/main.py` 逻辑相同但由本仓库实现（见 `docs/复现日志.md` 的说明） |
| `ratio_across_checkpoints.py` | 跨 checkpoint 一致性检验 |
| `sweep_phi.py` | 留出 IC 敏感性扫描 |
| `ce_precision.py` | CE 残差量级诊断 |
| `check_seed_overlap.py` | 续训数据重叠检验 |
| `profile_gp.py` | 性能剖析 |
| `docs/*.md` | 文档 |

## 4. 论文内容的引用

`docs/代码与论文逐模块对应详解.md` 与课程报告正文引用了论文的公式、算法与实验设定，
均标注出处（章节号 / 公式号 / 附录号）。译文部分以课程报告形式单独提交。
