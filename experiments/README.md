# experiments

PCFM 复现的实验代码。上游 `PCFM/` 保持原样，仅 `models/fno.py` 有一处 `neuraloperator>=1.0` 兼容改动。

## 目录

| 路径 | 内容 |
|---|---|
| `core/pcfm_compat.py` | 官方接口适配（库模块，不改算法） |
| `core/gen_burgers_data.py` | Burgers 数据集生成 |
| `heat/run_heat_official.py` | 热方程训练 / 评测 / 消融 |
| `heat/ood_heat.py` | 分布外泛化 |
| `heat/baselines_heat.py` | 多方法对照 |
| `burgers/burgers_run.py` | Burgers 训练与评测 |
| `burgers/burgers_extend.py` | Burgers 续训 |
| `ablation/supplementary2.py` | 配点密度、TVD、采样步数消融 |
| `pipeline/run_pipeline.py` | 串行总控 |
| `experiment.py` | 统一入口 |
| `_bootstrap.py` | 路径引导 |
| `_probe.py` | JSON 落盘辅助 |

## 使用

```bash
python experiments/experiment.py env            # 环境检查
python experiments/experiment.py list           # 列出全部实验
python experiments/experiment.py run <name> -- <args>
python experiments/experiment.py suite main     # 或 supp
```

`--` 之后的参数原样透传给目标脚本。也可直接运行子目录里的脚本。

## 实验一览

| name | 脚本 | 做什么 |
|---|---|---|
| `heat-train` | `heat/run_heat_official.py train` | 热方程训练，支持 `--resume` 续训 |
| `heat-ablate` | `heat/run_heat_official.py ablate` | 采样步数 / 牛顿步 / 罚项 λ 的消融 |
| `heat-ood` | `heat/ood_heat.py` | 分布外 φ 的泛化检验 |
| `heat-baselines` | `heat/baselines_heat.py` | vanilla / PCFM / ECI / DiffusionPDE / D-Flow 对照 |
| `burgers-train` | `burgers/burgers_run.py train` | Burgers 训练（另有 `smoke`/`eval`） |
| `burgers-eval` | `burgers/burgers_run.py eval` | Burgers IC/BC 两情形评测 |
| `burgers-extend` | `burgers/burgers_extend.py` | 从 checkpoint 续训，补齐数据遍历次数 |
| `burgers-data` | `core/gen_burgers_data.py` | 生成训练/测试/采样数据集 |
| `supp2` | `ablation/supplementary2.py` | 配点密度、TVD 约束、采样步数 |
| `pipeline` | `pipeline/run_pipeline.py` | 串行总控 |

## 关键参数

### `heat/run_heat_official.py`

| 参数 | 说明 | 默认 |
|---|---|---|
| `--phi` | 受约束采样的固定相位 | π/4（与论文一致，勿改） |
| `--n-ref` | 参考解族条数 | 32 |
| `--n-sample` | 生成样本条数 | 16 |
| `--n-step` | 采样步数 | 20 |
| `--resume` | 从 checkpoint 续训，累计步数写入文件名 | — |
| `--steps-list` / `--lam-list` | `ablate` 的网格取值 | — |

### `burgers/burgers_run.py`

| 参数 | 说明 | 默认 |
|---|---|---|
| `--ic-index` / `--bc-index` | 固定哪个初值 / 边界 | 0 |
| `--n-step` | 采样步数（非线性约束建议 ≥20） | 20 |
| `--out` | 输出文件 | `out/burgers_results.txt` |

### `burgers/burgers_extend.py`

| 参数 | 说明 | 默认 |
|---|---|---|
| `--base` | 起点 checkpoint | `out/burgers_20000.pt` |
| `--lr` | 续训起始学习率 | 1e-4 |
| `--save-every` | 存盘间隔 | 2500 |

### `heat/baselines_heat.py`

| 参数 | 说明 |
|---|---|
| `--mode` | `smoke` 快速自检 / `formal` 正式 |
| `--eci-n-mix` | ECI 每步混合次数（论文未公开，默认 10） |

## 输出约定

- 产物写入 `out/`：checkpoint 为 `heat_{累计步数}.pt` / `burgers_{步数}.pt`，结果文件为 `out/<name>.txt|json`。
- 续训使用 `seed_all(seed + resume_step)`，与上一段数据流不重叠。
- 指标统一乘 `1e2`，与论文 Table 3 的 `1e-2` 单位一致。

## 依赖

```bash
conda activate <env>
python -m pip install -r requirements.txt
```
