# -*- coding: utf-8 -*-
"""路径引导模块。

对象：`experiments/` 下所有脚本的 import 环境。
算法：定位 `repro/` 根，将 `PCFM/` 与各子目录加入 `sys.path`，导出 `OUT`/`REPO`/`PCFM`。"""
import os
import sys

EXP = os.path.dirname(os.path.abspath(__file__))     # .../experiments
REPO = os.path.dirname(EXP)                          # .../repro
OUT = os.path.join(REPO, "out")
PCFM = os.path.join(REPO, "PCFM")

_SUBS = ("core", "heat", "burgers", "ablation", "pipeline")

# 顺序有意义：越靠前优先级越高。PCFM/ 放最前，保证 import pcfm 优先命中上游包。
for _p in [PCFM, EXP] + [os.path.join(EXP, d) for d in _SUBS]:
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

os.makedirs(OUT, exist_ok=True)

# ── 把临时目录指到工作区内 ──────────────────────────────────────────
# neuralop 会连带 import wandb，后者在进程退出时清理它建的临时目录；
# 若临时目录在系统 %TEMP%（工作区外），清理会抛 PermissionError，
# 使解释器以退出码 1 结束，看起来像失败。指到工作区内即可正常清理。
_TMP = os.path.join(REPO, ".cache", "tmp")
os.makedirs(_TMP, exist_ok=True)
os.environ["TMP"] = _TMP
os.environ["TEMP"] = _TMP
os.environ.setdefault("WANDB_MODE", "disabled")
os.environ.setdefault("WANDB_SILENT", "true")

__all__ = ["EXP", "REPO", "OUT", "PCFM"]
