# -*- coding: utf-8 -*-
"""探测实验的输出辅助模块。

对象：`probes.py` 等脚本产生的指标字典。
算法：JSON 序列化（含 numpy/torch 标量转换）、分节执行与容错、等宽表格渲染。"""
import json
import os
import time
import traceback

from _bootstrap import OUT


def save(name, payload):
    """把 payload 写为 out/<name>.json（缩进 2、保留中文），返回路径。"""
    p = os.path.join(OUT, f"{name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(_to_builtin(payload), f, ensure_ascii=False, indent=2)
    return p


def load(name):
    with open(os.path.join(OUT, f"{name}.json"), encoding="utf-8") as f:
        return json.load(f)


def _to_builtin(value):
    """numpy / torch 标量转原生类型，保证可 JSON 化。"""
    if isinstance(value, dict):
        return {k: _to_builtin(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(v) for v in o]
    # torch 标量有 .item()，numpy 标量两者都有；张量取 .item() 会抛 ValueError
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, RuntimeError):
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def run_section(payload, key, fn):
    """执行 fn()，结果存入 payload[key]；失败则记录 traceback 的错误尾部。"""
    t0 = time.time()
    try:
        payload[key] = fn()
        payload.setdefault("_elapsed_s", {})[key] = round(time.time() - t0, 1)
        print(f"  [{key}] ok  {time.time()-t0:.0f}s", flush=True)
    except Exception:
        payload[key] = {"error": traceback.format_exc()[-800:]}
        print(f"  [{key}] FAILED", flush=True)
    return payload


def table(rows, cols, headers=None, fmt=None):
    """把 list[dict] 打成对齐的表，返回多行字符串列表。

    rows    : [{...}, ...]
    cols    : 列名（dict 的键，按顺序）
    headers : 表头文字，默认用 cols
    fmt     : {列名: "格式串"}，例如 {"mmse": ".4f", "ce": ".3e"}
    """
    fmt = fmt or {}
    headers = headers or cols
    cells = [[str(h) for h in headers]]
    for r in rows:
        line = []
        for c in cols:
            v = r.get(c, "")
            f = fmt.get(c)
            if f and isinstance(v, (int, float)):
                line.append(format(v, f))
            else:
                line.append(str(v))
        cells.append(line)
    width = [max(len(row[i]) for row in cells) for i in range(len(cols))]
    out = []
    for i, row in enumerate(cells):
        out.append("  " + "  ".join(c.ljust(width[j]) for j, c in enumerate(row)))
        if i == 0:
            out.append("  " + "  ".join("-" * w for w in width))
    return out
