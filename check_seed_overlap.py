# -*- coding: utf-8 -*-
"""验证：续训不会重放旧数据（种子偏移是否真的产生不重叠的数据流）

Heating 数据集的 (ν, φ) 来自 Python random；本脚本对两个种子各抽前 N 条，
比较是否重叠，并把结果写入 UTF-8 文件。
"""
import os, sys, random
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "PCFM"))

import numpy as np

L = []
def p(s=""):
    L.append(str(s))

def draw(seed, n, nx=5):
    """复现 DiffusionDataset.__getitem__ 的抽样逻辑：ν~U(1,5), φ~U(0,π)"""
    random.seed(seed)
    out = []
    for _ in range(n):
        v = random.uniform(1.0, 5.0)
        phi = random.uniform(0.0, np.pi)
        out.append((round(v, 6), round(phi, 6)))
    return out

N = 2000
SEED0 = 42        # 原始训练用的种子
OFF = 20000       # 续训时的 resume_step 偏移

a = draw(SEED0, N)
b = draw(SEED0 + OFF, N)

p("=" * 70)
p("续训数据流重叠性检验")
p("=" * 70)
p(f"原始训练 seed = {SEED0}, 取前 {N} 条 (ν, φ)")
p(f"续训     seed = {SEED0 + OFF}, 取前 {N} 条")
p("")
p("原始前 5 条       续训前 5 条")
for i in range(5):
    p(f"  ν={a[i][0]:.4f} φ={a[i][1]:.4f}   |   ν={b[i][0]:.4f} φ={b[i][1]:.4f}")
p("")

# 逐位比较是否完全相同
same_pos = sum(1 for i in range(N) if a[i] == b[i])
p(f"同一位置完全相同的条数: {same_pos} / {N}")

# 集合层面的重叠（ν,φ 二元组是否重复出现）
set_a, set_b = set(a), set(b)
inter = set_a & set_b
p(f"两个数据流共有的 (ν,φ) 组合数: {len(inter)}")
p(f"  占比: {len(inter)/N*100:.4f}%")
p("")

# 对照：用同一个种子会怎样（说明「如果只用一个 seed 就会重放」）
c = draw(SEED0, N)
same_same = sum(1 for i in range(N) if a[i] == c[i])
p(f"[对照] 同一种子两次抽取，相同位置一致条数: {same_same} / {N}")
p("  -> 若续训沿用同一 seed，这 {}/{} 条会被原样重放".format(same_same, N))
p("")

p("=" * 70)
p("结论")
p("=" * 70)
if same_same == N and same_pos < N * 0.01 and len(inter) < N * 0.01:
    p("  ✓ 种子偏移有效：同种子会完全重放（对照），偏移后几乎不重叠（实测），")
    p("    因此续训是在「见新数据」，不是在重复已有样本。")
else:
    p("  ⚠ 结果不符合预期，需检查抽样逻辑")

with open(os.path.join(HERE, "out", "seed_overlap_check.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("written")
