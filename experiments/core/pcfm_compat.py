# -*- coding: utf-8 -*-
"""官方代码的接口适配层，不改变任何算法。

对象：`pcfm.constraints.Residuals` 与 `pcfm.pcfm_sampling.fast_project_batched`。
算法：为 `Residuals` 补 `__call__`；把调用方的 `newtonsteps` 映射到投影函数的 `max_iter`；将张量由 `(B,nx,nt)` 展平为 `(B,n)`。
参考：论文 Algorithm 1 的 `h(u)` 调用约定。"""

import torch
import os, sys  # noqa: E401
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, REPO, PCFM  # noqa: E402,F401

# ── 缺口 1：可调用的约束残差 ──────────────────────────────
try:
    from pcfm.constraints import Residuals, Residuals2D
    _HAVE_REPO = True
except Exception:  # noqa: BLE001
    _HAVE_REPO = False
    Residuals = object
    Residuals2D = object


class CallableResiduals(Residuals):
    """Heat / RD / Burgers 的一维残差，补上 __call__。

    继承官方 Residuals，保留其全部方法（ic_residual / mass_residual_* /
    bc_residual_burgers / full_residual_*），只新增 __call__。
    """

    def __call__(self, u_flat):
        return self.full_residual_heat(u_flat)


class CallableResiduals2D(Residuals2D):
    """Navier–Stokes 的二维残差，补上 __call__。"""

    def __call__(self, u_flat):
        return self.full_residual_ns(u_flat)


# ── 缺口 2：fast_project_batched 兼容包装 ─────────────────
def patch_fast_project(verbose=True):
    """把 pcfm_sampling 与 ffm_sampler 里的 fast_project_batched 换成兼容版本。

    幂等：重复调用只生效一次。
    """
    try:
        import pcfm.pcfm_sampling as ps
        import pcfm.ffm_sampler as fs
    except Exception as e:  # noqa: BLE001
        if verbose:
            print(f"[compat] 跳过 fast_project 补丁：{e}")
        return False

    if getattr(ps, "_compat_patched", False):
        return True

    _orig = ps.fast_project_batched

    def fast_project_batched_compat(xi_batch, h_func, max_iter=1, **kw):
        # 调用方用 newtonsteps，定义方用 max_iter —— 语义相同，都是牛顿迭代次数
        if "newtonsteps" in kw:
            max_iter = kw.pop("newtonsteps")
        if "max_iter" in kw:
            max_iter = kw.pop("max_iter")
        # 以下三个参数仓库该函数实现用不到，直接丢弃
        kw.pop("mode", None)
        kw.pop("eps", None)
        kw.pop("use_vmap", None)
        if kw:
            if verbose:
                print(f"[compat] fast_project_batched 忽略未知参数: {list(kw)}")

        # 形状适配：仓库实现要求 (B, n)，调用方传的是 (B, nx, nt)
        orig_shape = xi_batch.shape
        flat = xi_batch.reshape(orig_shape[0], -1) if xi_batch.dim() > 2 else xi_batch
        out = _orig(flat, h_func, max_iter=max(1, int(max_iter)))
        return out.reshape(orig_shape) if xi_batch.dim() > 2 else out

    ps.fast_project_batched = fast_project_batched_compat
    # ffm_sampler 在 import 时已绑定旧引用，必须同步替换
    fs.fast_project_batched = fast_project_batched_compat
    ps._compat_patched = True
    if verbose:
        print("[compat] fast_project_batched 已适配（newtonsteps -> max_iter）")
    return True


# ── 自动应用 ─────────────────────────────────────────────
patch_fast_project(verbose=False)


def report():
    """打印当前补丁状态，便于确认。"""
    import pcfm.pcfm_sampling as ps
    ok = getattr(ps, "_compat_patched", False)
    return {
        "repo_available": _HAVE_REPO,
        "callable_residuals": _HAVE_REPO,
        "fast_project_patched": ok,
    }


if __name__ == "__main__":
    print(report())
