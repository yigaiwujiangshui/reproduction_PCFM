# -*- coding: utf-8 -*-
"""
PCFM 调用层适配（不改官方仓库算法，只补两处接口缺口）
=====================================================

背景
----
官方仓库 `pcfpengfei/PCFM` 的推理脚本未公开，因此三块组件（模型 / 约束 / 采样器）
之间缺少一层胶水。本文件只做接口适配，不重写任何算法。

缺口 1：Residuals 不可调用
    pcfm_sampling.py:145 写的是 `res = hfunc(u_corr)`，要求 hfunc 像函数一样可调用；
    但 pcfm.constraints.Residuals 只提供 full_residual_heat() 这类方法，没有 __call__。
    报错：TypeError: 'Residuals' object is not callable
    修法：定义子类 CallableResiduals 补 __call__，转发到 full_residual_heat。

缺口 2：fast_project_batched 签名不一致（仓库内部两处代码各行其是）
    调用方 ffm_sampler.py:36
        fast_project_batched(u, hfunc, mode=..., newtonsteps=..., eps=..., use_vmap=...)
    定义方 pcfm_sampling.py:17
        def fast_project_batched(xi_batch, h_func, max_iter: int = 1)
    报错：TypeError: fast_project_batched() got an unexpected keyword argument 'mode'
    修法：包装一层，把 newtonsteps 映射为 max_iter，其余仓库实现用不到的参数丢弃。

用法
----
    import pcfm_compat          # 必须在 import pcfm.ffm_sampler 之前
    from pcfm_compat import CallableResiduals
"""

import torch

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
