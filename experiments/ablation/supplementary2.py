# -*- coding: utf-8 -*-
"""配点密度、TVD 约束与采样步数的消融。

对象：Heat（`heat_80000.pt`）与 Burgers-IC（`burgers_20000.pt`）。
算法：
  A. 配点密度——Burgers 局部 Godunov 展开步数 `k∈{5,10,20}`，记录约束维数与 PCFM 保真度；
  B. TVD——按 `TV(u_T)=γ·TV(u_0)` 施加硬约束，离散 `TV=Σ|u_{i+1}-u_i|`，`γ∈{0.5,0.3}`；
  C. 采样步数——Heat 上 `n_step∈{5,10,20,50}`。
参考：论文 §4.3、附录 TVD 一节、Table 6/7。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, PCFM  # noqa: E402,F401

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

import pcfm_compat  # noqa: F401,E402
from pcfm.constraints import Residuals  # noqa: E402
from pcfm.ffm_sampler import FFM_sampler  # noqa: E402
from pcfm.pcfm_sampling import make_grid  # noqa: E402
from heat.run_heat_official import build as build_heat, heat_family, load_cfg, make_residuals  # noqa: E402
from burgers.burgers_run import build as build_burgers, seed_all  # noqa: E402
from _probe import save  # noqa: E402

PHI = 0.7853981633974483
SCALE = 1e2


class HeatTVD(Residuals):
    """IC + 质量守恒 + TVD 硬约束。"""

    def __init__(self, *a, gamma=0.5, **kw):
        super().__init__(*a, **kw)
        self.gamma = float(gamma)

    def tv(self, u_flat):
        u = u_flat.reshape(self.nx, self.nt)
        return (u[1:, :] - u[:-1, :]).abs().sum(dim=0)

    def __call__(self, u_flat):
        base = self.full_residual_heat(u_flat)
        tv = self.tv(u_flat)
        return torch.cat([base, (tv[-1] - self.gamma * tv[0]).reshape(1)])


class BurgersICK(Residuals):
    """IC + 局部 Godunov(k) + 质量守恒。"""

    def __init__(self, *a, k=5, **kw):
        super().__init__(*a, **kw)
        self.k = int(k)

    def __call__(self, u_flat):
        return self.full_residual_burgers(u_flat, k=self.k)


def stats(gen, ref):
    g = np.asarray(gen, np.float64)
    r = np.asarray(ref, np.float64)
    mu_g, mu_r = generated.mean(0), r.mean(0)
    return (float(((mu_g - mu_r) ** 2).mean() * SCALE),
            float(((generated.std(0) - r.std(0)) ** 2).mean() * SCALE))


def ce_split(hfunc, u, head_n):
    ce_head, ce_cl = [], []
    for i in range(u.shape[0]):
        r = hfunc(u[i].reshape(-1))
        ce_head.append(float(r[:head_n].pow(2).sum().sqrt()))
        ce_cl.append(float(r[head_n:].pow(2).sum().sqrt()))
    return ce_head, ce_cl


def tv_ratio(u_np):
    d = np.abs(np.diff(u_np, axis=1)).sum(axis=1)
    return (variation[:, -1] / np.maximum(variation[:, 0], 1e-12))


def heat_ctx():
    dev = "cuda"
    cfg = load_cfg(os.path.join(PCFM, "configs", "heat.yml"))
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]
    model = build_heat(cfg, dev)
    state_dict = {k: v for k, v in torch.load(os.path.join(OUT, "heat_80000.pt"),
                                     map_location=dev, weights_only=False)["model"].items()
          if k != "_metadata"}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)
    torch.manual_seed(2026)
    noise = model.gp.sample(grid, (nx, nt), n_samples=16).to(dev)
    fam = heat_family(PHI, torch.linspace(1.0, 5.0, 32), nx, nt, dev)
    return dict(cfg=cfg, nx=nx, nt=nt, dev=dev, model=model, sampler=sampler,
                noise=noise, fam=fam, ref=fam.cpu().numpy(),
                hf=make_residuals(cfg, fam[0].cpu(), dev))


def burgers_ctx():
    dev = "cuda"
    with open(os.path.join(PCFM, "configs", "burgers1d.yml"), encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    with h5py.File(os.path.join(PCFM, "datasets", "data",
                                "burgers_sampling_diffICs_nIC20_nBC512.h5"), "r") as f:
        u_ic = f["u"][...].astype(np.float32)
        x = f["x"][...].astype(np.float32)
        t = f["t"][...].astype(np.float32)
    model = build_burgers(cfg, dev)
    state_dict = {k: v for k, v in torch.load(os.path.join(OUT, "burgers_20000.pt"),
                                     map_location=dev, weights_only=False)["model"].items()
          if k != "_metadata"}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    seed_all(0)
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((len(x), len(t)), dev)
    return dict(cfg=cfg, x=x, t=t, ref=u_ic[0], model=model, sampler=sampler,
                noise=model.gp.sample(grid, (len(x), len(t)), n_samples=8).to(dev), dev=dev)


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    out = {}

    # ── A. 配点密度（Burgers-IC） ──
    b = burgers_ctx()
    nx, nt, dev = len(burgers["x"]), len(burgers["t"]), burgers["dev"]
    uv = burgers["sampler"].vanilla_sample(burgers["noise"].clone(), 20).cpu().numpy()
    vanilla_mmse, _ = stats(uv, burgers["ref"])
    rows = []
    for k in (5, 10, 20):
        hk = BurgersICK(torch.from_numpy(burgers["ref"][0]).unsqueeze(0).to(dev),
                        torch.from_numpy(burgers["x"]).to(dev),
                        torch.from_numpy(burgers["t"]).to(dev), nx=nx, nt=nt, k=k)
        n_con = int(hk(torch.zeros(nx * nt, device=dev)).numel())
        up = burgers["sampler"].pcfm_sample(burgers["noise"].clone(), 20, hk, guided_interpolation=False)
        pcfm_mmse, _ = stats(up.cpu().numpy(), burgers["ref"])
        ce_head, ce_cl = ce_split(hk, up, nx)
        rows.append({"k": k, "n_constraints": n_con, "pcfm_mmse": pcfm_mmse,
                     "vanilla_mmse": vanilla_mmse, "ratio": vanilla_mmse / pcfm_mmse,
                     "ce_head": float(np.mean(ce_head)), "ce_cl": float(np.mean(ce_cl))})
    out["collocation"] = rows

    # ── B. TVD（Heat） ──
    h = heat_ctx()
    dev = heat["dev"]
    tv_ref = float(np.mean(tv_ratio(heat["ref"])))
    xg = torch.from_numpy(np.linspace(0, 2 * np.pi, heat["nx"], dtype=np.float32)).to(dev)
    tg = torch.from_numpy(np.linspace(0, 1, heat["nt"], dtype=np.float32)).to(dev)
    rows = []
    for name, hf in (
        ("ic_mass", heat["hf"]),
        ("ic_mass_tvd_0.5", HeatTVD(heat["fam"][0].unsqueeze(0).cpu(), xg, tg,
                                    nx=heat["nx"], nt=heat["nt"], gamma=0.5)),
        ("ic_mass_tvd_0.3", HeatTVD(heat["fam"][0].unsqueeze(0).cpu(), xg, tg,
                                    nx=heat["nx"], nt=heat["nt"], gamma=0.3)),
    ):
        u = heat["sampler"].pcfm_sample(heat["noise"].clone(), 20, hf, guided_interpolation=False)
        a, s = stats(u.cpu().numpy(), heat["ref"])
        ce_head, ce_cl = ce_split(hf, u, heat["nx"])
        tvr = float(np.mean(tv_ratio(u.cpu().numpy())))
        rows.append({"config": name, "mmse": a, "smse": s,
                     "ce_ic": float(np.mean(ce_head)), "ce_cl": float(np.mean(ce_cl)),
                     "tv_ratio": tvr, "tv_ratio_ref": tv_ref,
                     "tv_dev": abs(tvr - tv_ref)})
    out["tvd"] = rows

    # ── C. 采样步数（Heat） ──
    rows = []
    for ns in (5, 10, 20, 50):
        uvn = heat["sampler"].vanilla_sample(heat["noise"].clone(), ns).cpu().numpy()
        upn = heat["sampler"].pcfm_sample(heat["noise"].clone(), ns, heat["hf"], guided_interpolation=False)
        vanilla_mmse, _ = stats(uvn, heat["ref"])
        pcfm_mmse, _ = stats(upn.cpu().numpy(), heat["ref"])
        ce_head, ce_cl = ce_split(heat["hf"], upn, heat["nx"])
        rows.append({"n_step": ns, "vanilla_mmse": vanilla_mmse, "pcfm_mmse": pcfm_mmse,
                     "ratio": vanilla_mmse / pcfm_mmse,
                     "ce_ic": float(np.mean(ce_head)), "ce_cl": float(np.mean(ce_cl))})
    out["n_step"] = rows

    save("ablation_repro", out)


if __name__ == "__main__":
    main()
