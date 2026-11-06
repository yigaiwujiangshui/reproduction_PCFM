# -*- coding: utf-8 -*-
"""分布外泛化检验。

对象：受约束采样的初始相位 `φ`，取训练区间 `[0,π]` 内 3 点与区间外 4 点；参考解族 32 条（ν∈[1,5]）。
算法：对每个 `φ` 分别运行 vanilla FFM 与 PCFM（硬投影），记录 MMSE 与约束残差 CE(IC)/CE(CL)，再按区间汇总。
参考：论文附录 L.4 / Table 8。"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _bootstrap import OUT, PCFM  # noqa: E402,F401

import torch  # noqa: E402

import pcfm_compat  # noqa: F401,E402
from pcfm.ffm_sampler import FFM_sampler  # noqa: E402
from pcfm.pcfm_sampling import make_grid  # noqa: E402
from heat.run_heat_official import build, heat_family, load_cfg, make_residuals, metrics  # noqa: E402
from _probe import save  # noqa: E402

CKPT = os.path.join(OUT, "heat_30000.pt")

# (φ, 标签, 是否分布内)
CASES = [
    (0.7853981633974483, "pi/4", True),
    (0.30, "0.30", True),
    (2.80, "2.80", True),
    (-1.20, "-1.20", False),
    (3.50, "3.50", False),
    (4.80, "4.80", False),
    (6.00, "6.00", False),
]


def main():
    os.environ.setdefault("WANDB_MODE", "disabled")
    cfg = load_cfg(os.path.join(PCFM, "configs", "heat.yml"))
    dev = "cuda"
    nx, nt = cfg["datasets"]["nx"], cfg["datasets"]["nt"]

    model = build(cfg, dev)
    state_dict = {k: v for k, v in torch.load(CKPT, map_location=dev, weights_only=False)["model"].items()
          if k != "_metadata"}
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    sampler = FFM_sampler(model, model.gp)
    grid = make_grid((nx, nt), dev)

    torch.manual_seed(4242)
    noise = model.gp.sample(grid, (nx, nt), n_samples=8).to(dev)

    cases = []
    for phi, tag, in_dist in CASES:
        family = heat_family(phi, torch.linspace(1.0, 5.0, 32), nx, nt, dev)
        hfunc = make_residuals(cfg, family[0].cpu(), dev)
        u_vanilla = sampler.vanilla_sample(noise, 20)
        u_pcfm = sampler.pcfm_sample(noise, 20, hfunc, guided_interpolation=False)
        vanilla_metrics = metrics(u_vanilla, family, hfunc)
        pcfm_metrics = metrics(u_pcfm, family, hfunc)
        ratio = vanilla_metrics["MMSE"] / pcfm_metrics["MMSE"] if pcfm_metrics["MMSE"] > 1e-12 else None
        cases.append({"phi": phi, "tag": tag, "in_dist": in_dist,
                      "vanilla": vanilla_metrics, "pcfm": pcfm_metrics, "ratio": ratio})

    summary = {}
    for label, sel in (("in_dist", [c for c in cases if c["in_dist"]]),
                       ("out_of_dist", [c for c in cases if not c["in_dist"]])):
        if not sel:
            continue
        v = [c["vanilla"]["MMSE"] for c in sel]
        p = [c["pcfm"]["MMSE"] for c in sel]
        summary[label] = {
            "n": len(sel),
            "vanilla_mmse_mean": sum(v) / len(v),
            "pcfm_mmse_mean": sum(p) / len(p),
            "ratio_mean": sum(a / b for a, b in zip(v, p)) / len(v),
            "pcfm_ce_ic_max": max(abs(c["pcfm"]["CE_IC"]) for c in sel),
            "pcfm_ce_cl_range": [min(c["pcfm"]["CE_CL"] for c in sel),
                                 max(c["pcfm"]["CE_CL"] for c in sel)],
            "vanilla_ce_ic_range": [min(c["vanilla"]["CE_IC"] for c in sel),
                                    max(c["vanilla"]["CE_IC"] for c in sel)],
        }

    save("ood_heat", {"ckpt": os.path.basename(CKPT), "n_sample": 8, "n_step": 20,
                      "cases": cases, "summary": summary})


if __name__ == "__main__":
    main()
