"""Blinded data checks of mass-fit inputs (qcd_abcd_yields.root with data, estimate_qcd.py --data-skim-dir).

Reads only the ROOT file, whose pass-region data bins overlapping the blind window are empty by
construction (MassfitAccumulator.fill_data), and refuses a file in which any is not: nothing here
can see data inside an SR box and the blind window. Per category:

- blinding: massfit/blinded/sr<i> flags, all zero in the data bins they flag, equal to the bins
  overlapping [blind_low, blind_high]; with --scores-json (scripts/check_data_scores.py, an
  independent pass over the data skims restricted to msoftdrop_1 < 50 or >= 150 GeV) every
  sideband bin of every region equals its count, and the file holds no data elsewhere in the pass
  regions;
- data vs MC: yields in the fail region (low sideband, window, high sideband) and in the pass
  sidebands (20-50, 150-250 GeV) per SR, and normalised-shape chi2;
- the data-driven QCD model (ABCD categories, i.e. with a TF_MC fit): per SR,
  n0_i x TF_MC shape x (fail data - non-QCD MC fail) + non-QCD MC pass, against the pass sideband
  data (the pre-fit model of combine.C with the Bernstein residual at 1 and the fail QCD taken from
  data instead of free).

Writes <output-dir>/massfit_<category>.json and massfit_<category>.pdf/.png.

Usage: pixi run python scripts/check_blinded_massfit.py CATEGORY=ROOT_FILE [...] --output-dir DIR
           [--scores-dir DIR (scores_<category>.json of check_data_scores.py)]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import uproot

from vvv.background.qcd_massfit import slugify
from vvv.ml.inference import CLASSES

GROUPS = [slugify(c) for c in CLASSES]
COLORS = {"vvv": "#e42536", "vh": "#f89c20", "top": "#5790fc", "vv": "#964a8b", "vjets": "#9c9ca1", "qcd": "#7a21dd"}
SIDEBANDS = ((20.0, 50.0), (150.0, 250.0))


def load(path):
    with uproot.open(path) as f:
        sr_ids = f["metadata/signal_regions"]["bin_index"].array(library="np").tolist()
        h = f["massfit/fail/data/yield"]
        edges = h.axis().edges()
        if f["massfit/settings/has_data"].values()[0] != 1.0:
            raise RuntimeError(f"{path}: no data")
        blind = (f["massfit/settings/blind_low"].values()[0], f["massfit/settings/blind_high"].values()[0])
        d = {"edges": edges, "sr_ids": sr_ids, "blind": blind, "regions": ["fail"] + [f"sr{s}" for s in sr_ids]}
        expected = (edges[:-1] < blind[1]) & (edges[1:] > blind[0])
        data, mc, mc2 = [h.values()], {}, {}
        for s in sr_ids:
            flags = f[f"massfit/blinded/sr{s}"].values() != 0
            if not np.array_equal(flags, expected):
                raise RuntimeError(f"{path}: blinded flags of sr{s} {flags.astype(int).tolist()} are not the bins "
                                   f"overlapping {blind}")
            v = f[f"massfit/pass/data/sr{s}/yield"].values()
            if np.any(v[flags] != 0):
                raise RuntimeError(f"{path}: blinded data bins of sr{s} are not empty; not reading it")
            data.append(v)
        d["blinded"] = expected
        d["data"] = np.array(data)
        for g in GROUPS:
            mc[g] = np.array([f[f"massfit/fail/groups/{g}/yield"].values()]
                             + [f[f"massfit/pass/groups/{g}/sr{s}/yield"].values() for s in sr_ids])
            mc2[g] = np.array([f[f"massfit/fail/groups/{g}/sumw2"].values()]
                              + [f[f"massfit/pass/groups/{g}/sr{s}/sumw2"].values() for s in sr_ids])
        d["mc"], d["mc2"] = mc, mc2
        d["qcd_from_mc"] = "massfit/settings/qcd_from_mc" in f
        if "massfit/tfmc/shape" in f:
            d["shape"] = f["massfit/tfmc/shape"].values()
            d["n0"] = f["massfit/tfmc/n0"].values()
    return d


def ratio(a, b, var_a, var_b):
    if b <= 0:
        return None, None
    r = a / b
    return float(r), float(r * np.sqrt((var_a / a ** 2 if a > 0 else 0.0) + var_b / b ** 2))


def check(cat, d, scores=None):
    edges, data, blinded = d["edges"], d["data"], d["blinded"]
    lo, hi = d["blind"]
    out = {"category": cat, "blind_window": [lo, hi], "blinded_bins": np.flatnonzero(blinded).tolist(),
           "regions": d["regions"]}
    side = {"low": (edges[1:] <= SIDEBANDS[0][1]), "high": (edges[:-1] >= SIDEBANDS[1][0])}
    window = ~(side["low"] | side["high"])
    assert np.array_equal(window, blinded)
    mc_tot = sum(d["mc"].values())
    mc_tot2 = sum(d["mc2"].values())
    nonqcd = sum(v for g, v in d["mc"].items() if g != "qcd")
    nonqcd2 = sum(v for g, v in d["mc2"].items() if g != "qcd")

    # blinding cross-check against the independent sideband-only pass over the skims
    if scores is not None:
        ok = True
        for r, name in enumerate(d["regions"]):
            ref = np.asarray(scores["sr_sideband_counts"][name], dtype=float)
            sb = ~blinded
            ok &= bool(np.array_equal(ref[sb], data[r, sb]))
            if r > 0:
                ok &= bool(np.all(data[r, blinded] == 0))
        out["sideband_counts_match_independent_skim_pass"] = ok
        if not ok:
            raise RuntimeError(f"{cat}: data sideband bins differ from check_data_scores.py's counts")

    rows = []
    for r, name in enumerate(d["regions"]):
        parts = {"low": side["low"], "high": side["high"]}
        if r == 0:
            parts["window"] = window
        row = {"region": name}
        for key, sel in parts.items():
            dv, mv, mv2 = data[r, sel].sum(), mc_tot[r, sel].sum(), mc_tot2[r, sel].sum()
            rr, re = ratio(dv, mv, dv, mv2)
            row[key] = {"data": float(dv), "mc": float(mv), "mc_stat": float(np.sqrt(mv2)), "ratio": rr, "ratio_err": re,
                        "mc_qcd_fraction": float(d["mc"]["qcd"][r, sel].sum() / mv) if mv > 0 else None}
        sel = parts["low"] | parts["high"] | (parts.get("window", np.zeros_like(blinded)))
        md, mm, mm2 = data[r, sel], mc_tot[r, sel], mc_tot2[r, sel]
        nd, nm = md.sum(), mm.sum()
        if nd > 0 and nm > 0:
            # Pearson: the expected variance of the data fraction (MC shape / nd) plus the MC statistics
            var = (mm / nm) / nd + mm2 / nm ** 2
            use = var > 0
            row["shape_chi2"] = float(np.sum((md / nd - mm / nm)[use] ** 2 / var[use]))
            row["shape_ndf"] = int(use.sum() - 1)
        rows.append(row)
    out["data_vs_mc"] = rows

    if "shape" in d:
        shape, n0 = d["shape"], d["n0"]
        qcd_fail_data = data[0] - nonqcd[0]
        qcd_fail_var = data[0] + nonqcd2[0]
        model = []
        pred_all = []
        for i, s in enumerate(d["sr_ids"]):
            r = i + 1
            pred = n0[i] * shape * qcd_fail_data + nonqcd[r]
            pvar = (n0[i] * shape) ** 2 * qcd_fail_var + nonqcd2[r]
            mcpred = n0[i] * shape * d["mc"]["qcd"][0] + nonqcd[r]
            pred_all.append((pred, pvar, mcpred))
            row = {"region": f"sr{s}", "n0": float(n0[i])}
            for key, sel in side.items():
                dv, pv, pv2 = data[r, sel].sum(), pred[sel].sum(), pvar[sel].sum()
                rr, re = ratio(dv, pv, dv, pv2)
                row[key] = {"data": float(dv), "pred": float(pv), "pred_err": float(np.sqrt(pv2)), "ratio": rr,
                            "ratio_err": re, "pred_mc_fail": float(mcpred[sel].sum()),
                            "nonqcd_mc": float(nonqcd[r, sel].sum())}
            sb = side["low"] | side["high"]
            use = sb & (pred > 0)
            row["chi2_sidebands"] = float(np.sum((data[r, use] - pred[use]) ** 2 / (pred[use] + pvar[use])))
            row["nbins_sidebands"] = int(use.sum())
            model.append(row)
        out["qcd_model_vs_data"] = model
        out["fail_qcd_data_over_mc"] = float(qcd_fail_data.sum() / d["mc"]["qcd"][0].sum())
        d["pred"] = pred_all
    return out


def plot(cat, d, path_stem):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edges, data, blinded = d["edges"], d["data"], d["blinded"]
    cen = 0.5 * (edges[:-1] + edges[1:])
    n = len(d["regions"])
    fig, axes = plt.subplots(2, n, figsize=(4.2 * n, 5.5), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05},
                             sharex=True, squeeze=False)
    mc_tot2 = sum(d["mc2"].values())
    for r, name in enumerate(d["regions"]):
        ax, rx = axes[0, r], axes[1, r]
        bottom = np.zeros(len(cen))
        for g in ["qcd", "vjets", "top", "vv", "vh", "vvv"]:
            y = d["mc"][g][r]
            ax.stairs(bottom + y, edges, baseline=bottom, fill=True, color=COLORS[g], alpha=0.85, label=g)
            bottom = bottom + y
        dv = data[r].astype(float)
        if r > 0:
            dv = np.where(blinded, np.nan, dv)
            for a in (ax, rx):
                a.axvspan(*d["blind"], color="grey", alpha=0.25, lw=0)
        ax.errorbar(cen, dv, yerr=np.sqrt(np.nan_to_num(dv)), fmt="o", ms=3, color="black", label="data")
        with np.errstate(divide="ignore", invalid="ignore"):
            rx.errorbar(cen, dv / bottom, yerr=np.sqrt(np.nan_to_num(dv)) / bottom, fmt="o", ms=3, color="black",
                        label="data / MC")
            rx.fill_between(edges, np.r_[1 - np.sqrt(mc_tot2[r]) / bottom, np.nan],
                            np.r_[1 + np.sqrt(mc_tot2[r]) / bottom, np.nan], step="post", color="grey", alpha=0.4)
            if r > 0 and "pred" in d:
                pred, pvar, _ = d["pred"][r - 1]
                ax.stairs(pred, edges, color="#bd1f01", lw=1.8)
                rx.errorbar(cen + 2, dv / pred, yerr=np.sqrt(np.nan_to_num(dv)) / pred, fmt="s", ms=3, color="#bd1f01")
        rx.axhline(1, color="black", lw=0.8)
        rx.set_ylim(0, 2.5)
        rx.set_xlabel("msoftdrop_1 [GeV]")
        ax.set_yscale("log")
        ax.set_title(f"{cat} {name}", fontsize=9)
        if r == 0:
            handles, labels = ax.get_legend_handles_labels()
            if "pred" in d:
                handles.append(plt.Line2D([], [], color="#bd1f01", lw=1.8))
                labels.append("QCD model: n0 TF$_{MC}$ (fail data - nonQCD MC) + nonQCD MC")
                handles.append(plt.Line2D([], [], color="#bd1f01", marker="s", ls="", ms=3))
                labels.append("data / QCD model (ratio panel)")
            ax.legend(handles, labels, fontsize=6, ncol=1, frameon=False, loc="lower left")
            rx.set_ylabel("data / MC")
    fig.savefig(f"{path_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_stem}.png", dpi=80, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", help="CATEGORY=qcd_abcd_yields.root")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--scores-dir", type=Path, help="scores_<category>.json of check_data_scores.py")
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for item in args.inputs:
        cat, path = item.split("=", 1)
        d = load(path)
        scores = None
        if args.scores_dir and (args.scores_dir / f"scores_{cat}.json").exists():
            scores = json.loads((args.scores_dir / f"scores_{cat}.json").read_text())
        out = check(cat, d, scores)
        out["root_file"] = path
        (args.output_dir / f"massfit_{cat}.json").write_text(json.dumps(out, indent=1))
        plot(cat, d, args.output_dir / f"massfit_{cat}")
        print(f"{cat}: blinded bins {out['blinded_bins']}, sideband cross-check "
              f"{out.get('sideband_counts_match_independent_skim_pass', 'not run')}", flush=True)


if __name__ == "__main__":
    main()
