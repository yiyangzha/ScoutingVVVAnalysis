"""Validation of rescored data skims (scripts/rescore_skims.py) for one category, blinded.

1. Exact check: in --n-files random files of the rescored data skim, every non-proba column equals
   the source skim's (same rows, same order), and for --n-rows random rows per file the stored
   proba_<class> equal CategoryModel.predict on the source features bit for bit. The same check on
   --n-mc-files random files of the MC fit-region skims shows that MC and data carry scores of
   the same models.
2. Data vs MC in the msoftdrop_1 sidebands (m < 50 or m >= 150 GeV, the bins the mass fit never
   blinds): the proba_<class> and control-variable distributions, the yields, and per SR box the
   sideband counts per mass bin (sr_sideband_counts, to compare with the pass/data bins of
   qcd_abcd_yields.root). MC (skim weights x --skim-lumi) is scaled to --data-lumi.

Blinding: every data table is cut to the sidebands right after it is read, before any
selection, count or histogram; no data with 50 <= msoftdrop_1 < 150 GeV reaches any output.

Writes <output-dir>/scores_<category>.json and scores_<category>.pdf/.png.

Usage: pixi run python scripts/check_data_scores.py --category CAT --data-skim-dir DIR --source-skim-dir DIR
           --mc-skim-dir DIR --model-dir DIR --training-config JSON --signal-region-csv CSV
           [--exclude-samples S ...] [--data-lumi L] [--skim-lumi L] --output-dir DIR
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from vvv.background.qcd_massfit import assign_regions, load_sr_boxes, mass_bin
from vvv.ml.inference import CLASSES, CategoryModel

MSD = "ScoutingFatPFJetRecluster_msoftdrop_1"
BLIND = (50.0, 150.0)
EDGES = np.arange(20.0, 251.0, 10.0)
CONTROLS = ["ScoutingFatPFJetRecluster_pt_1", "ScoutingFatPFJetRecluster_pt_2", "ScoutingFatPFJetRecluster_eta_1",
            "ScoutingFatPFJetRecluster_msoftdrop_2", "ScoutingFatPFJetRecluster_VvsQCD_1",
            "ScoutingFatPFJetRecluster_VvsQCD_2", "ScoutingFatPFJetRecluster_selected_pt_sum",
            "ScoutingFatPFJetRecluster_sphere_mass", "ScoutingFatPFJetRecluster_mass_fraction_1",
            "ScoutingPFJetRecluster2_selected_HT", "ScoutingMET_pt", "nScoutingPFJetRecluster2_selected"]
PROBA = [f"proba_{c}" for c in CLASSES]


def log(message):
    print(message, flush=True)


def sidebands(table):
    """The blinding cut: only events with msoftdrop_1 < 50 or >= 150 GeV, as numpy columns."""
    m = table[MSD].to_numpy().astype(np.float64)
    keep = (m < BLIND[0]) | (m >= BLIND[1])
    out = {c: table[c].to_numpy()[keep] for c in table.column_names}
    assert not np.any((out[MSD] >= BLIND[0]) & (out[MSD] < BLIND[1]))
    return out


def exact_check(src_files, dst_root, src_root, model, n_files, n_rows, rng, label):
    files = sorted(rng.choice(len(src_files), size=min(n_files, len(src_files)), replace=False))
    res = {"files": [], "rows_checked": 0, "proba_mismatches": 0, "column_mismatches": []}
    for i in files:
        src = src_files[i]
        s = pq.read_table(src)
        if dst_root is None:  # MC: the stored proba of the skim itself
            d = s
        else:
            d = pq.read_table(dst_root / src.relative_to(src_root))
            if d.num_rows != s.num_rows:
                res["column_mismatches"].append(f"{src.name}: {d.num_rows} != {s.num_rows} rows")
                continue
            for c in d.column_names:
                if c not in PROBA and not d[c].equals(s[c]):
                    res["column_mismatches"].append(f"{src.name}:{c}")
        rows = np.sort(rng.choice(s.num_rows, size=min(n_rows, s.num_rows), replace=False))
        cols = {f: s[f].to_numpy()[rows] for f in model.features}
        ref = model.predict(cols, s["event"].to_numpy()[rows])
        got = np.column_stack([d[p].to_numpy()[rows] for p in PROBA])
        bad = int(np.sum(np.any(got != ref, axis=1)))
        res["files"].append(str(src))
        res["rows_checked"] += len(rows)
        res["proba_mismatches"] += bad
        res["max_abs_diff"] = max(res.get("max_abs_diff", 0.0), float(np.max(np.abs(got - ref))) if len(rows) else 0.0)
    log(f"exact check ({label}): {len(res['files'])} files, {res['rows_checked']} rows, "
        f"{res['proba_mismatches']} proba mismatches (max |diff| {res.get('max_abs_diff', 0.0):.3g}), "
        f"{len(res['column_mismatches'])} column mismatches")
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--category", required=True)
    p.add_argument("--data-skim-dir", type=Path, required=True, help="rescored data skims (<dir>/data_2024/...)")
    p.add_argument("--source-skim-dir", type=Path, required=True, help="the data skims they were made from")
    p.add_argument("--mc-skim-dir", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--training-config", type=Path, required=True)
    p.add_argument("--signal-region-csv", type=Path, required=True)
    p.add_argument("--exclude-samples", nargs="*", default=[])
    p.add_argument("--data-sample", default="data_2024")
    p.add_argument("--data-lumi", type=float, default=None, help="default: the catalogue's data_2024 lumi_fb")
    p.add_argument("--skim-lumi", type=float, default=255.8, help="luminosity of the MC skim weights")
    p.add_argument("--n-files", type=int, default=8)
    p.add_argument("--n-rows", type=int, default=5000)
    p.add_argument("--n-mc-files", type=int, default=4)
    p.add_argument("--nthread", type=int, default=16)
    p.add_argument("--seed", type=int, default=20261110)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    t0 = time.time()
    cat = args.category
    if args.data_lumi is None:
        from vvv.samples import load_samples

        args.data_lumi = load_samples()[args.data_sample].lumi_fb
    scale = args.data_lumi / args.skim_lumi
    tcfg = json.loads(args.training_config.read_text())
    groups = {c: [s for s in v if s not in set(args.exclude_samples)] for c, v in tcfg["class_groups"].items()}
    boxes = load_sr_boxes(args.signal_region_csv, list(CLASSES))
    model = CategoryModel.load(args.model_dir, cat)
    model.set_nthread(args.nthread)
    rng = np.random.default_rng(args.seed)
    out = {"category": cat, "data_lumi": args.data_lumi, "mc_scale": scale, "blind_window": BLIND,
           "sideband_definition": "msoftdrop_1 < 50 or >= 150 GeV", "exclude_samples": args.exclude_samples}

    # 1. exact scores
    src_root = args.source_skim_dir
    src_files = sorted((src_root / args.data_sample / cat / "nominal").glob("*.parquet"))
    out["exact_data"] = exact_check(src_files, args.data_skim_dir, src_root, model, args.n_files, args.n_rows, rng, "data")
    mc_files = [f for c in groups.values() for s in c
                for f in sorted((args.mc_skim_dir / s / cat / "nominal").glob("*.parquet"))]
    mc_files = [f for f in mc_files if pq.ParquetFile(f).metadata.num_rows > 0]
    out["exact_mc"] = exact_check(mc_files, None, None, model, args.n_mc_files, args.n_rows, rng, "MC")

    # 2. data vs MC in the sidebands: MC first (it fixes the control-variable binning)
    columns = PROBA + [MSD] + CONTROLS
    mc = {}
    for c, samples in groups.items():
        parts = []
        for s in samples:
            path = args.mc_skim_dir / s / cat / "nominal"
            if not path.exists():
                continue
            for f in sorted(path.glob("*.parquet")):
                t = pq.read_table(f, columns=columns + ["weight"])
                if t.num_rows:
                    parts.append(sidebands(t))
        if parts:
            mc[c] = {k: np.concatenate([q[k] for q in parts]) for k in parts[0]}
            mc[c]["weight"] = mc[c]["weight"] * scale
    allmc = {k: np.concatenate([v[k] for v in mc.values()]) for k in columns + ["weight"]}
    bins = {pcol: np.linspace(0.0, 1.0, 41) for pcol in PROBA}
    for v in CONTROLS:
        x = allmc[v].astype(np.float64)
        x = x[x > -990]
        if v.startswith("n"):
            bins[v] = np.arange(-0.5, min(np.percentile(x, 99.8), 30) + 1.5)
        else:
            lo, hi = np.percentile(x, [0.2, 99.8])
            bins[v] = np.linspace(lo, hi, 41)
    bins[MSD] = EDGES

    def fill(cols, w=None):
        h = {}
        for v, e in bins.items():
            x = np.clip(cols[v].astype(np.float64), e[0], e[-1] - 1e-9 * (e[-1] - e[0]))
            h[v] = (np.histogram(x, e, weights=w)[0], np.histogram(x, e, weights=None if w is None else w ** 2)[0])
        region = assign_regions(np.column_stack([cols[pc] for pc in PROBA]), boxes, list(CLASSES))
        mb = mass_bin(cols[MSD], EDGES)
        ok = mb >= 0
        flat = region[ok] * (len(EDGES) - 1) + mb[ok]
        size = (len(boxes) + 1) * (len(EDGES) - 1)
        wk = None if w is None else w[ok]
        h["region_mass"] = (np.bincount(flat, weights=wk, minlength=size).reshape(len(boxes) + 1, -1),
                            np.bincount(flat, weights=None if wk is None else wk ** 2,
                                        minlength=size).reshape(len(boxes) + 1, -1))
        return h

    def add(a, b):
        return b if a is None else {k: (a[k][0] + b[k][0], a[k][1] + b[k][1]) for k in a}

    hmc = {c: fill(v, v["weight"]) for c, v in mc.items()}
    hdata, n_data = None, 0
    dfiles = sorted((args.data_skim_dir / args.data_sample / cat / "nominal").glob("*.parquet"))
    for f in dfiles:
        d = sidebands(pq.read_table(f, columns=columns))  # blinding: nothing else is read
        n_data += len(d[MSD])
        hdata = add(hdata, fill(d))
    log(f"sidebands: {n_data} data events, MC {sum(float(v['weight'].sum()) for v in mc.values()):.1f} "
        f"(x {scale:.4f} to {args.data_lumi:.2f} fb^-1)")

    mc_tot = sum(hmc[c]["region_mass"][0] for c in hmc)
    mc_tot2 = sum(hmc[c]["region_mass"][1] for c in hmc)
    d_rm = hdata["region_mass"][0]
    assert not d_rm[1:, (EDGES[:-1] < BLIND[1]) & (EDGES[1:] > BLIND[0])].any()
    side_bins = (EDGES[1:] <= BLIND[0]) | (EDGES[:-1] >= BLIND[1])
    lo_bins, hi_bins = EDGES[1:] <= BLIND[0], EDGES[:-1] >= BLIND[1]
    regions = ["fail"] + [f"sr{b[0]}" for b in boxes]
    table = []
    for r, name in enumerate(regions):
        row = {"region": name}
        for key, sel in (("low", lo_bins), ("high", hi_bins), ("both", side_bins)):
            dv, mv, mv2 = d_rm[r, sel].sum(), mc_tot[r, sel].sum(), mc_tot2[r, sel].sum()
            row[key] = {"data": float(dv), "mc": float(mv), "mc_stat": float(np.sqrt(mv2)),
                        "ratio": float(dv / mv) if mv > 0 else None}
        row["mc_by_class"] = {c: float(hmc[c]["region_mass"][0][r, side_bins].sum()) for c in hmc}
        table.append(row)
    out["sideband_yields"] = table
    out["sr_sideband_counts"] = {name: d_rm[r].tolist() for r, name in enumerate(regions)}
    mean = {}
    for pc in PROBA:
        c = 0.5 * (bins[pc][:-1] + bins[pc][1:])
        md, mm = hdata[pc][0], sum(hmc[k][pc][0] for k in hmc)
        mean[pc] = {"data": float((md * c).sum() / md.sum()), "mc": float((mm * c).sum() / mm.sum())}
    out["proba_mean_sidebands"] = mean
    shape = {}
    for v in list(PROBA) + CONTROLS:
        md, mm, mm2 = hdata[v][0].astype(float), sum(hmc[k][v][0] for k in hmc), sum(hmc[k][v][1] for k in hmc)
        nd, nm = md.sum(), mm.sum()
        ok = (md > 0) | (mm > 0)
        var = md / nd ** 2 + mm2 / nm ** 2
        chi2 = float(np.sum(((md / nd - mm / nm) ** 2 / np.where(var > 0, var, 1.0))[ok]))
        shape[v] = {"chi2": chi2, "ndf": int(ok.sum() - 1), "data_over_mc": float(nd / nm)}
    out["shape_chi2_sidebands"] = shape

    # plots: proba and control variables, data vs stacked MC, sidebands only
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    colors = {"VVV": "#e42536", "VH": "#f89c20", "Top": "#5790fc", "VV": "#964a8b", "VJets": "#9c9ca1", "QCD": "#7a21dd"}
    order = [c for c in ("VVV", "VH", "VV", "Top", "VJets", "QCD") if c in hmc][::-1]
    variables = list(PROBA) + CONTROLS + [MSD]
    pdf_path = args.output_dir / f"scores_{cat}.pdf"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with PdfPages(pdf_path) as pdf:
        n = len(variables)
        ncol = 4
        nrow = (n + ncol - 1) // ncol
        fig = plt.figure(figsize=(5 * ncol, 4.6 * nrow))
        gs = fig.add_gridspec(nrow, ncol, hspace=0.3, wspace=0.3, top=0.96)
        for k, v in enumerate(variables):
            sub = gs[k // ncol, k % ncol].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(sub[0])
            rx = fig.add_subplot(sub[1], sharex=ax)
            e = bins[v]
            bottom = np.zeros(len(e) - 1)
            for c in order:
                y = hmc[c][v][0]
                ax.stairs(bottom + y, e, baseline=bottom, fill=True, color=colors[c], label=c, alpha=0.85)
                bottom = bottom + y
            mm2 = sum(hmc[c][v][1] for c in hmc)
            dv = hdata[v][0].astype(float)
            cen = 0.5 * (e[:-1] + e[1:])
            if v == MSD:  # only the sideband bins exist
                dv = np.where(side_bins, dv, np.nan)
                ax.axvspan(BLIND[0], BLIND[1], color="grey", alpha=0.25, label="blinded")
                rx.axvspan(BLIND[0], BLIND[1], color="grey", alpha=0.25)
            ax.errorbar(cen, dv, yerr=np.sqrt(np.nan_to_num(dv)), fmt="o", ms=3, color="black", label="data")
            ax.set_yscale("log")
            ax.tick_params(labelbottom=False)
            ax.set_title(v.replace("ScoutingFatPFJetRecluster", "AK8").replace("ScoutingPFJetRecluster2", "AK4"),
                         fontsize=9)
            with np.errstate(divide="ignore", invalid="ignore"):
                rx.errorbar(cen, dv / bottom, yerr=np.sqrt(np.nan_to_num(dv)) / bottom, fmt="o", ms=3, color="black")
                rx.fill_between(e, np.r_[1 - np.sqrt(mm2) / bottom, np.nan], np.r_[1 + np.sqrt(mm2) / bottom, np.nan],
                                step="post", color="grey", alpha=0.4)
            rx.axhline(1.0, color="black", lw=0.8)
            rx.set_ylim(0, 2.5)
            rx.set_ylabel("data/MC", fontsize=8)
            if k == 0:
                ax.legend(fontsize=7, ncol=2, frameon=False)
        fig.suptitle(f"{cat}: data vs MC (x {scale:.3f} to {args.data_lumi:.1f} fb$^{{-1}}$), msoftdrop_1 sidebands "
                     f"(< 50 or >= 150 GeV) only", y=0.985)
        pdf.savefig(fig, bbox_inches="tight")
        fig.savefig(args.output_dir / f"scores_{cat}.png", dpi=80, bbox_inches="tight")
        plt.close(fig)
    out["seconds"] = time.time() - t0
    (args.output_dir / f"scores_{cat}.json").write_text(json.dumps(out, indent=1))
    log(f"wrote {pdf_path} and scores_{cat}.json ({out['seconds']:.0f} s)")


if __name__ == "__main__":
    main()
