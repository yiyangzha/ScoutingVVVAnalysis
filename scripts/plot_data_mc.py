"""Blinded data/MC comparison plots (review booklet) for the baseline categories.

Two stages, both per category:

1. fill (default; --plot-only skips it): one pass over the MC fit-region skims and one over the
   rescored data skims, file by file and batch by batch, accumulating histograms only. MC weights
   (physics weight x --skim-lumi) are scaled to --data-lumi. Saves <output-dir>/cache/<category>.npz.
2. plot: <output-dir>/data_mc_<category>.pdf (multi-page), <output-dir>/summary.pdf,
   <output-dir>/summary.json, and the focus study (--focus CAT:SR ...) as focus_<category>_sr<i>.pdf.

Blinding (vvv.validation.blinding). A data event is never used if it is inside any SR box AND has
50 <= msoftdrop_1 < 150 GeV. The cut is applied to every data batch right after it is read, before
anything else (apply_blinding asserts it), and every data histogram of an SR selection is checked to
be empty in the window again before saving and before plotting. What is shown:
  - fail region (outside every SR box): all msoftdrop_1;
  - SR-box events, and events of any score, only in the msoftdrop_1 sidebands (< 50 or >= 150 GeV).

Selections (region r: 0 = fail, i = SR box i; sb = msoftdrop_1 sideband):
  fail (r == 0), fail_sb (r == 0 & sb), sb (all r, sb), sru_sb (r > 0 & sb), sr<i>_sb (r == i & sb).

Focus extras (all categories, all SRs; only sideband events): N-1 score distributions per box axis,
the box loosened/tightened by a factor (vvv.background.qcd_massfit.box_mask semantics), data counts
per era (2024G/H/I by run number) and per run, the list of SR-box sideband data events (duplicate
check), and per-sample MC yields and maximal weights.

Caveat: the MC skims predate the JMS update (1.071 vs current 1.049); this is printed on every page.

Usage: pixi run python scripts/plot_data_mc.py [--categories CAT ...] [--plot-only] [--workers 8]
           [--focus boosted2_unresolved:1 boosted3:4] [--output-dir DIR]
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

MSD = "ScoutingFatPFJetRecluster_msoftdrop_1"
CONTROLS = ["ScoutingFatPFJetRecluster_pt_1", "ScoutingFatPFJetRecluster_pt_2", "ScoutingFatPFJetRecluster_eta_1",
            "ScoutingFatPFJetRecluster_msoftdrop_2", "ScoutingFatPFJetRecluster_VvsQCD_1",
            "ScoutingFatPFJetRecluster_VvsQCD_2", "ScoutingFatPFJetRecluster_selected_pt_sum",
            "ScoutingFatPFJetRecluster_sphere_mass", "ScoutingFatPFJetRecluster_mass_fraction_1",
            "ScoutingPFJetRecluster2_selected_HT", "ScoutingMET_pt", "nScoutingPFJetRecluster2_selected"]
CLASSES = ("VVV", "VH", "Top", "VV", "VJets", "QCD")  # == vvv.ml.inference.CLASSES (checked in main)
PROBA = [f"proba_{c}" for c in CLASSES]
VARS = [MSD] + CONTROLS + PROBA
BINS = {
    MSD: np.arange(20.0, 251.0, 10.0),  # the mass-fit binning
    "ScoutingFatPFJetRecluster_pt_1": np.arange(175.0, 1001.0, 25.0),
    "ScoutingFatPFJetRecluster_pt_2": np.arange(175.0, 801.0, 25.0),
    "ScoutingFatPFJetRecluster_eta_1": np.linspace(-2.4, 2.4, 25),
    "ScoutingFatPFJetRecluster_msoftdrop_2": np.arange(40.0, 151.0, 5.0),
    "ScoutingFatPFJetRecluster_VvsQCD_1": np.linspace(0.0, 1.0, 26),
    "ScoutingFatPFJetRecluster_VvsQCD_2": np.linspace(0.0, 1.0, 26),
    "ScoutingFatPFJetRecluster_selected_pt_sum": np.arange(350.0, 2001.0, 50.0),
    "ScoutingFatPFJetRecluster_sphere_mass": np.arange(0.0, 181.0, 5.0),
    "ScoutingFatPFJetRecluster_mass_fraction_1": np.linspace(0.0, 0.4, 41),
    "ScoutingPFJetRecluster2_selected_HT": np.arange(0.0, 1001.0, 25.0),
    "ScoutingMET_pt": np.arange(0.0, 1201.0, 25.0),  # to 1.2 TeV: the unresolved SR1 events have MET ~ 0.5 TeV
    "nScoutingPFJetRecluster2_selected": np.arange(-0.5, 11.0, 1.0),
    **{p: np.linspace(0.0, 1.0, 41) for p in PROBA},
}
LABELS = {
    MSD: r"AK8 $m_{SD,1}$ [GeV]",
    "ScoutingFatPFJetRecluster_pt_1": r"AK8 $p_{T,1}$ [GeV]",
    "ScoutingFatPFJetRecluster_pt_2": r"AK8 $p_{T,2}$ [GeV]",
    "ScoutingFatPFJetRecluster_eta_1": r"AK8 $\eta_1$",
    "ScoutingFatPFJetRecluster_msoftdrop_2": r"AK8 $m_{SD,2}$ [GeV]",
    "ScoutingFatPFJetRecluster_VvsQCD_1": "AK8 VvsQCD$_1$",
    "ScoutingFatPFJetRecluster_VvsQCD_2": "AK8 VvsQCD$_2$",
    "ScoutingFatPFJetRecluster_selected_pt_sum": r"AK8 $\sum p_T$ (selected) [GeV]",
    "ScoutingFatPFJetRecluster_sphere_mass": "AK8 sphere mass [GeV]",
    "ScoutingFatPFJetRecluster_mass_fraction_1": "AK8 mass fraction$_1$",
    "ScoutingPFJetRecluster2_selected_HT": r"AK4 $H_T$ (selected) [GeV]",
    "ScoutingMET_pt": r"$p_T^{miss}$ [GeV]",
    "nScoutingPFJetRecluster2_selected": "AK4 multiplicity (selected)",
    **{p: f"BDT score {p}" for p in PROBA},
}
NM1_EDGES = np.linspace(0.0, 1.0, 101)
LOOSEN = (0.8, 0.9, 1.0, 1.25, 1.5, 2.0, 3.0)
ERAS = (("2024G", 383780, 385813), ("2024H", 385814, 386408), ("2024I", 386409, 386951))
RUN0, RUN1 = ERAS[0][1], ERAS[-1][2] + 1
BATCH = 4_000_000
JMS_CAVEAT = "MC skims predate the JMS update (JMS 1.071 vs current 1.049)"
COLORS = {"VVV": "#e42536", "VH": "#f89c20", "Top": "#5790fc", "VV": "#964a8b", "VJets": "#9c9ca1", "QCD": "#7a21dd"}
STACK = ("QCD", "VJets", "Top", "VV", "VH", "VVV")  # bottom to top
CATEGORIES = ("boosted3", "boosted2_resolved", "boosted2_unresolved", "boosted2_leptonic_1lep",
              "boosted2_leptonic_2lep")
CAT_LABEL = {"boosted3": "boosted3", "boosted2_resolved": "boosted2 resolved", "boosted2_unresolved": "boosted2 unresolved",
             "boosted2_leptonic_1lep": "boosted2 leptonic 1$\\ell$", "boosted2_leptonic_2lep": "boosted2 leptonic 2$\\ell$"}


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --- filling ------------------------------------------------------------------


def bin_index(x, edges):
    """Bin index with under/overflow folded into the first/last bin; NaN -> -1 (dropped)."""
    x = np.asarray(x, dtype=np.float64)
    idx = np.searchsorted(edges, x, side="right") - 1
    idx = np.clip(idx, 0, len(edges) - 2)
    idx[~np.isfinite(x)] = -1
    return idx


def era_index(run):
    out = np.full(len(run), -1, dtype=np.int64)
    for k, (_, lo, hi) in enumerate(ERAS):
        out[(run >= lo) & (run <= hi)] = k
    return out


def merge(a, b):
    """Add nested dicts of arrays; keys 'maxw' take the maximum, keys 'ev_*' are concatenated."""
    if a is None:
        return b
    for k, v in b.items():
        if k not in a:
            a[k] = v
        elif isinstance(v, dict):
            a[k] = merge(a[k], v)
        elif k.startswith("ev_"):
            a[k] = np.concatenate([a[k], v])
        elif k == "maxw":
            a[k] = np.maximum(a[k], v)
        else:
            a[k] = a[k] + v
    return a


def hist(idx, mask, nb, w):
    i = idx[mask]
    ok = i >= 0
    if w is None:
        c = np.bincount(i[ok], minlength=nb).astype(np.float64)
        return c, c.copy()
    wm = w[mask][ok]
    return np.bincount(i[ok], weights=wm, minlength=nb), np.bincount(i[ok], weights=wm * wm, minlength=nb)


def fill_batch(cols, region, w, boxes, is_data):
    """All histograms of one batch. cols: numpy columns, region from assign_regions, w None for data."""
    from vvv.background.qcd_massfit import box_mask
    from vvv.validation.blinding import in_sideband

    m = cols[MSD]
    sb = in_sideband(m)
    low = sb & (m.astype(np.float64) < 50.0)
    high = sb & ~low
    sels = {"fail": region == 0, "fail_sb": (region == 0) & sb, "sb": sb, "sru_sb": (region > 0) & sb}
    for k, (sid, _) in enumerate(boxes):
        sels[f"sr{sid}_sb"] = (region == k + 1) & sb
    idx = {v: bin_index(cols[v], BINS[v]) for v in VARS}
    out = {"h": {}, "n": {}}
    for s, mask in sels.items():
        out["h"][s] = {}
        for v in VARS:
            sw, sw2 = hist(idx[v], mask, len(BINS[v]) - 1, w)
            out["h"][s][v] = {"sw": sw, "sw2": sw2}
        out["n"][s] = np.array([float(mask.sum())])
    proba = np.column_stack([cols[p] for p in PROBA])
    out["nm1"], out["loosen"] = {}, {}
    for sid, bounds in boxes:
        axis_pass = {a: box_mask(proba, {a: b}, list(CLASSES)) for a, b in bounds.items()}
        out["nm1"][f"sr{sid}"] = {}
        for a in bounds:
            mask = sb.copy()
            for o, p in axis_pass.items():
                if o != a:
                    mask &= p
            sw, sw2 = hist(bin_index(cols[f"proba_{a}"], NM1_EDGES), mask, len(NM1_EDGES) - 1, w)
            out["nm1"][f"sr{sid}"][a] = {"sw": sw, "sw2": sw2}
        lo_w, lo_w2, hi_w, hi_w2 = (np.zeros(len(LOOSEN)) for _ in range(4))
        for j, f in enumerate(LOOSEN):
            inbox = box_mask(proba, bounds, list(CLASSES), factor=f)
            for sel, a1, a2 in ((low, lo_w, lo_w2), (high, hi_w, hi_w2)):
                mk = inbox & sel
                if w is None:
                    a1[j] = a2[j] = mk.sum()
                else:
                    a1[j], a2[j] = w[mk].sum(), (w[mk] ** 2).sum()
        out["loosen"][f"sr{sid}"] = {"low_sw": lo_w, "low_sw2": lo_w2, "high_sw": hi_w, "high_sw2": hi_w2}
    if is_data:
        run = cols["run"].astype(np.int64)
        era = era_index(run)
        assert np.all(era >= 0), f"runs outside {ERAS}: {np.unique(run[era < 0])[:10]}"
        mb = idx[MSD]
        nmb = len(BINS[MSD]) - 1
        out["era"], out["run"] = {}, {}
        for s, mask in sels.items():
            flat = era[mask] * nmb + mb[mask]
            out["era"][s] = np.bincount(flat[mb[mask] >= 0], minlength=len(ERAS) * nmb).reshape(len(ERAS), nmb).astype(float)
            out["run"][s] = np.bincount(run[mask] - RUN0, minlength=RUN1 - RUN0).astype(float)
        ev = sels["sru_sb"]
        out["events"] = {f"ev_{k}": np.asarray(v)[ev] for k, v in cols.items()}
        out["events"]["ev_region"] = region[ev]
    return out


def _batches(path, columns):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for b in pf.iter_batches(batch_size=BATCH, columns=columns):
        yield {c: b.column(c).to_numpy(zero_copy_only=False) for c in columns}


def process_file(task):
    """Worker: one parquet file -> (key, accumulated histograms)."""
    import pyarrow as pa

    from vvv.background.qcd_massfit import assign_regions
    from vvv.validation.blinding import apply_blinding

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    kind, key, path, boxes, scale = task
    columns = [MSD] + CONTROLS + PROBA
    is_data = kind == "data"
    columns += ["run", "luminosityBlock", "event"] if is_data else ["weight"]
    acc, rows_in = None, 0
    for cols in _batches(path, columns):
        rows_in += len(cols[MSD])
        region = assign_regions(np.column_stack([cols[p] for p in PROBA]), boxes, list(CLASSES))
        if is_data:
            # BLINDING: first thing after reading; apply_blinding asserts the result
            cols, region = apply_blinding(cols, region, MSD)
            w = None
        else:
            w = cols.pop("weight").astype(np.float64) * scale
        part = fill_batch(cols, region, w, boxes, is_data)
        if not is_data:
            from vvv.validation.blinding import in_sideband

            sbm = in_sideband(cols[MSD])
            masks = {"fail_sb": (region == 0) & sbm, "sb": sbm, "sru_sb": (region > 0) & sbm}
            for k, (sid, _) in enumerate(boxes):
                masks[f"sr{sid}_sb"] = (region == k + 1) & sbm
            part["samplestats"] = {}
            for s, mk in masks.items():
                ww = w[mk]
                part["samplestats"][s] = {"sw": np.array([ww.sum()]), "sw2": np.array([(ww ** 2).sum()]),
                                          "n": np.array([float(mk.sum())]),
                                          "maxw": np.array([np.abs(ww).max() if len(ww) else 0.0])}
        acc = merge(acc, part)
    return kind, key, acc, rows_in


def flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        name = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            out.update(flatten(v, name))
        else:
            out[name] = np.asarray(v)
    return out


def unflatten(flat):
    out = {}
    for k, v in flat.items():
        parts = k.split("/")
        d = out
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v
    return out


def fill_category(cat, args):
    import pyarrow.parquet as pq

    from vvv.background.qcd_massfit import load_sr_boxes
    from vvv.validation.blinding import BLIND_WINDOW

    srcfg = json.loads((args.sr_config_dir / f"{cat}.json").read_text())
    exclude = set(srcfg.get("exclude_samples", []))
    tcfg = json.loads(args.training_config.read_text())
    groups = {c: [s for s in v if s not in exclude] for c, v in tcfg["class_groups"].items()}
    boxes = load_sr_boxes(args.sr_dir / cat / "signal_region.csv", list(CLASSES))
    scale = args.data_lumi / args.skim_lumi
    tasks = []
    for c, samples in groups.items():
        for s in samples:
            for f in sorted((args.mc_skim_dir / s / cat / "nominal").glob("*.parquet")):
                if pq.ParquetFile(f).metadata.num_rows:
                    tasks.append(("mc", (c, s), f, boxes, scale))
    dfiles = sorted((args.data_skim_dir / "data_2024" / cat / "nominal").glob("*.parquet"))
    tasks += [("data", ("data", "data_2024"), f, boxes, 1.0) for f in dfiles]
    tasks.sort(key=lambda t: -t[2].stat().st_size)  # big files first
    log(f"{cat}: {len(tasks) - len(dfiles)} MC files, {len(dfiles)} data files, excluded {sorted(exclude)}, "
        f"MC x {scale:.5f}, boxes {[b[0] for b in boxes]}")
    mc, samples, data = {}, {}, None
    nd_in = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (kind, key, acc, rin) in enumerate(pool.map(process_file, tasks)):
            if acc is None:
                continue
            if kind == "data":
                data = merge(data, acc)
                nd_in += rin
            else:
                c, s = key
                stats = acc.pop("samplestats")
                samples[(c, s)] = merge(samples.get((c, s)), stats)
                mc[c] = merge(mc.get(c), acc)
            if (i + 1) % 20 == 0:
                log(f"  {i + 1}/{len(tasks)} files ({time.time() - t0:.0f} s)")
    # blinding re-check on the accumulated data: SR selections empty in the window
    edges = BINS[MSD]
    win = (edges[:-1] < BLIND_WINDOW[1]) & (edges[1:] > BLIND_WINDOW[0])
    for s, h in data["h"].items():
        if s != "fail":
            assert not h[MSD]["sw"][win].any(), f"{cat}: data in blind window for selection {s}"
    for s, e in data["era"].items():
        if s != "fail":
            assert not e[:, win].any()
    assert np.all(~((data["events"]["ev_region"] > 0)
                    & (data["events"][f"ev_{MSD}"] >= 50) & (data["events"][f"ev_{MSD}"] < 150)))
    # never log or store the number of rows dropped by the blinding cut: it is the blinded SR count
    log(f"{cat}: data done ({len(dfiles)} files); {time.time() - t0:.0f} s")
    flat = flatten({"mc": mc, "data": data})
    for (c, s), st in samples.items():
        for sel, v in st.items():
            for q, a in v.items():
                flat[f"samples/{c}/{s}/{sel}/{q}"] = a
    out = args.output_dir / "cache" / f"{cat}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **flat)
    meta = {"category": cat, "boxes": boxes, "exclude_samples": sorted(exclude), "mc_scale": scale,
            "data_lumi": args.data_lumi, "skim_lumi": args.skim_lumi, "seconds": time.time() - t0}
    (args.output_dir / "cache" / f"{cat}.json").write_text(json.dumps(meta, indent=1))


def load_cache(cat, outdir):
    with np.load(outdir / "cache" / f"{cat}.npz") as z:
        d = unflatten({k: z[k] for k in z.files})
    meta = json.loads((outdir / "cache" / f"{cat}.json").read_text())
    meta["boxes"] = [(int(b[0]), {a: tuple(v) for a, v in b[1].items()}) for b in meta["boxes"]]
    return d, meta


# --- plotting -----------------------------------------------------------------


def garwood(n):
    from scipy.stats import chi2

    n = np.asarray(n, dtype=float)
    lo = np.where(n > 0, chi2.ppf(0.15865, 2 * n) / 2, 0.0)
    hi = chi2.ppf(0.84135, 2 * (n + 1)) / 2
    return n - lo, hi - n


def nice_scale(target):
    if not np.isfinite(target) or target <= 1:
        return 1
    e = 10 ** np.floor(np.log10(target))
    for f in (1, 2, 5, 10):
        if f * e >= target:
            return int(f * e) if f * e >= 1 else 1
    return int(10 * e)


def ratio_err(dv, mv, mv2):
    """data/MC, its data error (Garwood) and MC relative stat error."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = dv / mv
        el, eh = garwood(dv)
        return r, el / mv, eh / mv, np.sqrt(mv2) / mv


def data_mc_ratio(dv, mv, mv2):
    """Total data/MC with the combined error (data Poisson, MC stat)."""
    if mv <= 0:
        return None, None
    r = dv / mv
    return float(r), float(np.sqrt(dv / mv ** 2 + r ** 2 * mv2 / mv ** 2))


class Plotter:
    def __init__(self, cat, d, meta, lumi):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import mplhep as hep

        plt.style.use(hep.style.CMS)
        self.plt, self.hep = plt, hep
        self.cat, self.d, self.meta, self.lumi = cat, d, meta, lumi

    def mc(self, sel, var, src="h"):
        return {c: self.d["mc"][c][src][sel][var] for c in STACK if c in self.d["mc"]}

    def stack_plot(self, edges, mc, data, title, xlabel, blinded_bins=None, logy=None, extra_text=None,
                   ax_pair=None, small=False):
        """mc: {class: {sw, sw2}}; data: counts or None; blinded_bins: data hidden there (grey band)."""
        plt, hep = self.plt, self.hep
        if ax_pair is None:
            fig, (ax, rx) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06},
                                         sharex=True)
        else:
            fig = None
            ax, rx = ax_pair
        nb = len(edges) - 1
        tot = np.zeros(nb)
        tot2 = np.zeros(nb)
        bottom = np.zeros(nb)
        for c in STACK:
            if c not in mc:
                continue
            y = mc[c]["sw"]
            ax.stairs(bottom + y, edges, baseline=bottom, fill=True, color=COLORS[c], label=c, alpha=0.9)
            bottom = bottom + y
            tot += y
            tot2 += mc[c]["sw2"]
        ax.stairs(tot, edges, color="black", lw=0.8)
        band_lo, band_hi = tot - np.sqrt(tot2), tot + np.sqrt(tot2)
        ax.stairs(band_hi, edges, baseline=band_lo, fill=False, hatch="////", color="grey", lw=0, label="MC stat.")
        # signal lines, scaled
        peak = tot.max() if tot.max() > 0 else 1.0
        sigtxt = []
        for c, ls in (("VVV", "-"), ("VH", "--")):
            if c in mc and mc[c]["sw"].max() > 0:
                k = nice_scale(0.3 * peak / mc[c]["sw"].max())
                ax.stairs(mc[c]["sw"] * k, edges, color=COLORS[c], lw=1.2 if small else 2, ls=ls, label=f"{c} x {k}")
                sigtxt.append(f"{c} x{k} ({'solid' if ls == '-' else 'dashed'})")
        cen = 0.5 * (edges[:-1] + edges[1:])
        hidden = np.zeros(nb, bool) if blinded_bins is None else blinded_bins
        rtxt = ""
        if data is not None:
            dv = np.where(hidden, np.nan, data.astype(float))
            el, eh = garwood(np.nan_to_num(dv))
            show = ~hidden & ~((np.nan_to_num(dv) == 0) & (tot < 0.1))
            ax.errorbar(cen[show], dv[show], yerr=[el[show], eh[show]], fmt="o", ms=4 if small else 6, color="black",
                        label="Data", zorder=5)
            r, rl, rh, rm = ratio_err(dv, tot, tot2)
            ok = show & (tot > 0)
            rx.errorbar(cen[ok], r[ok], yerr=[rl[ok], rh[ok]], fmt="o", ms=4 if small else 6, color="black")
            dsum, msum, m2sum = np.nansum(dv[show]), tot[show].sum(), tot2[show].sum()
            rr, re = data_mc_ratio(dsum, msum, m2sum)
            if rr is not None:
                rtxt = f"data/MC = {rr:.3f} $\\pm$ {re:.3f} ({int(dsum)} / {msum:.1f})"
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(tot > 0, np.sqrt(tot2) / tot, np.nan)
        rx.stairs(1 + rel, edges, baseline=1 - rel, fill=True, color="grey", alpha=0.4)
        rx.axhline(1.0, color="black", lw=0.8)
        if blinded_bins is not None and blinded_bins.any():
            lo, hi = edges[:-1][blinded_bins].min(), edges[1:][blinded_bins].max()
            for a in (ax, rx):
                a.axvspan(lo, hi, color="grey", alpha=0.25, zorder=0)
            ax.text(0.5 * (lo + hi), 0.5, "data blinded", transform=ax.get_xaxis_transform(), ha="center",
                    fontsize=10 if small else 14, color="dimgrey")
        if logy is None:
            pos = tot[tot > 0]
            logy = len(pos) > 0 and pos.max() / max(pos.min(), 1e-9) > 50
        if logy:
            ax.set_yscale("log")
            ymin = max(tot[tot > 0].min() * 0.3 if (tot > 0).any() else 0.1, 1e-2 * (1 if peak > 1 else peak))
            ax.set_ylim(ymin, peak * (1e3 if not small else 5e2))
        else:
            top = peak
            if data is not None and (~hidden).any():
                top = max(top, float(np.max(data[~hidden] + np.sqrt(data[~hidden]))))
            ax.set_ylim(0, top * (1.6 if small else 2.2))
        rmax = 2.5
        if data is not None:
            ok = show & (tot > 0) & np.isfinite(r)
            if ok.any():
                rmax = float(np.clip(1.15 * np.max(r[ok]), 2.5, 8.0))
        rx.set_ylim(0, rmax)
        rx.set_ylabel("Data/MC", fontsize=10 if small else None)
        rx.set_xlabel(xlabel, fontsize=10 if small else None)
        ax.set_ylabel("Events / bin", fontsize=10 if small else None)
        ax.set_xlim(edges[0], edges[-1])
        if small:
            ax.tick_params(labelsize=8)
            rx.tick_params(labelsize=8)
            ax.set_title(title, fontsize=9)
            ax.text(0.02, 0.97, "\n".join([t for t in (rtxt, ", ".join(sigtxt)) if t]), transform=ax.transAxes,
                    fontsize=7, va="top")
            return fig
        hep.cms.label(ax=ax, llabel="Private Work", data=True, lumi=round(self.lumi, 1), com=13.6, fontsize=18)
        ax.legend(ncol=2, fontsize=11, loc="upper right", title=rtxt or None, title_fontsize=10)
        txt = f"{CAT_LABEL[self.cat]}\n{title}"
        if extra_text:
            txt += "\n" + extra_text
        ax.text(0.03, 0.96, txt, transform=ax.transAxes, fontsize=11, va="top")
        fig.text(0.5, 0.005, f"Caveat: {JMS_CAVEAT}. MC x {self.meta['mc_scale']:.4f} to {self.lumi:.2f} fb$^{{-1}}$; "
                 "VVV and VH also included in the stack at x1.", ha="center", fontsize=10, color="darkred")
        return fig

    def text_page(self, lines, title):
        fig = self.plt.figure(figsize=(11, 8.5))
        fig.text(0.04, 0.96, title, fontsize=16, weight="bold", va="top")
        fig.text(0.04, 0.91, "\n".join(lines), fontsize=9, va="top", family="monospace")
        fig.text(0.5, 0.01, f"Caveat: {JMS_CAVEAT}.", ha="center", fontsize=10, color="darkred")
        return fig


SEL_TEXT = {"fail": "fail region (outside all SR boxes), all $m_{SD,1}$",
            "fail_sb": "fail region, $m_{SD,1}$ sidebands",
            "sb": "all events, $m_{SD,1}$ sidebands (<50 or $\\geq$150 GeV)",
            "sru_sb": "union of SR boxes, $m_{SD,1}$ sidebands (<50 or $\\geq$150 GeV)"}


def sel_text(s):
    return SEL_TEXT.get(s) or f"SR{s[2:-3]} box, $m_{{SD,1}}$ sidebands (<50 or $\\geq$150 GeV)"


def yields(d, boxes):
    """Data/MC per region and sideband from the msd histograms (sideband bins only for SRs)."""
    from vvv.validation.blinding import BLIND_WINDOW

    e = BINS[MSD]
    parts = {"low": e[1:] <= BLIND_WINDOW[0], "window": (e[:-1] < BLIND_WINDOW[1]) & (e[1:] > BLIND_WINDOW[0]),
             "high": e[:-1] >= BLIND_WINDOW[1]}
    rows = []
    for s in ["fail"] + [f"sr{b[0]}_sb" for b in boxes] + ["sru_sb"]:
        row = {"region": s}
        dv = d["data"]["h"][s][MSD]["sw"]
        for p, sel in parts.items():
            if p == "window" and s != "fail":
                continue
            mv = sum(d["mc"][c]["h"][s][MSD]["sw"][sel].sum() for c in d["mc"])
            mv2 = sum(d["mc"][c]["h"][s][MSD]["sw2"][sel].sum() for c in d["mc"])
            r, re = data_mc_ratio(dv[sel].sum(), mv, mv2)
            row[p] = {"data": float(dv[sel].sum()), "mc": float(mv), "mc_stat": float(np.sqrt(mv2)), "ratio": r,
                      "ratio_err": re,
                      "mc_frac": {c: float(d["mc"][c]["h"][s][MSD]["sw"][sel].sum() / mv) if mv > 0 else None
                                  for c in d["mc"]}}
        rows.append(row)
    return rows


def _f(x):
    """2-decimal number or '-' (None: the lowest/highest third is a single bin holding more than a third)."""
    return "-" if x is None else f"{x:.2f}"


def shape_stats(d, sel, var):
    """data/MC normalisation and normalised-shape chi2/ndf of one histogram."""
    dv = d["data"]["h"][sel][var]["sw"]
    mv = sum(d["mc"][c]["h"][sel][var]["sw"] for c in d["mc"])
    mv2 = sum(d["mc"][c]["h"][sel][var]["sw2"] for c in d["mc"])
    nd, nm = dv.sum(), mv.sum()
    if nd <= 0 or nm <= 0:
        return {"data": float(nd), "mc": float(nm), "ratio": None, "chi2": None, "ndf": 0}
    ok = (dv > 0) | (mv > 0)
    var_ = dv / nd ** 2 + mv2 / nm ** 2
    chi2 = float(np.sum(((dv / nd - mv / nm) ** 2 / np.where(var_ > 0, var_, 1.0))[ok]))
    # slope of data/MC (normalised) across the histogram: ratio in the first and last third (by MC content)
    cum = np.cumsum(mv) / nm
    t1, t3 = cum <= 1 / 3, cum > 2 / 3
    r1 = (dv[t1].sum() / nd) / (mv[t1].sum() / nm) if mv[t1].sum() > 0 else None
    r3 = (dv[t3].sum() / nd) / (mv[t3].sum() / nm) if mv[t3].sum() > 0 else None
    return {"data": float(nd), "mc": float(nm), "ratio": float(nd / nm), "chi2": chi2, "ndf": int(ok.sum() - 1),
            "shape_low_third": r1, "shape_high_third": r3}


class Booklet:
    """PdfPages that can also write every page as <png_dir>/<stem>_pNN.png."""

    def __init__(self, path, png_dir=None):
        from matplotlib.backends.backend_pdf import PdfPages

        self.pdf = PdfPages(path)
        self.stem, self.png_dir, self.n = Path(path).stem, png_dir, 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.pdf.close()

    def savefig(self, fig):
        self.n += 1
        self.pdf.savefig(fig)
        if self.png_dir is not None:
            self.png_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(self.png_dir / f"{self.stem}_p{self.n:02d}.png", dpi=50)


def plot_category(cat, args, summary):
    PdfPages = lambda path: Booklet(path, args.png_dir)  # noqa: E731

    from vvv.validation.blinding import BLIND_WINDOW

    d, meta = load_cache(cat, args.output_dir)
    boxes = meta["boxes"]
    P = Plotter(cat, d, meta, args.data_lumi)
    e = BINS[MSD]
    win = (e[:-1] < BLIND_WINDOW[1]) & (e[1:] > BLIND_WINDOW[0])
    # blinding check before any plot: no data in the window outside the fail region
    for s, h in d["data"]["h"].items():
        if s != "fail":
            assert not h[MSD]["sw"][win].any(), f"{cat} {s}: data in the blind window"
    ys = yields(d, boxes)
    out = {"category": cat, "mc_scale": meta["mc_scale"], "exclude_samples": meta["exclude_samples"],
           "boxes": {f"sr{b[0]}": b[1] for b in boxes}, "yields": ys, "shapes": {}}
    sels_ctrl = ["fail", "sru_sb"]
    for s in sels_ctrl + ["sb", "fail_sb"]:
        out["shapes"][s] = {v: shape_stats(d, s, v) for v in VARS if not (s == "sb" and v == MSD)}
    pdf_path = args.output_dir / f"data_mc_{cat}.pdf"
    plt = P.plt
    with PdfPages(pdf_path) as pdf:
        # page 1: summary text
        lines = [f"Data: data_2024, {args.data_lumi:.2f} fb^-1 (rescored fit-region skims, read file by file).",
                 f"MC: skims_mc_fit_baseline, weights x {meta['mc_scale']:.5f} ({args.skim_lumi} -> {args.data_lumi:.2f} fb^-1).",
                 f"Excluded samples (SR config): {', '.join(meta['exclude_samples']) or 'none'}",
                 "Blinding: data inside any SR box with 50 <= msoftdrop_1 < 150 GeV is dropped right after reading.",
                 "", "SR boxes (score axis: [low, high); high = 1 means open):"]
        for sid, b in boxes:
            lines.append(f"  SR{sid}: " + ", ".join(f"{a} [{lo:g},{hi:g})" for a, (lo, hi) in b.items()))
        lines += ["", f"{'region':10s} {'part':6s} {'data':>12s} {'MC':>14s} {'data/MC':>16s}   MC composition"]
        for row in ys:
            for p in ("low", "window", "high"):
                if p not in row:
                    continue
                v = row[p]
                comp = " ".join(f"{c}:{(f or 0) * 100:.0f}%" for c, f in v["mc_frac"].items() if (f or 0) >= 0.01)
                rr = f"{v['ratio']:.3f}+-{v['ratio_err']:.3f}" if v["ratio"] is not None else "-"
                lines.append(f"{row['region']:10s} {p:6s} {v['data']:12.0f} {v['mc']:10.1f}+-{v['mc_stat']:<5.1f} "
                             f"{rr:>16s}   {comp}")
        lines += ["", "low = 20-50 GeV, high = 150-250 GeV, window = 50-150 GeV (fail region only).",
                  "Pages: msd1 fail; msd1 per SR (sidebands); 12 controls in the fail region; 12 controls in the SR-union",
                  "sidebands; 6 BDT scores in the sidebands (all events); summary of the control-variable shapes."]
        fig = P.text_page(lines, f"{cat}: blinded data vs MC")
        pdf.savefig(fig)
        plt.close(fig)

        def page(fig):
            pdf.savefig(fig)
            plt.close(fig)

        page(P.stack_plot(e, P.mc("fail", MSD), d["data"]["h"]["fail"][MSD]["sw"], sel_text("fail"), LABELS[MSD]))
        for sid, b in boxes:
            s = f"sr{sid}_sb"
            box = "\n".join(f"{a} in [{lo:g}, {hi:g})" for a, (lo, hi) in b.items() if not (lo == 0 and hi >= 1))
            page(P.stack_plot(e, P.mc(s, MSD), d["data"]["h"][s][MSD]["sw"], f"SR{sid} box: data only in sidebands",
                              LABELS[MSD], blinded_bins=win, logy=False, extra_text=box))
        for s in sels_ctrl:
            for v in CONTROLS:
                page(P.stack_plot(BINS[v], P.mc(s, v), d["data"]["h"][s][v]["sw"], sel_text(s), LABELS[v]))
        for v in PROBA:
            page(P.stack_plot(BINS[v], P.mc("sb", v), d["data"]["h"]["sb"][v]["sw"], sel_text("sb"), LABELS[v]))
        # shape summary page
        lines = [f"{'variable':45s} " + "  ".join(f"{s:>30s}" for s in ("fail", "sru_sb", "sb"))]
        lines.append(f"{'':45s} " + "  ".join(f"{'data/MC  chi2/ndf  lo3  hi3':>30s}" for _ in range(3)))
        for v in VARS:
            cells = []
            for s in ("fail", "sru_sb", "sb"):
                st = out["shapes"][s].get(v)
                if st is None or st["ratio"] is None:
                    cells.append(f"{'-':>30s}")
                    continue
                cells.append(f"{st['ratio']:7.3f} {st['chi2']:8.0f}/{st['ndf']:<3d} "
                             f"{_f(st['shape_low_third']):>5s} {_f(st['shape_high_third']):>5s}")
            lines.append(f"{v.replace('ScoutingFatPFJetRecluster', 'AK8').replace('ScoutingPFJetRecluster2', 'AK4'):45s} "
                         + "  ".join(f"{c:>30s}" for c in cells))
        lines += ["", "data/MC: total normalisation; chi2/ndf: normalised shapes, data Poisson + MC stat;",
                  "lo3/hi3: (data/MC shape) in the lowest/highest third of the MC distribution, normalised to 1 overall.",
                  "sb = all events in the msd1 sidebands (msd1 itself omitted)."]
        page(P.text_page(lines, f"{cat}: control-variable shape summary"))
    log(f"wrote {pdf_path}")
    summary[cat] = out
    return P


# --- focus study --------------------------------------------------------------


def focus_study(cat, sid, args, summary):
    PdfPages = lambda path: Booklet(path, args.png_dir)  # noqa: E731
    from scipy.stats import chi2 as chi2d, poisson

    from vvv.validation.blinding import BLIND_WINDOW

    d, meta = load_cache(cat, args.output_dir)
    boxes = dict(meta["boxes"])
    bounds = boxes[sid]
    P = Plotter(cat, d, meta, args.data_lumi)
    plt = P.plt
    s = f"sr{sid}_sb"
    e = BINS[MSD]
    win = (e[:-1] < BLIND_WINDOW[1]) & (e[1:] > BLIND_WINDOW[0])
    lowb, highb = e[1:] <= BLIND_WINDOW[0], e[:-1] >= BLIND_WINDOW[1]
    res = {"category": cat, "sr": sid, "box": bounds}
    # class composition and samples
    comp = {}
    for c in d["mc"]:
        h = d["mc"][c]["h"][s][MSD]
        comp[c] = {"sw": float(h["sw"].sum()), "stat": float(np.sqrt(h["sw2"].sum())), "n": float(d["mc"][c]["n"][s][0])}
    res["mc_by_class"] = comp
    samp = []
    for c, ss in d.get("samples", {}).items():
        for name, st in ss.items():
            if s in st and st[s]["n"][0] > 0:
                samp.append({"sample": name, "class": c, "sw": float(st[s]["sw"][0]), "stat": float(np.sqrt(st[s]["sw2"][0])),
                             "n": int(st[s]["n"][0]), "max_w": float(st[s]["maxw"][0])})
    samp.sort(key=lambda r: -r["sw"])
    res["mc_samples"] = samp
    dsb = d["data"]["h"][s][MSD]["sw"]
    res["data"] = {"low": float(dsb[lowb].sum()), "high": float(dsb[highb].sum())}
    # controls in this SR's sidebands
    ctrl = {v: shape_stats(d, s, v) for v in CONTROLS + PROBA}
    res["controls"] = ctrl
    # data/MC per bin of each control (where is the excess): ratio in low/high third
    # era dependence: SR sideband counts vs fail-sideband counts per era
    era_sr = d["data"]["era"][s][:, ~win].sum(axis=1)
    era_fail = d["data"]["era"]["fail_sb"][:, ~win].sum(axis=1)
    era_sru = d["data"]["era"]["sru_sb"][:, ~win].sum(axis=1)
    exp = era_sr.sum() * era_fail / era_fail.sum()
    res["era"] = {}
    for k, (name, _, _) in enumerate(ERAS):
        res["era"][name] = {"sr_sb": float(era_sr[k]), "fail_sb": float(era_fail[k]), "sru_sb": float(era_sru[k]),
                            "expected_from_fail_share": float(exp[k]),
                            "ratio_obs_exp": float(era_sr[k] / exp[k]) if exp[k] > 0 else None}
    chi2v = float(np.sum((era_sr - exp) ** 2 / np.where(exp > 0, exp, 1)))
    res["era_chi2"] = {"chi2": chi2v, "ndf": len(ERAS) - 1, "p": float(chi2d.sf(chi2v, len(ERAS) - 1))}
    # per run: KS-like comparison of the SR-sideband run distribution with the fail-sideband one
    rs, rf = d["data"]["run"][s], d["data"]["run"]["fail_sb"]
    cs, cf = np.cumsum(rs) / max(rs.sum(), 1), np.cumsum(rf) / rf.sum()
    dks = float(np.max(np.abs(cs - cf)))
    n_eff = rs.sum()
    from scipy.stats import kstwo

    res["run_ks"] = {"D": dks, "n": float(n_eff), "p": float(kstwo.sf(dks, int(n_eff))) if n_eff > 0 else None}
    # expected count per run vs observed: most populated runs
    expr = rs.sum() * rf / rf.sum()
    top = np.argsort(-rs)[:8]
    res["top_runs"] = [{"run": int(RUN0 + i), "observed": int(rs[i]), "expected": float(expr[i]),
                        "p_ge": float(poisson.sf(rs[i] - 1, expr[i]))} for i in top if rs[i] > 0]
    # events: duplicates and lumisection clustering
    ev = d["data"]["events"]
    k = [b for b in sorted(boxes)].index(sid) + 1
    sel = ev["ev_region"] == k
    assert np.all((ev[f"ev_{MSD}"][sel] < 50) | (ev[f"ev_{MSD}"][sel] >= 150))
    trip = np.stack([ev["ev_run"][sel].astype(np.int64), ev["ev_luminosityBlock"][sel].astype(np.int64),
                     ev["ev_event"][sel].astype(np.int64)], axis=1)
    _, cnt = np.unique(trip, axis=0, return_counts=True)
    _, lcnt = np.unique(trip[:, :2], axis=0, return_counts=True)
    res["events"] = {"n": int(sel.sum()), "duplicate_run_lumi_event": int((cnt > 1).sum()),
                     "lumisections_with_ge2_events": int((lcnt > 1).sum()), "max_events_per_lumisection": int(lcnt.max()) if len(lcnt) else 0,
                     "distinct_runs": int(len(np.unique(trip[:, 0])))}
    # box edges: N-1 and loosening
    ln = d["data"]["loosen"][f"sr{sid}"]
    lm = {c: d["mc"][c]["loosen"][f"sr{sid}"] for c in d["mc"]}
    res["loosen"] = []
    for j, f in enumerate(LOOSEN):
        row = {"factor": f}
        for p in ("low", "high"):
            dv = float(ln[f"{p}_sw"][j])
            mv = float(sum(lm[c][f"{p}_sw"][j] for c in lm))
            mv2 = float(sum(lm[c][f"{p}_sw2"][j] for c in lm))
            r, re = data_mc_ratio(dv, mv, mv2)
            row[p] = {"data": dv, "mc": mv, "ratio": r, "ratio_err": re,
                      "qcd_frac": float(lm["QCD"][f"{p}_sw"][j] / mv) if mv > 0 and "QCD" in lm else None}
        res["loosen"].append(row)
    # N-1: data/MC inside vs just outside each cut
    res["nm1"] = {}
    ce = NM1_EDGES
    for a, (lo, hi) in bounds.items():
        hd = d["data"]["nm1"][f"sr{sid}"][a]["sw"]
        hm = sum(d["mc"][c]["nm1"][f"sr{sid}"][a]["sw"] for c in d["mc"])
        hm2 = sum(d["mc"][c]["nm1"][f"sr{sid}"][a]["sw2"] for c in d["mc"])
        inside = (ce[:-1] >= lo - 1e-9) & ((ce[1:] <= hi + 1e-9) if hi < 1 - 1e-12 else True)
        info = {"cut": [lo, hi]}
        width = 0.1
        for name, msk in (("inside", inside),
                          ("inside_near_low_edge", inside & (ce[:-1] < lo + width)),
                          ("inside_near_high_edge", inside & (ce[1:] > hi - width) if hi < 1 - 1e-12 else np.zeros_like(inside)),
                          ("below_low_edge", (ce[1:] <= lo + 1e-9) & (ce[:-1] >= lo - width - 1e-9)),
                          ("above_high_edge", (ce[:-1] >= hi - 1e-9) & (ce[1:] <= hi + width + 1e-9) if hi < 1 - 1e-12 else np.zeros_like(inside))):
            msk = np.asarray(msk, bool)
            if not msk.any():
                continue
            r, re = data_mc_ratio(hd[msk].sum(), hm[msk].sum(), hm2[msk].sum())
            info[name] = {"data": float(hd[msk].sum()), "mc": float(hm[msk].sum()), "ratio": r, "ratio_err": re}
        res["nm1"][a] = info
    # comparison: other SRs of the category
    res["other_srs"] = {}
    for osid in sorted(boxes):
        h = d["data"]["h"][f"sr{osid}_sb"][MSD]["sw"]
        mv = sum(d["mc"][c]["h"][f"sr{osid}_sb"][MSD]["sw"] for c in d["mc"])
        mv2 = sum(d["mc"][c]["h"][f"sr{osid}_sb"][MSD]["sw2"] for c in d["mc"])
        r, re = data_mc_ratio(h.sum(), mv.sum(), mv2.sum())
        res["other_srs"][f"sr{osid}"] = {"data": float(h.sum()), "mc": float(mv.sum()), "ratio": r, "ratio_err": re,
                                         "qcd_frac": float(d["mc"]["QCD"]["h"][f"sr{osid}_sb"][MSD]["sw"].sum() / mv.sum())}

    pdf_path = args.output_dir / f"focus_{cat}_sr{sid}.pdf"
    with PdfPages(pdf_path) as pdf:
        def page(fig):
            pdf.savefig(fig)
            plt.close(fig)

        lines = [f"Box SR{sid}: " + ", ".join(f"{a} [{lo:g},{hi:g})" for a, (lo, hi) in bounds.items()),
                 f"Data in the sidebands: low (20-50) {res['data']['low']:.0f}, high (150-250) {res['data']['high']:.0f}",
                 "", "MC by class in the SR sidebands (scaled to data lumi):"]
        for c, v in comp.items():
            lines.append(f"  {c:6s} {v['sw']:9.2f} +- {v['stat']:6.2f}   raw MC entries {v['n']:8.0f}")
        lines += ["", "Top MC samples (sumw, stat, raw entries, max |w|):"]
        for r in samp[:10]:
            lines.append(f"  {r['sample']:32s} {r['class']:6s} {r['sw']:8.2f} +- {r['stat']:5.2f}  n={r['n']:6d}  max|w|={r['max_w']:.3f}")
        lines += ["", "Era dependence (data only; expected = SR-sideband total x era share of fail-region sideband data):"]
        for name, v in res["era"].items():
            lines.append(f"  {name}: SR{sid} sb {v['sr_sb']:4.0f}, expected {v['expected_from_fail_share']:6.1f}, "
                         f"obs/exp {v['ratio_obs_exp'] or 0:.2f};  fail sb {v['fail_sb']:12.0f}; SR-union sb {v['sru_sb']:5.0f}")
        lines.append(f"  chi2 = {res['era_chi2']['chi2']:.2f} / {res['era_chi2']['ndf']}, p = {res['era_chi2']['p']:.3f};  "
                     f"run-by-run KS vs fail sb: D = {res['run_ks']['D']:.3f}, p = {res['run_ks']['p']:.3f}")
        lines.append(f"  events: {res['events']}")
        lines += ["", "Box loosened (factor > 1) / tightened (< 1), sidebands only:",
                  f"  {'factor':>6s} {'low data':>9s} {'low MC':>9s} {'ratio':>12s} {'QCD%':>5s} {'high data':>10s} {'high MC':>9s} {'ratio':>12s} {'QCD%':>5s}"]
        for row in res["loosen"]:
            lo_, hi_ = row["low"], row["high"]
            lines.append(f"  {row['factor']:6.2f} {lo_['data']:9.0f} {lo_['mc']:9.1f} {lo_['ratio'] or 0:6.2f}+-{lo_['ratio_err'] or 0:4.2f} "
                         f"{(lo_['qcd_frac'] or 0) * 100:5.0f} {hi_['data']:10.0f} {hi_['mc']:9.1f} {hi_['ratio'] or 0:6.2f}+-{hi_['ratio_err'] or 0:4.2f} "
                         f"{(hi_['qcd_frac'] or 0) * 100:5.0f}")
        page(P.text_page(lines, f"Focus: {cat} SR{sid} (sidebands only)"))
        page(P.stack_plot(e, P.mc(s, MSD), dsb, f"SR{sid} box: data only in sidebands", LABELS[MSD], blinded_bins=win,
                          logy=False))
        # controls grid
        for group, title in ((CONTROLS, "control variables"), (PROBA, "BDT scores")):
            ncol = 4
            nrow = (len(group) + ncol - 1) // ncol
            fig = plt.figure(figsize=(22, 6.2 * nrow))
            gs = fig.add_gridspec(nrow, ncol, hspace=0.35, wspace=0.3)
            for i, v in enumerate(group):
                sub = gs[i // ncol, i % ncol].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
                ax = fig.add_subplot(sub[0])
                rx = fig.add_subplot(sub[1], sharex=ax)
                ax.tick_params(labelbottom=False)
                edges = BINS[v]
                hd = d["data"]["h"][s][v]["sw"]
                mc = P.mc(s, v)
                # coarser bins for the few-event SR sidebands
                if len(edges) > 21:
                    f = int(np.ceil((len(edges) - 1) / 10))
                    nb = (len(edges) - 1) // f * f
                    edges = edges[: nb + 1: f]
                    hd = hd[:nb].reshape(-1, f).sum(1)
                    mc = {c: {q: x[:nb].reshape(-1, f).sum(1) for q, x in h.items()} for c, h in mc.items()}
                P.stack_plot(edges, mc, hd, LABELS[v], LABELS[v], ax_pair=(ax, rx), small=True, logy=False)
            fig.suptitle(f"{cat} SR{sid} box, $m_{{SD,1}}$ sidebands only: {title}  ({JMS_CAVEAT})", fontsize=16)
            page(fig)
        # N-1 plots
        axes_ = list(bounds)
        ncol = 3
        nrow = (len(axes_) + ncol - 1) // ncol
        fig = plt.figure(figsize=(18, 6.2 * nrow))
        gs = fig.add_gridspec(nrow, ncol, hspace=0.35, wspace=0.3)
        for i, a in enumerate(axes_):
            sub = gs[i // ncol, i % ncol].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax = fig.add_subplot(sub[0])
            rx = fig.add_subplot(sub[1], sharex=ax)
            ax.tick_params(labelbottom=False)
            edges = NM1_EDGES[::2]
            mc = {c: {q: x.reshape(-1, 2).sum(1) for q, x in d["mc"][c]["nm1"][f"sr{sid}"][a].items()} for c in STACK
                  if c in d["mc"]}
            hd = d["data"]["nm1"][f"sr{sid}"][a]["sw"].reshape(-1, 2).sum(1)
            P.stack_plot(edges, mc, hd, f"N-1: proba_{a} (other cuts applied)", f"proba_{a}", ax_pair=(ax, rx),
                         small=True, logy=True)
            lo, hi = bounds[a]
            for x in (lo, hi):
                if 0 < x < 1:
                    for q in (ax, rx):
                        q.axvline(x, color="red", ls="--", lw=1.5)
        fig.suptitle(f"{cat} SR{sid}: N-1 score distributions, $m_{{SD,1}}$ sidebands only; red = box edges ({JMS_CAVEAT})",
                     fontsize=15)
        page(fig)
        # loosening and era plot
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 7))
        for p, mk in (("low", "o"), ("high", "s")):
            r = [row[p]["ratio"] or np.nan for row in res["loosen"]]
            er = [row[p]["ratio_err"] or 0 for row in res["loosen"]]
            a1.errorbar(np.array(LOOSEN) * (0.98 if p == "low" else 1.02), r, yerr=er, fmt=mk, label=f"{p} sideband")
        a1.axhline(1, color="black", lw=0.8)
        a1.set_xscale("log")
        a1.set_xticks(LOOSEN)
        a1.set_xticklabels([f"{f:g}" for f in LOOSEN])
        a1.set_xlabel("box factor (<1 tighter, >1 looser; box_mask semantics)")
        a1.set_ylabel("data/MC in sideband")
        a1.legend()
        a1.set_title(f"SR{sid}: data/MC vs box size", fontsize=14)
        names = [n for n, _, _ in ERAS]
        r = [res["era"][n]["ratio_obs_exp"] or np.nan for n in names]
        er = [np.sqrt(res["era"][n]["sr_sb"]) / res["era"][n]["expected_from_fail_share"]
              if res["era"][n]["expected_from_fail_share"] > 0 else 0 for n in names]
        a2.errorbar(range(len(names)), r, yerr=er, fmt="o", color="black")
        a2.axhline(1, color="black", lw=0.8)
        a2.set_xticks(range(len(names)))
        a2.set_xticklabels([f"{n}\n({res['era'][n]['sr_sb']:.0f} ev)" for n in names])
        a2.set_ylabel("observed / expected from fail-sb share")
        a2.set_title(f"SR{sid} sideband data per era (p = {res['era_chi2']['p']:.2f})", fontsize=14)
        fig.text(0.5, 0.01, f"Caveat: {JMS_CAVEAT}.", ha="center", fontsize=10, color="darkred")
        page(fig)
        # run distribution: cumulative
        fig, ax = plt.subplots(figsize=(14, 7))
        runs = np.arange(RUN0, RUN1)
        ax.plot(runs, cf, label="fail-region sideband data", color="grey")
        ax.plot(runs, np.cumsum(d["data"]["run"]["sru_sb"]) / d["data"]["run"]["sru_sb"].sum(),
                label="SR-union sideband data", color="tab:blue")
        ax.plot(runs, cs, label=f"SR{sid} sideband data", color="red")
        for name, lo, _ in ERAS[1:]:
            ax.axvline(lo, color="black", ls=":")
            ax.text(lo, 0.02, name, fontsize=10)
        ax.set_xlabel("run")
        ax.set_ylabel("cumulative fraction")
        ax.legend(loc="upper left")
        ax.set_title(f"{cat} SR{sid}: run distribution, KS D = {dks:.3f} (p = {res['run_ks']['p']:.2f})", fontsize=14)
        page(fig)
    (args.output_dir / f"focus_{cat}_sr{sid}.json").write_text(json.dumps(res, indent=1))
    log(f"wrote {pdf_path}")
    summary.setdefault("focus", {})[f"{cat}:sr{sid}"] = res


def summary_pdf(summary, args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mplhep as hep
    from matplotlib.backends.backend_pdf import PdfPages

    plt.style.use(hep.style.CMS)
    cats = [c for c in CATEGORIES if c in summary]
    path = args.output_dir / "summary.pdf"
    with PdfPages(path) as pdf:
        # overview: data/MC per region
        fig, axes = plt.subplots(len(cats), 1, figsize=(14, 3.2 * len(cats) + 1), squeeze=False)
        for ax, cat in zip(axes[:, 0], cats):
            labels, r, er, col = [], [], [], []
            for row in summary[cat]["yields"]:
                for p, cl in (("low", "tab:blue"), ("window", "tab:green"), ("high", "tab:red")):
                    if p in row and row[p]["ratio"] is not None:
                        labels.append(f"{row['region'].replace('_sb', '')}\n{p}")
                        r.append(row[p]["ratio"])
                        er.append(row[p]["ratio_err"])
                        col.append(cl)
            x = np.arange(len(r))
            ax.scatter(x, r, c=col, zorder=3)
            ax.errorbar(x, r, yerr=er, fmt="none", ecolor="black", zorder=2)
            ax.axhline(1, color="black", lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylabel("data/MC", fontsize=12)
            ax.set_yscale("log")
            ax.set_ylim(0.3, 8)
            ax.set_yticks([0.5, 1, 2, 4])
            ax.set_yticklabels(["0.5", "1", "2", "4"])
            ax.minorticks_off()
            ax.set_title(f"{cat}  (fail: all masses; SRs: sidebands only; blue low 20-50, green window, red high 150-250)",
                         fontsize=11)
        fig.suptitle(f"Data/MC overview, {args.data_lumi:.1f} fb$^{{-1}}$. Caveat: {JMS_CAVEAT}", fontsize=13)
        fig.tight_layout()
        pdf.savefig(fig)
        fig.savefig(args.output_dir / "summary_overview.png", dpi=70)
        plt.close(fig)
        # one text page per category: yields
        for cat in cats:
            lines = [f"{'region':10s} {'part':6s} {'data':>12s} {'MC':>18s} {'data/MC':>16s}   MC composition"]
            for row in summary[cat]["yields"]:
                for p in ("low", "window", "high"):
                    if p in row:
                        v = row[p]
                        comp = " ".join(f"{c}:{(f or 0) * 100:.0f}%" for c, f in v["mc_frac"].items() if (f or 0) >= 0.01)
                        rr = f"{v['ratio']:.3f}+-{v['ratio_err']:.3f}" if v["ratio"] is not None else "-"
                        lines.append(f"{row['region']:10s} {p:6s} {v['data']:12.0f} {v['mc']:12.1f}+-{v['mc_stat']:<5.1f} "
                                     f"{rr:>16s}   {comp}")
            lines += ["", "Controls (fail region; SR-union sidebands): data/MC norm, shape chi2/ndf, shape ratio lowest/highest third"]
            for v in CONTROLS:
                a, b = summary[cat]["shapes"]["fail"][v], summary[cat]["shapes"]["sru_sb"][v]
                fmt = lambda st: (f"{st['ratio']:.3f} {st['chi2']:9.0f}/{st['ndf']:<3d} {_f(st['shape_low_third'])} "
                                  f"{_f(st['shape_high_third'])}") if st["ratio"] else "-"
                lines.append(f"  {v.replace('ScoutingFatPFJetRecluster', 'AK8').replace('ScoutingPFJetRecluster2', 'AK4'):40s} "
                             f"{fmt(a):>36s}   {fmt(b):>36s}")
            fig = plt.figure(figsize=(11, 8.5))
            fig.text(0.04, 0.96, f"{cat}: data/MC yields", fontsize=16, weight="bold", va="top")
            fig.text(0.04, 0.91, "\n".join(lines), fontsize=8.5, va="top", family="monospace")
            fig.text(0.5, 0.01, f"Caveat: {JMS_CAVEAT}.", ha="center", fontsize=10, color="darkred")
            pdf.savefig(fig)
            plt.close(fig)
    log(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--categories", nargs="*", default=list(CATEGORIES))
    p.add_argument("--plot-only", action="store_true", help="skip filling; use <output-dir>/cache")
    p.add_argument("--fill-only", action="store_true")
    p.add_argument("--workers", type=int, default=8, help="processes (2 Arrow threads each)")
    p.add_argument("--focus", nargs="*", default=["boosted2_unresolved:1", "boosted3:4"])
    p.add_argument("--mc-skim-dir", type=Path, default=Path("/depot/cms/users/schul105/VVV/skims_mc_fit_baseline"))
    p.add_argument("--data-skim-dir", type=Path, default=Path("/depot/cms/users/schul105/VVV/skims_data_baseline"))
    p.add_argument("--sr-dir", type=Path, default=Path("/work/users/schul105/vvv/sr/baseline"))
    p.add_argument("--sr-config-dir", type=Path, default=Path("configs/signal_regions"))
    p.add_argument("--training-config", type=Path, default=Path("configs/training/baseline.json"))
    p.add_argument("--data-lumi", type=float, default=None, help="default: data_2024 lumi_fb of data/samples.yaml")
    p.add_argument("--skim-lumi", type=float, default=255.8)
    p.add_argument("--output-dir", type=Path, default=Path("/work/users/schul105/vvv/data_mc_plots"))
    p.add_argument("--png", action="store_true", help="also write every page as PNG to <output-dir>/png")
    args = p.parse_args()
    args.png_dir = args.output_dir / "png" if args.png else None
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(k, "1")
    from vvv.ml.inference import CLASSES as MODEL_CLASSES

    assert tuple(MODEL_CLASSES) == CLASSES
    if args.data_lumi is None:
        from vvv.samples import load_samples

        args.data_lumi = load_samples()["data_2024"].lumi_fb
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        for cat in args.categories:
            fill_category(cat, args)
    if args.fill_only:
        return
    summary = {"data_lumi": args.data_lumi, "skim_lumi": args.skim_lumi, "caveat": JMS_CAVEAT}
    for cat in args.categories:
        plot_category(cat, args, summary)
    for f in args.focus:
        cat, sid = f.split(":")
        if cat in args.categories:
            focus_study(cat, int(sid), args, summary)
    summary_pdf(summary, args)
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
