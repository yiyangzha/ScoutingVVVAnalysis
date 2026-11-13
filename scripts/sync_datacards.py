"""Card-for-card sync of the Python datacard writer (vvv.stats.datacards, scripts/write_datacards.py and
scripts/run_combine.py) against the legacy combine.C on the same configs and inputs.

SPEC (example: configs/datacards_sync.json) has legacy_repo, cmssw_src, and cases {name: {config,
cards_only_scopes, synthetic}}:

- config: a combine.C config (a dict, or the path of a legacy combine.json); "registry": "<training
  config>" in a channel instead of bdt_root makes a stub registry (class_groups of that training config +
  the legacy sample.json), as scripts/fitcheck_massfit.py does; "syst_dir" sets the six *_syst_json;
- synthetic (optional): {channel: {from: qcd_abcd_yields.root, data: {seed, blind: [lo, hi]},
  sr_slope: [ids], residual_order, tfmc_free}}: a modified copy of an input (Poisson data from the MC
  of each bin, blinded pass bins; per-SR slopes; the Bernstein residual), written to OUT/synthetic/, to
  exercise the paths no production input has (data_mode observed, slopes, residual, constrained TF_MC).

Per case, in OUT/<case>/:
- legacy/: combine.C (compiled into OUT/bin) with Combine, on the config's scenario_scopes;
- python/: write_datacards + run_combine on the same config;
- legacy_cards/, python_cards/: cards only, every scope (combined, class, sample), combine.C run with
  combine_cmd = combine_cards_cmd = "true" (its scenarios then fail at reading the Combine output, after
  the cards are written; keep_work keeps them);
- comparison: every work/<tag>/ card (comment lines and the shape-file path aside, token by token, with
  numbers compared exactly), every shape ROOT file histogram by histogram (names, contents, errors), the
  combined datacard.txt, and the CSVs (Z, r95 quantiles).

Writes OUT/sync.json and OUT/sync.md.

Usage: pixi run python scripts/sync_datacards.py SPEC --out DIR [--steps build,synthetic,legacy,python,compare]
           [--cases A,B] [--jobs N] [--force]
"""

import argparse
import copy
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from vvv.stats.combine_run import cms_shell, run_all  # noqa: E402
from vvv.stats.datacards import load_config, write_all  # noqa: E402

STEPS = ("build", "synthetic", "legacy", "python", "compare")
SYST_JSON_KEYS = ("theory_syst_json", "pileup_syst_json", "jes_syst_json", "jer_syst_json", "jms_syst_json",
                  "jmr_syst_json")
ALL_SCOPES = ["combined", "class", "sample"]


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def quiet(_message):
    pass


# -------------------- setup --------------------


def build(spec, out, force):
    binary = out / "bin" / "combine_sync"
    if binary.exists() and not force:
        return binary
    binary.parent.mkdir(parents=True, exist_ok=True)
    src = Path(spec["legacy_repo"]) / "combine" / "combine.C"
    log(f"compiling {src} -> {binary}")
    r = cms_shell(spec["cmssw_src"], f"c++ -O2 -DNDEBUG -std=c++17 $(root-config --cflags) {src} -o {binary} "
                                     f"$(root-config --libs)", out)
    if r.returncode != 0:
        sys.exit(f"compilation failed:\n{r.stdout}")
    head = subprocess.run(["git", "-C", spec["legacy_repo"], "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()
    md5 = subprocess.run(["md5sum", str(src)], capture_output=True, text=True).stdout.split()[0]
    (binary.parent / "combine_sync.source").write_text(f"{src}\nlegacy HEAD {head}\nmd5 {md5}\n")
    return binary


def registry_stub(spec, out, training_config):
    tcfg = json.loads((REPO / training_config).read_text())
    d = out / "registry" / Path(training_config).stem
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(json.dumps({
        "class_groups": tcfg["class_groups"],
        "sample_config": str(Path(spec["legacy_repo"]) / "src" / "sample.json"),
        "note": f"combine.C class registry stub from {training_config}"}, indent=1))
    return str(d)


def synthetic_root(src, dst, mod):
    """A copy of a qcd_abcd_yields.root with data, per-SR slopes or a Bernstein residual added."""
    import uproot

    one = np.array([0.0, 1.0])
    with uproot.open(src) as fin:
        objs = {}
        for k, c in fin.classnames(cycle=False).items():
            if c.startswith("TH1") or c.startswith("TH2"):
                objs[k] = fin[k]
        sr_ids = fin["metadata/signal_regions"]["bin_index"].array(library="np").astype(np.int32)
        settings = {k.split("/")[-1]: fin[k].values()[0] for k in objs if k.startswith("massfit/settings/")}
        fail = fin["massfit/fail/qcd_true/yield"]
        edges = fail.axis().edges()
        n_bins = len(edges) - 1
        new = {}
        if "data" in mod:
            rng = np.random.default_rng(mod["data"]["seed"])
            lo, hi = mod["data"]["blind"]
            blinded = (edges[:-1] < hi) & (edges[1:] > lo)
            groups = sorted({k.split("/")[3] for k in objs if k.startswith("massfit/fail/groups/")})
            for region, path in [(0, "fail")] + [(i + 1, f"pass/{{g}}/sr{s}") for i, s in enumerate(sr_ids)]:
                mc = np.zeros(n_bins)
                for g in groups:
                    p = "massfit/fail/groups/" + g if region == 0 else "massfit/" + path.format(g="groups/" + g)
                    mc += np.maximum(fin[p + "/yield"].values(), 0.0)
                counts = rng.poisson(mc).astype(float)
                if region == 0:
                    new["massfit/fail/data/yield"] = (counts, edges)
                else:
                    counts[blinded] = 0.0
                    s = sr_ids[region - 1]
                    new[f"massfit/pass/data/sr{s}/yield"] = (counts, edges)
                    new[f"massfit/blinded/sr{s}"] = (blinded.astype(float), edges)
            for key, value in (("has_data", 1.0), ("blind_low", lo), ("blind_high", hi)):
                new[f"massfit/settings/{key}"] = (np.array([float(value)]), one)
        if "sr_slope" in mod:
            new["massfit/sr_slope"] = (np.array([1.0 if int(s) in mod["sr_slope"] else 0.0 for s in sr_ids]),
                                       np.arange(len(sr_ids) + 1, dtype=float))
        for key in ("residual_order", "tfmc_free", "bern_low", "bern_high"):
            if key in mod:
                if key not in settings:
                    raise ValueError(f"{src}: no massfit/settings/{key} to modify (qcd_from_mc input?)")
                new[f"massfit/settings/{key}"] = (np.array([float(mod[key])]), one)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with uproot.recreate(dst) as fout:
            fout.mktree("metadata/signal_regions", {"bin_index": np.int32}).extend({"bin_index": sr_ids})
            for k, obj in objs.items():
                if k not in new:
                    fout[k] = obj
            for k, v in new.items():
                fout[k] = v
    return dst


def case_config(spec, out, name, case):
    """The combine.C config of a case (absolute paths; output_dir set per run)."""
    cfg = case["config"]
    if isinstance(cfg, str):
        cfg = json.loads(Path(cfg).read_text())
    cfg = copy.deepcopy(cfg)
    for ch in cfg["channels"]:
        if "registry" in ch:
            ch["bdt_root"] = registry_stub(spec, out, ch.pop("registry"))
        if ch["name"] in case.get("synthetic", {}):
            ch["root_file"] = str(out / "synthetic" / name / ch["name"] / "qcd_abcd_yields.root")
    if "syst_dir" in cfg:
        sd = Path(cfg.pop("syst_dir"))
        cfg.update({k: str(sd / k.replace("_json", "_yields.json")) for k in SYST_JSON_KEYS})
    cfg.setdefault("xsec_unc_json", str(Path(spec["legacy_repo"]) / "combine" / "xsec_unc.json"))
    cfg["keep_work"] = True
    return cfg


def write_config(cfg, run_dir, **overrides):
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(cfg, output_dir=str(run_dir), **overrides)
    path = run_dir / "combine.json"
    path.write_text(json.dumps(cfg, indent=2))
    return path


# -------------------- runs --------------------


def run_legacy(spec, binary, run_dir, cfg, force, **overrides):
    done = run_dir / "significance.csv"
    if done.exists() and not force:
        return 0
    path = write_config(cfg, run_dir, **overrides)
    t0 = time.time()
    r = cms_shell(spec["cmssw_src"], f"COMBINE_CONFIG_PATH={path} OMP_NUM_THREADS=1 {binary}", run_dir,
                  run_dir / "job.log")
    log(f"  combine.C {run_dir}: status {r.returncode} ({time.time() - t0:.0f} s)")
    return r.returncode


def run_python(spec, run_dir, cfg, force, combine, jobs, **overrides):
    if (run_dir / ("summary.json" if combine else "datacards.json")).exists() and not force:
        return 0
    path = write_config(cfg, run_dir, **overrides)
    t0 = time.time()
    with open(run_dir / "job.log", "w") as f:
        def flog(m):
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {m}\n")
            f.flush()

        manifest = write_all(load_config(path), log=flog)
        status = int(any(r["status"] == "failed" for r in manifest["runs"]))
        if combine:
            summary = run_all(manifest, cmssw_src=spec["cmssw_src"], jobs=jobs, log=flog)
            status = int(any(r["result"]["status"] == "failed" for r in summary["runs"]))
    log(f"  python {run_dir}: status {status} ({time.time() - t0:.0f} s)")
    return status


# -------------------- comparison --------------------


def _tokens(path, work):
    """Card tokens without comment lines, with the shape path relative to the work directory."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            continue
        parts = line.split()
        # absolute paths (the shape file; combineCards.py's "Combination of" line) relative to the work directory
        out.append([os.path.relpath(p, work) if p.startswith("/") else p for p in parts])
    return out


def _same_token(a, b):
    if a == b:
        return True
    try:
        return float(a) == float(b)
    except ValueError:
        return False


def compare_cards(a, b, wa, wb):
    ta, tb = _tokens(a, wa), _tokens(b, wb)
    if len(ta) != len(tb):
        return {"identical": False, "detail": f"{len(ta)} vs {len(tb)} lines"}
    text_identical = True
    for i, (la, lb) in enumerate(zip(ta, tb)):
        if la != lb:
            text_identical = False
            if len(la) != len(lb) or not all(_same_token(x, y) for x, y in zip(la, lb)):
                return {"identical": False, "detail": f"line {i}: {' '.join(la)[:200]} | {' '.join(lb)[:200]}"}
    return {"identical": True, "text_identical": text_identical, "lines": len(ta)}


def _hist_table(path):
    import uproot

    out = {}
    with uproot.open(path) as f:
        for k, c in f.classnames(cycle=False).items():
            if not c.startswith("TH1"):
                continue
            h = f[k]
            ax = h.member("fXaxis")
            out[k] = (h.member("fName"), h.member("fTitle"), ax.member("fNbins"), ax.member("fXmin"), ax.member("fXmax"),
                      np.asarray(h.values(flow=True), dtype=float), np.asarray(h.variances(flow=True), dtype=float))
    return out


def compare_shapes(a, b):
    ha, hb = _hist_table(a), _hist_table(b)
    if set(ha) != set(hb):
        return {"identical": False, "detail": f"keys differ: {sorted(set(ha) ^ set(hb))[:10]}"}
    max_rel = 0.0
    n_exact = 0
    for k in ha:
        xa, xb = ha[k], hb[k]
        if xa[:5] != xb[:5]:
            return {"identical": False, "detail": f"{k}: name/title/axis {xa[:5]} vs {xb[:5]}"}
        exact = True
        for va, vb in ((xa[5], xb[5]), (xa[6], xb[6])):
            if np.array_equal(va, vb):
                continue
            exact = False
            rel = np.max(np.abs(va - vb) / np.maximum(np.abs(va), 1e-300))
            max_rel = max(max_rel, float(rel))
        n_exact += exact
    return {"identical": max_rel <= 1e-12, "histograms": len(ha), "bit_identical": n_exact, "max_rel_diff": max_rel}


def compare_work(da, db):
    """All tags of two work directories: cards, shapes, combined datacard.txt."""
    tags_a = {p.name for p in (da / "work").iterdir()} if (da / "work").exists() else set()
    tags_b = {p.name for p in (db / "work").iterdir()} if (db / "work").exists() else set()
    res = {"tags": len(tags_a), "tags_only_legacy": sorted(tags_a - tags_b), "tags_only_python": sorted(tags_b - tags_a),
           "cards": 0, "cards_identical": 0, "cards_text_identical": 0, "shapes": 0, "shapes_identical": 0,
           "histograms": 0, "histograms_bit_identical": 0, "max_rel_diff": 0.0, "datacards": 0,
           "datacards_identical": 0, "failures": []}
    for tag in sorted(tags_a & tags_b):
        wa, wb = da / "work" / tag, db / "work" / tag
        fa = {p.name for p in wa.glob("card_*.txt")} | {p.name for p in wa.glob("shapes_*.root")}
        fb = {p.name for p in wb.glob("card_*.txt")} | {p.name for p in wb.glob("shapes_*.root")}
        if fa != fb:
            res["failures"].append(f"{tag}: files {sorted(fa ^ fb)}")
        for name in sorted(fa & fb):
            if name.endswith(".txt"):
                c = compare_cards(wa / name, wb / name, wa, wb)
                res["cards"] += 1
                res["cards_identical"] += c["identical"]
                res["cards_text_identical"] += c.get("text_identical", False)
                if not c["identical"]:
                    res["failures"].append(f"{tag}/{name}: {c['detail']}")
            else:
                c = compare_shapes(wa / name, wb / name)
                res["shapes"] += 1
                res["shapes_identical"] += c["identical"]
                res["histograms"] += c.get("histograms", 0)
                res["histograms_bit_identical"] += c.get("bit_identical", 0)
                res["max_rel_diff"] = max(res["max_rel_diff"], c.get("max_rel_diff", 0.0))
                if not c["identical"]:
                    res["failures"].append(f"{tag}/{name}: {c['detail'] if 'detail' in c else c}")
        if (wa / "datacard.txt").exists() and (wb / "datacard.txt").exists() and (wa / "datacard.txt").stat().st_size:
            c = compare_cards(wa / "datacard.txt", wb / "datacard.txt", wa, wb)
            res["datacards"] += 1
            res["datacards_identical"] += c["identical"]
            if not c["identical"]:
                res["failures"].append(f"{tag}/datacard.txt: {c['detail']}")
    return res


def read_csv(path):
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


CSVS = ("significance.csv", "limits.csv", "significance_abcd_mc.csv", "limits_abcd_mc.csv",
        "significance_by_channel.csv", "limits_by_channel.csv", "significance_by_channel_abcd_mc.csv",
        "limits_by_channel_abcd_mc.csv", "significance_postfit.csv", "limits_postfit.csv",
        "significance_abcd_mc_postfit.csv", "limits_abcd_mc_postfit.csv")


def _rel(a, b):
    if a == b or (math.isnan(a) and math.isnan(b)):
        return 0.0
    if math.isinf(a) or math.isinf(b) or math.isnan(a) or math.isnan(b):
        return math.inf
    return abs(a - b) / max(abs(a), 1e-12)


def compare_results(da, db):
    """The CSVs of both runs: per row Z and r95 (median), and the largest relative difference of any number."""
    rows, max_rel, identical_files, files = [], 0.0, 0, 0
    for name in CSVS:
        ra, rb = read_csv(da / name), read_csv(db / name)
        if not ra and not rb:
            continue
        files += 1
        identical_files += (da / name).read_text() == (db / name).read_text() if (db / name).exists() else 0
        for a, b in zip(ra, rb):
            key = {k: a[k] for k in ("channel", "scope", "name") if k in a}
            for k in a:
                if k in key:
                    continue
                rel = _rel(float(a[k]), float(b.get(k, "nan")))
                max_rel = max(max_rel, rel)
            if name.startswith("significance") or name.startswith("limits"):
                rows.append(dict(key, file=name, **{f"legacy_{k}": float(a[k]) for k in a if k not in key},
                                 **{f"python_{k}": float(b[k]) for k in b if k not in key}))
        if len(ra) != len(rb):
            max_rel = math.inf
    return {"csv_files": files, "csv_files_identical": identical_files, "max_rel_diff": max_rel, "rows": rows}


def headline(da):
    """Z and r95 of the combined scenario, fitted-QCD (abcd) mode, from a run's CSVs."""
    out = {}
    for mode, suffix in (("mc", ""), ("abcd", "_abcd_mc")):
        z = [r for r in read_csv(da / f"significance{suffix}.csv") if r["scope"] == "combined"]
        lim = [r for r in read_csv(da / f"limits{suffix}.csv") if r["scope"] == "combined"]
        if z and lim:
            out[mode] = {"Z": float(z[0]["significance"]), "r95": float(lim[0]["exp_50"])}
        zp = [r for r in read_csv(da / f"significance{suffix}_postfit.csv") if r["scope"] == "combined"]
        lp = [r for r in read_csv(da / f"limits{suffix}_postfit.csv") if r["scope"] == "combined"]
        if zp and lp:
            out[mode + "_postfit"] = {"Z": float(zp[0]["significance"]), "r95": float(lp[0]["exp_50"])}
    return out


# -------------------- report --------------------


def compare_case(spec, out, name, case):
    d = out / name
    entry = {"channels": [c["name"] for c in case_config(spec, out, name, case)["channels"]]}
    if (d / "legacy").exists() and (d / "python").exists():
        entry["full"] = compare_work(d / "legacy", d / "python")
        entry["results"] = compare_results(d / "legacy", d / "python")
        entry["legacy"] = headline(d / "legacy")
        entry["python"] = headline(d / "python")
    if (d / "legacy_cards").exists() and (d / "python_cards").exists():
        entry["all_scopes"] = compare_work(d / "legacy_cards", d / "python_cards")
    if "recorded" in case:
        entry["recorded"] = headline(Path(case["recorded"]))
    log(f"{name}: " + json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("rows", "failures")}
                                  if isinstance(v, dict) else v for k, v in entry.items()}, default=str)[:600])
    return entry


def report(spec, out, cases, jobs=6):
    with ProcessPoolExecutor(jobs) as pool:
        entries = list(pool.map(compare_case, [spec] * len(cases), [out] * len(cases), list(cases), cases.values()))
    doc = {"spec": spec, "cases": dict(zip(cases, entries))}
    (out / "sync.json").write_text(json.dumps(doc, indent=1, default=str))
    lines = ["# Datacard writer sync: Python (vvv.stats.datacards) vs combine.C", "",
             "Cards compared token by token (comment lines and the absolute shape-file path aside), shape files "
             "histogram by histogram (name, title, axis, contents and errors incl. flow), the combined datacard.txt "
             "of combineCards.py, and every number of the combine.C CSVs.", "",
             "| case | channels | tags | cards identical | shapes identical (hists bit-identical) | datacard.txt | "
             "all-scope tags | all-scope cards / shapes identical | CSV max rel diff | Z mc / abcd (legacy) | "
             "Z mc / abcd (python) | r95 mc / abcd (legacy) | r95 mc / abcd (python) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def f(x):
        return "-" if x is None else f"{x:.5g}"

    for name, e in doc["cases"].items():
        fu, al, rs = e.get("full", {}), e.get("all_scopes", {}), e.get("results", {})
        lg, py = e.get("legacy", {}), e.get("python", {})
        lines.append(
            f"| {name} | {len(e['channels'])} | {fu.get('tags', '-')} | {fu.get('cards_identical', '-')}/"
            f"{fu.get('cards', '-')} | {fu.get('shapes_identical', '-')}/{fu.get('shapes', '-')} "
            f"({fu.get('histograms_bit_identical', '-')}/{fu.get('histograms', '-')}) | "
            f"{fu.get('datacards_identical', '-')}/{fu.get('datacards', '-')} | {al.get('tags', '-')} | "
            f"{al.get('cards_identical', '-')}/{al.get('cards', '-')} / {al.get('shapes_identical', '-')}/"
            f"{al.get('shapes', '-')} | {f(rs.get('max_rel_diff'))} | "
            f"{f(lg.get('mc', {}).get('Z'))} / {f(lg.get('abcd', {}).get('Z'))} | "
            f"{f(py.get('mc', {}).get('Z'))} / {f(py.get('abcd', {}).get('Z'))} | "
            f"{f(lg.get('mc', {}).get('r95'))} / {f(lg.get('abcd', {}).get('r95'))} | "
            f"{f(py.get('mc', {}).get('r95'))} / {f(py.get('abcd', {}).get('r95'))} |")
    fails = [(n, x) for n, e in doc["cases"].items() for k in ("full", "all_scopes") for x in e.get(k, {}).get(
        "failures", [])]
    lines += ["", f"Failures: {len(fails)}"] + [f"- {n}: {x}" for n, x in fails[:50]]
    (out / "sync.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out / 'sync.json'} and {out / 'sync.md'}")
    return doc


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("spec", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--steps", default=",".join(STEPS))
    p.add_argument("--cases", help="comma-separated subset")
    p.add_argument("--jobs", type=int, default=6, help="cases (combine.C) or Combine runs (python) in parallel")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    spec = json.loads(args.spec.read_text())
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "spec.json").write_text(json.dumps(spec, indent=1))
    steps = args.steps.split(",")
    cases = {k: v for k, v in spec["cases"].items() if not args.cases or k in args.cases.split(",")}
    binary = build(spec, out, args.force and "build" in steps) if "build" in steps or "legacy" in steps \
        else out / "bin" / "combine_sync"
    if "synthetic" in steps:
        for name, case in cases.items():
            for ch, mod in case.get("synthetic", {}).items():
                dst = out / "synthetic" / name / ch / "qcd_abcd_yields.root"
                if not dst.exists() or args.force:
                    synthetic_root(mod["from"], dst, mod)
                    log(f"synthetic input {dst}")
    cfgs = {name: case_config(spec, out, name, case) for name, case in cases.items()}
    if "legacy" in steps:
        tasks = []
        for name, case in cases.items():
            tasks.append((out / name / "legacy", cfgs[name], {}))
            if case.get("cards_only_scopes", True):
                tasks.append((out / name / "legacy_cards", cfgs[name], {
                    "scenario_scopes": ALL_SCOPES, "combine_cmd": "true", "combine_cards_cmd": "true"}))
        with ThreadPoolExecutor(args.jobs) as pool:
            list(pool.map(lambda t: run_legacy(spec, binary, t[0], t[1], args.force, **t[2]), tasks))
    if "python" in steps:
        for name, case in cases.items():
            run_python(spec, out / name / "python", cfgs[name], args.force, True, args.jobs)
            if case.get("cards_only_scopes", True):
                run_python(spec, out / name / "python_cards", cfgs[name], args.force, False, args.jobs,
                           scenario_scopes=ALL_SCOPES, combine_cmd="true", combine_cards_cmd="true")
    if "compare" in steps:
        report(spec, out, cases, args.jobs)


if __name__ == "__main__":
    main()
