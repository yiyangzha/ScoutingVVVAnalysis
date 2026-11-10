"""Combine runs on the cards of vvv.stats.datacards: the command part of the legacy combine.C buildAndRun.

Per manifest run (one scenario, QCD mode and channel set), in its work directory and with the same
commands as combine.C:

1. combineCards.py card_<ch>.txt ... > datacard.txt (the k-th card's bins become ch<k>_<bin>);
2. combine -M Significance -t -1 --expectSignal 1 -m 120 --rMin -5 --rMax 10
   --cminDefaultMinimizerStrategy 0 --setParameters 'rgx{zc_.*}=0' -n sig_<tag> datacard.txt;
3. combine -M AsymptoticLimits -t -1 -m 120 --run expected --cminDefaultMinimizerStrategy 0
   --setParameters 'rgx{zc_.*}=0' -n lim_<tag> datacard.txt;
4. data_mode observed only: text2workspace.py --channel-masks, a background-only MultiDimFit to the
   unblinded bins (with per-SR slopes first frozen, then released), and Significance /
   AsymptoticLimits on the Asimov dataset of that fit (post-fit expected).

The results go into the CSVs combine.C writes (significance.csv, limits.csv, *_abcd_mc.csv,
*_by_channel*.csv, with observed data also *_postfit.csv), in the same format, and into
summary.json. A failed run is stored as nan (the others still run), a run whose signal is identically
zero as Z = 0 and infinite limits, as in combine.C.
"""

import json
import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ZERO_FLOOR_START = "rgx{zc_.*}=0"
SIGNIFICANCE_R_RANGE = "--rMin -5 --rMax 10"
MINIMIZER_STRATEGY = "--cminDefaultMinimizerStrategy 0"
QUANTILES = (("exp_2p5", 0.025), ("exp_16", 0.16), ("exp_50", 0.5), ("exp_84", 0.84), ("exp_97p5", 0.975))
DEFAULT_CMSSW_SRC = Path(__file__).resolve().parents[4] / "CMSSW_16_1_0_pre4" / "src"


def log_message(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def shell_quote(s):
    """combine.C shellQuote."""
    return "'" + s.replace("'", "'\\''") + "'"


def cms_shell(cmssw_src, command, cwd, log_path=None, append=False):
    """Runs command in bash after cmsenv in cmssw_src (as scripts/fitcheck_massfit.py): a clean environment
    without the pixi variables, stdin from /dev/null (else ~/.bashrc's AF banner breaks scram)."""
    script = (f"source /cvmfs/cms.cern.ch/cmsset_default.sh && cd {shell_quote(str(cmssw_src))} && "
              "eval \"$(scramv1 runtime -sh)\" && command -v combine >/dev/null && "
              "command -v combineCards.py >/dev/null || { echo 'cmsenv failed: no combine on PATH' >&2; exit 97; }; "
              f"cd {shell_quote(str(cwd))} && {command}")
    env = {k: os.environ[k] for k in ("HOME", "USER", "LOGNAME") if k in os.environ}
    env["PATH"] = "/usr/local/bin:/usr/bin:/bin"
    for k in ("OMP_NUM_THREADS",):
        if k in os.environ:
            env[k] = os.environ[k]
    out = open(log_path, "a" if append else "w") if log_path else None
    try:
        return subprocess.run(["/bin/bash", "-c", script], stdin=subprocess.DEVNULL, stdout=out or subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, env=env)
    finally:
        if out:
            out.close()


def combine_commands(manifest, run):
    """The shell commands combine.C runs for one run, in order: [(label, command)]."""
    tag = run["tag"]
    work = run["work"]
    cd = f"cd {shell_quote(work)} && "
    combine = manifest["combine_cmd"]
    cmds = [("combineCards", cd + manifest["combine_cards_cmd"] + "".join(" " + shell_quote(c) for c in run["cards"])
             + " > " + shell_quote("datacard.txt"))]
    cmds.append(("significance", cd + f"{combine} -M Significance -t -1 --expectSignal 1 -m 120 {SIGNIFICANCE_R_RANGE} "
                 f"{MINIMIZER_STRATEGY} --setParameters {shell_quote(ZERO_FLOOR_START)} -n "
                 f"{shell_quote('sig_' + tag)} {shell_quote('datacard.txt')}"))
    cmds.append(("limits", cd + f"{combine} -M AsymptoticLimits -t -1 -m 120 --run expected {MINIMIZER_STRATEGY} "
                 f"--setParameters {shell_quote(ZERO_FLOOR_START)} -n {shell_quote('lim_' + tag)} "
                 f"{shell_quote('datacard.txt')}"))
    if manifest["data_mode"] != "observed":
        return cmds
    masks_on = "".join(f",mask_{m}=1" for m in run["masked_bins"])
    masks_off = "".join(f",mask_{m}=0" for m in run["masked_bins"])
    snap_name = "snap_" + tag
    slopes = "".join("," + s for s in run["slope_params"])
    first_name = "snap0_" + tag if slopes else snap_name
    fit = (cd + f"text2workspace.py {shell_quote('datacard.txt')} --channel-masks -m 120 -o "
           f"{shell_quote('workspace_masked.root')} && {combine} -M MultiDimFit {shell_quote('workspace_masked.root')} "
           f"-m 120 {MINIMIZER_STRATEGY} --saveWorkspace --freezeParameters {shell_quote('r' + slopes)} "
           f"--setParameters {shell_quote(ZERO_FLOOR_START + ',r=0' + masks_on)} -n {shell_quote(first_name)}")
    if slopes:
        # The saved workspace keeps the frozen terms constant: --floatParameters releases them.
        fit += (f" && {combine} -M MultiDimFit {shell_quote('higgsCombine' + first_name + '.MultiDimFit.mH120.root')} "
                f"--snapshotName MultiDimFit -m 120 {MINIMIZER_STRATEGY} --saveWorkspace --freezeParameters r "
                f"--floatParameters {shell_quote(slopes[1:])} --setParameters "
                f"{shell_quote(ZERO_FLOOR_START + ',r=0' + masks_on)} -n {shell_quote(snap_name)}")
    cmds.append(("postfit_fit", fit))
    snap_root = f"higgsCombine{snap_name}.MultiDimFit.mH120.root"
    post = (f" --snapshotName MultiDimFit -t -1 --toysFrequentist --bypassFrequentistFit -m 120 {MINIMIZER_STRATEGY}"
            f" --floatParameters r --setParameters {shell_quote(ZERO_FLOOR_START + masks_off)}")
    cmds.append(("significance_postfit", cd + f"{combine} -M Significance {shell_quote(snap_root)}{post} "
                 f"--expectSignal 1 {SIGNIFICANCE_R_RANGE} -n {shell_quote('sigpost_' + tag)}"))
    cmds.append(("limits_postfit", cd + f"{combine} -M AsymptoticLimits {shell_quote(snap_root)}{post} --run expected "
                 f"-n {shell_quote('limpost_' + tag)}"))
    return cmds


def read_significance(path):
    """combine.C readSingleLimit: the first entry of the limit tree."""
    import uproot

    with uproot.open(path) as f:
        if "limit" not in f:
            raise RuntimeError(f"Missing TTree 'limit' in {path}")
        v = f["limit"]["limit"].array(library="np")
    if len(v) < 1:
        raise RuntimeError(f"Empty 'limit' tree in {path}")
    return float(v[0])


def read_limits(path):
    """combine.C tryReadLimitQuantiles: the five expected quantiles (|q - target| < 5e-3 in float), or None
    when one is missing."""
    import uproot

    with uproot.open(path) as f:
        t = f["limit"]
        limit = t["limit"].array(library="np")
        q = t["quantileExpected"].array(library="np").astype(np.float32)
    out = {}
    for lim, qv in zip(limit, q):
        for key, target in QUANTILES:
            if abs(qv - np.float32(target)) < np.float32(5e-3):
                out[key] = float(lim)
                break
    return out if len(out) == len(QUANTILES) else None


def _nan_limits():
    return {k: math.nan for k, _ in QUANTILES}


def _inf_limits():
    return {k: math.inf for k, _ in QUANTILES}


def run_one(manifest, run, cmssw_src, log=log_message):
    """Runs the Combine commands of one run; returns {significance, limits[, significance_postfit,
    limits_postfit], status, error}."""
    observed = manifest["data_mode"] == "observed"
    res = {"tag": run["tag"], "status": run["status"], "error": run["error"]}
    if run["status"] == "failed":
        res.update(significance=math.nan, limits=_nan_limits())
        if observed:
            res.update(significance_postfit=math.nan, limits_postfit=_nan_limits())
        return res
    if run["status"] == "zero_signal":
        res.update(significance=0.0, limits=_inf_limits())
        if observed:
            res.update(significance_postfit=0.0, limits_postfit=_inf_limits())
        return res
    work = Path(run["work"])
    log_path = work / "combine_run.log"
    log_path.write_text("")
    t0 = time.time()
    try:
        for label, cmd in combine_commands(manifest, run):
            with open(log_path, "a") as f:
                f.write(f"$ {cmd}\n")
            r = cms_shell(cmssw_src, cmd, work, log_path, append=True)
            if r.returncode != 0:
                raise RuntimeError(f"Command failed (rc={r.returncode}): {cmd} (see {log_path})")
            if label == "combineCards" and observed:
                bins = datacard_bins(work / "datacard.txt")
                for m in run["masked_bins"]:
                    if m not in bins:
                        raise RuntimeError(f"Blinded bin '{m}' is not a bin of {work / 'datacard.txt'}")
            if label == "significance":
                res["significance"] = read_significance(work / f"higgsCombinesig_{run['tag']}.Significance.mH120.root")
            elif label == "limits":
                res["limits"] = _limits_or_nan(work / f"higgsCombinelim_{run['tag']}.AsymptoticLimits.mH120.root",
                                               run, res["significance"], log)
            elif label == "significance_postfit":
                res["significance_postfit"] = read_significance(
                    work / f"higgsCombinesigpost_{run['tag']}.Significance.mH120.root")
            elif label == "limits_postfit":
                res["limits_postfit"] = _limits_or_nan(
                    work / f"higgsCombinelimpost_{run['tag']}.AsymptoticLimits.mH120.root", run,
                    res["significance_postfit"], log)
    except Exception as e:  # noqa: BLE001 - combine.C stores any failure as nan and goes on
        log(f"ERROR: scenario failed, storing nan: {run['tag']}: {e}")
        res.update(status="failed", error=str(e), significance=math.nan, limits=_nan_limits())
        if observed:
            res.update(significance_postfit=math.nan, limits_postfit=_nan_limits())
        return res
    log(f"  {run['tag']}: Z = {res['significance']:.6g}, r95 = {res['limits']['exp_50']:.6g} "
        f"({time.time() - t0:.0f} s)")
    return res


def _limits_or_nan(path, run, significance, log):
    lim = read_limits(path)
    if lim is None:
        log(f"WARNING: AsymptoticLimits output is missing expected quantiles; keeping the significance and storing "
            f"nan expected limits: {run['tag']} significance={significance:.17g}")
        return _nan_limits()
    return lim


def datacard_bins(path):
    """The bins of a datacard (its first 'bin' line)."""
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts and parts[0] == "bin":
            return set(parts[1:])
    raise RuntimeError(f"No bin line in {path}")


def cxx_double(value):
    """std::ostream << double (default precision 6)."""
    return f"{value:g}"


def csv_double(value):
    """combine.C csvDouble: inf, else setprecision(17)."""
    return "inf" if math.isinf(value) else f"{value:.17g}"


def write_csvs(manifest, results, out_dir, log=log_message):
    """The CSVs of combine.C runMain, from the results keyed by tag."""
    out_dir = Path(out_dir)
    scen = manifest["scenarios"]
    runs = manifest["runs"]
    observed = manifest["data_mode"] == "observed"

    def pick(mode, channel):
        return [(r, results[r["tag"]]) for r in runs if r["mode"] == mode and r["channel"] == channel]

    channels = [c["name"] for c in manifest["channels"]]
    suffixes = [("", "significance", "limits")]
    if observed:
        suffixes.append(("_postfit", "significance_postfit", "limits_postfit"))
    for suffix, skey, lkey in suffixes:
        for mode, mtag in (("mc", ""), ("abcd", "_abcd_mc")):
            rows = pick(mode, None)
            assert [(r["scope"], r["name"]) for r, _ in rows] == [(s["scope"], s["name"]) for s in scen]
            _write(out_dir / f"significance{mtag}{suffix}.csv", "scope,name,significance",
                   [f"{r['scope']},{r['name']},{cxx_double(x[skey])}" for r, x in rows], log)
            _write(out_dir / f"limits{mtag}{suffix}.csv", "scope,name,exp_2p5,exp_16,exp_50,exp_84,exp_97p5",
                   [f"{r['scope']},{r['name']}," + ",".join(csv_double(x[lkey][k]) for k, _ in QUANTILES)
                    for r, x in rows], log)
            rows = [rx for ch in channels for rx in pick(mode, ch)]
            _write(out_dir / f"significance_by_channel{mtag}{suffix}.csv", "channel,scope,name,significance",
                   [f"{r['channel']},{r['scope']},{r['name']},{cxx_double(x[skey])}" for r, x in rows], log)
            _write(out_dir / f"limits_by_channel{mtag}{suffix}.csv",
                   "channel,scope,name,exp_2p5,exp_16,exp_50,exp_84,exp_97p5",
                   [f"{r['channel']},{r['scope']},{r['name']}," + ",".join(csv_double(x[lkey][k]) for k, _ in QUANTILES)
                    for r, x in rows], log)


def _write(path, header, rows, log):
    path.write_text("\n".join([header] + rows) + "\n")
    log(f"Wrote {path}")


def run_all(manifest, cmssw_src=DEFAULT_CMSSW_SRC, jobs=1, out_dir=None, tags=None, log=log_message):
    """Runs every run of a datacards.json manifest (jobs in parallel), writes the CSVs and summary.json into
    out_dir (default: the manifest's output_dir). Returns the summary."""
    out_dir = Path(out_dir or manifest["output_dir"])
    runs = [r for r in manifest["runs"] if tags is None or r["tag"] in tags]
    log(f"{len(runs)} Combine runs, {jobs} in parallel, CMSSW {cmssw_src}")
    with ThreadPoolExecutor(max(1, jobs)) as pool:
        results = {r["tag"]: x for r, x in zip(runs, pool.map(lambda r: run_one(manifest, r, cmssw_src, log=log),
                                                               runs))}
    if tags is None:
        write_csvs(manifest, results, out_dir, log=log)
    summary = {"manifest": str(Path(manifest["output_dir"]) / "datacards.json"), "cmssw_src": str(cmssw_src),
               "data_mode": manifest["data_mode"],
               "method": "Asimov (-t -1): -M Significance --expectSignal 1, -M AsymptoticLimits --run expected"
                         + ("; post-fit: Asimov from a background-only fit to the unblinded bins"
                            if manifest["data_mode"] == "observed" else ""),
               "runs": [dict(r, result=results[r["tag"]]) for r in runs]}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default))
    n_failed = sum(x["status"] == "failed" for x in results.values())
    log(f"Wrote {out_dir / 'summary.json'}: {len(runs)} runs, {n_failed} failed")
    if not manifest.get("keep_work", True) and tags is None:
        shutil.rmtree(manifest["work_dir"], ignore_errors=True)
    return summary


def _json_default(v):
    return float(v)
