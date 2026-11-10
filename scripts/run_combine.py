"""Runs Combine on the cards of scripts/write_datacards.py with the commands of the legacy combine.C
(vvv.stats.combine_run): combineCards.py, then expected significance and AsymptoticLimits on the Asimov
dataset (a-priori), and with data_mode observed also the post-fit expected results.

MANIFEST is the <output_dir>/datacards.json of write_datacards.py. Writes into its output_dir the
combine.C CSVs (significance.csv, limits.csv, significance_abcd_mc.csv, limits_abcd_mc.csv,
*_by_channel*.csv, *_postfit.csv) and summary.json. Combine runs in a cmsenv of --cmssw-src (default
../CMSSW_16_1_0_pre4/src, the legacy area with HiggsAnalysis/CombinedLimit).

Usage: pixi run python scripts/run_combine.py MANIFEST [--jobs N] [--cmssw-src DIR] [--tags TAG ...]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("manifest", type=Path, help="datacards.json of write_datacards.py")
    p.add_argument("--jobs", type=int, default=4, help="Combine runs in parallel (each single-threaded)")
    p.add_argument("--cmssw-src", type=Path)
    p.add_argument("--tags", nargs="*", help="only these runs (no CSVs then; summary.json has them)")
    args = p.parse_args()

    from vvv.stats.combine_run import DEFAULT_CMSSW_SRC, run_all

    manifest = json.loads(args.manifest.read_text())
    summary = run_all(manifest, cmssw_src=args.cmssw_src or DEFAULT_CMSSW_SRC, jobs=args.jobs, tags=args.tags)
    sys.exit(1 if any(r["result"]["status"] == "failed" for r in summary["runs"]) else 0)


if __name__ == "__main__":
    main()
