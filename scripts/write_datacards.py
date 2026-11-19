"""Datacards and shape ROOT files of the pass/fail mass fit from a legacy combine.C config.json
(vvv.stats.datacards: port of the card writing of combine.C).

CONFIG is a combine.C config (channels with root_file = qcd_abcd_yields.root and bdt_root = the class
registry, enabled_nuisances, lumi_unc, trigger_unc, xsec_unc_json, the six *_syst_json, scenario_scopes,
data_mode, ...); relative paths are relative to it, as in combine.C. Writes, per scenario and QCD mode,
<output_dir>/work/<tag>/card_<channel>.txt and shapes_<channel>.root (the combine.C layout and tags), and
<output_dir>/datacards.json, the manifest scripts/run_combine.py reads. With --run, runs Combine as
combine.C does (scripts/run_combine.py) afterwards.

Usage: pixi run python scripts/write_datacards.py CONFIG [--output-dir DIR] [--run] [--jobs N]
           [--cmssw-src DIR]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("config", type=Path, help="combine.C config.json")
    p.add_argument("--output-dir", type=Path, help="instead of the config's output_dir")
    p.add_argument("--run", action="store_true", help="also run combineCards.py and Combine (run_combine.py)")
    p.add_argument("--jobs", type=int, default=4, help="with --run: Combine runs in parallel")
    p.add_argument("--cmssw-src", type=Path, help="with --run: CMSSW src area with HiggsAnalysis/CombinedLimit")
    args = p.parse_args()

    from vvv.stats.datacards import CardError, load_config, write_all

    try:
        cfg = load_config(args.config, output_dir=args.output_dir)
        manifest = write_all(cfg)
    except CardError as e:
        sys.exit(f"[ERROR] {e}")
    status = 1 if any(r["status"] == "failed" for r in manifest["runs"]) else 0
    if args.run:
        from vvv.stats.combine_run import DEFAULT_CMSSW_SRC, run_all

        summary = run_all(manifest, cmssw_src=args.cmssw_src or DEFAULT_CMSSW_SRC, jobs=args.jobs)
        status = 1 if any(r["result"]["status"] == "failed" for r in summary["runs"]) else 0
    sys.exit(status)


if __name__ == "__main__":
    main()
