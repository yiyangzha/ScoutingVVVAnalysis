"""Out-of-fold scores of a K-fold training, (re)computed on existing fit-region skims.

The skims (vvv run --skim-cut [--model-dir]) carry every classifier input, so a retrained model
needs no new pass over NanoAOD: each <SRC>/<sample>/<category>/nominal/*.parquet is copied to
DEST with its proba_<class> columns replaced by the new model's, or added if the skim has none
(skims made without --model-dir, e.g. the data fit-region skims). Each event is scored by the
model of its fold, event % k (vvv.ml.inference.score_table). _DONE.json is copied, with the
category, model directory and kept columns recorded under "rescored" (one entry per category).
Only the given category is written, which is all the SR optimisation of that category reads.

Options:
- --samples: only these samples (default: every <SRC>/<sample>/_DONE.json).
- --columns: write only these columns plus proba_<class> (default: all), to keep a copy lean.
- --skip-existing: resume, leaving output files that exist already (every output file is written
  to a temporary name and renamed, so an existing one is complete).
- --threads: files in parallel; --nthread: XGBoost threads per prediction (default: all cores).

Usage: pixi run python scripts/rescore_skims.py SRC DEST --category CAT --model-dir DIR
           [--samples S ...] [--columns C ...] [--skip-existing] [--threads N] [--nthread N]
"""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from vvv.ml.inference import CategoryModel, score_table


def main():
    p = argparse.ArgumentParser()
    p.add_argument("src", type=Path)
    p.add_argument("dest", type=Path)
    p.add_argument("--category", required=True)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--samples", nargs="+", help="only these samples (default: every sample with a _DONE.json)")
    p.add_argument("--columns", nargs="+", help="write only these columns plus proba_<class> (default: all)")
    p.add_argument("--skip-existing", action="store_true", help="leave existing output files (resume)")
    p.add_argument("--threads", type=int, default=8, help="files in parallel")
    p.add_argument("--nthread", type=int, help="XGBoost threads per prediction (default: all cores)")
    args = p.parse_args()
    model = CategoryModel.load(args.model_dir, args.category)
    if args.nthread:
        model.set_nthread(args.nthread)
    if args.samples:
        samples = [args.src / s for s in args.samples]
        for s in samples:
            if not (s / "_DONE.json").exists():
                raise FileNotFoundError(f"{s}: no _DONE.json (skim incomplete)")
    else:
        samples = sorted(d.parent for d in args.src.glob("*/_DONE.json"))
    jobs = [(s, f) for s in samples for f in sorted((s / args.category / "nominal").glob("*.parquet"))]
    print(f"{len(samples)} samples, {len(jobs)} files", flush=True)

    def rescore(job):
        sample, f = job
        out = args.dest / f.relative_to(args.src)
        if args.skip_existing and out.exists():
            n = pq.ParquetFile(out).metadata.num_rows
            print(f"  {out}: exists ({n} rows), skipped", flush=True)
            return n
        t0 = time.time()
        t = score_table(pq.read_table(f), model, keep=args.columns)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.tmp")
        pq.write_table(t, tmp, compression="zstd")
        tmp.rename(out)
        print(f"  {out}: {t.num_rows} rows, {time.time() - t0:.1f} s", flush=True)
        return t.num_rows

    with ThreadPoolExecutor(args.threads) as ex:
        n = sum(ex.map(rescore, jobs))
    for s in samples:
        done = json.loads((s / "_DONE.json").read_text())
        dest_done = args.dest / s.name / "_DONE.json"
        previous = json.loads(dest_done.read_text()).get("rescored", {}) if dest_done.exists() else {}
        if "category" in previous:  # the earlier single-entry layout
            previous = {previous["category"]: previous}
        previous[args.category] = {"category": args.category, "model_dir": str(args.model_dir),
                                   "columns": args.columns}
        done["rescored"] = previous
        dest_done.parent.mkdir(parents=True, exist_ok=True)
        dest_done.write_text(json.dumps(done, indent=1))
    print(f"rescored {n} events", flush=True)


if __name__ == "__main__":
    main()
