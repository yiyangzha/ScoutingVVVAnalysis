"""QCD mass-fit inputs from the coffea MC fit-region skims (counterpart of legacy.LegacySetup).

The massfit settings and exclude_samples come from a legacy qcd_est.json (the production
point's), the classes and the msoftdrop_1 window from the coffea training config, the events
(proba_<class>, weight x lumi, msoftdrop_1; every other category threshold already applied) from
<skim_dir>/<sample>/<category>/nominal, and the SR boxes from a signal_region.csv. Data skims
need proba_<class> columns (vvv run --model-dir, or scripts/rescore_skims.py) to be used; they are
read file by file (the boosted2_unresolved data skim has ~485M rows), the blinding being applied by
MassfitAccumulator.fill_data to each file before anything is counted.

The skim MC weights are x the config's lumi (vvv run --lumi, 255.8 fb^-1 by default); mc_lumi
rescales them to another luminosity (e.g. the data's, for a fit to data).
"""

import json
from pathlib import Path

import pyarrow.dataset as ds

from vvv.background.qcd_massfit import MassfitAccumulator, MassfitConfig, load_sr_boxes
from vvv.ml.inference import CLASSES

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"


class SkimSetup:
    def __init__(self, qcd_est_config, training_config, category, skim_dir, signal_region_csv, output_dir,
                 data_skim_dir=None, data_sample="data_2024", catalogue=None, mc_lumi=None):
        from vvv.samples import load_samples

        raw = json.loads(Path(qcd_est_config).read_text())
        tcfg = json.loads(Path(training_config).read_text())
        if str(raw.get("abcd_mode", "union")).lower() != "massfit":
            raise ValueError(f"only abcd_mode 'massfit' is ported, got {raw.get('abcd_mode')!r}")
        cc = tcfg["categories"][category]
        self.raw, self.tree, self.branch = raw, category, MASS_BRANCH
        self.window = tuple(cc["thresholds"][MASS_BRANCH])
        self.massfit = MassfitConfig.from_dict(raw["massfit"][category], default_blind_window=self.window)
        self.class_groups = tcfg["class_groups"]
        self.class_names = list(self.class_groups)
        if self.class_names != list(CLASSES):
            raise ValueError(f"class_groups {self.class_names} are not in the model's class order {CLASSES}")
        self.qcd_classes = [c for c in self.class_names if "qcd" in c.lower()]
        self.sample_class = {s: c for c, members in self.class_groups.items() for s in members}
        self.exclude = set(raw.get("exclude_samples", []))
        catalogue = catalogue or load_samples()
        for c in self.massfit.smooth_templates:
            if c not in self.class_groups or c in self.qcd_classes or all(
                    catalogue[s].kind == "signal" for s in self.class_groups[c]):
                raise RuntimeError(f"massfit smooth_templates: '{c}' is not a non-QCD background class")
        self.lumi = float(raw.get("lumi", 0))  # the skim weights are already x lumi
        self.mc_scale = 1.0
        if mc_lumi is not None:
            if not self.lumi > 0:
                raise ValueError("mc_lumi needs the config's lumi (the luminosity of the skim weights)")
            self.mc_scale, self.lumi = float(mc_lumi) / self.lumi, float(mc_lumi)
        self.skim_dir, self.bdt_root = Path(skim_dir), Path(skim_dir)
        self.signal_region_csv = Path(signal_region_csv)
        self.output_dir = Path(output_dir)
        self.root_file_name = raw.get("root_file_name", "qcd_abcd_yields.root")
        self.data_skim_dir = Path(data_skim_dir) if data_skim_dir else None
        self.data_sample = data_sample if data_skim_dir else None
        if self.data_sample is not None and self.massfit.blind_window is None:
            raise RuntimeError("data needs massfit blind_window")

    def accumulator(self, log=print):
        boxes = load_sr_boxes(self.signal_region_csv, self.class_names)
        unknown = sorted(set(self.massfit.sr_slopes) - {b[0] for b in boxes})
        if unknown:
            raise RuntimeError(f"massfit sr_slopes names SRs not in the SR CSV: {unknown}")
        return MassfitAccumulator(self.massfit, boxes, self.class_names, self.sample_class, self.qcd_classes,
                                  self.window, log=log)

    def _read(self, root, sample, weight=True, per_file=False):
        """Events of one sample's category skim (None: no category directory); per_file: a generator
        over its files."""
        if not (root / sample / "_DONE.json").exists():
            raise FileNotFoundError(f"{root}/{sample}: no _DONE.json (skim incomplete)")
        path = root / sample / self.tree / "nominal"
        if not path.exists():
            return None
        columns = [f"proba_{c}" for c in CLASSES] + [MASS_BRANCH] + (["weight"] if weight else [])
        import numpy as np

        def events(t):
            return {"proba": np.column_stack([t[f"proba_{c}"].to_numpy() for c in CLASSES]),
                    "mass": t[MASS_BRANCH].to_numpy().astype(np.float64),
                    "weight": t["weight"].to_numpy() * self.mc_scale if weight else None, "n": t.num_rows}

        dataset = ds.dataset(path)
        if per_file:
            return (events(fragment.to_table(columns=columns)) for fragment in dataset.get_fragments())
        return events(dataset.to_table(columns=columns))

    def fill_mc(self, acc, check_fingerprint=True, log=print):
        n_tot = 0
        for sample in self.sample_class:
            if sample in self.exclude:
                log(f"  {sample}: excluded (exclude_samples)")
                continue
            ev = self._read(self.skim_dir, sample)
            if ev is None:
                log(f"  {sample}: no {self.tree} events")
                continue
            acc.fill(sample, ev["mass"], ev["weight"], proba=ev["proba"])
            n_tot += ev["n"]
            log(f"  {sample}: {ev['n']} skim events")
        log(f"Loaded MC events (fit-region skims): {n_tot}")

    def fill_data(self, acc, check_fingerprint=True, log=print):
        files = self._read(self.data_skim_dir, self.data_sample, weight=False, per_file=True)
        if files is None:
            raise FileNotFoundError(f"{self.data_skim_dir}/{self.data_sample}/{self.tree}: no data skim")
        n = 0
        for ev in files:
            acc.fill_data(ev["mass"], proba=ev["proba"])
            n += ev["n"]
        log(f"  {self.data_sample}: {n} skim events")
