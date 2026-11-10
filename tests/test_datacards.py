"""Unit tests of the datacard writer (vvv.stats.datacards) on small hand-built inputs; no Combine needed."""

import json
import math

import numpy as np
import pytest
import uproot

from vvv.stats.combine_run import combine_commands, csv_double, cxx_double, write_csvs
from vvv.stats.datacards import (RATIO_SPECS, CardError, FallbackCount, RegionSums, SampleSums, bernstein_basis,
                                 build_scenarios, fixed6, g10, load_config, load_json, load_registry, process_kappa,
                                 reduce_sums, slugify, write_all)

EDGES = np.array([20.0, 40.0, 60.0, 80.0])
ONE = np.array([0.0, 1.0])
SAMPLES = {"sigA": "VVV", "sigB": "VVV", "qcd1": "QCD", "top1": "Top", "top2": "Top"}
SPEC = {s.nuisance: s for s in RATIO_SPECS}


def _mass(sample, region):
    """Yields of a sample in a region (0 = fail, 1/2 = SR 1/2) per mass bin."""
    base = {"sigA": [0.5, 2.0, 1.0], "sigB": [0.0, 1.0, 0.5], "qcd1": [100.0, 80.0, 40.0],
            "top1": [3.0, 0.0, 2.0], "top2": [1.0, 1.0, 0.0]}[sample]
    scale = {0: 10.0, 1: 1.0, 2: 0.5}[region]
    if sample == "qcd1" and region > 0:
        scale = 0.01 * region
    return np.array(base) * scale


def make_inputs(tmp_path, qcd_from_mc=False, data=False, smooth=False, sr_slope=False, residual_order=0):
    """qcd_abcd_yields.root with 2 SRs and 3 mass bins, a registry stub and a sample.json."""
    root = tmp_path / "qcd_abcd_yields.root"
    sr_ids = np.array([1, 2], dtype=np.int32)
    groups = {"vvv": ["sigA", "sigB"], "qcd": ["qcd1"], "top": ["top1", "top2"]}
    with uproot.recreate(root) as f:
        f.mktree("metadata/signal_regions", {"bin_index": np.int32}).extend({"bin_index": sr_ids})
        f["metadata/sr_csv_hash"] = (np.array([float(0xABC)]), ONE)

        def bundle(prefix, regions):
            for i, sr in enumerate(sr_ids):
                y = float(np.sum(regions[i + 1]))
                for name, v in (("yield", y), ("stat_error", 0.1), ("scale_error", 0.0), ("n_entries", 10.0)):
                    f[f"{prefix}/sr{sr}/{name}"] = (np.array([v]), ONE)
            f[f"{prefix}/covariance_total"] = (np.eye(2) * 0.01, np.arange(3.0), np.arange(3.0))
            f[f"{prefix}/mean_event_weight"] = (np.array([0.1]), ONE)

        def mass(prefix, regions):
            for r, p in [(0, f"massfit/fail/{prefix}")] + [(i + 1, f"massfit/pass/{prefix}/sr{s}")
                                                            for i, s in enumerate(sr_ids)]:
                f[f"{p}/yield"] = (regions[r], EDGES)
                f[f"{p}/sumw2"] = (regions[r] * 0.04, EDGES)
                f[f"{p}/n_entries"] = (regions[r] * 25, EDGES)

        for s in SAMPLES:
            regs = [_mass(s, r) for r in range(3)]
            bundle(f"samples/{s}", regs)
            mass(f"samples/{s}", regs)
        for g, members in groups.items():
            regs = [sum(_mass(s, r) for s in members) for r in range(3)]
            bundle(f"groups/{g}", regs)
            mass(f"groups/{g}", regs)
        qcd = [_mass("qcd1", r) for r in range(3)]
        bundle("qcd_predict", qcd)
        f["qcd_predict/abcd_scale"] = (np.array([1.0]), ONE)
        f["qcd_predict/abcd_scale_error"] = (np.array([0.1]), ONE)
        mass("qcd_true", qcd)
        if qcd_from_mc:
            settings = {"enabled": 1.0, "qcd_from_mc": 1.0}
        else:
            settings = {"enabled": 1.0, "tfmc_order": 1.0, "residual_order": float(residual_order), "tfmc_free": 1.0,
                        "bern_low": 0.0, "bern_high": 10.0, "n_eig": 1.0}
            f["massfit/tfmc/x"] = (np.array([-2 / 3, 0.0, 2 / 3]), EDGES)
            f["massfit/tfmc/shape"] = (np.array([0.01, 0.02, 0.03]), EDGES)
            f["massfit/tfmc/eig0"] = (np.array([0.1, -0.05, 0.2]), EDGES)
            f["massfit/tfmc/n0"] = (np.array([1.0, 0.5]), np.arange(3.0))
            if sr_slope:
                f["massfit/sr_slope"] = (np.array([0.0, 1.0]), np.arange(3.0))
        for k, v in settings.items():
            f[f"massfit/settings/{k}"] = (np.array([v]), ONE)
        if smooth:
            f["massfit/smooth/top/factor"] = (np.array([2.0]), ONE)
            for s in sr_ids:
                f[f"massfit/smooth/top/sr{s}/yield"] = (np.array([4.0, 2.0, 2.0]), EDGES)
                f[f"massfit/smooth/top/sr{s}/sumw2"] = (np.array([0.4, 0.2, 0.2]), EDGES)
        if data:
            for k, v in (("has_data", 1.0), ("blind_low", 40.0), ("blind_high", 60.0)):
                f[f"massfit/settings/{k}"] = (np.array([v]), ONE)
            f["massfit/fail/data/yield"] = (np.array([1000.0, 800.0, 400.0]), EDGES)
            for s in sr_ids:
                f[f"massfit/pass/data/sr{s}/yield"] = (np.array([5.0, 0.0, 3.0]), EDGES)
                f[f"massfit/blinded/sr{s}"] = (np.array([0.0, 1.0, 0.0]), EDGES)
    (tmp_path / "sample.json").write_text(json.dumps({"sample": [
        {"name": s, "is_MC": True, "is_signal": c == "VVV", "has_theory_weights": c != "QCD"}
        for s, c in SAMPLES.items()]}))
    reg = tmp_path / "registry"
    reg.mkdir()
    (reg / "config.json").write_text(json.dumps({"sample_config": str(tmp_path / "sample.json"), "class_groups": {
        "VVV": ["sigA", "sigB"], "Top": ["top1", "top2"], "QCD": ["qcd1"]}}))
    (tmp_path / "xsec.json").write_text(json.dumps({"uncertainties": {"sig*": 0.1, "top*": 0.05, "qcd*": 0.5}}))
    return root, reg


def make_config(tmp_path, root, reg, **extra):
    cfg = {"channels": [{"name": "cat", "root_file": str(root), "bdt_root": str(reg)}], "output_dir": "out",
           "lumi_unc": 0.016, "trigger_unc": 0.02, "xsec_unc_json": "xsec.json", "mc_stat_threshold": 10,
           "enabled_nuisances": ["lumi", "trigger", "xsec", "autoMCStats"], "scenario_scopes": ["combined"]}
    cfg.update(extra)
    path = tmp_path / "combine.json"
    path.write_text(json.dumps(cfg))
    return path


def card(manifest, tag):
    run = next(r for r in manifest["runs"] if r["tag"] == tag)
    return open(run["cards"][0]).read().splitlines(), run


def row(lines, name):
    return next(line.split()[2:] for line in lines if line.split()[:1] == [name])


def cell(lines, name, bin_name, proc):
    """The value of row name in the (bin, process) cell."""
    keys = list(zip(lines[10].split()[1:], lines[11].split()[1:]))
    return row(lines, name)[keys.index((bin_name, proc))]


# -------------------- formatting, parsing --------------------


def test_formatting_as_cxx():
    assert slugify("VVV") == "vvv" and slugify("_V+jets_") == "v_jets" and slugify("GluGluZH-Zto2Q") == "gluglu" \
        "zh_zto2q"
    assert fixed6(1.016) == "1.016000" and fixed6(1.0 + 0.1 / 3) == "1.033333"
    assert g10(0.1) == "0.1" and g10(1e-5) == "1e-05" and g10(2 / 3) == "0.6666666667" and g10(-0.0) == "-0"
    assert cxx_double(0.956994123) == "0.956994" and cxx_double(math.nan) == "nan"
    assert csv_double(math.inf) == "inf" and csv_double(2.1875) == "2.1875"


def test_json_sorted_keys_and_long_double(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"b": 1, "A": {"z": 0.1, "a": 2}, "a": [3, 1e-3]}')
    d = load_json(p)
    assert list(d) == ["A", "a", "b"] and list(d["A"]) == ["a", "z"]  # std::map order
    assert d["A"]["z"] == float(np.longdouble("0.1")) and d["a"] == [3.0, 0.001]


def test_bernstein_basis_partition_of_unity():
    for n in range(4):
        for t in (0.0, 0.3, 1.0):
            assert sum(bernstein_basis(n, v, t) for v in range(n + 1)) == pytest.approx(1.0)
    assert bernstein_basis(2, 1, 0.5) == 0.5


# -------------------- kappas --------------------


def _sums(y, w2, varied, varied_w2, regions=None, n=10, pdf_method=""):
    rs = RegionSums(y, w2, list(varied), list(varied_w2))
    return SampleSums(rs, regions or {}, {}, {}, [], "", n, pdf_method)


def test_reduce_sums():
    k = reduce_sums(SPEC["jes"], RegionSums(10.0, 1.0, [11.0, 9.5], [1.0, 1.0]), "")
    assert (k.up, k.down) == (1.1, 0.95)
    scale = [10.0, 10.5, 99.0, 9.0, 10.0, 10.2, 0.1, 10.1, 9.8]  # corners 2 and 6 ignored
    k = reduce_sums(SPEC["theory_scale"], RegionSums(10.0, 1.0, scale, [1.0] * 9), "")
    assert (k.up, k.down) == (1.05, 0.9)
    k = reduce_sums(SPEC["theory_ps_isr"], RegionSums(10.0, 1.0, [9.0, 8.0, 12.0, 13.0], [1.0] * 4), "")
    assert (k.up, k.down) == (1.2, 0.9)
    k = reduce_sums(SPEC["theory_ps_fsr"], RegionSums(10.0, 1.0, [9.0, 8.0, 12.0, 13.0], [1.0] * 4), "")
    assert (k.up, k.down) == (1.3, 0.8)
    pdf = [10.0] + [10.0 + (0.3 if i == 0 else 0.0) for i in range(100)] + [9.8, 10.2]
    k = reduce_sums(SPEC["theory_pdf"], RegionSums(10.0, 1.0, pdf, [1.0] * 103), "hessian_symmetric")
    total = math.sqrt(0.03 ** 2 + 0.02 ** 2)
    assert k.up == pytest.approx(1.0 + total) and k.down == pytest.approx(1.0 - total)


def test_process_kappa_merges_samples_and_falls_back():
    spec = SPEC["jes"]
    info = {s: type("I", (), {"has_theory_weights": True})() for s in ("a", "b")}
    # SR 1: a alone has n_eff 4 (< 10), a + b merged 32: own kappa of the merged sums
    sums = {"a": {"cat": _sums(100.0, 1.0, [110.0, 90.0], [1.0, 1.0],
                               {1: RegionSums(2.0, 1.0, [3.0, 1.5], [1.0, 1.0])})},
            "b": {"cat": _sums(100.0, 1.0, [100.0, 100.0], [1.0, 1.0],
                               {1: RegionSums(6.0, 0.125, [6.0, 6.0], [0.125, 0.125])})}}
    count = FallbackCount()
    k = process_kappa(spec, sums, "cat", "p", ["a", "b"], info, lambda s: 1.0, 1.0, "window", 1, -1, count)
    assert (k.up, k.down) == (9.0 / 8.0, 7.5 / 8.0) and count.inclusive == 0
    # a alone: falls back to its inclusive ratios
    k = process_kappa(spec, sums, "cat", "p", ["a"], info, lambda s: 1.0, 1.0, "window", 1, -1, count)
    assert (k.up, k.down) == (1.1, 0.9) and count.inclusive == 1 and count.cells == 2
    # no yield in the cell: inactive; a sample with yield but no sums: an error
    assert not process_kappa(spec, sums, "cat", "p", ["a"], info, lambda s: 1.0, 0.0, "window", 1, -1, count).active
    with pytest.raises(CardError, match="no sums for sample 'c'"):
        process_kappa(spec, sums, "cat", "p", ["a", "c"], info, lambda s: 1.0, 1.0, "window", 1, -1, count)


# -------------------- cards --------------------


def test_scenarios_and_registry_order(tmp_path):
    root, reg = make_inputs(tmp_path)
    cfg = load_config(make_config(tmp_path, root, reg), log=lambda m: None)
    r = load_registry(cfg, log=lambda m: None)
    assert r.class_order == ["QCD", "Top", "VVV"]  # std::map order, not the JSON order
    assert r.signal_classes == {"VVV"} and r.signal_samples == ["sigA", "sigB"] and r.qcd_classes == ["QCD"]
    assert [(s.scope, s.name) for s in build_scenarios(r)] == [("combined", "combined"), ("class", "VVV"),
                                                                ("sample", "sigA"), ("sample", "sigB")]


def test_fitted_qcd_card(tmp_path):
    root, reg = make_inputs(tmp_path, residual_order=2, sr_slope=True)
    m = write_all(load_config(make_config(tmp_path, root, reg), log=lambda m: None), log=lambda m: None)
    assert [r["tag"] for r in m["runs"]] == ["mc_combined_combined", "abcd_combined_combined",
                                             "channel_cat_mc_combined_combined", "channel_cat_abcd_combined_combined"]
    lines, run = card(m, "abcd_combined_combined")
    assert run["slope_params"] == ["n_cat_sr2_slope"] and run["masked_bins"] == []
    assert lines[0].startswith("#") and lines[1] == "imax 9"
    bins = lines[7].split()[1:]
    assert bins == [f"cat_{r}_m{j:02d}" for r in ("fail", "sr1", "sr2") for j in range(3)]
    procs = lines[11].split()[1:]
    assert procs[:3] == ["signal", "bkg_qcd", "bkg_top"]
    # pass bin m01 of SR 1: top2 alone (top1 empty), signal, fitted QCD
    assert lines[10].split().count("cat_sr1_m01") == 3
    assert cell(lines, "lumi", "cat_fail_m00", "bkg_top") == "-"  # no MC-normalisation rows in the fail bins
    assert [cell(lines, "lumi", "cat_sr1_m00", p) for p in ("signal", "bkg_qcd", "bkg_top")] == [
        "1.016000", "-", "1.016000"]
    assert cell(lines, "xsec_signal", "cat_sr1_m00", "signal") == "1.100000"
    assert "xsec_bkg_qcd" not in [line.split()[0] for line in lines]
    assert "* autoMCStats 10" in lines
    assert "n_cat_sr1 extArg 1 [0,100]" in lines and "n_cat_sr2_slope extArg 0 [-10,10]" in lines
    assert "tf_cat_e0 extArg 0 [-20,20]" in lines and "bern_cat_2 extArg 1 [0,10]" in lines
    assert "qcdF_cat_m00 rateParam cat_fail_m00 bkg_qcd 1 [0,10]" in lines
    t = 0.5 * (2 / 3 + 1.0)
    formula = (f"(@0*@1*exp((0.2)*@2)*({g10(bernstein_basis(2, 0, t))}+({g10(bernstein_basis(2, 1, t))})*@3"
               f"+({g10(bernstein_basis(2, 2, t))})*@4)*exp(({g10(2 / 3)})*@5))")
    assert (f"qcd_cat_sr2_m02 rateParam cat_sr2_m02 bkg_qcd {formula} qcdF_cat_m02,n_cat_sr2,tf_cat_e0,bern_cat_1,"
            "bern_cat_2,n_cat_sr2_slope") in lines
    # the shape file: fitted QCD template n0 x F x shape, no error; data_obs = the Asimov sum
    with uproot.open(run["work"] + "/shapes_cat.root") as f:
        h = f["cat_sr2_m02/bkg_qcd"]
        assert h.values()[0] == 0.5 * 40.0 * 10.0 * 0.03 and h.variances()[0] == 0.0
        assert f["cat_fail_m00/bkg_qcd"].values()[0] == 1000.0 and f["cat_fail_m00/bkg_qcd"].variances()[0] == 0.0
        assert f["cat_sr1_m00/signal"].variances()[0] == pytest.approx(0.5 * 0.04)
        assert f["cat_sr1_m00/data_obs"].values()[0] == pytest.approx(0.5 + 1.0 * 1000 * 0.01 + 3.0 + 1.0)
    # MC mode: the QCD MC in the pass bins (error sqrt(sumw2)), the fail bins with free QCD
    lines, _ = card(m, "mc_combined_combined")
    assert "extArg" not in "\n".join(lines) and "qcdF_cat_m01 rateParam cat_fail_m01 bkg_qcd 1 [0,10]" in lines
    assert cell(lines, "xsec_bkg_qcd", "cat_sr1_m00", "bkg_qcd") == "1.500000"
    assert cell(lines, "xsec_bkg_qcd", "cat_fail_m00", "bkg_qcd") == "-"


def test_qcd_from_mc_smooth_and_observed(tmp_path):
    root, reg = make_inputs(tmp_path, qcd_from_mc=True, smooth=True, data=True)
    cfg = make_config(tmp_path, root, reg, data_mode="observed", scenario_scopes=["combined", "sample"])
    m = write_all(load_config(cfg, log=lambda m: None), log=lambda m: None)
    lines, run = card(m, "abcd_combined_combined")
    bins = lines[7].split()[1:]
    assert bins == [f"cat_sr{s}_m{j:02d}" for s in (1, 2) for j in range(3)]  # pass bins only
    assert run["masked_bins"] == ["ch1_cat_sr1_m01", "ch1_cat_sr2_m01"]
    assert "rateParam" not in "\n".join(lines)
    # smoothed top: box total over the card bins (7 in SR 1) x the loosened shape (4, 2, 2)/8, mcnorm row
    with uproot.open(run["work"] + "/shapes_cat.root") as f:
        assert [f[f"cat_sr1_m{j:02d}/bkg_top"].values()[0] for j in range(3)] == [3.5, 1.75, 1.75]
        assert f["cat_sr1_m00/bkg_top"].variances()[0] == pytest.approx(3.5 ** 2 * 0.4 / 16)
        assert [f[f"cat_sr1_m{j:02d}/data_obs"].values()[0] for j in range(3)] == [5.0, 0.0, 3.0]
    norm = fixed6(1.0 + math.sqrt(7 * 0.04) / 7)
    assert row(lines, "mcnorm_cat_sr1_bkg_top").count(norm) == 3
    # smoothed xsec kappa: weights of the box totals over the card bins (top1 5, top2 2 in SR 1)
    assert cell(lines, "xsec_bkg_top", "cat_sr1_m01", "bkg_top") == "1.050000"
    # sample scope: other signal samples are backgrounds, per-sample processes
    lines, _ = card(m, "abcd_sample_siga")
    assert lines[11].split()[1:4] == ["signal", "bkg_qcd1", "bkg_top1"]
    assert "bkg_sigb" in lines[11].split()
    assert "mcnorm" not in "\n".join(lines)  # single samples keep their box templates
    cmds = dict(combine_commands(m, run))
    assert "--channel-masks" in cmds["postfit_fit"] and "mask_ch1_cat_sr1_m01=1" in cmds["postfit_fit"]
    assert cmds["significance"].endswith("-M Significance -t -1 --expectSignal 1 -m 120 --rMin -5 --rMax 10 "
                                         "--cminDefaultMinimizerStrategy 0 --setParameters 'rgx{zc_.*}=0' -n "
                                         "'sig_abcd_combined_combined' 'datacard.txt'")


def test_syst_rows_window_and_errors(tmp_path):
    root, reg = make_inputs(tmp_path, qcd_from_mc=True)
    entry = {"yield": 1.0, "sumw2": 0.01, "yield_jes_up": 1.1, "yield_jes_down": 0.9, "sumw2_jes_up": 0.01,
             "sumw2_jes_down": 0.01, "min_region_events": 10, "sr_csv_hash": "0000000000abc",
             "regions": {"1": {"yield": 1.0, "sumw2": 0.01, "yield_jes_up": 1.2, "yield_jes_down": 0.7,
                               "sumw2_jes_up": 0.01, "sumw2_jes_down": 0.01},
                         "2": {"yield": 1.0, "sumw2": 0.5, "yield_jes_up": 2.0, "yield_jes_down": 0.5,
                               "sumw2_jes_up": 0.5, "sumw2_jes_down": 0.5}}}
    (tmp_path / "jes.json").write_text(json.dumps({s: {"cat": entry} for s in SAMPLES}))
    cfg = make_config(tmp_path, root, reg, enabled_nuisances=["jes"], jes_syst_json="jes.json")
    m = write_all(load_config(cfg, log=lambda m: None), log=lambda m: None)
    lines, _ = card(m, "mc_combined_combined")
    # SR 1: own merged kappa; SR 2: n_eff 2 per sample -> merged signal 4 < 10 -> inclusive
    assert cell(lines, "jes", "cat_sr1_m00", "signal") == "0.700000/1.200000"
    assert cell(lines, "jes", "cat_sr2_m00", "signal") == "0.900000/1.100000"
    assert cell(lines, "jes", "cat_sr1_m00", "bkg_qcd") == "0.700000/1.200000"
    assert "autoMCStats" not in "\n".join(lines)
    # SR CSV hash mismatch is fatal, as in combine.C
    entry["sr_csv_hash"] = "0000000000abd"
    (tmp_path / "jes.json").write_text(json.dumps({s: {"cat": entry} for s in SAMPLES}))
    with pytest.raises(CardError, match="other signal regions"):
        write_all(load_config(cfg, log=lambda m: None), log=lambda m: None)


def test_zero_signal_and_csvs(tmp_path):
    root, reg = make_inputs(tmp_path, qcd_from_mc=True)
    m = write_all(load_config(make_config(tmp_path, root, reg, scenario_scopes=["sample"]), log=lambda m: None),
                  log=lambda m: None)
    # sigB has no yield in bin m00 but some elsewhere: a card; all runs ok
    assert {r["status"] for r in m["runs"]} == {"ok"}
    res = {r["tag"]: {"significance": 1.5, "limits": {k: 2.0 for k in ("exp_2p5", "exp_16", "exp_50", "exp_84",
                                                                       "exp_97p5")}} for r in m["runs"]}
    write_csvs(m, res, tmp_path, log=lambda m: None)
    assert (tmp_path / "significance.csv").read_text() == "scope,name,significance\nsample,sigA,1.5\nsample,sigB,1.5\n"
    assert (tmp_path / "limits_by_channel_abcd_mc.csv").read_text().splitlines()[1] == "cat,sample,sigA,2,2,2,2,2"


def test_not_ported_one_bin_cards(tmp_path):
    root, reg = make_inputs(tmp_path)
    with uproot.update(root) as f:
        f["massfit/settings/enabled"] = (np.array([0.0]), ONE)
    with pytest.raises(NotImplementedError, match="one-bin ABCD"):
        write_all(load_config(make_config(tmp_path, root, reg), log=lambda m: None), log=lambda m: None)
