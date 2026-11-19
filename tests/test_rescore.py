import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import xgboost as xgb

from vvv.ml.inference import CLASSES, CategoryModel, score_table

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["f_a", "f_b"]


def _model_dir(tmp_path, k=2, tree="cat"):
    """A legacy-layout k-fold model directory with tiny 6-class boosters (one per fold, different seeds)."""
    rng = np.random.default_rng(1)
    x = rng.normal(size=(600, 2))
    y = (np.abs(x[:, 0] * 3 + x[:, 1]).astype(int)) % len(CLASSES)
    for r in range(k):
        d = tmp_path / "models" / tree / f"fold{r}"
        d.mkdir(parents=True)
        booster = xgb.train({"objective": "multi:softprob", "num_class": len(CLASSES), "max_depth": 2, "seed": r,
                             "subsample": 0.7}, xgb.DMatrix(x, label=y, feature_names=FEATURES), num_boost_round=3 + r)
        booster.save_model(str(d / f"{tree}_model.json"))
        (d / "branch.json").write_text(json.dumps({tree: [{"name": f} for f in FEATURES]}))
        (d / "selection.json").write_text(json.dumps({tree: {"clip_ranges": {}, "log_transform": [], "thresholds": {}}}))
        (d / "test_ranges.json").write_text(json.dumps({"kfold": k, "fold": r, "tree_name": tree}))
    return tmp_path / "models"


def _table(n=50, with_proba=False):
    rng = np.random.default_rng(2)
    cols = {"run": pa.array(np.full(n, 386000, dtype=np.uint32)),
            "event": pa.array(rng.integers(0, 10 ** 9, n).astype(np.uint64)),
            "f_a": pa.array(rng.normal(size=n).astype(np.float32)),
            "f_b": pa.array(rng.normal(size=n).astype(np.float32)),
            "ScoutingFatPFJetRecluster_msoftdrop_1": pa.array(rng.uniform(20, 250, n).astype(np.float32))}
    if with_proba:
        cols.update({f"proba_{c}": pa.array(np.full(n, -1.0)) for c in CLASSES})
    return pa.table(cols)


def test_score_table_appends_missing_proba(tmp_path):
    model = CategoryModel.load(_model_dir(tmp_path), "cat")
    t = _table()
    out = score_table(t, model)
    assert out.column_names == t.column_names + [f"proba_{c}" for c in CLASSES]
    proba = np.column_stack([out[f"proba_{c}"].to_numpy() for c in CLASSES])
    ref = model.predict({f: t[f].to_numpy() for f in FEATURES}, t["event"].to_numpy())
    np.testing.assert_array_equal(proba, ref)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert out["event"].equals(t["event"])


def test_score_table_replaces_existing_proba_in_place(tmp_path):
    model = CategoryModel.load(_model_dir(tmp_path), "cat")
    t = _table(with_proba=True)
    out = score_table(t, model)
    assert out.column_names == t.column_names
    assert np.all(out["proba_VVV"].to_numpy() >= 0)
    np.testing.assert_array_equal(out["proba_QCD"].to_numpy(), score_table(_table(), model)["proba_QCD"].to_numpy())


def test_score_table_keep_columns(tmp_path):
    model = CategoryModel.load(_model_dir(tmp_path), "cat")
    t = _table()
    out = score_table(t, model, keep=["event", "ScoutingFatPFJetRecluster_msoftdrop_1"])
    # the features need not be kept: they are read before the selection
    assert out.column_names == ["event", "ScoutingFatPFJetRecluster_msoftdrop_1"] + [f"proba_{c}" for c in CLASSES]
    np.testing.assert_array_equal(out["proba_VH"].to_numpy(), score_table(t, model)["proba_VH"].to_numpy())
    with pytest.raises(KeyError):
        score_table(t, model, keep=["event", "not_a_column"])


def test_score_table_empty(tmp_path):
    model = CategoryModel.load(_model_dir(tmp_path), "cat")
    out = score_table(_table().slice(0, 0), model)
    assert out.num_rows == 0 and "proba_Top" in out.column_names


def test_rescore_script_without_proba_and_resume(tmp_path):
    models = _model_dir(tmp_path)
    src, dest = tmp_path / "src", tmp_path / "dest"
    for sample in ("data_2024", "other"):
        d = src / sample / "cat" / "nominal"
        d.mkdir(parents=True)
        (src / sample / "_DONE.json").write_text(json.dumps({"cutflow": {"cat_skim": 100}}))
        pq.write_table(_table(60), d / "part-00000.parquet")
        pq.write_table(_table(40), d / "part-00001.parquet")
    cmd = [sys.executable, str(ROOT / "scripts" / "rescore_skims.py"), str(src), str(dest), "--category", "cat",
           "--model-dir", str(models), "--samples", "data_2024", "--columns", "run", "event",
           "ScoutingFatPFJetRecluster_msoftdrop_1", "--threads", "1", "--nthread", "2"]
    subprocess.run(cmd, check=True, capture_output=True)
    assert not (dest / "other").exists()
    out = dest / "data_2024" / "cat" / "nominal" / "part-00000.parquet"
    t = pq.read_table(out)
    assert t.num_rows == 60 and t.column_names[:3] == ["run", "event", "ScoutingFatPFJetRecluster_msoftdrop_1"]
    ref = CategoryModel.load(models, "cat").predict({f: _table(60)[f].to_numpy() for f in FEATURES},
                                                    _table(60)["event"].to_numpy())
    np.testing.assert_array_equal(t["proba_VVV"].to_numpy(), ref[:, 0])
    done = json.loads((dest / "data_2024" / "_DONE.json").read_text())
    assert done["cutflow"] == {"cat_skim": 100} and done["rescored"]["cat"]["model_dir"] == str(models)
    assert not list(out.parent.glob(".*.tmp"))
    # resume: an existing output is left as it is
    mtime = out.stat().st_mtime_ns
    res = subprocess.run(cmd + ["--skip-existing"], check=True, capture_output=True, text=True)
    assert out.stat().st_mtime_ns == mtime and "skipped" in res.stdout
