import json

import pytest

from vvv.paths import environment, expand, load_config


def test_expand_and_defaults(tmp_path, monkeypatch):
    monkeypatch.setenv("NB_USER", "alice")
    for k in ("VVV_WORK", "VVV_DEPOT", "VVV_CMSSW", "VVV_LEGACY"):
        monkeypatch.delenv(k, raising=False)
    env = environment()
    assert env["VVV_WORK"] == "/work/users/alice/vvv"
    assert env["VVV_LEGACY"] == env["VVV_CMSSW"] + "/ScoutingVVVTools"
    monkeypatch.setenv("VVV_DEPOT", "/somewhere")
    cfg = {"skim_dir": "${VVV_DEPOT}/skims", "x": ["${VVV_WORK}/a", 3], "note": "no variable"}
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg))
    out = load_config(path)
    assert out == {"skim_dir": "/somewhere/skims", "x": ["/work/users/alice/vvv/a", 3], "note": "no variable"}
    with pytest.raises(KeyError):
        expand("${VVV_NOPE}/x")
