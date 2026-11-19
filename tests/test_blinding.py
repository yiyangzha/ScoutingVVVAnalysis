import json
from pathlib import Path

import numpy as np
import pytest

from vvv.validation.blinding import MASS_BRANCH, allowed_mask, apply_blinding, assert_blinded, blind_window, in_sideband

WINDOW = (50.0, 150.0)


def test_allowed_mask_semantics():
    msd = np.array([20.0, 49.999, 50.0, 100.0, 149.999, 150.0, 250.0, np.nan, 100.0, np.nan])
    region = np.array([1, 1, 1, 2, 3, 1, 2, 1, 0, 0])
    got = allowed_mask(region, msd, WINDOW)
    assert got.tolist() == [True, True, False, False, False, True, True, False, True, True]
    assert in_sideband(msd, WINDOW).tolist() == [True, True, False, False, False, True, True, False, False, False]
    # another window moves the edges
    assert in_sideband(msd, (40.0, 100.0)).tolist() == [True, False, False, True, True, True, True, False, True, False]


def test_apply_blinding_cuts_and_asserts():
    rng = np.random.default_rng(1)
    n = 100000
    msd = rng.uniform(0, 300, n).astype(np.float32)
    region = rng.integers(0, 4, n)
    cols = {"m": msd, "x": np.arange(n)}
    out, reg = apply_blinding(cols, region, "m", WINDOW)
    m = out["m"].astype(np.float64)
    assert not np.any((reg > 0) & (m >= 50) & (m < 150))
    # nothing allowed was dropped
    expected = (region == 0) | (msd < 50) | (msd >= 150)
    assert np.array_equal(out["x"], np.flatnonzero(expected))
    with pytest.raises(AssertionError):
        assert_blinded(region, msd, WINDOW)


def test_blind_window_from_training_config():
    tcfg = json.loads((Path(__file__).resolve().parents[1] / "configs" / "training" / "baseline.json").read_text())
    for cat in tcfg["categories"]:
        assert blind_window(tcfg, cat) == WINDOW
    for bad in (None, [50, None], [None, 150], [150, 50], [50]):
        cfg = {"categories": {"c": {"thresholds": {} if bad is None else {MASS_BRANCH: bad}}}}
        with pytest.raises(ValueError, match="closed window"):
            blind_window(cfg, "c")
