import numpy as np
import pytest

from vvv.validation.blinding import BLIND_WINDOW, allowed_mask, apply_blinding, assert_blinded, in_sideband


def test_allowed_mask_semantics():
    msd = np.array([20.0, 49.999, 50.0, 100.0, 149.999, 150.0, 250.0, np.nan, 100.0, np.nan])
    region = np.array([1, 1, 1, 2, 3, 1, 2, 1, 0, 0])
    got = allowed_mask(region, msd)
    assert got.tolist() == [True, True, False, False, False, True, True, False, True, True]
    assert in_sideband(msd).tolist() == [True, True, False, False, False, True, True, False, False, False]
    assert BLIND_WINDOW == (50.0, 150.0)


def test_apply_blinding_cuts_and_asserts():
    rng = np.random.default_rng(1)
    n = 100000
    msd = rng.uniform(0, 300, n).astype(np.float32)
    region = rng.integers(0, 4, n)
    cols = {"m": msd, "x": np.arange(n)}
    out, reg = apply_blinding(cols, region, "m")
    m = out["m"].astype(np.float64)
    assert not np.any((reg > 0) & (m >= 50) & (m < 150))
    # nothing allowed was dropped
    expected = (region == 0) | (msd < 50) | (msd >= 150)
    assert np.array_equal(out["x"], np.flatnonzero(expected))
    with pytest.raises(AssertionError):
        assert_blinded(region, msd)
