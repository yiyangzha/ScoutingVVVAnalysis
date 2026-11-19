"""Blinding of data events for data/MC comparisons.

An event is blinded when it is inside any SR box (region > 0, vvv.background.qcd_massfit.assign_regions)
AND its msoftdrop_1 is inside the blind window [low, high): the category's msoftdrop_1 threshold of
the training config (blind_window; [50, 150] GeV in the current configs). Everything else may be shown:

- the fail region (outside every SR box) at any msoftdrop_1;
- SR-box events in the msoftdrop_1 sidebands (< low or >= high).

An SR-box event whose msoftdrop_1 is not finite is treated as blinded (it cannot be shown to be
in a sideband).
"""

import numpy as np

MASS_BRANCH = "ScoutingFatPFJetRecluster_msoftdrop_1"


def blind_window(training_config, category):
    """The blind window (low, high) of a category: its msoftdrop_1 threshold in a training config
    (configs/training/*.json, as a dict); a missing or open-ended threshold is an error."""
    thresholds = training_config["categories"][category]["thresholds"]
    window = thresholds.get(MASS_BRANCH)
    if (not isinstance(window, list) or len(window) != 2
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in window)
            or not window[0] < window[1]):
        raise ValueError(f"category {category}: the {MASS_BRANCH} threshold must be a closed window [low, high], "
                         f"got {window}")
    return float(window[0]), float(window[1])


def in_window(msd, window):
    """msoftdrop_1 in [low, high), i.e. the mass-fit threshold window (NaN: False)."""
    m = np.asarray(msd, dtype=np.float64)
    return (m >= window[0]) & (m < window[1])


def in_sideband(msd, window):
    """msoftdrop_1 finite and outside [low, high)."""
    m = np.asarray(msd, dtype=np.float64)
    return np.isfinite(m) & ~in_window(m, window)


def allowed_mask(region, msd, window):
    """True for events that may be shown: fail region (region == 0) at any mass, or a sideband mass."""
    region = np.asarray(region)
    return (region == 0) | in_sideband(msd, window)


def assert_blinded(region, msd, window):
    """Raise if any event is in an SR box without a sideband mass."""
    region = np.asarray(region)
    bad = (region > 0) & ~in_sideband(msd, window)
    if np.any(bad):
        raise AssertionError(f"blinding violated: {int(bad.sum())} SR-box events without a sideband msoftdrop_1")


def apply_blinding(columns, region, msd_key, window):
    """Cut a dict of equal-length numpy columns (and region) to the allowed events, then assert it."""
    keep = allowed_mask(region, columns[msd_key], window)
    out = {k: np.asarray(v)[keep] for k, v in columns.items()}
    reg = np.asarray(region)[keep]
    assert_blinded(reg, out[msd_key], window)
    return out, reg
