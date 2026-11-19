"""Storage locations in configs, without user-specific absolute paths.

Configs write ${VVV_WORK}, ${VVV_DEPOT} and ${VVV_LEGACY} instead of absolute paths; load_config
expands them from the environment, with these defaults:
- VVV_WORK:   /work/users/<user>/vvv           (outputs, models, staging)
- VVV_DEPOT:  /depot/cms/users/<user>/VVV      (skims and other large shared outputs)
- VVV_CMSSW:  <repository>/../CMSSW_16_1_0_pre4/src  (the CMSSW area with Combine)
- VVV_LEGACY: ${VVV_CMSSW}/ScoutingVVVTools  (the legacy chain)
<user> is NB_USER, else USER, else the login name. Jobs without the session's environment (Ray)
get the resolved values from the submitter (scripts/submit_training.py: environment()).
"""

import getpass
import json
import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
_VAR = re.compile(r"\$\{(VVV_[A-Z_]+)\}")


def _user():
    return os.environ.get("NB_USER") or os.environ.get("USER") or getpass.getuser()


def environment():
    """The VVV_* locations, environment over defaults."""
    cmssw = os.environ.get("VVV_CMSSW", str(REPO.parent / "CMSSW_16_1_0_pre4" / "src"))
    defaults = {"VVV_WORK": f"/work/users/{_user()}/vvv", "VVV_DEPOT": f"/depot/cms/users/{_user()}/VVV",
                "VVV_CMSSW": cmssw, "VVV_LEGACY": f"{cmssw}/ScoutingVVVTools"}
    return {k: os.environ.get(k, v) for k, v in defaults.items()}


def expand(value, env=None):
    """${VVV_*} in a string replaced (unknown names are an error), recursively in lists and dicts."""
    env = env or environment()
    if isinstance(value, str):
        def sub(m):
            if m[1] not in env:
                raise KeyError(f"unknown location variable ${{{m[1]}}} in {value!r}")
            return env[m[1]]
        return _VAR.sub(sub, value)
    if isinstance(value, list):
        return [expand(v, env) for v in value]
    if isinstance(value, dict):
        return {k: expand(v, env) for k, v in value.items()}
    return value


def load_config(path):
    """A JSON config with its ${VVV_*} locations expanded."""
    return expand(json.loads(Path(path).read_text()))
