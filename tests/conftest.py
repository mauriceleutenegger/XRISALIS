"""Shared fixtures for the XRISALIS test suite."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from helpers import SEED, energy_grid, write_arf, write_rmf, write_simput  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "src" / "xrisalis" / "models"


@pytest.fixture(autouse=True)
def _seed_and_no_show(monkeypatch):
    """Seed the global numpy RNG, make plt.show a no-op and close figures."""
    np.random.seed(SEED)
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


@pytest.fixture(scope="session")
def bundled_paths():
    return {
        "arf": MODELS_DIR / "arf.fits",
        "rmf": MODELS_DIR / "rmf.fits",
        "simput": MODELS_DIR / "double_point_source.fits",
        "notebook": REPO_ROOT / "simple_example.ipynb",
    }


@pytest.fixture
def synthetic_responses(tmp_path):
    """Flat spectrum, two-baseline ARF (ratio 1:3) and identity RMF."""
    lo, hi, mid = energy_grid()
    areas = np.vstack([np.full(mid.size, 10.0), np.full(mid.size, 30.0)])
    return {
        "lo": lo,
        "hi": hi,
        "mid": mid,
        "areas": areas,
        "flux": np.full(mid.size, 0.05),
        "simput": write_simput(tmp_path / "src.fits", mid, np.full(mid.size, 0.05)),
        "arf": write_arf(tmp_path / "arf.fits", lo, hi, areas),
        "rmf": write_rmf(tmp_path / "rmf.fits", lo, hi),
    }
