"""Package-level smoke tests and checks of the bundled model files.

Covers: importability of all `xrisalis` submodules and the structure of
`src/xrisalis/models/{arf,rmf,double_point_source}.fits`.
Fixtures: `bundled_paths`.
"""

import importlib

import numpy as np
import pytest
from astropy.io import fits
from matplotlib.colors import Colormap


@pytest.mark.parametrize("name", ["instrument", "images", "process", "analysis"])
def test_submodule_imports(name):
    """Each submodule imports without error."""
    importlib.import_module(f"xrisalis.{name}")


def test_xri_cmap_builds():
    """The module-level colormap is a valid Colormap with 10^4 levels."""
    from xrisalis import analysis

    assert isinstance(analysis.xri_cmap, Colormap)
    assert analysis.xri_cmap.N == 10_000


def test_bundled_arf_structure(bundled_paths):
    """ARF has SPECRESP plus SPECRESP_BL1..10 with BASELINE 0.05 -> 1.0 m on a shared grid."""
    with fits.open(bundled_paths["arf"]) as h:
        names = [hdu.name for hdu in h[1:]]
        assert names == ["SPECRESP"] + [f"SPECRESP_BL{i}" for i in range(1, 11)]
        baselines = [h[f"SPECRESP_BL{i}"].header["BASELINE"] for i in range(1, 11)]
        lo = h["SPECRESP"].data["ENERG_LO"]
        for i in range(1, 11):
            assert np.array_equal(h[f"SPECRESP_BL{i}"].data["ENERG_LO"], lo)
    assert baselines[0] == pytest.approx(0.05)
    assert baselines[-1] == pytest.approx(1.0)
    assert np.all(np.diff(baselines) > 0)


def test_bundled_arf_total_is_sum_of_baselines(bundled_paths):
    """Total SPECRESP equals the sum of the per-baseline effective areas (float32 precision)."""
    with fits.open(bundled_paths["arf"]) as h:
        total = h["SPECRESP"].data["SPECRESP"]
        summed = sum(h[f"SPECRESP_BL{i}"].data[f"SPECRESP_BL{i}"] for i in range(1, 11))
    np.testing.assert_allclose(summed, total, rtol=1e-5)


def test_bundled_rmf_structure(bundled_paths):
    """RMF has MATRIX and EBOUNDS on the ARF energy grid, channels 0..950, no negative entries."""
    with (
        fits.open(bundled_paths["rmf"]) as h,
        fits.open(bundled_paths["arf"]) as a,
    ):
        matrix = h["MATRIX"].data
        ebounds = h["EBOUNDS"].data
        assert matrix["MATRIX"].shape == (951, 951)
        np.testing.assert_array_equal(ebounds["CHANNEL"], np.arange(951))
        np.testing.assert_allclose(matrix["ENERG_LO"], a["SPECRESP"].data["ENERG_LO"])
        assert not (matrix["MATRIX"] < 0).any()


def test_bundled_simput_structure(bundled_paths):
    """SIMPUT has 2 sources ~400 uas apart; spectra share a grid whose points lie in the ARF bins."""
    with (
        fits.open(bundled_paths["simput"]) as h,
        fits.open(bundled_paths["arf"]) as a,
    ):
        cat = h["SRC_CAT"].data
        energies = h["SPECTRUM"].data["ENERGY"]
        lo = a["SPECRESP"].data["ENERG_LO"]
        hi = a["SPECRESP"].data["ENERG_HI"]
    assert len(cat) == 2
    sep_uas = np.hypot(np.diff(cat["RA"]), np.diff(cat["DEC"]))[0] * 3600e6
    assert sep_uas == pytest.approx(400, rel=0.01)
    assert np.array_equal(energies[0], energies[1])
    assert np.all((energies[0] >= lo) & (energies[0] <= hi))
