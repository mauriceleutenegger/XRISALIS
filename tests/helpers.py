"""Helpers shared by the tests: unit constants, issue markers, FITS builders."""

import os

import numpy as np
import pytest
from astropy.io import fits

# Global test seed; override with XRISALIS_TEST_SEED to check that statistical
# tests do not depend on a particular seed.
SEED = int(os.environ.get("XRISALIS_TEST_SEED", 12345))

# keV -> J, the conversion used throughout the package
KEV = 1e3 * 1.602176634e-19
# arcsec -> rad, as written in images.py
ARCSEC = 2 * np.pi / (3600 * 360)


def known_bug(summary):
    """Strict xfail marker for a test that exposes a known bug in the package."""
    return pytest.mark.xfail(strict=True, reason=f"known bug: {summary}")


# ---------------------------------------------------------------------------
# Synthetic OGIP / SIMPUT file builders
# ---------------------------------------------------------------------------


def energy_grid(e_min=1.0, e_max=7.0, n=60):
    """Return (lo, hi, mid) edges of a linear energy grid in keV."""
    edges = np.linspace(e_min, e_max, n + 1)
    return edges[:-1], edges[1:], 0.5 * (edges[:-1] + edges[1:])


def write_simput(path, energies, fluxes, ra=None, dec=None):
    """Write a minimal SIMPUT file.

    energies: 1D grid (keV) shared by all sources.
    fluxes: 2D array (n_src, n_E) in photons/cm^2/s/keV.
    ra, dec: per-source positions (deg).
    """
    fluxes = np.atleast_2d(fluxes)
    n_src, n_e = fluxes.shape
    ra = np.zeros(n_src) if ra is None else np.asarray(ra, float)
    dec = np.zeros(n_src) if dec is None else np.asarray(dec, float)
    cat = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="SRC_ID", format="J", array=np.arange(1, n_src + 1)),
            fits.Column(name="RA", format="D", array=ra),
            fits.Column(name="DEC", format="D", array=dec),
        ],
        name="SRC_CAT",
    )
    spec = fits.BinTableHDU.from_columns(
        [
            fits.Column(
                name="ENERGY", format=f"{n_e}E", array=np.tile(energies, (n_src, 1))
            ),
            fits.Column(name="FLUXDENSITY", format=f"{n_e}E", array=fluxes),
        ],
        name="SPECTRUM",
    )
    fits.HDUList([fits.PrimaryHDU(), cat, spec]).writeto(path, overwrite=True)
    return path


def write_arf(path, lo, hi, areas, baselines=None):
    """Write an ARF with a total SPECRESP plus one SPECRESP_BL{i} per baseline.

    areas: 2D array (n_bl, n_E) in cm^2.
    """
    areas = np.atleast_2d(areas)
    n_bl = areas.shape[0]
    baselines = np.linspace(0.05, 1.0, n_bl) if baselines is None else baselines
    hdus = [
        fits.PrimaryHDU(),
        fits.BinTableHDU.from_columns(
            [
                fits.Column(name="ENERG_LO", format="E", array=lo),
                fits.Column(name="ENERG_HI", format="E", array=hi),
                fits.Column(name="SPECRESP", format="E", array=areas.sum(axis=0)),
            ],
            name="SPECRESP",
        ),
    ]
    for i in range(n_bl):
        name = f"SPECRESP_BL{i + 1}"
        hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="ENERG_LO", format="E", array=lo),
                fits.Column(name="ENERG_HI", format="E", array=hi),
                fits.Column(name=name, format="E", array=areas[i]),
            ],
            name=name,
        )
        hdu.header["BASELINE"] = float(baselines[i])
        hdus.append(hdu)
    fits.HDUList(hdus).writeto(path, overwrite=True)
    return path


def write_rmf(path, lo, hi, matrix=None, tlmin=0):
    """Write an RMF with a full (non-sparse) matrix; identity by default."""
    n = len(lo)
    matrix = np.eye(n) if matrix is None else np.asarray(matrix, float)
    mat = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="ENERG_LO", format="E", array=lo),
            fits.Column(name="ENERG_HI", format="E", array=hi),
            fits.Column(name="N_GRP", format="I", array=np.ones(n, int)),
            fits.Column(name="F_CHAN", format="I", array=np.full(n, tlmin)),
            fits.Column(name="N_CHAN", format="I", array=np.full(n, n)),
            fits.Column(name="MATRIX", format=f"{n}E", array=matrix),
        ],
        name="MATRIX",
    )
    mat.header["TLMIN4"] = tlmin
    ebounds = fits.BinTableHDU.from_columns(
        [
            fits.Column(name="CHANNEL", format="I", array=np.arange(n) + tlmin),
            fits.Column(name="E_MIN", format="E", array=lo),
            fits.Column(name="E_MAX", format="E", array=hi),
        ],
        name="EBOUNDS",
    )
    fits.HDUList([fits.PrimaryHDU(), mat, ebounds]).writeto(path, overwrite=True)
    return path
