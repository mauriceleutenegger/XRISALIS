"""Tests for `xrisalis.images`: the image container and all photon generators.

Helpers: `KEV`, `ARCSEC`, `known_bug`. Synthetic PNG and FITS inputs are written
to `tmp_path`. Statistical checks use binomial / normal 5-sigma bounds.
"""

import numpy as np
import pandas as pd
import pytest
from astropy.io import fits
from helpers import ARCSEC, KEV, known_bug
from PIL import Image

from xrisalis import images

DEG = np.pi / 180


def assert_binomial(count, n, p, nsigma=5):
    """count is consistent with Binomial(n, p) within nsigma."""
    sigma = np.sqrt(n * p * (1 - p))
    assert abs(count - n * p) <= nsigma * sigma, (count, n * p, sigma)


def energy_is(im, e_kev):
    """Boolean mask of photons whose energy equals e_kev (relative comparison; energies in J)."""
    return np.isclose(im.energies, e_kev * KEV, rtol=1e-9, atol=0)


def flat_spectrum(e_min=1.0, e_max=7.0, n=13):
    """Two-column (energy keV, flux) spectrum array."""
    e = np.linspace(e_min, e_max, n)
    return np.column_stack([e, np.ones(n)])


def write_png(path, array):
    """Save a uint8 grayscale array as PNG and return its path."""
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="L").save(path)
    return str(path)


# ---------------------------------------------------------------------------
# image container
# ---------------------------------------------------------------------------


def test_image_container_defaults():
    """image(size) allocates zeroed arrays of the right shape and empty metadata."""
    im = images.image(7)
    assert im.size == 7
    assert im.toa.shape == (7,) and im.toa.dtype.kind == "i" and not im.toa.any()
    assert im.energies.shape == (7,) and not im.energies.any()
    assert im.loc.shape == (7, 2) and not im.loc.any()
    assert im.spectrum is None and im.baseline_indices is None
    assert im.bkg_indices == [] and im.bkg_spect is None


# ---------------------------------------------------------------------------
# point sources
# ---------------------------------------------------------------------------


def test_point_source_units_and_times():
    """point_source: energies keV -> J, offsets arcsec -> rad, toa = 0..size-1."""
    im = images.point_source(50, 3.0, -2.0, 6.4)
    np.testing.assert_allclose(im.energies, 6.4 * KEV)
    np.testing.assert_allclose(im.loc, np.tile([3.0 * ARCSEC, -2.0 * ARCSEC], (50, 1)))
    np.testing.assert_array_equal(im.toa, np.arange(50))
    assert im.spectrum is None


def test_point_source_spectrum_array_and_default_energy():
    """With a spectrum array and energy=None the mean spectrum energy is used."""
    spec = flat_spectrum()
    im = images.point_source(10, 0, 0, None, spectrum=spec)
    assert im.spectrum is spec
    np.testing.assert_allclose(im.energies, spec[:, 0].mean() * KEV)


def test_point_source_spectrum_from_file(tmp_path):
    """A spectrum file path is loaded with np.loadtxt."""
    spec = flat_spectrum()
    path = tmp_path / "spec.txt"
    np.savetxt(path, spec)
    im = images.point_source(10, 0, 0, 2.0, spectrum=str(path))
    np.testing.assert_allclose(im.spectrum, spec)
    np.testing.assert_allclose(im.energies, 2.0 * KEV)


@pytest.mark.parametrize(
    "generator, args",
    [
        (images.point_source, (10, 0, 0)),
        (images.double_point_source, (10, [0, 1], [0, 1])),
        (images.m_point_sources, (10, 2, [0, 1], [0, 1])),
        (images.point_source_multichromatic_range, (10, 0, 0)),
        (images.disc, (10, 0, 0)),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
def test_generators_require_energy_or_spectrum(generator, args):
    """Generators raise when neither energy nor spectrum is given."""
    kwargs = {"radius": 1.0} if generator is images.disc else {}
    with pytest.raises(Exception, match="spectrum or energy"):
        generator(*args, None, **kwargs)


@pytest.mark.parametrize(
    "generator, args, energy",
    [
        (images.double_point_source, (20, [0, 1], [0, 0]), [1.0, 2.0]),
        (images.m_point_sources, (20, 2, [0, 1], [0, 0]), [1.0, 2.0]),
        (images.point_source_multichromatic_range, (20, 0, 0), [1.0, 2.0]),
        (images.disc, (20, 0, 0), 1.5),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
@pytest.mark.parametrize("as_file", [True, False], ids=["file", "array"])
def test_generators_store_spectrum(tmp_path, generator, args, energy, as_file):
    """Generators store a spectrum given as array or file path; explicit energies still apply."""
    spec = flat_spectrum()
    path = tmp_path / "spec.txt"
    np.savetxt(path, spec)
    kwargs = {"radius": 1.0} if generator is images.disc else {}
    im = generator(*args, energy, spectrum=str(path) if as_file else spec, **kwargs)
    np.testing.assert_allclose(im.spectrum, spec)
    e = im.energies / KEV
    assert e.min() >= 1.0 - 1e-9 and e.max() <= 2.0 + 1e-9


def test_disc_spectrum_default_energy():
    """disc with a spectrum and energy=None uses the mean spectrum energy."""
    spec = flat_spectrum()
    im = images.disc(10, 0, 0, None, radius=1.0, spectrum=spec)
    np.testing.assert_allclose(im.energies, spec[:, 0].mean() * KEV)


def test_generate_from_image_spectrum_from_file(two_pixel_png, tmp_path):
    """generate_from_image loads a spectrum given as a file path."""
    spec = flat_spectrum()
    path = tmp_path / "spec.txt"
    np.savetxt(path, spec)
    im, _ = images.generate_from_image(two_pixel_png, 10, 1.0, 2.0, spectrum=str(path))
    np.testing.assert_allclose(im.spectrum, spec)


def test_double_point_source_assignment_and_fractions():
    """double_point_source: energy and location of each photon belong to the same source;
    each source gets ~1/2 of the photons (binomial 5 sigma)."""
    n = 4000
    im = images.double_point_source(n, [1.0, -4.0], [2.0, 0.5], [1.0, 6.0])
    first = energy_is(im, 1.0)
    second = energy_is(im, 6.0)
    assert np.all(first ^ second)
    np.testing.assert_allclose(
        im.loc[first], [[1.0 * ARCSEC, 2.0 * ARCSEC]] * first.sum()
    )
    np.testing.assert_allclose(
        im.loc[second], [[-4.0 * ARCSEC, 0.5 * ARCSEC]] * second.sum()
    )
    np.testing.assert_array_equal(im.toa, np.arange(n))
    assert_binomial(first.sum(), n, 0.5)


def test_m_point_sources_fractions():
    """m_point_sources with m=3: each source gets ~1/3 of the photons (binomial 5 sigma)."""
    n = 6000
    energies = [1.0, 2.0, 3.0]
    im = images.m_point_sources(n, 3, [0, 1, 2], [0, 0, 0], energies)
    for k, e in enumerate(energies):
        sel = energy_is(im, e)
        assert_binomial(sel.sum(), n, 1 / 3)
        np.testing.assert_allclose(im.loc[sel, 0], k * ARCSEC)


@pytest.mark.parametrize(
    "generator, args",
    [
        (images.double_point_source, (100, [0, 1], [0, 0])),
        (images.m_point_sources, (100, 2, [0, 1], [0, 0])),
        (images.point_source_multichromatic_range, (100, 0, 0)),
    ],
    ids=lambda x: getattr(x, "__name__", ""),
)
@known_bug("energy=None with spectrum gives a scalar energy")
def test_multi_generators_accept_spectrum_without_energy(generator, args):
    """With a spectrum and energy=None the generators still produce finite energies."""
    im = generator(*args, None, spectrum=flat_spectrum())
    assert np.all(np.isfinite(im.energies)) and np.all(im.energies > 0)


def test_multichromatic_range_uniform():
    """Energies are uniform in [E0, E1]: within bounds, mean (E0+E1)/2 and
    variance (E1-E0)^2/12 (5 sigma on the mean, 5 % on the variance for n=20000)."""
    n, e0, e1 = 20_000, 2.0, 8.0
    im = images.point_source_multichromatic_range(n, 1.0, 1.0, [e0, e1])
    e = im.energies / KEV
    assert e.min() >= e0 and e.max() <= e1
    assert abs(e.mean() - (e0 + e1) / 2) <= 5 * (e1 - e0) / np.sqrt(12 * n)
    assert e.var() == pytest.approx((e1 - e0) ** 2 / 12, rel=0.05)
    np.testing.assert_allclose(im.loc, ARCSEC)


def test_multichromatic_gauss_mean_and_spread():
    """Energies ~ N(E, sigma) in keV (converted to J): mean within 5 sigma/sqrt(n),
    std within 3 % (relative s.e. 0.5 % for n=20000)."""
    n, e, s = 20_000, 6.4, 0.2
    im = images.point_source_multichromatic_gauss(n, 0.0, 2.0, e, s)
    en = im.energies / KEV
    assert abs(en.mean() - e) <= 5 * s / np.sqrt(n)
    assert en.std() == pytest.approx(s, rel=0.03)
    np.testing.assert_allclose(im.loc, np.tile([0.0, 2.0 * ARCSEC], (n, 1)))
    np.testing.assert_array_equal(im.toa, np.arange(n))


# ---------------------------------------------------------------------------
# disc
# ---------------------------------------------------------------------------


def disc_radii(im, alpha, beta):
    return np.hypot(im.loc[:, 0] - alpha * ARCSEC, im.loc[:, 1] - beta * ARCSEC)


def test_disc_photons_within_radius():
    """All disc photons lie within `radius` of the centre; energies are monochromatic."""
    im = images.disc(5000, 1.0, -1.0, 3.0, radius=0.5)
    assert disc_radii(im, 1.0, -1.0).max() <= 0.5 * ARCSEC * (1 + 1e-12)
    np.testing.assert_allclose(im.energies, 3.0 * KEV)
    np.testing.assert_array_equal(im.toa, np.arange(5000))


def test_disc_centroid():
    """The disc centroid is at the centre (5 sigma; sigma_x <= R/sqrt(n))."""
    n, r = 5000, 0.5
    im = images.disc(n, 1.0, -1.0, 3.0, radius=r)
    centroid = im.loc.mean(axis=0) / ARCSEC
    assert np.all(np.abs(centroid - [1.0, -1.0]) <= 5 * r / np.sqrt(n))


@pytest.mark.convention_pending
def test_disc_radius_distribution_current():
    """Current behaviour: r = U * R, so half the photons are inside R/2 (binomial 5 sigma)."""
    n = 4000
    im = images.disc(n, 0, 0, 3.0, radius=1.0)
    assert_binomial(np.sum(disc_radii(im, 0, 0) < 0.5 * ARCSEC), n, 0.5)


@pytest.mark.convention_pending
@known_bug("disc not uniform in area")
def test_disc_uniform_areal_density():
    """A uniform disc has a quarter of the photons inside R/2 (binomial 5 sigma)."""
    n = 4000
    im = images.disc(n, 0, 0, 3.0, radius=1.0)
    assert_binomial(np.sum(disc_radii(im, 0, 0) < 0.5 * ARCSEC), n, 0.25)


@known_bug("disc energy_spread added in keV to energies in J")
def test_disc_energy_spread_in_kev():
    """disc energy_spread is a keV standard deviation (3 % on the std for n=20000)."""
    im = images.disc(20_000, 0, 0, 6.0, radius=1.0, energy_spread=0.1)
    assert (im.energies / KEV).std() == pytest.approx(0.1, rel=0.03)


# ---------------------------------------------------------------------------
# generate_from_image
# ---------------------------------------------------------------------------


@pytest.fixture
def two_pixel_png(tmp_path):
    """10x10 image with pixel (2, 3) = 100 and pixel (2, 7) = 200."""
    a = np.zeros((10, 10))
    a[2, 3], a[2, 7] = 100, 200
    return write_png(tmp_path / "two.png", a)


def test_generate_from_image_single_pixel(tmp_path):
    """One bright pixel: every photon has the same origin; energies keV -> J; pix_scale = shape."""
    a = np.zeros((8, 10))
    a[5, 2] = 255
    im, pix_scale = images.generate_from_image(
        write_png(tmp_path / "one.png", a), 200, 1.0, 2.0
    )
    assert im.size == 200
    assert len(np.unique(im.loc, axis=0)) == 1
    np.testing.assert_allclose(im.energies, 2.0 * KEV)
    np.testing.assert_array_equal(im.toa, np.arange(200))
    np.testing.assert_array_equal(pix_scale, [8, 10])


def test_generate_from_image_pmf_and_scale(two_pixel_png):
    """Photons are drawn in proportion to pixel brightness (2/3 from the brighter pixel,
    binomial 5 sigma); the two origins are 4 pixels * img_scale/10 apart."""
    n = 3000
    im, _ = images.generate_from_image(two_pixel_png, n, 1.0, 2.0)
    origins, counts = np.unique(im.loc, axis=0, return_counts=True)
    assert len(origins) == 2
    sep = np.hypot(*(origins[1] - origins[0])) / ARCSEC
    assert sep == pytest.approx(0.4)
    assert_binomial(counts.max(), n, 2 / 3)


def test_generate_from_image_offset(two_pixel_png):
    """`offset` (arcsec) shifts every photon origin by offset * ARCSEC."""
    np.random.seed(7)
    base, _ = images.generate_from_image(two_pixel_png, 500, 1.0, 2.0)
    np.random.seed(7)
    shifted, _ = images.generate_from_image(
        two_pixel_png, 500, 1.0, 2.0, offset=[0.3, -0.2]
    )
    np.testing.assert_allclose(
        shifted.loc - base.loc, [[0.3 * ARCSEC, -0.2 * ARCSEC]] * 500
    )


@pytest.mark.convention_pending
def test_generate_from_image_orientation_current(tmp_path):
    """Current mapping: pixel (row, col) -> ((row - H/2), (col - W/2)) * scale / max(H, W)
    in (loc[:, 0], loc[:, 1]); no row flip (orientation convention undecided)."""
    a = np.zeros((8, 10))
    a[5, 2] = 255
    im, _ = images.generate_from_image(write_png(tmp_path / "one.png", a), 10, 2.0, 2.0)
    expected = np.array([(5 - 4) * 2.0 / 10, (2 - 5) * 2.0 / 10]) * ARCSEC
    np.testing.assert_allclose(im.loc[0], expected)


def test_generate_from_image_energy_spread(two_pixel_png):
    """energy_spread is a keV standard deviation (3 % on the std for n=20000)."""
    im, _ = images.generate_from_image(
        two_pixel_png, 20_000, 1.0, 5.0, energy_spread=0.2
    )
    assert (im.energies / KEV).std() == pytest.approx(0.2, rel=0.03)
    assert abs((im.energies / KEV).mean() - 5.0) <= 5 * 0.2 / np.sqrt(20_000)


def test_generate_from_image_spectrum_default_energy(two_pixel_png):
    """With a spectrum and energy=None the mean spectrum energy is used."""
    spec = flat_spectrum()
    im, _ = images.generate_from_image(two_pixel_png, 50, 1.0, None, spectrum=spec)
    assert im.spectrum is spec
    np.testing.assert_allclose(im.energies, spec[:, 0].mean() * KEV)


def test_generate_from_image_requires_energy_or_spectrum(two_pixel_png):
    """generate_from_image raises when neither energy nor spectrum is given."""
    with pytest.raises(Exception, match="spectrum or energy"):
        images.generate_from_image(two_pixel_png, 50, 1.0, None)


@known_bug("np.in1d removed in NumPy 2.4")
def test_generate_from_image_background_photons(two_pixel_png):
    """bkg_phot (int) adds that many background photons, indexed in bkg_indices,
    with energy bkg_energy."""
    im, _ = images.generate_from_image(
        two_pixel_png, 100, 1.0, 2.0, bkg_phot=50, bkg_energy=0.5
    )
    assert im.size == 150
    assert len(im.bkg_indices) == 50
    np.testing.assert_allclose(im.energies[im.bkg_indices], 0.5 * KEV)


@known_bug("np.in1d removed in NumPy 2.4")
def test_generate_from_image_background_fraction(two_pixel_png):
    """A float bkg_phot is a fraction of no_photons."""
    im, _ = images.generate_from_image(two_pixel_png, 100, 1.0, 2.0, bkg_phot=0.25)
    assert im.size == 125
    assert len(im.bkg_indices) == 25


@known_bug("np.in1d removed in NumPy 2.4; noise array excludes background photons")
def test_generate_from_image_background_with_energy_spread(two_pixel_png):
    """Background photons combined with energy_spread produce finite energies."""
    im, _ = images.generate_from_image(
        two_pixel_png, 100, 1.0, 2.0, energy_spread=0.1, bkg_phot=50
    )
    assert im.size == 150 and np.all(np.isfinite(im.energies))


@known_bug("pixel_locations undefined when bkg_spect is given")
def test_generate_from_image_background_spectrum(two_pixel_png):
    """A background spectrum is stored and the image is still generated."""
    spec = flat_spectrum()
    im, _ = images.generate_from_image(
        two_pixel_png, 100, 1.0, 2.0, bkg_phot=50, bkg_spect=spec
    )
    assert im.bkg_spect is spec


# ---------------------------------------------------------------------------
# draw_photons_from_cube
# ---------------------------------------------------------------------------


@pytest.fixture
def cube():
    """Two energy slices: E=1 keV (total flux 4, voxels 1:3), E=2 keV (total flux 4, 2:2)."""
    return pd.DataFrame(
        {
            "energy_bin": [1.0, 1.0, 2.0, 2.0],
            "flux": [1.0, 3.0, 2.0, 2.0],
            "alpha": [0.0, 1.0, 0.0, 1.0],
            "beta": [0.0, 0.0, 1.0, 1.0],
        }
    )


def test_draw_photons_from_cube_zero_flux_raises(cube):
    """A cube with no positive flux raises ValueError."""
    cube["flux"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        images.draw_photons_from_cube(cube, 10)


@known_bug("read-only array with pandas Copy-on-Write")
def test_draw_photons_from_cube_energy_and_spatial_distribution(cube):
    """Energies follow the summed flux per slice (1/2 each); within the 1 keV slice the
    voxels are drawn 1:3 (binomial 5 sigma). Not seedable: uses its own default_rng."""
    n = 4000
    im = images.draw_photons_from_cube(cube, n)
    assert im.size == n
    e1 = energy_is(im, 1.0)
    assert np.all(e1 | energy_is(im, 2.0))
    assert_binomial(e1.sum(), n, 0.5)
    assert_binomial(np.sum(im.loc[e1, 0] == 1.0), e1.sum(), 0.75)


@known_bug("read-only array with pandas Copy-on-Write")
def test_draw_photons_from_cube_does_not_mutate_input(cube):
    """The caller's DataFrame is unchanged."""
    before = cube.copy()
    images.draw_photons_from_cube(cube, 100)
    pd.testing.assert_frame_equal(cube, before)


@pytest.mark.convention_pending
@known_bug("read-only array with pandas Copy-on-Write")
def test_draw_photons_from_cube_loc_units_current(cube):
    """Current behaviour (unit convention undecided): loc takes alpha/beta verbatim,
    without unit conversion."""
    im = images.draw_photons_from_cube(cube, 200)
    assert set(map(tuple, im.loc)) <= {(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)}


# ---------------------------------------------------------------------------
# append
# ---------------------------------------------------------------------------


def test_append_concatenates_and_offsets_toa():
    """append: arrays concatenated in order; image2 toa offset by image1.toa[-1] + 1."""
    a = images.point_source(10, 1.0, 0.0, 1.0)
    b = images.point_source(5, -1.0, 0.0, 6.0)
    c = images.append(a, b)
    assert c.size == 15
    np.testing.assert_allclose(c.energies, np.r_[a.energies, b.energies])
    np.testing.assert_allclose(c.loc, np.r_[a.loc, b.loc])
    np.testing.assert_array_equal(c.toa, np.arange(15))


@known_bug("append drops baseline_indices")
def test_append_keeps_baseline_indices():
    """append concatenates baseline_indices when both images have them."""
    a = images.point_source(3, 0, 0, 1.0)
    b = images.point_source(2, 0, 0, 1.0)
    a.baseline_indices = np.array([0, 1, 2])
    b.baseline_indices = np.array([3, 4])
    np.testing.assert_array_equal(images.append(a, b).baseline_indices, np.arange(5))


@known_bug("append fails for an empty first image")
def test_append_empty_first_image():
    """Appending to an empty image returns the second image's photons."""
    b = images.point_source(4, 0, 0, 1.0)
    c = images.append(images.image(0), b)
    np.testing.assert_array_equal(c.toa, b.toa)


# ---------------------------------------------------------------------------
# create_image_from_events
# ---------------------------------------------------------------------------


def event_table(with_baseline=True):
    """Small EVENTS table as produced by process.generate_event_list."""
    cols = [
        fits.Column(name="DETECTED_ENERGY", format="E", array=[1.0, 2.5, 6.0]),
        fits.Column(name="TIME", format="E", array=[0.2, 3.7, 9.999]),
        fits.Column(name="RA_POS", format="D", array=[1e-7, 0.0, -2e-7]),
        fits.Column(name="DEC_POS", format="D", array=[0.0, 3e-7, 5e-8]),
    ]
    if with_baseline:
        cols.insert(0, fits.Column(name="BASELINE_ID", format="I", array=[0, 2, 1]))
    return fits.BinTableHDU.from_columns(cols).data


def test_create_image_from_events_energies_times_baselines():
    """Energies keV -> J, toa = int(TIME) (truncation), BASELINE_ID -> baseline_indices."""
    im = images.create_image_from_events(event_table())
    assert im.size == 3
    np.testing.assert_allclose(im.energies, np.array([1.0, 2.5, 6.0]) * KEV, rtol=1e-6)
    np.testing.assert_array_equal(im.toa, [0, 3, 9])
    np.testing.assert_array_equal(im.baseline_indices, [0, 2, 1])


def test_create_image_from_events_without_baseline_column():
    """Without a BASELINE_ID column baseline_indices is None."""
    im = images.create_image_from_events(event_table(with_baseline=False))
    assert im.baseline_indices is None


@pytest.mark.convention_pending
def test_create_image_from_events_positions_current():
    """Current behaviour (axis convention undecided): RA/DEC read as degrees;
    DEC -> loc[:, 0], RA -> loc[:, 1]."""
    im = images.create_image_from_events(event_table())
    np.testing.assert_allclose(im.loc[:, 0], np.array([0.0, 3e-7, 5e-8]) * DEG)
    np.testing.assert_allclose(im.loc[:, 1], np.array([1e-7, 0.0, -2e-7]) * DEG)
