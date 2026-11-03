"""Tests for `xrisalis.analysis`: Fourier helpers, image reconstruction
(`image_recon_smooth`, `reconstruction_data`), `filter_data` and the plot helpers.

Reconstruction tests use a 10-baseline Willingale interferometer (D = 0.1 ... 1 m)
with the smooth roller, 1 keV point sources with baseline indices set by hand
(generator images lack them), and a 64 x 64 image with 2e-10 rad pixels (finer
than the lambda / (2 D_max) = 6.2e-10 rad Nyquist limit).
Fixtures: `synthetic_responses`. Helpers: `ARCSEC`, `KEV`, `known_bug`.
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy.constants as spc
from helpers import ARCSEC, KEV, SEED, known_bug
from matplotlib.figure import Figure

from xrisalis import analysis, images, instrument, process

PIXEL_RAD = 2e-10
SAMPLES = np.array([64, 64])
FOV_ARCSEC = PIXEL_RAD * SAMPLES[0] / ARCSEC
CENTRE = (32, 32)
BASELINES = np.linspace(0.1, 1.0, 10)


def rolling_interferometer():
    inst = instrument.interferometer(roller=instrument.interferometer.smooth_roller)
    for d in BASELINES:
        inst.add_willingale_baseline(d)
    return inst


def observe(img, inst=None, **kwargs):
    """Spread photons over all baselines (every baseline sees every roll) and process."""
    inst = rolling_interferometer() if inst is None else inst
    img.baseline_indices = np.arange(img.size) % len(inst.baselines)
    return inst, process.interferometer_data(inst, img, **kwargs)


def reconstruct(data, inst, **kwargs):
    kwargs.setdefault("samples", SAMPLES)
    kwargs.setdefault("fov", FOV_ARCSEC)
    kwargs.setdefault("verbose", False)
    return analysis.image_recon_smooth(data, inst, **kwargs)


def peak(image):
    return tuple(int(i) for i in np.unravel_index(np.argmax(image), image.shape))


@pytest.fixture(scope="module")
def on_axis():
    """10000 on-axis 1 keV photons, observed and reconstructed."""
    np.random.seed(SEED + 2)
    inst, data = observe(images.point_source(10_000, 0, 0, 1.0))
    return inst, data, reconstruct(data, inst)


# ---------------------------------------------------------------------------
# ft_data, plot_ft, hist_data
# ---------------------------------------------------------------------------


def test_ft_data_sinusoid():
    """FFT of cos(2 pi f0 x) on an exact grid peaks at +-f0 with amplitude 1/2."""
    n, spacing, k = 256, 1e-6, 20
    x = np.arange(n) * spacing
    f0 = k / (n * spacing)
    freqs, amp = analysis.ft_data(np.cos(2 * np.pi * f0 * x), n, spacing)
    np.testing.assert_allclose(freqs, np.fft.fftfreq(n, spacing))
    assert set(np.argsort(np.abs(amp))[-2:]) == {k, n - k}
    assert np.abs(amp[k]) == pytest.approx(0.5)


@pytest.mark.parametrize(
    "log, xscale, yscale",
    [(0, "linear", "linear"), (1, "linear", "log"), (2, "log", "log")],
)
def test_plot_ft_axis_scales(log, xscale, yscale):
    """plot_ft draws one fftshifted line with 0, 1 or 2 log axes."""
    n = 16
    freqs, amp = analysis.ft_data(np.arange(n, dtype=float) + 1, n, 1.0)
    fig, ax = plt.subplots()
    analysis.plot_ft(freqs, np.abs(amp), ax, log=log, num=3)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_xdata(), np.fft.fftshift(freqs))
    assert line.get_label() == "Baseline 3"
    assert (ax.get_xscale(), ax.get_yscale()) == (xscale, yscale)


@pytest.mark.parametrize("pixs, unit", [(True, "pixels"), (False, "micrometers")])
def test_hist_data_labels(pixs, unit):
    """hist_data draws a labelled histogram with the right x-axis unit."""
    analysis.hist_data(np.random.normal(size=500), 20, pixs=pixs, num=2)
    ax = plt.gca()
    assert unit in ax.get_xlabel()
    assert len(ax.patches) == 20


# ---------------------------------------------------------------------------
# image_recon_smooth
# ---------------------------------------------------------------------------


def test_recon_output_structure(on_axis):
    """Returns ((image, scale), uv, reconstruction_data); image is real with shape `samples`."""
    _, data, (img, uv, obj) = on_axis
    assert isinstance(obj, analysis.reconstruction_data)
    assert img[0].shape == tuple(SAMPLES) and np.isrealobj(img[0])
    assert np.all(np.isfinite(img[0]))
    assert uv.shape == (2 * data.size, 2)
    assert obj.f_values.shape == (2 * data.size,)
    assert obj.uv is uv and obj.image is img
    n = obj.ft_image.shape[0]
    assert obj.ft_image.shape == (n, n) and obj.fft_freqs.shape == (n,)
    assert obj.uv_conv.shape == uv.shape


def test_recon_uv_hermitian_pairs(on_axis):
    """Each photon contributes (uv, f) and (-uv, conj f), interleaved."""
    _, _, (_, uv, obj) = on_axis
    np.testing.assert_allclose(uv[1::2], -uv[0::2])
    np.testing.assert_allclose(obj.f_values[1::2], np.conj(obj.f_values[0::2]))
    np.testing.assert_allclose(np.abs(obj.f_values), 1.0, rtol=1e-6)


def test_recon_uv_coordinates(on_axis):
    """|uv| = D / lambda and (u, v) = |uv| (sin roll, cos roll), photons grouped by baseline."""
    inst, data, (_, uv, _) = on_axis
    order = np.concatenate(
        [np.flatnonzero(data.baseline_indices == k) for k in range(len(inst.baselines))]
    )
    lam = spc.h * spc.c / data.energies[order]
    d = BASELINES[data.baseline_indices[order]]
    roll = data.pointing[data.toa[order], 2]
    np.testing.assert_allclose(uv[0::2, 0], d / lam * np.sin(roll), rtol=1e-9)
    np.testing.assert_allclose(uv[0::2, 1], d / lam * np.cos(roll), rtol=1e-9)


def test_recon_grid_size_formula(on_axis):
    """Fourier grid: n = ceil(1 / (2 d (D_min / lambda_max) error)), d = fov / max(samples)."""
    _, _, (_, _, obj) = on_axis
    lam = spc.h * spc.c / (1.0 * KEV)
    d = FOV_ARCSEC * np.pi / (180 * 3600) / SAMPLES.max()
    n = int(np.ceil(1 / (2 * d * (BASELINES.min() / lam) * 0.1)))
    assert obj.ft_image.shape == (n, n)
    np.testing.assert_allclose(obj.fft_freqs, np.fft.fftfreq(n, d))


def test_recon_on_axis_peak_at_centre(on_axis):
    """An on-axis point source reconstructs to a peak at the image centre."""
    _, _, (img, _, _) = on_axis
    assert peak(img[0]) == CENTRE


@pytest.mark.parametrize("alpha_px, beta_px", [(10, 0), (0, 10), (-10, 5), (7, -12)])
def test_recon_offset_source_position(alpha_px, beta_px):
    """A source at (alpha, beta) reconstructs at centre + (alpha, beta) / pixel along
    array axes (0, 1) -- i.e. the image recovers loc in loc's own coordinates (+-1 px)."""
    px = PIXEL_RAD / ARCSEC
    inst, data = observe(images.point_source(10_000, alpha_px * px, beta_px * px, 1.0))
    img, _, _ = reconstruct(data, inst)
    got = np.array(peak(img[0]))
    assert np.all(np.abs(got - (np.array(CENTRE) + [alpha_px, beta_px])) <= 1)


def test_recon_two_sources_resolved():
    """Two sources 20 px apart (resolution lambda/(2 D_max) ~ 3 px) give two peaks at the
    right places with a dip between them."""
    px = PIXEL_RAD / ARCSEC
    img_in = images.double_point_source(20_000, [0, 0], [-10 * px, 10 * px], [1.0, 1.0])
    inst, data = observe(img_in)
    img, _, _ = reconstruct(data, inst)
    a = img[0]
    left, right, mid = a[32, 22], a[32, 42], a[32, 32]
    assert min(left, right) > 0.6 * a.max()
    assert mid < 0.5 * min(left, right)


def test_recon_verbose_output(on_axis, capsys):
    """verbose=False prints nothing; verbose=True reports progress."""
    inst, data, _ = on_axis
    reconstruct(data, inst, verbose=False)
    assert capsys.readouterr().out == ""
    reconstruct(data, inst, verbose=True)
    out = capsys.readouterr().out
    assert "Image reconstruction complete." in out and "sample spacing" in out


@pytest.mark.convention_pending
def test_recon_scale_value_current(on_axis):
    """Current behaviour: the second image element is samples.max() * (n d in uas)."""
    _, _, (img, _, obj) = on_axis
    n = obj.ft_image.shape[0]
    d = FOV_ARCSEC * np.pi / (180 * 3600) / SAMPLES.max()
    assert img[1] == pytest.approx(
        SAMPLES.max() * (1e6 * 3600 * 360 / (2 * np.pi)) * d * n
    )


@pytest.mark.parametrize("pixel_rad", [1e-9, 2e-9], ids=["aliasing", "out-of-grid"])
@known_bug("uv beyond grid Nyquist aliases silently or raises IndexError")
def test_recon_rejects_pixels_coarser_than_nyquist(on_axis, pixel_rad):
    """A pixel coarser than 1 / (2 |uv|_max) is rejected with a clear ValueError."""
    inst, data, _ = on_axis
    with pytest.raises(ValueError):
        reconstruct(data, inst, fov=pixel_rad * 64 / ARCSEC)


@known_bug("crop clamp drops the last row/column")
def test_recon_small_grid_returns_full_grid(on_axis):
    """When the Fourier grid is smaller than `samples`, the whole n x n grid is returned."""
    inst, data, _ = on_axis
    fov = 1e-9 * 64 / ARCSEC
    img, _, obj = reconstruct(data, inst, fov=fov, samples=np.array([64, 64]))
    assert img[0].shape == obj.ft_image.shape


# ---------------------------------------------------------------------------
# filter_data
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def broadband():
    """3000 photons, uniform 1-7 keV, over the 10 rolling baselines."""
    np.random.seed(SEED + 3)
    return observe(images.point_source_multichromatic_range(3000, 0, 0, [1.0, 7.0]))


def test_filter_data_energy_and_baseline_masks(broadband, capsys):
    """Keeps photons with E_lo < E < E_hi (keV) and b_lo <= baseline <= b_hi; all
    per-photon arrays stay aligned; the event count is printed."""
    _, data = broadband
    out = analysis.filter_data(data, (2.0, 5.0), baseline_bounds=(3, 6))
    e = data.energies / KEV
    mask = (
        (e > 2.0)
        & (e < 5.0)
        & (data.baseline_indices >= 3)
        & (data.baseline_indices <= 6)
    )
    assert out.size == mask.sum() > 0
    np.testing.assert_array_equal(out.pos, data.pos[mask])
    np.testing.assert_array_equal(out.energies, data.energies[mask])
    np.testing.assert_array_equal(out.toa, data.toa[mask])
    np.testing.assert_array_equal(out.baseline_indices, data.baseline_indices[mask])
    assert f"Events in filtered data: {mask.sum()}" in capsys.readouterr().out


def test_filter_data_without_baseline_bounds(broadband):
    """Without baseline_bounds only the energy cut applies."""
    _, data = broadband
    out = analysis.filter_data(data, (0.0, 4.0))
    e = data.energies / KEV
    assert out.size == np.sum(e < 4.0)


def test_filter_data_leaves_input_unchanged(broadband):
    """The input object keeps all its photons."""
    _, data = broadband
    size, pos = data.size, data.pos.copy()
    analysis.filter_data(data, (2.0, 3.0), baseline_bounds=(0, 1))
    assert data.size == size
    np.testing.assert_array_equal(data.pos, pos)


def test_filter_data_output_reconstructs(broadband):
    """Filtered data can be fed to image_recon_smooth (on-axis peak at centre).
    Cut to 1-1.5 keV so the 2e-10 rad pixel stays below Nyquist (4.1e-10 rad at 1.5 keV)."""
    inst, data = broadband
    out = analysis.filter_data(data, (1.0, 1.5), baseline_bounds=(0, 9))
    img, _, _ = reconstruct(out, inst)
    assert peak(img[0]) == CENTRE


@known_bug("filter_data leaves actual_pos / actual_energies unfiltered")
def test_filter_data_filters_all_per_photon_arrays(broadband):
    """actual_pos and actual_energies are filtered like pos and energies."""
    _, data = broadband
    out = analysis.filter_data(data, (2.0, 5.0))
    assert out.actual_pos.shape == out.pos.shape
    assert out.actual_energies.shape == out.energies.shape


# ---------------------------------------------------------------------------
# plot helpers (smoke tests)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("return_fig", [True, False])
def test_plot_inf_image(on_axis, return_fig):
    """plot_inf_image shows the normalised image with a uas extent of the given fov."""
    _, _, (img, _, _) = on_axis
    fig = analysis.plot_inf_image(img, FOV_ARCSEC, return_fig=return_fig)
    if not return_fig:
        assert fig is None
        return
    assert isinstance(fig, Figure)
    (im,) = fig.axes[0].get_images()
    half = FOV_ARCSEC * 1e6 / 2
    np.testing.assert_allclose(im.get_extent(), [-half, half, -half, half])
    assert np.nanmax(im.get_array()) == pytest.approx(1.0)


def test_plot_uv(on_axis):
    """plot_uv plots the uv points in G-lambda."""
    _, _, (_, uv, _) = on_axis
    fig = analysis.plot_uv(uv, return_fig=True)
    (line,) = fig.axes[0].get_lines()
    np.testing.assert_allclose(line.get_xdata(), uv[:, 0] / 1e9)


def test_plot_res_hist(on_axis):
    """plot_res_hist histograms 1 / (2 |uv|) in uas on log axes."""
    _, _, (_, uv, _) = on_axis
    fig = analysis.plot_res_hist(uv, return_fig=True)
    ax = fig.axes[0]
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert len(ax.patches) == 1  # histtype="stepfilled"


def test_subplots(on_axis):
    """subplots draws the image and the uv coverage side by side (plus a colorbar)."""
    _, _, (img, uv, _) = on_axis
    fig = analysis.subplots(img, FOV_ARCSEC, uv, return_fig=True)
    assert len(fig.axes) == 3
    assert fig.axes[0].get_images() and fig.axes[1].get_lines()


def test_plot_spec_per_baseline(synthetic_responses, tmp_path):
    """plot_spec_per_baseline draws the total spectrum plus one histogram per baseline."""
    r = synthetic_responses
    out = tmp_path / "events.fits"
    process.generate_event_list(r["simput"], r["arf"], r["rmf"], 20.0, str(out))
    analysis.plot_spec_per_baseline(str(out), 20.0)
    ax = plt.gca()
    assert [t.get_text() for t in ax.get_legend().get_texts()] == [
        "All Baselines",
        "Baseline 1",
        "Baseline 2",
    ]
