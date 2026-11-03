"""Tests for `xrisalis.process`: Fresnel diffraction, detector-position sampling
(`interferometer_data`) and event-list generation (`generate_event_list`).

Fixtures: `synthetic_responses` (flat SIMPUT, 2-baseline ARF 10/30 cm^2, identity RMF).
Helpers: `KEV`, `ARCSEC`, `known_bug`, `energy_grid`, `write_simput`, `write_arf`,
`write_rmf`. Statistical checks use Poisson / binomial / normal 5-sigma bounds.
"""

import numpy as np
import pytest
import scipy.constants as spc
from astropy.io import fits
from helpers import (
    ARCSEC,
    KEV,
    SEED,
    energy_grid,
    write_arf,
    write_rmf,
    write_simput,
    known_bug,
)

from xrisalis import images, instrument, process
from xrisalis.instrument import baseline

fre_dif = process.interferometer_data.fre_dif  # does not use `self`

N_PHOTONS = 20_000
PIXEL = 5e-6


def wavelength(e_kev):
    return spc.h * spc.c / (e_kev * KEV)


def make_baseline(W_um, L, D=0.5):
    return baseline(30, D=D, W=W_um, L=L, grazing_angle=1e-3)


def willingale(*ds):
    inst = instrument.interferometer()
    for d in ds:
        inst.add_willingale_baseline(d)
    return inst


def point_image(n, e_kev=1.0, beta_arcsec=0.0, alpha_arcsec=0.0, baselines=None):
    """On-axis (or offset) monochromatic point source with baseline indices set
    (generator images lack them; see test_interferometer_data_accepts_generator_image)."""
    im = images.point_source(n, alpha_arcsec, beta_arcsec, e_kev)
    im.baseline_indices = np.zeros(n, int) if baselines is None else baselines
    return im


def fringe_phasor(y, spacing):
    """Mean of exp(2 pi i y / s): magnitude = fringe visibility / 2, arg = fringe phase."""
    return np.mean(np.exp(2j * np.pi * y / spacing))


def phasor_sigma(n):
    """Standard error of each component of fringe_phasor for n photons (Var <= 1/2)."""
    return np.sqrt(0.5 / n)


# ---------------------------------------------------------------------------
# fre_dif
# ---------------------------------------------------------------------------


def test_fre_dif_symmetric_and_u_coordinate():
    """Intensity is symmetric in y; u = y * sqrt(2 / (lambda L))."""
    bl = make_baseline(300, 10)
    lam = wavelength(1.0)
    y = np.linspace(-bl.W, bl.W, 201)
    inten, u = fre_dif(None, lam, bl, 1, y_pos=y)
    np.testing.assert_allclose(inten, inten[::-1], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(u, y * np.sqrt(2 / (lam * bl.L)))


def test_fre_dif_default_samples_span_slit():
    """Without y_pos, `samples` points spanning [-W/2, W/2] are evaluated."""
    bl = make_baseline(300, 10)
    lam = wavelength(1.0)
    inten, u = fre_dif(None, lam, bl, 11)
    assert inten.shape == (11,)
    np.testing.assert_allclose(
        u[[0, -1]], np.array([-1, 1]) * bl.W / 2 * np.sqrt(2 / (lam * bl.L))
    )


def test_fre_dif_geometric_limit():
    """Large Fresnel number (u0 ~ 120): I ~ 2 at the slit centre, ~ 0 far outside."""
    bl = make_baseline(3000, 1)
    lam = wavelength(1.0)
    inten, _ = fre_dif(None, lam, bl, 1, y_pos=np.array([0.0, 3 * bl.W]))
    assert inten[0] == pytest.approx(2.0, rel=0.02)
    assert inten[1] < 1e-3


def test_fre_dif_fraunhofer_limit():
    """Small Fresnel number (u0 = 0.1): I(y)/I(0) = sinc^2(W y / (lambda L)) to < 1 %."""
    lam = wavelength(1.0)
    L = 10.0
    W = 0.1 / np.sqrt(2 / (lam * L))
    bl = make_baseline(W * 1e6, L)
    y = np.linspace(-3, 3, 61) * lam * L / W
    inten, _ = fre_dif(None, lam, bl, 1, y_pos=y)
    np.testing.assert_allclose(
        inten / inten[30], np.sinc(W * y / (lam * L)) ** 2, atol=0.01
    )


@pytest.mark.parametrize("e_kev", [0.5, 1.0, 3.0, 7.0, 10.0])
def test_fre_dif_below_accept_reject_box(e_kev):
    """The accept-reject box in process_photon_dpos (2.7 * I(0)) bounds I over the slit."""
    bl = willingale(0.5).baselines[0]
    lam = wavelength(e_kev)
    inten, _ = fre_dif(None, lam, bl, 2001)
    i0, _ = fre_dif(None, lam, bl, 1, y_pos=0)
    assert inten.max() <= 2.7 * i0


# ---------------------------------------------------------------------------
# interferometer_data / process_photon_dpos
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def on_axis_data():
    """20000 on-axis 1 keV photons through one Willingale baseline (D = 0.5 m)."""
    np.random.seed(SEED + 1)
    inst = willingale(0.5)
    im = point_image(N_PHOTONS)
    return inst, im, process.interferometer_data(inst, im)


def test_interferometer_data_passthrough_and_shapes(on_axis_data):
    """Energies, toa and baseline indices are passed through; positions have one entry per photon."""
    inst, im, data = on_axis_data
    assert data.size == im.size
    assert data.energies is im.energies and data.toa is im.toa
    np.testing.assert_array_equal(data.baseline_indices, im.baseline_indices)
    assert data.actual_pos.shape == data.pos.shape == (im.size,)
    assert data.pointing.shape == (im.toa.max() + 2, 3)


def test_interferometer_data_pixel_quantisation(on_axis_data):
    """pos is actual_pos rounded to the 5 um pixel grid."""
    _, _, data = on_axis_data
    np.testing.assert_allclose(data.pos / PIXEL, np.round(data.pos / PIXEL), atol=1e-6)
    assert np.abs(data.pos - data.actual_pos).max() <= PIXEL / 2 * (1 + 1e-9)


def test_interferometer_data_positions_inside_detector_box(on_axis_data):
    """On-axis photons land within +-W * num_pairs / 2."""
    inst, _, data = on_axis_data
    half = inst.baselines[0].W * inst.baselines[0].num_pairs / 2
    assert np.abs(data.actual_pos).max() <= half


def test_interferometer_data_fringe_period(on_axis_data):
    """The periodogram of positions peaks at 1/s with s = lambda / theta_b (to 1 %)."""
    inst, _, data = on_axis_data
    s = wavelength(1.0) / inst.baselines[0].beam_angle
    freqs = np.linspace(0.5, 1.5, 401) / s
    power = np.abs(np.exp(2j * np.pi * np.outer(freqs, data.actual_pos)).mean(axis=1))
    assert freqs[np.argmax(power)] * s == pytest.approx(1.0, abs=0.01)


def test_interferometer_data_on_axis_fringe_phase(on_axis_data):
    """On axis the central fringe maximum is at y = 0: <exp(2 pi i y/s)> ~ 0.5 (real),
    as for an intensity 1 + cos(2 pi y/s); 5 sigma per component."""
    inst, _, data = on_axis_data
    s = wavelength(1.0) / inst.baselines[0].beam_angle
    ph = fringe_phasor(data.actual_pos, s)
    tol = 5 * phasor_sigma(data.size)
    assert abs(ph.real - 0.5) <= tol + 0.01  # +0.01: partial fringes at the box edges
    assert abs(ph.imag) <= tol


@pytest.mark.parametrize("e_kev", [1.0, 6.0])
def test_pure_fringes_visibility(e_kev):
    """pure_fringes: no diffraction envelope; <cos(2 pi y/s)> = 0.5 (5 sigma + edge allowance)."""
    n = N_PHOTONS
    inst = willingale(0.5)
    data = process.interferometer_data(inst, point_image(n, e_kev), pure_fringes=True)
    s = wavelength(e_kev) / inst.baselines[0].beam_angle
    ph = fringe_phasor(data.actual_pos, s)
    assert abs(ph.real - 0.5) <= 5 * phasor_sigma(n) + 0.01
    assert abs(ph.imag) <= 5 * phasor_sigma(n)


def test_pure_diffraction_has_no_fringes():
    """pure_diffraction: no power at the fringe frequency (|phasor| < 5 sigma + 0.01)."""
    n = N_PHOTONS
    inst = willingale(0.5)
    data = process.interferometer_data(inst, point_image(n), pure_diffraction=True)
    s = wavelength(1.0) / inst.baselines[0].beam_angle
    assert abs(fringe_phasor(data.actual_pos, s)) <= 5 * phasor_sigma(n) + 0.01


@pytest.mark.parametrize("num_pairs", [30, 31], ids=["even", "odd"])
def test_pure_diffraction_envelope_matches_fresnel_profile(num_pairs):
    """pure_diffraction: the position folded into one slit is distributed as the Fresnel
    intensity (chi^2, 20 bins, < dof + 5 sqrt(2 dof)). Folding: (|y| mod W) - W/2 for
    even num_pairs, (|y + W/2| mod W) - W/2 for odd (slits centred on y = 0)."""
    n = N_PHOTONS
    inst = instrument.interferometer()
    inst.add_baseline(num_pairs, D=0.5, W=300, L=10, grazing_angle=1e-3)
    bl = inst.baselines[0]
    lam = wavelength(1.0)
    data = process.interferometer_data(inst, point_image(n), pure_diffraction=True)
    shift = 0 if num_pairs % 2 == 0 else bl.W / 2
    folded = (np.abs(data.actual_pos + shift) % bl.W) - bl.W / 2
    edges = np.linspace(-bl.W / 2, bl.W / 2, 21)
    counts, _ = np.histogram(folded, edges)
    fine = np.linspace(-bl.W / 2, bl.W / 2, 20 * 200 + 1)
    inten, _ = fre_dif(None, lam, bl, 1, y_pos=0.5 * (fine[1:] + fine[:-1]))
    expected = inten.reshape(20, 200).sum(axis=1)
    expected = expected / expected.sum() * n
    chi2 = np.sum((counts - expected) ** 2 / expected)
    dof = 19
    assert chi2 < dof + 5 * np.sqrt(2 * dof)


def test_off_axis_fringe_shift_magnitude():
    """A source offset by beta shifts the fringes by F_eff * theta with
    F_eff = D / (2 sin(theta_b/2)); beta chosen for a quarter-fringe shift, so the
    phasor phase is +-pi/2 (5 sigma)."""
    n = N_PHOTONS
    inst = willingale(0.5)
    bl = inst.baselines[0]
    lam = wavelength(1.0)
    s = lam / bl.beam_angle
    f_eff = bl.D / (2 * np.sin(bl.beam_angle / 2))
    theta = s / 4 / f_eff  # small angle: sin(theta) = theta
    data = process.interferometer_data(
        inst, point_image(n, beta_arcsec=theta / ARCSEC), pure_fringes=True
    )
    ph = fringe_phasor(data.actual_pos, s)
    assert abs(ph) == pytest.approx(0.5, abs=5 * phasor_sigma(n) + 0.01)
    assert abs(np.angle(ph)) == pytest.approx(np.pi / 2, abs=10 * phasor_sigma(n))


@pytest.mark.convention_pending
def test_off_axis_projection_current():
    """Current behaviour (axis/sign convention undecided): at roll 0 only loc[:, 1]
    (beta) shifts the fringes, with theta = -beta, so beta > 0 gives phase +pi/2;
    an alpha offset gives none."""
    n = N_PHOTONS
    inst = willingale(0.5)
    bl = inst.baselines[0]
    s = wavelength(1.0) / bl.beam_angle
    theta = s / 4 / (bl.D / (2 * np.sin(bl.beam_angle / 2)))
    beta = process.interferometer_data(
        inst, point_image(n, beta_arcsec=theta / ARCSEC), pure_fringes=True
    )
    alpha = process.interferometer_data(
        inst, point_image(n, alpha_arcsec=theta / ARCSEC), pure_fringes=True
    )
    assert np.angle(fringe_phasor(beta.actual_pos, s)) == pytest.approx(
        np.pi / 2, abs=10 * phasor_sigma(n)
    )
    assert abs(np.angle(fringe_phasor(alpha.actual_pos, s))) <= 10 * phasor_sigma(n)


def test_far_off_axis_warns():
    """A source whose zero-path-difference point falls outside the detector warns."""
    inst = willingale(0.5)
    with pytest.warns(UserWarning, match="outside of the FOV"):
        process.interferometer_data(inst, point_image(200, beta_arcsec=1.0))


def test_multiple_baselines_use_own_fringe_spacing():
    """Each photon is processed with its own baseline: spacing lambda/theta_b differs
    for theta_b = 3e-5 vs 6e-5 and each subset peaks at its own 1/s."""
    n = 10_000
    inst = instrument.interferometer()
    inst.add_baseline(30, D=0.5, W=300, L=10, grazing_angle=1e-3)
    inst.add_baseline(30, D=0.5, W=300, L=5, grazing_angle=1e-3)
    idx = np.repeat([0, 1], n // 2)
    data = process.interferometer_data(inst, point_image(n, baselines=idx))
    for k, bl in enumerate(inst.baselines):
        y = data.actual_pos[idx == k]
        s = wavelength(1.0) / bl.beam_angle
        assert abs(fringe_phasor(y, s)) > 0.4
        assert abs(fringe_phasor(y, 2 * s if k == 0 else s / 2)) < 0.1


def test_interferometer_data_reproducible_with_seed():
    """Same seed, same positions."""
    inst = willingale(0.5)
    np.random.seed(5)
    a = process.interferometer_data(inst, point_image(500)).actual_pos
    np.random.seed(5)
    b = process.interferometer_data(inst, point_image(500)).actual_pos
    np.testing.assert_array_equal(a, b)


def test_interferometer_data_copy_is_independent_container():
    """copy() returns a new object sharing arrays (shallow)."""
    inst = willingale(0.5)
    data = process.interferometer_data(inst, point_image(50))
    c = data.copy()
    assert c is not data and c.pos is data.pos


@known_bug("generator images have no baseline_indices")
def test_interferometer_data_accepts_generator_image():
    """An image straight from a generator can be processed."""
    inst = willingale(0.5)
    data = process.interferometer_data(inst, images.point_source(50, 0, 0, 1.0))
    assert data.pos.shape == (50,)


# ---------------------------------------------------------------------------
# generate_event_list
# ---------------------------------------------------------------------------


def read_events(path):
    with fits.open(path) as h:
        return h["EVENTS"].header, h["EVENTS"].data.copy()


@pytest.fixture
def identity_events(synthetic_responses, tmp_path):
    """Event list from the flat synthetic setup, T = 100 s."""
    r = synthetic_responses
    out = tmp_path / "events.fits"
    process.generate_event_list(r["simput"], r["arf"], r["rmf"], 100.0, str(out))
    return r, read_events(out)


def test_event_list_counts_per_baseline(identity_events):
    """Counts per baseline = sum(F * A * dE) * T: 0.05*10*0.1*60*100 = 300 and 900
    (Poisson 5 sigma)."""
    r, (_, ev) = identity_events
    de = r["hi"] - r["lo"]
    for k in range(2):
        expected = np.sum(r["flux"] * r["areas"][k] * de) * 100.0
        count = np.sum(ev["BASELINE_ID"] == k)
        assert abs(count - expected) <= 5 * np.sqrt(expected)
    assert set(np.unique(ev["BASELINE_ID"])) <= {0, 1}


def test_event_list_identity_rmf_energies(identity_events):
    """With an identity RMF the detected energy is the true energy (both bin centres)."""
    r, (_, ev) = identity_events
    np.testing.assert_allclose(ev["DETECTED_ENERGY"], ev["TRUE_ENERGY"], rtol=1e-6)
    # stored as float32: compare by distance to the nearest bin centre
    dist = np.abs(ev["TRUE_ENERGY"][:, None] - r["mid"][None, :]).min(axis=1)
    assert dist.max() < 1e-5


def test_event_list_flat_spectrum_energy_distribution(identity_events):
    """A flat spectrum and flat ARF give a flat energy distribution: the fraction above
    4 keV is 1/2 (binomial 5 sigma)."""
    _, (_, ev) = identity_events
    n = len(ev)
    assert abs(np.sum(ev["TRUE_ENERGY"] > 4.0) - n / 2) <= 5 * np.sqrt(n / 4)


def test_event_list_times(identity_events):
    """TIME in [0, T), rounded to 1 ms."""
    _, (_, ev) = identity_events
    t = ev["TIME"].astype(float)
    assert t.min() >= 0 and t.max() <= 100.0
    np.testing.assert_allclose(t * 1e3, np.round(t * 1e3), atol=0.02)


def test_event_list_file_format(identity_events):
    """EVENTS extension with the documented columns, units and header keywords."""
    _, (hdr, ev) = identity_events
    assert ev.columns.names == [
        "BASELINE_ID",
        "TRUE_ENERGY",
        "DETECTED_ENERGY",
        "TIME",
        "RA_POS",
        "DEC_POS",
    ]
    assert hdr["TELESCOP"] == "XRI" and hdr["INSTRUME"] == "CIS"
    assert hdr["TUNIT2"] == "keV" and hdr["TUNIT3"] == "keV" and hdr["TUNIT4"] == "s"


def test_event_list_default_output_and_message(
    synthetic_responses, tmp_path, monkeypatch, capsys
):
    """Without output_file, event_list.fits is written to the working directory and
    the photon count is printed."""
    r = synthetic_responses
    monkeypatch.chdir(tmp_path)
    process.generate_event_list(r["simput"], r["arf"], r["rmf"], 10.0)
    _, ev = read_events(tmp_path / "event_list.fits")
    assert f"Total photons generated: {len(ev)}" in capsys.readouterr().out


def test_event_list_source_positions_and_fractions(tmp_path):
    """Two sources with flux ratio 1:2: each event carries one source's (RA, DEC) and
    the fraction from source 2 is 2/3 (binomial 5 sigma)."""
    lo, hi, mid = energy_grid()
    fluxes = np.vstack([np.full(mid.size, 0.05), np.full(mid.size, 0.10)])
    simput = write_simput(
        tmp_path / "s.fits", mid, fluxes, ra=[1e-7, -1e-7], dec=[0, 2e-7]
    )
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, mid.size), 20.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    _, ev = read_events(out)
    pairs = set(zip(ev["RA_POS"], ev["DEC_POS"]))
    assert pairs <= {(1e-7, 0.0), (-1e-7, 2e-7)}
    n = len(ev)
    count2 = np.sum(ev["RA_POS"] < 0)
    assert abs(count2 - 2 * n / 3) <= 5 * np.sqrt(n * 2 / 9)


def test_event_list_rmf_redistribution(tmp_path):
    """RMF with 0.7 on the diagonal and 0.3 one channel up: 70 % of events keep their
    energy, the rest move up exactly one bin (binomial 5 sigma)."""
    lo, hi, mid = energy_grid()
    n_e = mid.size
    matrix = 0.7 * np.eye(n_e) + 0.3 * np.eye(n_e, k=1)
    matrix[-1, -1] = 1.0
    simput = write_simput(tmp_path / "s.fits", mid, np.full(n_e, 0.05))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, n_e), 30.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi, matrix=matrix)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    _, ev = read_events(out)
    ev = ev[ev["TRUE_ENERGY"] < mid[-1] - 1e-3]  # last row is diagonal-only
    shift = np.round((ev["DETECTED_ENERGY"] - ev["TRUE_ENERGY"]) / (hi[0] - lo[0]))
    assert set(np.unique(shift)) <= {0.0, 1.0}
    n = len(ev)
    assert abs(np.sum(shift == 0) - 0.7 * n) <= 5 * np.sqrt(n * 0.21)


@pytest.mark.convention_pending
def test_event_list_rmf_rows_renormalised(tmp_path):
    """Current behaviour: RMF rows are renormalised, so a uniformly scaled matrix (row
    sums 1e-3) gives the same events as the identity. OGIP rows may sum to < 1, so
    whether to renormalise is undecided."""
    lo, hi, mid = energy_grid()
    simput = write_simput(tmp_path / "s.fits", mid, np.full(mid.size, 0.05))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, mid.size), 10.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi, matrix=1e-3 * np.eye(mid.size))
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    _, ev = read_events(out)
    expected = np.sum(0.05 * 10.0 * (hi - lo)) * 100.0
    assert abs(len(ev) - expected) <= 5 * np.sqrt(expected)
    np.testing.assert_allclose(ev["DETECTED_ENERGY"], ev["TRUE_ENERGY"], rtol=1e-6)


@known_bug("TRUE_ENERGY and RA/DEC not filtered with dropped photons")
def test_event_list_photons_without_response_dropped_consistently(tmp_path):
    """Energies whose RMF row is zero are dropped from every column alike."""
    lo, hi, mid = energy_grid()
    matrix = np.eye(mid.size)
    matrix[:10] = 0.0
    simput = write_simput(tmp_path / "s.fits", mid, np.full(mid.size, 0.05))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, mid.size), 30.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi, matrix=matrix)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    _, ev = read_events(out)
    assert ev["TRUE_ENERGY"].min() > hi[9]
    np.testing.assert_allclose(ev["DETECTED_ENERGY"], ev["TRUE_ENERGY"], rtol=1e-6)


@known_bug("no events -> crash")
def test_event_list_zero_flux_writes_empty_list(tmp_path):
    """A source with zero flux yields an empty EVENTS table."""
    lo, hi, mid = energy_grid()
    simput = write_simput(tmp_path / "s.fits", mid, np.zeros(mid.size))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, mid.size), 30.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    assert len(read_events(out)[1]) == 0


@known_bug("SIMPUT grid must equal the ARF grid")
def test_event_list_different_source_grid(tmp_path):
    """A SIMPUT spectrum on a coarser grid than the ARF gives counts = int F A dE * T
    (Poisson 5 sigma)."""
    lo, hi, _ = energy_grid(n=60)
    _, _, src_mid = energy_grid(n=30)
    simput = write_simput(tmp_path / "s.fits", src_mid, np.full(src_mid.size, 0.05))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, lo.size), 10.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    expected = 0.05 * 10.0 * 6.0 * 100.0
    assert abs(len(read_events(out)[1]) - expected) <= 5 * np.sqrt(expected)


@known_bug("F_CHAN used as 0-based index regardless of TLMIN")
def test_event_list_rmf_channels_starting_at_one(tmp_path):
    """An identity RMF whose channels start at 1 (TLMIN = 1) still maps E -> E."""
    lo, hi, mid = energy_grid()
    simput = write_simput(tmp_path / "s.fits", mid, np.full(mid.size, 0.05))
    arf = write_arf(tmp_path / "a.fits", lo, hi, np.full((1, mid.size), 30.0))
    rmf = write_rmf(tmp_path / "r.fits", lo, hi, tlmin=1)
    out = tmp_path / "e.fits"
    process.generate_event_list(simput, arf, rmf, 100.0, str(out))
    _, ev = read_events(out)
    np.testing.assert_allclose(ev["DETECTED_ENERGY"], ev["TRUE_ENERGY"], rtol=1e-6)


def test_event_list_to_detector_positions(synthetic_responses, tmp_path):
    """Event list -> create_image_from_events -> interferometer_data runs and keeps
    every photon on its baseline's detector."""
    r = synthetic_responses
    out = tmp_path / "events.fits"
    process.generate_event_list(r["simput"], r["arf"], r["rmf"], 20.0, str(out))
    _, ev = read_events(out)
    im = images.create_image_from_events(ev)
    inst = willingale(0.05, 1.0)
    data = process.interferometer_data(inst, im)
    assert data.pos.shape == (len(ev),)
    half = inst.baselines[0].W * inst.baselines[0].num_pairs / 2
    assert np.all(np.isfinite(data.pos)) and np.abs(data.actual_pos).max() <= half
