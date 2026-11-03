"""Slow end-to-end tests with the bundled model files, and execution of the example
notebook.

Workflow (as in simple_example.ipynb, with a 1000 s exposure): generate_event_list ->
create_image_from_events -> interferometer (Willingale baselines from the ARF
headers, smooth roller) -> interferometer_data -> filter_data -> image_recon_smooth.
The notebook is executed unchanged (10^4 s exposure) in a temporary copy of the repo
layout so it does not write into the repository (~70 s, ~7 GB peak RAM).
Fixtures: `bundled_paths`. All tests are marked `slow`.
"""

import shutil

import numpy as np
import pytest
from astropy.io import fits
from helpers import ARCSEC, SEED

from xrisalis import analysis, images, instrument, process

pytestmark = pytest.mark.slow

EXPOSURE = 1000.0
FOV_ARCSEC = 1.6e-3  # as in the notebook; 512 x 512 image
SAMPLES = 512


def expected_counts(arf_path, simput_path, exposure, e_min=None):
    """Expected counts per baseline: sum_j sum_E F_j(E) A_bl(E) dE_arf T (the model
    generate_event_list implements), optionally only for E > e_min."""
    with fits.open(arf_path) as a, fits.open(simput_path) as s:
        spec = s["SPECTRUM"].data
        n_bl = len(a) - 2
        out = np.zeros(n_bl)
        for k in range(n_bl):
            name = f"SPECRESP_BL{k + 1}"
            arf = a[name].data
            mid = 0.5 * (arf["ENERG_LO"] + arf["ENERG_HI"])
            de = arf["ENERG_HI"] - arf["ENERG_LO"]
            for j in range(len(spec)):
                e = spec["ENERGY"][j]
                rate = spec["FLUXDENSITY"][j] * np.interp(e, mid, arf[name], 0, 0) * de
                if e_min is not None:
                    rate = rate * (e > e_min)
                out[k] += rate.sum() * exposure
    return out


@pytest.fixture(scope="module")
def workflow(bundled_paths, tmp_path_factory):
    """Run the notebook workflow once (1000 s) and share all intermediate products."""
    np.random.seed(SEED + 4)
    p = bundled_paths
    out = tmp_path_factory.mktemp("workflow") / "event_list.fits"
    process.generate_event_list(p["simput"], p["arf"], p["rmf"], EXPOSURE, str(out))
    with fits.open(out) as h:
        events = h["EVENTS"].data.copy()
    with fits.open(p["arf"]) as h:
        baselines = [
            h[f"SPECRESP_BL{i + 1}"].header["BASELINE"] for i in range(len(h) - 2)
        ]
    img = images.create_image_from_events(events)
    inst = instrument.interferometer(roller=instrument.interferometer.smooth_roller)
    for b in baselines:
        inst.add_willingale_baseline(b)
    data = process.interferometer_data(inst, img)
    return {"events": events, "baselines": baselines, "inst": inst, "data": data}


def test_workflow_counts_per_baseline(workflow, bundled_paths):
    """Counts per baseline match sum F A dE T (Poisson 5 sigma each)."""
    exp = expected_counts(bundled_paths["arf"], bundled_paths["simput"], EXPOSURE)
    obs = np.bincount(workflow["events"]["BASELINE_ID"].astype(int), minlength=exp.size)
    assert obs.size == exp.size == 10
    assert np.all(np.abs(obs - exp) <= 5 * np.sqrt(exp)), (obs, exp)


def test_workflow_hard_fraction_per_baseline(workflow, bundled_paths):
    """The fraction of events with E > 2 keV per baseline matches the ARF-folded
    spectrum (binomial 5 sigma); it falls steeply with baseline length."""
    p = bundled_paths
    tot = expected_counts(p["arf"], p["simput"], EXPOSURE)
    hard = expected_counts(p["arf"], p["simput"], EXPOSURE, e_min=2.0)
    frac = hard / tot
    ev = workflow["events"]
    for k in range(10):
        sel = ev["BASELINE_ID"] == k
        n = sel.sum()
        count = np.sum(ev["TRUE_ENERGY"][sel] > 2.0)
        assert abs(count - n * frac[k]) <= 5 * np.sqrt(n * frac[k] * (1 - frac[k])) + 1
    assert frac[0] > 10 * frac[-1]


def test_workflow_source_positions(workflow, bundled_paths):
    """Every event carries one of the two catalogue positions; both sources contribute."""
    with fits.open(bundled_paths["simput"]) as h:
        cat = h["SRC_CAT"].data
        sources = set(zip(cat["RA"], cat["DEC"]))
    ev = workflow["events"]
    pairs = list(zip(ev["RA_POS"], ev["DEC_POS"]))
    assert set(pairs) == sources


def test_workflow_detector_positions(workflow):
    """All photons land on their baseline's detector and are pixel-quantised."""
    data, inst = workflow["data"], workflow["inst"]
    half = inst.baselines[0].W * inst.baselines[0].num_pairs / 2
    assert np.all(np.isfinite(data.pos))
    assert np.abs(data.actual_pos).max() <= half * (1 + 1e-6)


def test_workflow_long_baselines_resolve_double_source(workflow, bundled_paths):
    """Baselines 7-9, 1-10 keV (as in the notebook, ~400 photons): two sources 400 uas
    apart (+-64 px of 3.125 uas) are resolved. Peaks are taken as the maximum within
    +-8 px of each source; noise is the rms in a corner far from the sources.
    The brighter source holds the global maximum; the fainter peak is > 3 rms; the
    peak ratio equals the photon-count ratio of the two sources within 5 rms; the
    centre between them is below the fainter peak. The axis along which the sources
    lie depends on an undecided axis convention, so either orientation is accepted."""
    data, inst, ev = workflow["data"], workflow["inst"], workflow["events"]
    filt = analysis.filter_data(data, (1.0, 10.0), baseline_bounds=(7, 9))
    img, _, _ = analysis.image_recon_smooth(
        filt, inst, fov=FOV_ARCSEC, samples=np.array([SAMPLES, SAMPLES]), verbose=False
    )
    a = img[0] / img[0].max()
    rms = np.std(a[:100, :100])
    with fits.open(bundled_paths["simput"]) as h:
        cat = h["SRC_CAT"].data
    # sources at RA = -+5.56e-8 deg, DEC = 0: offset of each from the centre in pixels
    pixel_rad = FOV_ARCSEC * ARCSEC / SAMPLES
    off = int(round(np.abs(cat["RA"]).max() * np.pi / 180 / pixel_rad))
    assert off == 64
    c, w = SAMPLES // 2, 8

    def window(i, j):
        return a[i - w : i + w + 1, j - w : j + w + 1].max()

    # photons per source in the filtered set; source order = increasing RA
    sel = (
        (ev["DETECTED_ENERGY"] > 1.0)
        & (ev["DETECTED_ENERGY"] < 10.0)
        & (ev["BASELINE_ID"] >= 7)
        & (ev["BASELINE_ID"] <= 9)
    )
    counts = np.array(
        [np.sum(sel & (ev["RA_POS"] < 0)), np.sum(sel & (ev["RA_POS"] > 0))]
    )
    along_1 = np.array([window(c, c - off), window(c, c + off)])
    along_0 = np.array([window(c - off, c), window(c + off, c)])
    peaks = along_1 if along_1.min() >= along_0.min() else along_0

    bright, faint = np.argmax(counts), np.argmin(counts)
    assert peaks[bright] == pytest.approx(1.0)
    assert peaks[faint] > 3 * rms
    assert abs(peaks[faint] - counts[faint] / counts[bright]) <= 5 * rms
    assert a[c, c] < peaks[faint]


def test_example_notebook_executes(bundled_paths, tmp_path):
    """simple_example.ipynb runs top to bottom without errors (nbclient, unchanged
    notebook, executed in a temporary copy of the repo layout)."""
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    repo = bundled_paths["notebook"].parent
    shutil.copy(bundled_paths["notebook"], tmp_path)
    shutil.copy(repo / "xri_plot_white.mplstyle", tmp_path)
    (tmp_path / "src" / "xrisalis").mkdir(parents=True)
    (tmp_path / "src" / "xrisalis" / "models").symlink_to(bundled_paths["arf"].parent)
    nb = nbformat.read(tmp_path / bundled_paths["notebook"].name, as_version=4)
    client = nbclient.NotebookClient(
        nb,
        timeout=1800,
        kernel_name="python3",
        resources={"metadata": {"path": str(tmp_path)}},
    )
    client.execute()
    assert (tmp_path / "event_list.fits").exists()
