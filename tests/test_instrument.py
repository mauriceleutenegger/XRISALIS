"""Tests for `xrisalis.instrument`: baseline geometry, interferometer pointing, plot_arf.

Fixtures: `bundled_paths` (for plot_arf). Helpers: `ARCSEC`, `known_bug`.
"""

import math

import numpy as np
import pytest
from helpers import ARCSEC, known_bug

from xrisalis import instrument
from xrisalis.instrument import baseline, interferometer

# Willingale (2004) geometry: W = 300 um, L = 10 m -> beam angle 3e-5 rad
W_UM, L_M, THETA_B = 300.0, 10.0, 3e-5


def assert_consistent(bl):
    """The two geometric constraints enforced by baseline hold."""
    assert math.isclose(bl.D, bl.F * bl.beam_angle)
    assert math.isclose(bl.W, bl.beam_angle * bl.L)


# ---------------------------------------------------------------------------
# baseline geometry
# ---------------------------------------------------------------------------

D_M = 0.5
F_M = D_M / THETA_B


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(D=D_M, W=W_UM, L=L_M),
        dict(F=F_M, W=W_UM, L=L_M),
        dict(D=D_M, beam_angle=THETA_B, L=L_M),
        dict(D=D_M, beam_angle=THETA_B, W=W_UM),
        dict(F=F_M, beam_angle=THETA_B, W=W_UM),
        dict(F=F_M, beam_angle=THETA_B, L=L_M),
    ],
    ids=lambda kw: "+".join(kw),
)
def test_baseline_sufficient_inputs_give_same_geometry(kwargs):
    """Any sufficient subset of (D, F, W, L, beam_angle) yields the same consistent geometry."""
    bl = baseline(5, grazing_angle=1e-3, **kwargs)
    assert_consistent(bl)
    assert bl.D == pytest.approx(D_M)
    assert bl.F == pytest.approx(F_M)
    assert bl.W == pytest.approx(W_UM * 1e-6)  # input in um, stored in m
    assert bl.L == pytest.approx(L_M)
    assert bl.beam_angle == pytest.approx(THETA_B)
    assert bl.num_pairs == 5


def test_baseline_willingale_parameters():
    """add_willingale_baseline: 30 pairs, L=10 m, W=300 um, theta_b=3e-5, F=D/theta_b,
    grazing angle arctan(D/2/7 m)/2."""
    inst = interferometer()
    inst.add_willingale_baseline(0.3)
    bl = inst.baselines[0]
    assert bl.num_pairs == 30
    assert bl.L == 10
    assert bl.W == pytest.approx(300e-6)
    assert bl.beam_angle == pytest.approx(3e-5)
    assert bl.F == pytest.approx(0.3 / 3e-5)
    assert bl.bench_length == 7
    assert bl.grazing_angle == pytest.approx(np.arctan(0.3 / 2 / 7) / 2)
    assert bl.mirr_reflec is None


def test_baseline_mismatched_d_f_beam_angle_raises():
    """Over-specified D, F, beam_angle that violate D = F*theta_b raise."""
    with pytest.raises(Exception, match="baseline length"):
        baseline(5, D=1.0, F=1.0, beam_angle=THETA_B, L=L_M, grazing_angle=1e-3)


def test_baseline_mismatched_w_l_beam_angle_raises():
    """Over-specified W, L, beam_angle that violate W = theta_b*L raise."""
    with pytest.raises(Exception, match="beam width"):
        baseline(5, D=D_M, W=W_UM, L=2 * L_M, beam_angle=THETA_B, grazing_angle=1e-3)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(D=D_M, W=W_UM), "beam angle"),
        (dict(D=D_M, L=L_M), "beam angle"),
        (dict(W=W_UM, L=L_M), "baseline length"),
        (dict(D=D_M, beam_angle=THETA_B), "beam width"),
    ],
    ids=["no-L", "no-W", "no-D-no-F", "no-W-no-L"],
)
def test_baseline_underspecified_raises(kwargs, match):
    """Insufficient geometry inputs raise with an explanatory message."""
    with pytest.raises(Exception, match=match):
        baseline(5, grazing_angle=1e-3, **kwargs)


def test_baseline_without_grazing_angle_or_bench_warns():
    """Without grazing_angle or bench_length a warning is issued and grazing_angle is None."""
    with pytest.warns(UserWarning, match="grazing angle"):
        bl = baseline(5, D=D_M, W=W_UM, L=L_M)
    assert bl.grazing_angle is None


def test_baseline_grazing_angle_from_bench_length():
    """grazing_angle = arctan(D / 2 / bench_length) / 2 when only bench_length is given."""
    bl = baseline(5, D=D_M, W=W_UM, L=L_M, bench_length=4.0)
    assert bl.grazing_angle == pytest.approx(np.arctan(D_M / 2 / 4.0) / 2)


def test_baseline_explicit_grazing_angle_wins():
    """An explicit grazing_angle is kept even when bench_length is given."""
    bl = baseline(5, D=D_M, W=W_UM, L=L_M, bench_length=4.0, grazing_angle=0.01)
    assert bl.grazing_angle == 0.01


@pytest.mark.parametrize("num_pairs", [0, -3, 2.0])
def test_baseline_invalid_num_pairs_raises(num_pairs):
    """num_pairs must be a positive int."""
    with pytest.raises(Exception, match="number of pairs"):
        baseline(num_pairs, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3)


@known_bug("numpy integer num_pairs rejected")
def test_baseline_accepts_numpy_int_num_pairs():
    """A numpy integer is a valid num_pairs."""
    bl = baseline(np.int64(5), D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3)
    assert bl.num_pairs == 5


def write_henke(path, energies, reflectivity):
    """Write a reflectivity table in the Henke (henke.lbl.gov) format: 2 header lines."""
    with open(path, "w") as f:
        f.write(" Pt Rough=0 Angle=x\n Photon Energy (eV), Reflectivity\n")
        for e, r in zip(energies, reflectivity):
            f.write(f" {e:.1f} {r:.6f}\n")
    return str(path)


def test_baseline_mirror_reflectivity_files_parsed(tmp_path):
    """mirr_reflec: angle taken from the file name, data read as (energy, reflectivity) rows."""
    e = np.linspace(1000, 7000, 7)
    files = [
        write_henke(tmp_path / "Pt_0.5deg.dat", e, np.linspace(0.9, 0.5, 7)),
        write_henke(tmp_path / "Pt_1.5deg.dat", e, np.linspace(0.8, 0.1, 7)),
    ]
    bl = baseline(5, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3, mirr_reflec=files)
    assert bl.mirr_reflec.shape == (2, 2)
    assert list(bl.mirr_reflec[:, 0]) == [0.5, 1.5]
    np.testing.assert_allclose(bl.mirr_reflec[1, 1][0], e)
    # written with 6 decimals
    np.testing.assert_allclose(
        bl.mirr_reflec[1, 1][1], np.linspace(0.8, 0.1, 7), atol=1e-6
    )


@known_bug("greedy regex truncates multi-digit angles")
def test_baseline_mirror_reflectivity_multidigit_angle(tmp_path):
    """A file name with angle 12.5 is parsed as 12.5."""
    e = np.linspace(1000, 7000, 7)
    files = [write_henke(tmp_path / "Pt_12.5deg.dat", e, np.full(7, 0.5))]
    bl = baseline(5, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3, mirr_reflec=files)
    assert bl.mirr_reflec[0, 0] == 12.5


# ---------------------------------------------------------------------------
# interferometer: construction and baselines
# ---------------------------------------------------------------------------


def test_interferometer_defaults():
    """A new interferometer has no baselines, no wobble/roll and time_step 1."""
    inst = interferometer()
    assert inst.baselines == []
    assert inst.time_step == 1
    assert inst.wobbler is None and inst.roller is None
    assert inst.wobble_I == 0.0 and inst.roll_speed == 0.0 and inst.roll_init == 0


def test_add_baseline_forwards_parameters():
    """add_baseline appends a baseline built from the given geometry."""
    inst = interferometer()
    inst.add_baseline(7, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3)
    assert len(inst.baselines) == 1
    bl = inst.baselines[0]
    assert bl.num_pairs == 7 and bl.D == pytest.approx(D_M)
    assert_consistent(bl)


def test_add_baseline_mirror_files_positional(tmp_path):
    """The mirror reflectivity file list passed as `mirr_mater` reaches the baseline."""
    files = [write_henke(tmp_path / "Pt_0.5deg.dat", [1000, 2000], [0.9, 0.8])]
    inst = interferometer()
    inst.add_baseline(5, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3, mirr_mater=files)
    assert inst.baselines[0].mirr_reflec[0, 0] == 0.5


@known_bug("add_baseline keyword is mirr_mater, baseline uses mirr_reflec")
def test_add_baseline_accepts_mirr_reflec_keyword(tmp_path):
    """add_baseline accepts the same `mirr_reflec` keyword as baseline."""
    files = [write_henke(tmp_path / "Pt_0.5deg.dat", [1000, 2000], [0.9, 0.8])]
    inst = interferometer()
    inst.add_baseline(5, D=D_M, W=W_UM, L=L_M, grazing_angle=1e-3, mirr_reflec=files)
    assert inst.baselines[0].mirr_reflec[0, 0] == 0.5


def test_add_willingale_baselines_and_clear():
    """Several Willingale baselines are kept in order; clear_baselines empties the list."""
    inst = interferometer()
    for d in [0.05, 0.5, 1.0]:
        inst.add_willingale_baseline(d)
    assert [bl.D for bl in inst.baselines] == [0.05, 0.5, 1.0]
    inst.clear_baselines()
    assert inst.baselines == []


# ---------------------------------------------------------------------------
# interferometer: pointing
# ---------------------------------------------------------------------------


def test_gen_pointing_without_roller_is_zero():
    """gen_pointing(t) returns a (t+2, 3) zero array when no roller is set."""
    p = interferometer().gen_pointing(10)
    assert p.shape == (12, 3)
    assert not p.any()


def test_smooth_roller_linear_roll():
    """smooth_roller: roll[k] = roll_init + k * pi/N * time_step for N = t+2 rows."""
    inst = interferometer(roller=interferometer.smooth_roller, roll_init=0.1)
    p = inst.gen_pointing(98)
    n = 100
    np.testing.assert_allclose(p[:, 2], 0.1 + np.arange(n) * np.pi / n)
    assert not p[:, :2].any()
    assert inst.roll_speed == pytest.approx(np.pi / n)


def test_smooth_roller_scales_with_time_step():
    """smooth_roller multiplies the per-row increment by time_step."""
    inst = interferometer(roller=interferometer.smooth_roller, time_step=2)
    p = inst.gen_pointing(8)
    np.testing.assert_allclose(np.diff(p[:, 2]), 2 * np.pi / 10)


@pytest.mark.convention_pending
@known_bug("smooth_roller overwrites roll_speed with pi/N")
def test_smooth_roller_respects_roll_speed():
    """With roll_speed given, roll[k] = roll_init + k * roll_speed * time_step."""
    inst = interferometer(roller=interferometer.smooth_roller, roll_speed=0.01)
    p = inst.gen_pointing(98)
    np.testing.assert_allclose(p[:, 2], np.arange(100) * 0.01)


def test_discrete_roller_single_step():
    """discrete_roller: roll is 0 up to roll_stop_t and roll_stop_a afterwards."""
    inst = interferometer(
        roller=interferometer.discrete_roller, roll_stop_t=5, roll_stop_a=0.3
    )
    p = inst.gen_pointing(18)
    t = np.arange(20)
    np.testing.assert_allclose(p[:, 2], np.where(t > 5, 0.3, 0.0))


@pytest.mark.convention_pending
@known_bug("discrete_roller steps only once")
def test_discrete_roller_repeats_steps():
    """Over many stop intervals the roll takes more than one non-zero value."""
    inst = interferometer(
        roller=interferometer.discrete_roller, roll_stop_t=5, roll_stop_a=0.3
    )
    p = inst.gen_pointing(48)
    assert len(np.unique(p[p[:, 2] > 0, 2])) > 1


def test_random_wobble_first_row_and_step_size():
    """random_wobble keeps row 0 and draws offsets with sigma = wobble_I arcsec (in rad).

    Expectation: std of rows 1.. equals wobble_I * ARCSEC; with 2x9999 samples the
    relative standard error of the std is ~0.5 %, tolerance 3 % (> 5 sigma).
    """
    inst = interferometer(wobble_I=2.0)
    p = inst.random_wobble(np.zeros((10_000, 3)))
    assert not p[0].any()
    assert not p[:, 2].any()
    assert np.std(p[1:, :2]) == pytest.approx(2.0 * ARCSEC, rel=0.03)


@known_bug("random_wobble is not a random walk")
def test_random_wobble_is_random_walk():
    """Consecutive wobble differences have std wobble_I (random walk), not sqrt(2)*wobble_I."""
    inst = interferometer(wobble_I=2.0)
    p = inst.random_wobble(np.zeros((10_000, 3)))
    assert np.std(np.diff(p[:, :2], axis=0)) == pytest.approx(2.0 * ARCSEC, rel=0.03)


def test_file_wobble_reads_csv(tmp_path):
    """file_wobble fills pitch/yaw from a comma-separated file, leaving roll untouched."""
    data = np.random.normal(size=(6, 2))
    path = tmp_path / "wobble.csv"
    np.savetxt(path, data, delimiter=",")
    inst = interferometer(wobble_file=str(path))
    p = np.zeros((6, 3))
    p[:, 2] = 7.0
    p = inst.file_wobble(p)
    np.testing.assert_allclose(p[:, :2], data)
    assert np.all(p[:, 2] == 7.0)


@known_bug("gen_pointing never applies the wobbler")
def test_gen_pointing_applies_wobbler():
    """gen_pointing with a wobbler and wobble_I > 0 yields non-zero pitch/yaw."""
    inst = interferometer(wobbler=interferometer.random_wobble, wobble_I=1.0)
    assert inst.gen_pointing(50)[:, :2].any()


@known_bug("gen_pointing swallows roller exceptions")
def test_gen_pointing_reports_roller_errors():
    """An exception raised by the roller propagates out of gen_pointing."""

    def broken_roller(self, pointing):
        raise RuntimeError("roller failed")

    inst = interferometer(roller=broken_roller)
    with pytest.raises(RuntimeError):
        inst.gen_pointing(5)


# ---------------------------------------------------------------------------
# plot_arf
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("baseline_mode, n_lines", [(False, 1), (True, 11)])
def test_plot_arf_smoke(bundled_paths, baseline_mode, n_lines):
    """plot_arf draws the total area, plus one line per baseline in baseline mode."""
    import matplotlib.pyplot as plt

    instrument.plot_arf(str(bundled_paths["arf"]), baseline_mode=baseline_mode)
    assert len(plt.gca().get_lines()) == n_lines
