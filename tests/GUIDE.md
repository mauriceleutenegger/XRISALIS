# XRISALIS test suite — a guided tour

This is a longer companion to `tests/README.md`. The README says *how to run* and
*how to maintain* the suite; this document explains *what every function in the
suite does and why*. It is written for a reader who knows the physics and the
package but has not worked much with pytest. (Written 2026-11-03; see
`AI_DISCLOSURE.md` for how it was produced.)

The suite has 138 test functions in 6 files. Several are *parametrised* (run once
per set of inputs), so pytest reports 179 test cases: 148 pass and 31 are
expected failures documenting known bugs.

---

## 0. Reading a pytest test: the vocabulary used below

- **Test function.** Any function whose name starts with `test_` in a file whose
  name starts with `test_`. pytest finds and runs it; it *passes* if it returns
  without raising an exception, and *fails* otherwise. Almost all checks are made
  with `assert <condition>`, which raises `AssertionError` when the condition is
  false.
- **Fixture.** A function decorated with `@pytest.fixture` that prepares an input.
  A test asks for it simply by naming it as an argument (e.g. `def test_x(on_axis):`);
  pytest calls the fixture first and passes in its return value. A fixture's
  *scope* says how often it is rebuilt: `function` (default, fresh for every test),
  `module` (once per test file, shared by the tests in it — used for expensive
  simulations), or `session` (once per pytest run). `tmp_path` is a fixture
  built into pytest that gives each test its own empty temporary directory.
  `autouse=True` means the fixture is applied to every test without being named.
- **Parametrisation.** `@pytest.mark.parametrize("x", [a, b, c])` runs the test
  once for each value, as separate test cases (`test_name[a]`, `test_name[b]`, ...).
  Stacking two parametrize decorators runs every combination.
- **Expected failure (xfail).** `@known_bug("short description")` marks a test that
  states the *correct* behaviour but currently fails because of a known bug in the
  package; the description appears in pytest's report. It is *strict*: if the test
  unexpectedly passes (e.g. because the bug was fixed), pytest reports a failure,
  prompting someone to remove the marker.
- **`convention_pending` marker.** Labels a test that pins down the *current*
  behaviour where the intended convention (sign, axis order, units) has not been
  decided. Such a test is not a statement that the current behaviour is right; it
  makes sure any change of behaviour is noticed and deliberate.
- **`slow` marker.** End-to-end tests using the bundled model files. Skipped with
  `pytest -m "not slow"`.
- **Comparisons.** `pytest.approx(v, rel=..., abs=...)` and
  `np.testing.assert_allclose(a, b, rtol=..., atol=...)` compare floating-point
  numbers within a tolerance (exact `==` is fragile for computed floats).
  `assert_array_equal` requires exact equality. `pytest.raises(E, match="text")`
  passes only if the code inside raises exception `E` whose message contains
  `text`; `pytest.warns` does the same for warnings.
- **Statistical tests and "5σ".** Simulations are random. Every test starts from a
  fixed seed, so a run is reproducible; but each statistical assertion is written
  so that it holds for *any* seed, by comparing against an analytic expectation
  with a tolerance of at least 5 standard deviations (σ) of the expected
  fluctuation (Poisson: σ = √N; binomial: σ = √(Np(1−p)); a mean of n samples:
  σ/√n). A correct program then fails such a check with probability < 10⁻⁶,
  while a real bug usually misses by many σ.
- **Smoke test.** A test that only checks a function runs and its main outputs are
  sensible (used for the plotting functions).

Units to keep in mind: the package stores photon energies in **joules**
(`KEV = 1.602e-16` J per keV), sky offsets in **radians** (`ARCSEC = 4.85e-6` rad),
and FITS files use keV and degrees.

---

## 1. Shared infrastructure

### `conftest.py`
pytest loads this file automatically before any test. It first switches
matplotlib to the non-interactive `Agg` backend, so plots are drawn in memory and
no windows open.

- **`_seed_and_no_show`** (autouse fixture). Runs around every test. Before the
  test it seeds NumPy's global random generator with `SEED` and replaces
  `plt.show` by a do-nothing function (the package's plot functions call
  `plt.show()`, which would otherwise block or warn). After the test (the code after
  `yield`) it closes all figures so memory does not accumulate. The replacement of
  `plt.show` uses pytest's `monkeypatch`, which undoes the change automatically.
- **`bundled_paths`** (session fixture). A dictionary of paths to the bundled
  `arf.fits`, `rmf.fits`, `double_point_source.fits` and to `simple_example.ipynb`.
  No files are opened here; it only centralises the locations.
- **`synthetic_responses`** (fixture). Builds a small, fully understood
  observation setup in the test's temporary directory: a SIMPUT file with one
  on-axis source of flat spectrum 0.05 ph/cm²/s/keV on a 1–7 keV grid of 60 bins,
  an ARF with two baselines of constant area 10 and 30 cm² (so their count ratio
  is exactly 1:3), and an identity RMF (every photon is detected at its true
  energy). Because everything is flat, expected counts can be computed by hand.

### `helpers.py`
Plain functions and constants imported by the test files (unlike fixtures they
are called directly).

- **`SEED`.** The global test seed, 12345 unless the environment variable
  `XRISALIS_TEST_SEED` is set. Changing it is how one checks that statistical
  tests do not secretly depend on a lucky seed.
- **`KEV`, `ARCSEC`.** The unit conversions keV → J and arcsec → rad, written
  exactly as the package writes them, so that comparisons are not off by rounding
  of a constant.
- **`known_bug(summary)`.** Returns a strict xfail marker whose reason text is
  "known bug: <summary>". Used as a decorator.
- **`energy_grid(e_min=1, e_max=7, n=60)`.** Returns the lower edges, upper edges
  and centres of a linear energy grid in keV — the grid used by all synthetic FITS
  files.
- **`write_simput(path, energies, fluxes, ra, dec)`.** Writes a minimal SIMPUT file
  with a source catalogue (`SRC_CAT`: id, RA, DEC in degrees) and a `SPECTRUM`
  table (one row per source with the energy grid and flux density). Positions
  default to the origin.
- **`write_arf(path, lo, hi, areas, baselines)`.** Writes an ARF in the package's
  layout: a total `SPECRESP` extension (the sum over baselines) plus one
  `SPECRESP_BL{i}` extension per baseline with a `BASELINE` header (length in m;
  default evenly spaced from 0.05 to 1 m).
- **`write_rmf(path, lo, hi, matrix, tlmin)`.** Writes an RMF with a full (one group
  per row) redistribution matrix — the identity unless a matrix is supplied — and
  an `EBOUNDS` table giving each channel's energy range. `tlmin` sets the first
  channel number (0 or 1), used to test RMFs whose channels start at 1.

---

## 2. `test_package.py` — imports and bundled data files

- **`test_submodule_imports`** (parametrised over the four submodules). Imports
  each of `instrument`, `images`, `process`, `analysis`. This catches syntax errors,
  missing dependencies and import-time crashes separately from functional tests.
- **`test_xri_cmap_builds`.** Checks that `analysis.xri_cmap`, the custom colormap
  built when the module is imported, is a genuine matplotlib `Colormap` with 10 000
  levels.
- **`test_bundled_arf_structure`.** Opens the bundled ARF and checks its layout:
  the extensions are exactly `SPECRESP` followed by `SPECRESP_BL1` … `BL10`; every
  baseline table uses the same energy grid as the total; and the `BASELINE` header
  values increase from 0.05 to 1.0 m. `generate_event_list` and the notebook rely on
  this layout.
- **`test_bundled_arf_total_is_sum_of_baselines`.** Checks that the total effective
  area equals the sum of the ten per-baseline areas, to the precision of the
  32-bit floats stored in the file (relative 10⁻⁵).
- **`test_bundled_rmf_structure`.** Checks the bundled RMF: a 951 × 951 matrix,
  channels numbered 0 … 950, an energy grid identical to the ARF's, and no negative
  probabilities. (The known problems with this file — rows summing to ~10⁻³ rather
  than 1 and a misplaced `TLMIN` keyword — are not asserted here; the
  renormalisation is characterised in `test_process.py`.)
- **`test_bundled_simput_structure`.** Checks the bundled source model: two sources
  separated by 400 µas (±1 %); both spectra on the same energy grid; and every
  spectral energy inside the ARF's energy range. The last property matters because
  the event-list generator currently assumes the two grids coincide (a known bug).

---

## 3. `test_instrument.py` — baselines, the interferometer and pointing

Module constants: the Willingale geometry `W_UM = 300` µm, `L_M = 10` m and the
resulting beam angle `THETA_B = W/L = 3·10⁻⁵` rad; a test baseline `D_M = 0.5` m
with focal length `F_M = D/θ_b`.

- **`assert_consistent(bl)`** (helper). Asserts the two geometric relations the
  `baseline` class is supposed to enforce: D = F·θ_b and W = θ_b·L.

### Baseline geometry
- **`test_baseline_sufficient_inputs_give_same_geometry`** (6 cases). A baseline
  is determined by five linked quantities (D, F, W, L, θ_b) with two relations
  between them, so three well-chosen inputs suffice. The test gives six different
  sufficient combinations (e.g. D+W+L, F+θ_b+W, …) and checks that each produces
  the same complete geometry, satisfies both relations, and stores W in metres even
  though it is given in µm. This exercises every branch of the constructor's
  "fill in the missing quantities" logic.
- **`test_baseline_willingale_parameters`.** `add_willingale_baseline(0.3)` builds a
  baseline whose other parameters are fixed inside the method (30 pairs, L = 10 m,
  W = 300 µm, bench length 7 m). The test checks those values, and the derived
  ones: θ_b = 3·10⁻⁵, F = D/θ_b, and the grazing angle arctan(D/2 / 7 m)/2. No
  mirror reflectivity is attached.
- **`test_baseline_mismatched_d_f_beam_angle_raises`.** Gives D, F and θ_b that
  violate D = F·θ_b and checks that the constructor refuses with an error message
  mentioning the baseline length.
- **`test_baseline_mismatched_w_l_beam_angle_raises`.** Same idea for the second
  relation: inconsistent W, L, θ_b must raise an error mentioning the beam width.
- **`test_baseline_underspecified_raises`** (4 cases). Gives too few inputs (e.g. D
  and W but no L) and checks that each case raises an error whose message names the
  quantity that cannot be determined.
- **`test_baseline_without_grazing_angle_or_bench_warns`.** Without a grazing angle
  or a bench length the grazing angle cannot be computed; the constructor should
  warn (not fail) and leave it as `None`.
- **`test_baseline_grazing_angle_from_bench_length`.** With only a bench length,
  the grazing angle is computed as arctan(D / 2 / bench_length) / 2.
- **`test_baseline_explicit_grazing_angle_wins`.** If both are given, the explicit
  grazing angle is kept rather than recomputed from the bench length.
- **`test_baseline_invalid_num_pairs_raises`** (3 cases: 0, −3, 2.0). The number of
  mirror pairs must be a positive integer; zero, negative and float values raise.
- **`test_baseline_accepts_numpy_int_num_pairs`** — *known bug*. A NumPy integer
  (e.g. from `np.arange`) is a perfectly good number of pairs, but the constructor's
  type check accepts only Python `int`, so this currently raises.

### Mirror reflectivity files
- **`write_henke(path, energies, reflectivity)`** (helper). Writes a small file in
  the format downloaded from the Henke/CXRO web site: two header lines and then
  "energy reflectivity" rows. The grazing angle is encoded in the file name
  (e.g. `Pt_0.5deg.dat`), which is how the package reads it.
- **`test_baseline_mirror_reflectivity_files_parsed`.** Passes two such files to a
  baseline and checks that `mirr_reflec` becomes a 2 × 2 table of
  (angle, data) with the angles 0.5 and 1.5 taken from the file names, and that the
  energy and reflectivity columns were read correctly (to the 6 decimals written).
- **`test_baseline_mirror_reflectivity_multidigit_angle`** — *known bug*. A file
  named `Pt_12.5deg.dat` should give angle 12.5; the regular expression in the code
  is "greedy" and currently extracts 2.5.

### Interferometer construction
- **`test_interferometer_defaults`.** A freshly created interferometer has no
  baselines, a time step of 1, no wobble or roll functions, and zero wobble
  amplitude, roll speed and initial roll.
- **`test_add_baseline_forwards_parameters`.** `add_baseline` creates a baseline from
  its arguments and appends it to `inst.baselines`; the stored baseline has the
  requested number of pairs and D, and a consistent geometry.
- **`test_add_baseline_mirror_files_positional`.** `add_baseline` calls its mirror
  file argument `mirr_mater` (while `baseline` calls it `mirr_reflec`); the test
  checks that files passed under the current name do reach the baseline.
- **`test_add_baseline_accepts_mirr_reflec_keyword`** — *known bug*. The same,
  using the `baseline` class's keyword name `mirr_reflec`; currently rejected
  because the two names differ.
- **`test_add_willingale_baselines_and_clear`.** Adds Willingale baselines of 0.05,
  0.5 and 1 m, checks they are stored in that order, then checks `clear_baselines`
  empties the list.

### Pointing: roll and wobble
The pointing is an array with one row per time step (t_exp + 2 rows) and columns
pitch, yaw, roll in radians. The *roller* fills the roll column; the *wobbler*
should add random pointing jitter to pitch and yaw.

- **`test_gen_pointing_without_roller_is_zero`.** With no roller, `gen_pointing(10)`
  returns a 12 × 3 array of zeros.
- **`test_smooth_roller_linear_roll`.** The smooth roller rotates the instrument at
  a constant rate so that it turns through half a turn (π) over the observation:
  roll[k] = roll_init + k·π/N for N = 100 rows. The test checks the whole roll
  column, that pitch and yaw stay zero, and that the roller stores its speed π/N in
  `roll_speed`. (Half a turn suffices because the uv plane is symmetric.)
- **`test_smooth_roller_scales_with_time_step`.** With `time_step=2`, each row's roll
  increment doubles.
- **`test_smooth_roller_respects_roll_speed`** — *known bug, convention pending*.
  If the user specifies `roll_speed`, one would expect roll[k] = k·roll_speed; the
  roller currently overwrites the user's value with π/N. Whether a user value
  should take precedence is part of the deferred convention question.
- **`test_discrete_roller_single_step`.** The discrete roller holds the roll fixed
  and then jumps: currently roll is 0 up to time `roll_stop_t` and `roll_stop_a`
  afterwards. The test checks this exact step function.
- **`test_discrete_roller_repeats_steps`** — *known bug, convention pending*. Over
  a long observation a "stop-and-go" roller should step repeatedly, producing
  several distinct roll angles; it currently steps only once.
- **`test_random_wobble_first_row_and_step_size`.** Applies `random_wobble` with an
  amplitude of 2 arcsec to a 10 000-row pointing. Row 0 must stay unchanged, roll
  must stay zero, and the pitch/yaw offsets must have a standard deviation of
  2 arcsec in radians. With ~20 000 samples the standard deviation is known to
  ~0.5 %, so the 3 % tolerance is beyond 5σ.
- **`test_random_wobble_is_random_walk`** — *known bug*. A physical pointing
  jitter is a random walk: each step adds a small random increment, so differences
  between consecutive rows have standard deviation σ. The code instead draws an
  independent offset for every row, so consecutive differences have √2·σ. This
  test asserts the random-walk property.
- **`test_file_wobble_reads_csv`.** Writes random pitch/yaw values to a CSV file and
  checks that `file_wobble` copies them into the pitch/yaw columns while leaving a
  pre-set roll column (7.0) untouched.
- **`test_gen_pointing_applies_wobbler`** — *known bug*. Setting a wobbler with a
  non-zero amplitude should make pitch/yaw non-zero; `gen_pointing` never calls the
  wobbler, so they stay zero.
- **`test_gen_pointing_reports_roller_errors`** — *known bug*. Defines a roller
  that deliberately raises `RuntimeError` and checks that the error reaches the
  caller. `gen_pointing` wraps the roller call in a `try/except` that silently
  discards all errors, which hides genuine bugs.

### `plot_arf`
- **`test_plot_arf_smoke`** (2 cases). Plots the bundled ARF. In normal mode one line
  (the total area) is drawn; in baseline mode 11 lines (total plus ten baselines).
  Counting lines is a cheap check that the function plotted what was asked.

---

## 4. `test_images.py` — the image container and photon generators

An `image` holds per-photon arrays: `energies` (J), `loc` (n × 2 offsets in rad),
`toa` (integer arrival-time index) plus optional metadata (spectrum,
baseline indices, background indices).

Helpers in this file:
- **`assert_binomial(count, n, p)`.** Asserts that `count` successes out of `n` is
  consistent with probability `p` within 5σ, σ = √(np(1−p)). Used whenever photons
  are randomly assigned to sources or pixels.
- **`energy_is(im, e_kev)`.** Returns a true/false mask of photons whose energy
  equals `e_kev` keV. It compares *relatively* (rtol 10⁻⁹, atol 0): energies in
  joules are ~10⁻¹⁶, far below `np.isclose`'s default absolute tolerance of 10⁻⁸,
  which would make every energy "equal" to every other.
- **`flat_spectrum(...)`.** A two-column (energy, flux) array: 13 energies from 1 to
  7 keV with flux 1 — the format the generators accept as `spectrum`.
- **`write_png(path, array)`.** Saves a grayscale array as a PNG for
  `generate_from_image`.
- **`disc_radii(im, alpha, beta)`.** Distance of each photon from a given centre.
- **`two_pixel_png`** (fixture). A 10 × 10 PNG that is black except for pixel (2, 3)
  of brightness 100 and pixel (2, 7) of brightness 200.
- **`cube`** (fixture). A four-row pandas table in the format of
  `draw_photons_from_cube`: two energy slices (1 and 2 keV), each with total flux 4;
  in the 1 keV slice the two positions have fluxes 1 and 3.
- **`event_table(with_baseline)`.** A three-event FITS table in the format written by
  `generate_event_list`, optionally without the `BASELINE_ID` column.

### Container
- **`test_image_container_defaults`.** `image(7)` allocates zero-filled arrays of the
  right shapes (with integer `toa`), and empty metadata.

### Point sources
- **`test_point_source_units_and_times`.** A point source at (3, −2) arcsec, 6.4 keV,
  50 photons: every photon has energy 6.4 keV converted to joules, offset converted
  to radians, and arrival indices 0 … 49.
- **`test_point_source_spectrum_array_and_default_energy`.** With a spectrum array
  and no energy, the generator stores the spectrum (the same object, not a copy)
  and assigns every photon the mean energy of the spectrum's energy column.
- **`test_point_source_spectrum_from_file`.** A spectrum given as a file path is read
  from the file; an explicit energy still sets the photon energies.
- **`test_generators_require_energy_or_spectrum`** (5 generators). Each generator
  must refuse to run with neither an energy nor a spectrum, with a message saying
  so.
- **`test_generators_store_spectrum`** (4 generators × array/file = 8 cases).
  Each generator stores a spectrum given either as an array or as a file name;
  meanwhile the explicitly given energies (1–2 keV) are still used for the photons
  — the spectrum (1–7 keV) is stored for later use but not sampled.
- **`test_disc_spectrum_default_energy`.** The disc generator, given a spectrum and
  no energy, uses the spectrum's mean energy (the same rule as `point_source`).
- **`test_generate_from_image_spectrum_from_file`.** `generate_from_image` loads a
  spectrum given as a file path.
- **`test_double_point_source_assignment_and_fractions`.** Two sources at different
  positions *and* different energies (1 and 6 keV), 4000 photons. Because energies
  differ, each photon's energy tells which source it came from; the test checks that
  every photon belongs to exactly one source, that its position is that source's
  position (energy and position were not mixed up), that times run 0 … n−1, and that
  each source received about half the photons (binomial 5σ).
- **`test_m_point_sources_fractions`.** Three sources with distinct energies, 6000
  photons: each gets about a third (binomial 5σ) and is at its own position.
- **`test_multi_generators_accept_spectrum_without_energy`** — *known bug*
  (3 generators). Given a spectrum and no energy, the multi-source generators
  compute a single scalar "mean energy" but then index it as if it were a list of
  per-source energies, which fails. The test asks for finite positive energies.
- **`test_multichromatic_range_uniform`.** `point_source_multichromatic_range` with
  [2, 8] keV, 20 000 photons: energies must lie in the range, have mean 5 keV within
  5σ (σ of a uniform distribution's mean = (E₁−E₀)/√(12n)) and variance
  (E₁−E₀)²/12 within 5 %. These three properties together identify a uniform
  distribution well.
- **`test_multichromatic_gauss_mean_and_spread`.** `point_source_multichromatic_gauss`
  with mean 6.4 keV and σ = 0.2 keV: sample mean within 5σ/√n of 6.4 and sample
  standard deviation within 3 % of 0.2 keV, plus positions and times. This also
  checks the spread is applied in keV before conversion to joules.

### Disc
- **`test_disc_photons_within_radius`.** All 5000 photons of a 0.5 arcsec disc lie
  within 0.5 arcsec of its centre, and with no energy spread all energies equal the
  given energy.
- **`test_disc_centroid`.** The mean photon position equals the disc centre within
  5σ, using R/√n as a (generous) σ for the centroid.
- **`test_disc_radius_distribution_current`** — *convention pending*. Characterises
  the current radial law: radius = U·R with U uniform, so half the photons fall
  inside R/2. This concentrates photons towards the centre.
- **`test_disc_uniform_areal_density`** — *known bug, convention pending*. If
  "disc" means uniform surface brightness, the fraction inside R/2 is (1/2)² = 1/4
  (achieved with radius = √U·R). Exactly one of this and the previous test passes;
  which one is right is the deferred convention question.
- **`test_disc_energy_spread_in_kev`** — *known bug*. A 0.1 keV energy spread
  should give a 0.1 keV standard deviation; the code adds the spread (a keV
  number, ~0.1) to energies already in joules (~10⁻¹⁶), so the noise completely
  swamps the energies and the result is meaningless (even negative).

### `generate_from_image`
- **`test_generate_from_image_single_pixel`.** An 8 × 10 image with one bright
  pixel: all 200 photons come from one position, have 2 keV in joules and times
  0 … 199, and the returned `pix_scale` equals the image shape.
- **`test_generate_from_image_pmf_and_scale`.** With the two-pixel image, photons
  must come only from the two lit pixels, in proportion to brightness (2/3 from the
  brighter pixel, binomial 5σ). The two origins are 4 pixels apart; with
  `img_scale` 1 arcsec spread over the 10-pixel image this is 0.4 arcsec, which
  checks the pixel-to-angle conversion.
- **`test_generate_from_image_offset`.** Runs the generator twice from the same seed,
  once with an offset of (0.3, −0.2) arcsec, and checks that every photon is shifted
  by exactly that offset (converted to radians). Reseeding makes both runs draw the
  same pixels, so the only difference is the offset.
- **`test_generate_from_image_orientation_current`** — *convention pending*. Pins the
  current mapping from pixel (row, column) to sky offset: row → `loc[:, 0]`, column →
  `loc[:, 1]`, measured from the image centre, scaled by `img_scale / max(H, W)`,
  with no vertical flip. Whether images should be flipped (FITS/PNG row order) and
  which axis is RA is undecided.
- **`test_generate_from_image_energy_spread`.** A 0.2 keV energy spread on 5 keV
  photons gives a standard deviation of 0.2 keV (3 %) and mean 5 keV (5σ). Unlike
  the disc, this generator handles the units correctly.
- **`test_generate_from_image_spectrum_default_energy`.** Spectrum and no energy →
  mean spectral energy, as for the other generators.
- **`test_generate_from_image_requires_energy_or_spectrum`.** Neither given → error.
- **`test_generate_from_image_background_photons`** — *known bug*. Asking for 50
  background photons at 0.5 keV should give 150 photons, 50 of them listed in
  `bkg_indices` with energy 0.5 keV. The background code calls `np.in1d`, which was
  removed in NumPy 2.4, so any use of background crashes.
- **`test_generate_from_image_background_fraction`** — *known bug*. A fractional
  `bkg_phot=0.25` means 25 % of the source photons; same crash.
- **`test_generate_from_image_background_with_energy_spread`** — *known bug*.
  Background plus energy spread; behind the crash there is a second bug
  (array size mismatch).
- **`test_generate_from_image_background_spectrum`** — *known bug*. With a
  background spectrum, a variable used later (`pixel_locations`) is never defined.

### `draw_photons_from_cube`
This function draws photons from a 3-D (energy, α, β) flux cube stored as a pandas
table. It uses its own random generator, which pytest's seeding does not control,
so these tests rely on the 5σ tolerances alone. All but the first currently fail
because the function crashes under pandas 3.

- **`test_draw_photons_from_cube_zero_flux_raises`.** A cube with no positive flux
  must raise `ValueError` (this check runs before the crash).
- **`test_draw_photons_from_cube_energy_and_spatial_distribution`** — *known bug*.
  Two energy slices with equal total flux → about half the photons at each energy;
  inside the 1 keV slice, positions drawn in the ratio 1:3 (binomial 5σ each).
- **`test_draw_photons_from_cube_does_not_mutate_input`** — *known bug*. The
  function normalises weights; it must not modify the caller's table while doing so.
- **`test_draw_photons_from_cube_loc_units_current`** — *known bug, convention
  pending*. Pins the current behaviour that α/β are copied into `loc` without unit
  conversion, unlike every other generator.

### `append`
- **`test_append_concatenates_and_offsets_toa`.** Appending a 5-photon image to a
  10-photon image gives 15 photons with energies and positions in order, and the
  second image's times shifted to follow the first (0 … 14).
- **`test_append_keeps_baseline_indices`** — *known bug*. If both images carry
  baseline indices, the result should too; `append` drops them.
- **`test_append_empty_first_image`** — *known bug*. Appending to an empty image
  should return the second image's photons; it fails because it reads the last time
  of an empty array.

### `create_image_from_events`
- **`test_create_image_from_events_energies_times_baselines`.** From a three-event
  table: energies converted keV → J, times truncated to integer indices
  (3.7 → 3, 9.999 → 9), and `BASELINE_ID` copied to `baseline_indices`.
- **`test_create_image_from_events_without_baseline_column`.** Without a
  `BASELINE_ID` column, `baseline_indices` stays `None`.
- **`test_create_image_from_events_positions_current`** — *convention pending*. Pins
  the current reading of positions: RA and DEC in degrees converted to radians, with
  **DEC in `loc[:, 0]` and RA in `loc[:, 1]`**. Whether this order (and the sign
  conventions downstream) is intended is undecided.

---

## 5. `test_process.py` — Fresnel diffraction, detector positions, event lists

Helpers and constants:
- **`fre_dif`** is `interferometer_data.fre_dif` taken from the class; it never uses
  `self`, so the tests call it directly with `None` in that place.
- **`N_PHOTONS = 20 000`**, **`PIXEL = 5 µm`** (the detector pixel hard-coded in the
  package).
- **`wavelength(e_kev)`** — λ = hc/E.
- **`make_baseline(W_um, L, D)`** — a 30-pair baseline with chosen slit width and
  length, for Fresnel tests.
- **`willingale(*ds)`** — an interferometer with Willingale baselines of the given
  lengths.
- **`point_image(n, e_kev, beta_arcsec, alpha_arcsec, baselines)`** — a monochromatic
  point source with baseline indices filled in by hand. (Generator images lack
  them and cannot be processed directly — a known bug.)
- **`fringe_phasor(y, s)`** — the average of exp(2πi·y/s) over photon positions y.
  This is the key analysis tool for fringes: for an intensity pattern
  1 + V·cos(2πy/s + φ), the average is (V/2)·e^{iφ}. So its magnitude measures the
  fringe visibility (0.5 for perfect fringes) and its angle the fringe phase
  (position). For photons with no fringe at spacing s it averages to ~0.
- **`phasor_sigma(n)`** — the statistical uncertainty of each component of the
  phasor for n photons, at most √(1/(2n)); the basis of the 5σ tolerances.

### `fre_dif` (Fresnel diffraction by one slit)
- **`test_fre_dif_symmetric_and_u_coordinate`.** The diffraction pattern of a slit
  centred at 0 is symmetric, I(y) = I(−y); and the returned dimensionless Fresnel
  coordinate is u = y·√(2/(λL)).
- **`test_fre_dif_default_samples_span_slit`.** Without explicit positions, the
  function evaluates `samples` evenly spaced points from −W/2 to +W/2.
- **`test_fre_dif_geometric_limit`.** A wide slit close to the detector (Fresnel
  number ~120) behaves like geometric optics: the intensity at the slit centre is the
  geometric value (2 in the code's normalisation, within 2 %) and essentially zero
  three slit widths outside.
- **`test_fre_dif_fraunhofer_limit`.** A very narrow slit (Fresnel number 0.1) gives
  the far-field pattern: the normalised intensity follows sinc²(W·y/(λL)) out to the
  third minimum within 1 %. These two limits bracket the regime used by the
  instrument and check the Fresnel-integral implementation against textbook optics.
- **`test_fre_dif_below_accept_reject_box`** (5 energies, 0.5–10 keV). Detector
  positions are drawn by *accept-reject sampling*: pick a random position, pick a
  random height between 0 and a ceiling of 2.7·I(0), and keep the position if the
  height is below I(y). This is only correct if the ceiling is above the curve
  everywhere (otherwise the peaks are clipped). The test checks that for a Willingale
  baseline the maximum of I over the slit is below 2.7·I(0) at all energies tested.

### `interferometer_data` (photon → detector position)
- **`on_axis_data`** (module fixture). Simulates 20 000 on-axis 1 keV photons through
  one Willingale baseline (D = 0.5 m) once, shared by the next five tests. It reseeds
  with `SEED + 1` so its result does not depend on which test runs first.
- **`test_interferometer_data_passthrough_and_shapes`.** The result has one detector
  position per photon; energies and times are the *same* arrays as in the input
  image (not copies); baseline indices are copied; and the pointing array has t+2
  rows.
- **`test_interferometer_data_pixel_quantisation`.** `pos` (the detected position) is
  the true position `actual_pos` rounded to the 5 µm pixel grid: every `pos` is a
  multiple of 5 µm, and no photon moved by more than half a pixel.
- **`test_interferometer_data_positions_inside_detector_box`.** All on-axis photons
  land within the detector, ±W·num_pairs/2 (the region covered by the slits).
- **`test_interferometer_data_fringe_period`.** Physics check of the fringe spacing.
  For beams crossing at angle θ_b the fringe spacing is s = λ/θ_b. The test computes
  the phasor magnitude (a periodogram) for 401 trial spacings from 0.67s to 2s and
  checks that the strongest one is s, within 1 %.
- **`test_interferometer_data_on_axis_fringe_phase`.** For an on-axis source the
  central bright fringe is at y = 0: the phasor must be ≈ 0.5 and real (phase 0).
  The tolerance is 5σ plus 0.01 for the incomplete fringes at the edges of the
  detector.
- **`test_pure_fringes_visibility`** (1 and 6 keV). With `pure_fringes=True` (no slit
  diffraction envelope) the fringes are perfect: phasor 0.5 and real, at both
  energies, which also checks that s scales with λ.
- **`test_pure_diffraction_has_no_fringes`.** With `pure_diffraction=True` (no
  interference) there must be no structure at the fringe spacing: |phasor| ≈ 0.
- **`test_pure_diffraction_envelope_matches_fresnel_profile`** (even and odd number of
  pairs). With pure diffraction, photons from all slits are "folded" back into one
  slit width (subtracting whole slit widths; the fold is offset by W/2 when the
  number of slits is odd because then a slit, not a gap, sits at y = 0). The folded
  positions are histogrammed in 20 bins and compared with the Fresnel intensity
  from `fre_dif` integrated over each bin, with a χ² test: χ² must be below
  dof + 5√(2·dof) (the mean plus 5σ of the χ² distribution, dof = 19). This shows
  that the accept-reject sampler reproduces the diffraction pattern, not just its
  width.
- **`test_off_axis_fringe_shift_magnitude`.** A source off axis by angle θ shifts the
  fringe pattern by F_eff·θ with F_eff = D / (2 sin(θ_b/2)). The test chooses θ so
  the shift is exactly a quarter fringe and checks the phasor phase is ±π/2 while its
  magnitude stays 0.5. (The phase tolerance is 10× the component σ because the
  error on the angle is σ/|phasor| = 2σ; so this is still 5σ in angle.) This is the
  core relation that makes imaging possible: fringe position encodes sky position.
- **`test_off_axis_projection_current`** — *convention pending*. At roll angle 0 only
  the second `loc` coordinate (β) shifts the fringes, with a sign such that β > 0
  gives phase +π/2; an offset in α gives no shift. Which sky axis that corresponds to
  and the sign are undecided.
- **`test_far_off_axis_warns`.** A source 1 arcsec off axis shifts its central
  fringe off the detector; the code should warn ("outside of the FOV").
- **`test_multiple_baselines_use_own_fringe_spacing`.** Two baselines with θ_b =
  3·10⁻⁵ and 6·10⁻⁵ (L = 10 and 5 m), half the photons each. Each half must show
  strong fringes at its *own* spacing (|phasor| > 0.4) and none at the other's
  (< 0.1). This catches a mix-up where all photons would be processed with one
  baseline's parameters.
- **`test_interferometer_data_reproducible_with_seed`.** Two runs from the same seed
  give identical positions — the precondition for every other reproducibility claim.
- **`test_interferometer_data_copy_is_independent_container`.** `copy()` returns a new
  object that shares the underlying arrays (a "shallow" copy); the test documents
  that sharing.
- **`test_interferometer_data_accepts_generator_image`** — *known bug*. An image
  straight from `images.point_source` has no baseline indices, so
  `interferometer_data` crashes with a `TypeError`; the test asks that it work.
  (How baselines should then be assigned is left to the fix; the code that used to
  do it, `process_photon_e_base`, is disabled dead code.)

### `generate_event_list` (SIMPUT × ARF → Poisson counts → RMF → FITS)
- **`read_events(path)`** (helper). Reads the `EVENTS` table and its header from a
  written event list.
- **`identity_events`** (fixture). Runs `generate_event_list` on the
  `synthetic_responses` setup for 100 s and returns both the setup and the events,
  for the next five tests.
- **`test_event_list_counts_per_baseline`.** Expected counts per baseline are
  Σ flux × area × bin width × time = 0.05 × 10 × 0.1 × 60 × 100 = 300 for baseline 0 and
  900 for baseline 1; the observed counts must agree within 5√expected (Poisson 5σ),
  and no other baseline ids appear.
- **`test_event_list_identity_rmf_energies`.** With the identity RMF the detected
  energy equals the true energy, and both are bin centres of the grid. (The file
  stores 32-bit floats, so "is a bin centre" is checked as "within 10⁻⁵ keV of the
  nearest centre" rather than by exact membership.)
- **`test_event_list_flat_spectrum_energy_distribution`.** Flat spectrum × flat area →
  flat energy distribution: half the events lie above 4 keV (binomial 5σ).
- **`test_event_list_times`.** Arrival times lie in [0, T] and are rounded to 1 ms.
- **`test_event_list_file_format`.** The `EVENTS` extension has exactly the
  documented columns in order (`BASELINE_ID`, `TRUE_ENERGY`, `DETECTED_ENERGY`,
  `TIME`, `RA_POS`, `DEC_POS`), the expected `TELESCOP`/`INSTRUME` keywords, and keV
  and s units.
- **`test_event_list_default_output_and_message`.** Without an output file name, the
  function writes `event_list.fits` into the current directory and prints the number
  of photons. The test moves into the temporary directory first (`monkeypatch.chdir`)
  so the repository is not touched, and captures printed output with `capsys`.
- **`test_event_list_source_positions_and_fractions`.** Two sources with flux ratio
  1:2: every event carries one of the two catalogue positions, and 2/3 of events
  come from the brighter source (binomial 5σ).
- **`test_event_list_rmf_redistribution`.** An RMF that keeps 70 % of photons in their
  channel and moves 30 % up by one channel: the detected minus true energy is always
  0 or exactly one bin, and 70 % of events have no shift (binomial 5σ). Events in the
  last bin are excluded because that row has nowhere to move to.
- **`test_event_list_rmf_rows_renormalised`** — *convention pending*. Pins the
  current behaviour that the code renormalises each RMF row to sum to 1: an identity
  scaled by 10⁻³ gives the same counts and energies as the identity. In the OGIP
  convention RMF rows can sum to less than 1 (detector efficiency); the bundled RMF's
  rows sum to ~10⁻³, so whether to renormalise is a real decision.
- **`test_event_list_photons_without_response_dropped_consistently`** — *known
  bug*. Gives the lowest 10 energy bins a zero RMF row. Photons there must be
  dropped from *all* columns; the code drops them from detected energy, baseline and
  time but not from true energy or RA/DEC, so the columns fall out of alignment.
- **`test_event_list_zero_flux_writes_empty_list`** — *known bug*. A zero-flux
  source should produce an empty event list; the code crashes on the empty case.
- **`test_event_list_different_source_grid`** — *known bug*. A source spectrum on a
  coarser energy grid than the ARF should still give the integrated number of counts
  (0.05 × 10 × 6 keV × 100 s = 300); the code assumes the two grids are identical.
- **`test_event_list_rmf_channels_starting_at_one`** — *known bug*. Many RMFs
  number channels from 1 (`TLMIN = 1`). An identity RMF numbered that way should still
  map each energy to itself; the code treats channel numbers as 0-based indices and
  shifts every event by one channel.
- **`test_event_list_to_detector_positions`.** A short integration test of the first
  half of the workflow on synthetic files: event list → `create_image_from_events` →
  `interferometer_data` with Willingale baselines of 0.05 and 1 m (matching the
  synthetic ARF headers). Every event gets a finite detector position on its
  baseline's detector.

---

## 6. `test_analysis.py` — reconstruction, filtering, plots

Setup for the reconstruction tests: ten Willingale baselines from 0.1 to 1 m with
the smooth roller, 1 keV photons, and a 64 × 64 image with 2·10⁻¹⁰ rad pixels. The
pixel must be finer than λ/(2·D_max) = 6.2·10⁻¹⁰ rad (the Nyquist limit set by the
longest baseline); coarser pixels alias or crash (a known bug).

Helpers:
- **`rolling_interferometer()`** — that ten-baseline rolling instrument.
- **`observe(img, inst)`** — assigns photons to baselines in turn
  (0, 1, …, 9, 0, 1, …) so that every baseline sees every roll angle, then runs
  `interferometer_data`.
- **`reconstruct(data, inst, ...)`** — calls `image_recon_smooth` with the standard
  64 × 64 / 2·10⁻¹⁰ rad setup and verbose output off, unless overridden.
- **`peak(image)`** — the (row, column) of the brightest pixel.
- **`on_axis`** (module fixture). 10 000 on-axis photons observed and reconstructed
  once, shared by many tests (seeded with `SEED + 2`).
- **`broadband`** (module fixture). 3000 photons uniform in 1–7 keV, observed but not
  reconstructed, for the `filter_data` tests.

### Fourier helpers
- **`test_ft_data_sinusoid`.** The Fourier transform of a cosine that fits exactly
  20 periods in the sample window has its two largest components at frequencies +f₀
  and −f₀, with amplitude 1/2; the frequency axis is NumPy's standard one. This pins
  the normalisation of `ft_data`.
- **`test_plot_ft_axis_scales`** (3 cases). `plot_ft` draws one line, with the
  frequencies reordered to run from negative to positive (`fftshift`), labelled
  "Baseline N", and with linear, semi-log or log-log axes for `log` = 0, 1, 2.
- **`test_hist_data_labels`** (2 cases). `hist_data` draws a 20-bin histogram whose
  x-axis label says pixels or micrometres as requested.

### `image_recon_smooth`
Reconstruction principle: each photon's detector position gives one complex
Fourier sample (a "visibility") at a uv point determined by its baseline length,
wavelength and the roll angle at its arrival time. The samples are gridded and
inverse-Fourier-transformed into an image.

- **`test_recon_output_structure`.** The function returns
  `((image, scale), uv, reconstruction_data)`. The image is real, finite and of the
  requested 64 × 64 shape; there are two uv points and two Fourier values per photon;
  and the `reconstruction_data` object holds consistent arrays (square Fourier grid,
  matching frequency axis, and the same `uv`/`image` objects that were returned).
- **`test_recon_uv_hermitian_pairs`.** A real sky has a Fourier transform with
  F(−u) = F(u)*. The code exploits this by adding, for every photon, the mirrored
  point −uv with the complex-conjugate value; the test checks the interleaving and
  that each value has unit magnitude (each photon is a pure phase).
- **`test_recon_uv_coordinates`.** The uv point of each photon has length D/λ and
  direction set by the roll angle at its arrival time: (u, v) = (D/λ)(sin roll,
  cos roll). The code orders photons by baseline; the test reproduces that order
  and compares to 10⁻⁹.
- **`test_recon_grid_size_formula`.** The Fourier grid size follows the formula in the
  code, n = ⌈1 / (2·d·(D_min/λ_max)·error)⌉ with pixel d and the default
  `error = 0.1`, and the frequency axis matches. This pins the grid sizing, which
  controls memory use (the notebook's ~7 GB comes from here).
- **`test_recon_on_axis_peak_at_centre`.** An on-axis source reconstructs to a peak in
  the central pixel (32, 32).
- **`test_recon_offset_source_position`** (4 offsets). A source displaced by a known
  number of pixels in α and β must reconstruct at centre + (α, β) in array (row,
  column), within one pixel. This is convention-independent in the sense that it
  checks the image reproduces the input `loc` coordinates; how `loc` relates to RA/DEC
  and to the displayed orientation is a separate, undecided convention.
- **`test_recon_two_sources_resolved`.** Two equal sources 20 pixels apart (the
  resolution λ/(2·D_max) is ~3 pixels): both expected positions are bright
  (> 0.6 of the maximum) and the midpoint is less than half as bright, i.e. the
  sources are resolved.
- **`test_recon_verbose_output`.** With `verbose=False` nothing is printed; with
  `verbose=True` progress messages appear.
- **`test_recon_scale_value_current`** — *convention pending*. Pins the current value
  of the second element of the returned image tuple (a "scale"), which equals
  samples × (grid size × pixel) in µas. What it is meant to be is unclear.
- **`test_recon_rejects_pixels_coarser_than_nyquist`** — *known bug* (2 cases).
  With pixels coarser than the Nyquist limit, the code either produces a silently
  aliased image (1·10⁻⁹ rad) or crashes with an obscure `IndexError` (2·10⁻⁹ rad).
  The test asks for a clear `ValueError` in both cases.
- **`test_recon_small_grid_returns_full_grid`** — *known bug*. When the Fourier grid
  is smaller than the requested image, the whole grid should be returned; the
  cropping code drops the last row and column.

### `filter_data`
- **`test_filter_data_energy_and_baseline_masks`.** Filtering to 2–5 keV and baselines
  3–6 keeps exactly the photons satisfying E_lo < E < E_hi and b_lo ≤ b ≤ b_hi
  (strict for energy, inclusive for baselines), keeps all per-photon arrays aligned,
  and prints the number kept.
- **`test_filter_data_without_baseline_bounds`.** With no baseline bounds only the
  energy cut is applied.
- **`test_filter_data_leaves_input_unchanged`.** Filtering returns a new object; the
  input keeps all its photons.
- **`test_filter_data_output_reconstructs`.** A filtered data set can be passed to
  `image_recon_smooth` and still gives a centred peak. The cut is 1–1.5 keV because at
  higher energies the 2·10⁻¹⁰ rad pixel would violate the Nyquist limit.
- **`test_filter_data_filters_all_per_photon_arrays`** — *known bug*. The
  `actual_pos` and `actual_energies` arrays (true values before pixelisation and
  energy redistribution) should be filtered too; they are currently left at full
  length.

### Plot helpers (smoke tests)
- **`test_plot_inf_image`** (`return_fig` True/False). Without `return_fig` nothing is
  returned. With it, a `Figure` is returned whose single image spans ±FOV/2 in µas on
  both axes and is normalised to a maximum of 1.
- **`test_plot_uv`.** The uv plot shows the uv points with u converted to units of
  10⁹ wavelengths (Gλ).
- **`test_plot_res_hist`.** `plot_res_hist` histograms the angular scale 1/(2|uv|) on
  log-log axes as a single filled step histogram.
- **`test_subplots`.** The combined figure contains the image, the uv coverage and a
  colorbar (three axes).
- **`test_plot_spec_per_baseline`.** Generates a synthetic event list with two
  baselines and checks the spectrum plot's legend: "All Baselines", "Baseline 1",
  "Baseline 2".

---

## 7. `test_workflow.py` — end-to-end runs (all `slow`)

These tests run the same chain as `simple_example.ipynb` on the bundled model files,
with a 1000 s exposure instead of 10⁴ s to keep runtime down.

- **`expected_counts(arf_path, simput_path, exposure, e_min)`** (helper). Computes the
  expected number of events per baseline directly from the files: for each source
  and baseline, Σ flux(E) × area(E) × bin width × exposure, optionally only above
  `e_min`. This is the same model `generate_event_list` implements, computed
  independently.
- **`workflow`** (module fixture). Runs the whole chain once — event list, image,
  interferometer with Willingale baselines read from the ARF headers and the smooth
  roller, detector positions — and shares the results with the tests below.
- **`test_workflow_counts_per_baseline`.** Observed event counts for each of the ten
  baselines agree with `expected_counts` within Poisson 5σ.
- **`test_workflow_hard_fraction_per_baseline`.** For each baseline, the fraction of
  events above 2 keV agrees with the fraction predicted from the ARF-folded spectrum
  (binomial 5σ, plus one count of slack for baselines with very few events). It also
  checks that this fraction is more than ten times larger for the shortest baseline
  than for the longest, reflecting that longer baselines have larger grazing angles
  (arctan(D/2 / bench length)/2) and so lose high-energy reflectivity.
- **`test_workflow_source_positions`.** Every event carries one of the two catalogue
  positions, and both appear.
- **`test_workflow_detector_positions`.** Every photon gets a finite detector position
  inside its detector.
- **`test_workflow_long_baselines_resolve_double_source`.** The scientific check of
  the notebook: with baselines 7–9 and 1–10 keV (about 400 photons), the two sources
  400 µas apart (±64 pixels of 3.125 µas from the centre) must be resolved. The test
  measures each source's peak as the maximum within ±8 pixels of its expected
  position and estimates the image noise from the rms in a corner far from the
  sources. It then requires that the brighter source (the one with more photons)
  holds the global maximum; the fainter peak exceeds 3× the noise; the ratio of the
  two peaks equals the ratio of their photon counts within 5× the noise; and the
  midpoint between them is fainter than the fainter peak. Because the axis on which
  the sources appear depends on an undecided axis convention, both orientations
  are tried and the one with the stronger peaks is used.
- **`test_example_notebook_executes`.** Executes `simple_example.ipynb` unchanged, top
  to bottom, at its full 10⁴ s exposure (~50 s, ~7 GB peak memory). It copies the
  notebook and its style file to a temporary directory and links the model files
  there, so the notebook's `event_list.fits` output does not land in the
  repository; then it runs every cell with `nbclient` and checks the output file was
  written. Any exception in any cell fails the test.
