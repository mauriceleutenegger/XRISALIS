# XRISALIS test suite

## Running

Environment: mamba env `xrisalis` (Python 3.13), package installed editable:

```bash
mamba create -n xrisalis -c conda-forge python=3.13 pip
mamba activate xrisalis
pip install -e ".[dev,test]"
```

| Command | What runs |
|---|---|
| `pytest -m "not slow"` | fast tier (unit + physics tests) |
| `pytest` | everything, including the `slow` tier |
| `pytest -m slow` | end-to-end workflow with bundled model files |
| `pytest --nbmake simple_example.ipynb` | execute the example notebook |
| `pytest --cov=xrisalis --cov-report=term-missing` | coverage report |
| `ruff format --check && ruff check` | formatting and lint (also covers `tests/`) |

## Layout

| File | Contents |
|---|---|
| `conftest.py` | Agg backend, autouse fixture (seed RNG, `plt.show` no-op, close figures), shared fixtures |
| `helpers.py` | unit constants (`KEV`, `ARCSEC`), `known_bug`, synthetic FITS builders |
| `test_<module>.py` | one file per source module in `src/xrisalis/` |
| `test_workflow.py` | `slow` end-to-end tests |

### Fixtures (`conftest.py`)
- `_seed_and_no_show` (autouse): `np.random.seed(12345)`, `plt.show` → no-op, `plt.close("all")` after each test.
- `bundled_paths`: paths to `models/arf.fits`, `rmf.fits`, `double_point_source.fits` and the notebook.
- `synthetic_responses`: flat SIMPUT spectrum (0.05 ph/cm²/s/keV, 1–7 keV, 60 bins),
  two-baseline ARF (10 and 30 cm²), identity RMF — all in `tmp_path`.

### Helpers (`helpers.py`)
- `energy_grid(e_min, e_max, n)` → `(lo, hi, mid)` in keV.
- `write_simput(path, energies, fluxes, ra, dec)`, `write_arf(path, lo, hi, areas, baselines)`,
  `write_rmf(path, lo, hi, matrix=None, tlmin=0)` (identity by default).
- `known_bug(summary)` → strict xfail marker for a test that exposes a known bug.

## Policies

- **Seeding and tolerances.** Every test starts from a fixed seed, but stochastic
  assertions must hold for *any* seed: compare against analytic expectations with
  ≥ 5σ tolerances, and state the expectation and σ in the test docstring.
- **Docstrings.** Every test module says which source module it covers; every test
  has a one-line docstring stating the property checked.
- **Known bugs.** A test exposing a known bug states the correct behaviour and is
  marked `@known_bug("short description")` (strict xfail). Tests never modify `src/`.
  When the bug is fixed the test passes, strict mode flags it, and the marker is removed.
- **`convention_pending`.** Tests that pin down current behaviour where the intended
  convention is undecided (disc radial law, units, axis order and signs, roll speed) carry
  `@pytest.mark.convention_pending`. Revisit them once the convention is decided.
- **Plots.** Plot functions are smoke-tested only (run under Agg, return a figure
  when asked).

## Adding a test

1. Put it in the `test_<module>.py` matching the source module.
2. Give it a docstring; for statistical tests include expectation and tolerance.
3. Use `helpers.py` builders for FITS input; write files only under `tmp_path`.
4. If it exposes a bug: state the correct behaviour and mark the test with `known_bug("short description")`.
5. Add it to the coverage map below.

## Coverage map

Public function → tests → known bugs and open conventions.

| Module | Function / class | Tests | Known bugs / open conventions |
|---|---|---|---|
| (package) | submodule imports, `analysis.xri_cmap` | `test_package.py::test_submodule_imports`, `test_xri_cmap_builds` | — |
| (data) | `models/arf.fits`, `rmf.fits`, `double_point_source.fits` | `test_package.py::test_bundled_*` | bundled RMF rows unnormalised |
| instrument | `baseline.__init__` (geometry) | `test_instrument.py::test_baseline_*` | numpy-int `num_pairs` rejected |
| instrument | `baseline.__init__` (`mirr_reflec`) | `test_baseline_mirror_reflectivity_*` | multi-digit angles misparsed |
| instrument | `interferometer.__init__` | `test_interferometer_defaults` | — |
| instrument | `interferometer.add_baseline` | `test_add_baseline_*` | keyword differs from `baseline` |
| instrument | `interferometer.add_willingale_baseline`, `clear_baselines` | `test_baseline_willingale_parameters`, `test_add_willingale_baselines_and_clear` | — |
| instrument | `interferometer.gen_pointing` | `test_gen_pointing_*` | wobbler never applied; roller errors swallowed |
| instrument | `interferometer.smooth_roller` | `test_smooth_roller_*` | `roll_speed` ignored |
| instrument | `interferometer.discrete_roller` | `test_discrete_roller_*` | steps only once |
| instrument | `interferometer.random_wobble` | `test_random_wobble_*` | not a random walk |
| instrument | `interferometer.file_wobble` | `test_file_wobble_reads_csv` | — |
| instrument | `plot_arf` | `test_plot_arf_smoke` | — |
