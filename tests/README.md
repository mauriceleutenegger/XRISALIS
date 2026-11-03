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
| `pytest -m slow` | end-to-end workflow with bundled files + example notebook (~1 min, ~7 GB peak RAM) |
| `pytest -m slow -k notebook` | only execute `simple_example.ipynb` (in a temp copy, via nbclient) |
| `pytest --cov=xrisalis --cov-report=term-missing` | coverage report |
| `XRISALIS_TEST_SEED=7 pytest` | run with another global seed (check seed independence) |
| `ruff format --check && ruff check` | formatting and lint (also covers `tests/`) |

## Layout

| File | Contents |
|---|---|
| `conftest.py` | Agg backend, autouse fixture (seed RNG, `plt.show` no-op, close figures), shared fixtures |
| `helpers.py` | unit constants (`KEV`, `ARCSEC`), `known_bug`, synthetic FITS builders |
| `test_<module>.py` | one file per source module in `src/xrisalis/` |
| `test_workflow.py` | `slow` end-to-end tests and notebook execution |

### Fixtures (`conftest.py`)
- `_seed_and_no_show` (autouse): `np.random.seed(SEED)` (12345 unless `XRISALIS_TEST_SEED` is set), `plt.show` → no-op, `plt.close("all")` after each test.
- `bundled_paths` (session scope): paths to `models/arf.fits`, `rmf.fits`, `double_point_source.fits` and the notebook.
- `synthetic_responses`: flat SIMPUT spectrum (0.05 ph/cm²/s/keV, 1–7 keV, 60 bins),
  two-baseline ARF (10 and 30 cm²), identity RMF — all in `tmp_path`.

### Helpers (`helpers.py`)
- `SEED`: global test seed (env `XRISALIS_TEST_SEED`, default 12345).
- `energy_grid(e_min, e_max, n)` → `(lo, hi, mid)` in keV.
- `write_simput(path, energies, fluxes, ra, dec)`, `write_arf(path, lo, hi, areas, baselines)`,
  `write_rmf(path, lo, hi, matrix=None, tlmin=0)` (identity by default).
- `known_bug(summary)` → strict xfail marker for a test that exposes a known bug.

## Policies

- **Seeding and tolerances.** Every test starts from a fixed seed, but stochastic
  assertions must hold for *any* seed (check with `XRISALIS_TEST_SEED`): compare against analytic expectations with
  ≥ 5σ tolerances, and state the expectation and σ in the test docstring.
- **Docstrings.** Every test module says which source module it covers; every test
  has a one-line docstring stating the property checked.
- **Known bugs.** A test exposing a known bug states the correct behaviour and is
  marked `@known_bug("short description")` (strict xfail). Tests never modify `src/`.
  When the bug is fixed the test passes, strict mode flags it, and the marker is removed.
- **`convention_pending`.** Tests that pin down current behaviour where the intended
  convention is undecided (disc radial law, units, axis order and signs, roll speed) carry
  `@pytest.mark.convention_pending`. Revisit them once the convention is decided.
- **Unseedable code.** `images.draw_photons_from_cube` uses its own
  `np.random.default_rng()`, so `np.random.seed` does not make it reproducible;
  its tests rely on the 5σ tolerances alone.
- **Comparing energies in J.** Energies are ~1e-16 J, below `np.isclose`'s default
  `atol=1e-8`; use relative comparisons (`assert_allclose`, or `rtol`, `atol=0`).
- **Reconstruction pixel size.** `image_recon_smooth` needs the pixel
  `fov / max(samples)` below λ_min / (2 D_max) (coarser pixels alias or crash); the analysis tests use
  2e-10 rad pixels with 1–1.5 keV photons and D ≤ 1 m.
- **Notebook.** Executed unchanged with nbclient in `tmp_path` (notebook, style
  file, symlink to `src/xrisalis/models`), so its `event_list.fits` never lands in
  the repo. `pytest --nbmake` would run it in the repo root and is not used.
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
| images | `image` | `test_images.py::test_image_container_defaults` | — |
| images | `point_source` | `test_point_source_*`, `test_generators_require_energy_or_spectrum` | — |
| images | spectrum handling (all generators) | `test_generators_store_spectrum`, `test_generators_require_energy_or_spectrum` | spectrum without energy fails |
| images | `double_point_source`, `m_point_sources` | `test_double_point_source_*`, `test_m_point_sources_fractions`, `test_multi_generators_accept_spectrum_without_energy` | spectrum without energy fails |
| images | `point_source_multichromatic_range` | `test_multichromatic_range_uniform`, `test_multi_generators_*` | spectrum without energy fails |
| images | `point_source_multichromatic_gauss` | `test_multichromatic_gauss_mean_and_spread` | — |
| images | `disc` | `test_disc_*` | energy-spread units; radial law (convention) |
| images | `generate_from_image` | `test_generate_from_image_*` | background photons crash; orientation (convention) |
| images | `draw_photons_from_cube` | `test_draw_photons_from_cube_*` | crashes on pandas 3; `loc` units (convention) |
| images | `append` | `test_append_*` | drops metadata; fails for empty first image |
| images | `create_image_from_events` | `test_create_image_from_events_*`, `test_process.py::test_event_list_to_detector_positions` | RA/DEC axis order and orientation (convention) |
| process | `interferometer_data.fre_dif` | `test_process.py::test_fre_dif_*` | — |
| process | `interferometer_data.__init__`, `copy` | `test_interferometer_data_*` | generator images lack baseline indices |
| process | `interferometer_data.process_photon_dpos` | `test_interferometer_data_*`, `test_pure_*`, `test_off_axis_*`, `test_far_off_axis_warns`, `test_multiple_baselines_use_own_fringe_spacing` | oversampled fringe range; axis/sign (convention) |
| process | `interferometer_data.process_photon_e_base` | — (dead code) | dead code |
| process | `generate_event_list` | `test_event_list_*` | dropped photons misalign columns; empty list crashes; grid mismatch; 1-based channels; RMF renormalisation (convention) |
| analysis | `ft_data` | `test_analysis.py::test_ft_data_sinusoid` | — |
| analysis | `plot_ft`, `hist_data` | `test_plot_ft_axis_scales`, `test_hist_data_labels` | — |
| analysis | `image_recon_smooth` (+ inner `inverse_fourier`) | `test_recon_*` | coarse pixels alias or crash; returned scale (convention) |
| analysis | `reconstruction_data` | `test_recon_output_structure` | — |
| analysis | `filter_data` | `test_filter_data_*` | true positions/energies not filtered |
| analysis | `plot_inf_image`, `plot_uv`, `plot_res_hist`, `subplots`, `plot_spec_per_baseline` | `test_plot_*`, `test_subplots` | RA/DEC axis order and orientation (convention) |
| (workflow) | full notebook chain with bundled files, 1000 s | `test_workflow.py::test_workflow_*` (slow) | RA/DEC axis order and orientation (convention) |
| (notebook) | `simple_example.ipynb` | `test_workflow.py::test_example_notebook_executes` (slow) | — |
