# AI assistance disclosure

The test suite in this directory was developed with substantial assistance from
an AI coding assistant: Claude (Anthropic; model Claude Opus 5.5), used through
Claude Code between 2026-11-03 and 2026-11-04.

## What the AI did

- Proposed the test plan (test categories, physics checks, statistical-tolerance
  policy, slow/fast tiers) for review by the maintainer before any code was written.
- Wrote all files in `tests/` — `conftest.py`, `helpers.py`, every `test_*.py`,
  `README.md`, `GUIDE.md` and this file — and the test-related changes outside it:
  the `test` extra and pytest configuration in `pyproject.toml`, test steps in
  `.github/workflows/ci.yml`, and coverage/pytest entries in `.gitignore`.
- Read the package source to identify behaviour that looked wrong, and recorded
  each suspected bug as a strict `xfail` test stating the expected behaviour.
- Ran the suite, coverage, ruff, and repeated runs under different random seeds
  (`XRISALIS_TEST_SEED`), and fixed its own test errors found that way.

## What the AI did not do

- It did not modify the package source in `src/xrisalis/`, the bundled model
  files, or the example notebook. Suspected bugs were recorded, not fixed.
- It did not decide open conventions (axis order, signs, units, disc radial
  law); tests of current behaviour in those areas are marked `convention_pending`.

## Human oversight

The maintainer of this fork directed the work, set the policies (environment,
record-don't-fix, deferring conventions, notebook in the slow tier), and reviewed
the plan and the results of each implementation step before the next began.

## Limitations

- The physics expectations encoded in the tests (fringe spacing, off-axis fringe
  shift, Fresnel limits, reconstruction geometry, event-count models) were derived
  by the AI from the code, standard optics and the reference notebook. They have
  passed the checks described above but should be reviewed by a domain expert,
  as should the classification of behaviours as bugs.
- Statistical tests use ≥ 5σ tolerances and were checked under 10–11 seeds; a
  very rare spurious failure remains possible by construction.
- The CI workflow was checked by reproducing its commands in a fresh local
  virtual environment, not by running it on GitHub Actions.
