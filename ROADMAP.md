# Roadmap

Mjölnir (distribution: `Thor-Hammer`) is an early-stage library (v0.0.10). This
roadmap tracks the work needed to make it a reliable, documented causal ML tool.
Milestones are ordered by dependency; each has an acceptance criterion.

## Status snapshot
- Core DAG regression fits correctly and recovers known linear relationships
  (verified by `tests/test_fit.py`).
- Symbolic math, do-calculus, and inverse methods are stubs.
- Conformal prediction works against both MAPIE `<1` and `>=1`.

## M1 — Restore a correct, tested fit path ✅ (in progress)
- [x] Relative imports; package imports cleanly with no side effects.
- [x] MAPIE `<1`/`>=1` compatibility shims.
- [x] Fixed parent ordering in `conformal_fit`/`conformal_predict`.
- [x] Fixed causality direction and edge sampling in `datasets`.
- [x] Reset per-fit state (`fitted_predictions`).
- [x] Add `tests/` and run them in CI.
- [ ] Wire coverage reporting.
- **Done when:** CI runs the fit and conformal tests green on every push.

## M2 — Complete the symbolic analysis API
- [ ] Implement `derivative(method)`, `gradient`, `divergence`, `jacobian`,
      `hessian` from `symb_model_exprs` using SymPy.
- [ ] Replace empty stubs (`inv`, `pinv`) with implementations or explicit
      `NotImplementedError` with a documented reason.
- [ ] Document which methods assume a triangular/lower-triangular Jacobian.
- **Done when:** analytic derivatives match finite-difference checks in tests.

## M3 — Do-calculus
- [ ] Implement `DAGModel.do(query)`: graph surgery (remove incoming edges to
      intervened nodes) and refit or evaluate the truncated factorization.
- [ ] Add identification checks for non-identifiable queries.
- **Done when:** `do(X=x)` on known SCMs returns the analytically expected
      interventional distribution.

## M4 — Conformal prediction hardening
- [ ] Drop the legacy MAPIE path once a minimum version is chosen; pin it.
- [ ] Overwrite `symb_model_exprs` from conformal fits (current TODO).
- [ ] Return pandas-formatted predictions/intervals (current TODO).
- [ ] Validate empirical coverage on held-out data across DAGs.
- **Done when:** coverage is within tolerance of nominal on held-out samples.

## M5 — Data handling
- [ ] Missing-data handling via multiple imputation along the DAG (current TODO).
- [ ] Measurement-error transforms (`AdditiveSciPyDistError`,
      `MultiplicativeSciPyDistError`) exercised in tests; declare dependencies.
- [ ] Latent-variable support or explicit non-support docs.
- **Done when:** missing-data tests show no bias vs. complete-case baseline.

## M6 — Model quality and tuning
- [ ] Sensible default GP parameters; current defaults under-fit.
- [ ] Optuna-based hyperparameter tuning hook (current TODO).
- [ ] Model diagnostics (residuals, per-node fit quality, R²).
- **Done when:** defaults recover linear SCMs within noise on the included
      dataset generator.

## M7 — Packaging and docs
- [ ] Fill in `docs/` (examples, methodology, API reference via mkdocstrings).
- [ ] Automate releases (see `update.sh`); the `rm -r dist` guard should not
      fail when `dist/` is absent.
- [ ] Add `StatsmodelsAPI`/`statsmodels` documentation or remove the wrapper.
- **Done when:** docs build and PyPI release runs from CI on tagged commits.

## M8 — Code quality
- [ ] Add linting (ruff) and type checking (mypy) with pre-commit hooks.
- [ ] Broaden the test suite (edge cases: empty DAGs, single node, multi-char
      node prefixes, non-numeric dtypes).
- **Done when:** lint/type/test all pass in CI.

## Known issues / technical debt
- `requires-python = ">=3.7"` but the dependency set (scikit-learn >=1.6 via
  gplearn) effectively needs a newer Python; reconcile the floor.
- `graphviz` is now declared; `statsmodels` and `scipy` are used but undeclared.
- `Datasets.make_dag_regression` uses beta scale 10, which produces very large
  descendant values and makes GP recovery hard; consider a smaller default.
