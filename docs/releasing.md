# Release checklist

The next prepared version is **0.2.0**. Creating release artifacts is separate
from publishing them; no tag or package upload happens during the checks below.

## Open release blocker

Adaptive ODE reverse differentiation fails for a basic parameterized decay
example. The custom VJP stores a Python controller in dynamic residuals; removing
that reveals additional filtered-state handling failures. Resolve and test the
adjoint path before publishing 0.2.0. Fixed-grid RK4 and its gradients are covered
by the release regression tests. This candidate is not yet publication-ready.

## Check the candidate

1. Review `CHANGELOG.md` and the [migration guide](guides/migration.md). Describe
   implemented behavior, including compatibility breaks and unverified backends.
2. Match the version in `pyproject.toml` and `probjax/__init__.py`.
3. Run the required Python 3.11, 3.12 and 3.13 CI checks on the release commit.
   Run accelerator/mesh suites on the hardware being claimed as validated.
4. Generate tutorials, execute documentation examples and build the site:

```bash
python -m pip install -e ".[dev]"
python -m pip install -r docs/requirements.txt matplotlib
python scripts/convert_notebooks.py
python scripts/convert_notebooks.py --check
JAX_PLATFORMS=cpu pytest tests/test_docs.py --benchmark-disable
zensical build --strict
```

Tutorial conversion reuses notebook outputs. CI executes generated Python blocks,
but does not update stored plots or compare them to regenerated output. Refresh
notebook outputs explicitly when their presentation changes.

## Build and inspect distributions

Use a clean checkout of the candidate and a fresh output directory:

```bash
python -m pip install build twine
python -m build --outdir /tmp/probjax-release-dist
python -m twine check /tmp/probjax-release-dist/*
```

Inspect both the wheel and source archive: package files, license and README must
be present; caches, local environments and retired kernel packages must not be
included. Install the wheel into a fresh environment, from outside the checkout,
and check that `probjax.__version__` equals
`importlib.metadata.version("probjax")`. Smoke-test distributions and an ODE solve.

## Publish the reviewed candidate

After checks and release review, replace the Unreleased heading with the chosen
version and actual release date, add a new empty Unreleased section, and update
changelog comparison links. Commit that metadata before tagging. Create the
matching `v0.2.0` tag and GitHub release with the migration notes, then upload the
reviewed artifacts to PyPI using the maintainer's configured publishing method.
This repository currently has no package-publishing workflow.

Finally verify installation from the index and the documentation deployment.
The public site follows `main`, not a versioned documentation snapshot; the tag
retains the matching documentation source.

Codecov repository setup is a separate integration issue: uploads currently fail
with `Repository not found`. CI preserves `coverage.xml` as an artifact. Confirm
the coverage test step passed independently of that upload failure.
