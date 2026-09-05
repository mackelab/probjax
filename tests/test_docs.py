"""The documentation, held to the same standard as the code.

Two things rot in docs: examples that stop working, and a reference that drifts
out of step with the package. Both are checkable, so both are checked here.

Python blocks are executed **cumulatively per page**, the way a reader follows
them, so a snippet may build on the one above it. A block that cannot run in
this environment is opted out with a ``python skip`` fence, which keeps the
exclusion visible in the source rather than hidden in a list here.
"""

import importlib
import pathlib
import re
import shutil
import subprocess
import sys
import textwrap

import pytest

DOCS = pathlib.Path(__file__).resolve().parent.parent / "docs"
REFERENCE = DOCS / "reference"

#: Public names that are deliberately absent from the reference, with a reason.
#: Keeping this explicit means new public API cannot slip in undocumented, and
#: every omission is a decision someone made rather than an oversight.
UNDOCUMENTED = {
    "probjax.core": {
        # Submodules, reachable but not API surface in their own right.
        "custom_primitives",
        "interpreters",
        "jaxpr_propagation",
        "registry",
        "transformation",
    },
    "probjax.utils": {
        # Type aliases, documented where they are used.
        "Array",
        "ArrayLike",
        "PyTree",
        "RngKey",
        # Internal helpers.
        "find_ancestors_jax",
        "faithfull_mask",
        "split_drift",
        "ravel_args",
    },
    "probjax.stats": {
        # Typing machinery, not called directly.
        "DistributionAPI",
        "DistributionParams",
        "InvertibleTransformProtocol",
        "TransformProtocol",
        # Returned by freezing a generator rather than constructed; the
        # behaviour is documented on rv_frozen.
        "rv_continuous_frozen",
        "rv_discrete_frozen",
        "rv_multivariate_frozen",
        "rv_spherical_frozen",
    },
    "probjax.inference": {
        # Extension points for writing new kernels and filters, rather than
        # things a user of the existing ones calls.
        "FilterAPI",
        "make_filter_api",
        "AdaptationTrace",
        "adaptive_persistent_smc_kernel",
        "adaptive_smc_kernel",
        "persistent_smc_kernel",
        # EKF model builders, used when defining a state-space model.
        "make_continuous_transition",
        "make_linearized_observation",
        "make_linearized_transition",
        # Adaptive rejection sampling internals; the sampler is `ars`.
        "ARSState",
        "init_ars_state",
        "update_ars_state",
        # Submodules re-exported for convenience.
        "tuning",
        "ukf",
        # blackjax's own types, re-exported here; griffe cannot follow the
        # alias into blackjax, and their reference lives with blackjax.
        "State",
        "Params",
    },
}


def markdown_pages():
    return sorted(DOCS.rglob("*.md"))


def python_blocks(text):
    """Executable blocks, in order. ``python skip`` fences are excluded."""
    return [
        textwrap.dedent(body)
        for info, body in re.findall(r"```(python[^\n]*)\n(.*?)```", text, re.S)
        if info.strip() == "python"
    ]


def reference_targets():
    targets = []
    for page in sorted(REFERENCE.glob("*.md")):
        targets += [
            (page.name, t) for t in re.findall(r"^::: ([\w.]+)", page.read_text(), re.M)
        ]
    return targets


# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "page", markdown_pages(), ids=lambda p: str(p.relative_to(DOCS))
)
def test_every_python_block_runs(page):
    """Run each page's blocks cumulatively, as a reader would.

    Isolated execution would report false failures for continuation snippets,
    and would miss the case where a page's blocks contradict each other.
    """
    blocks = python_blocks(page.read_text())
    if not blocks:
        pytest.skip("no executable python blocks")

    script = ""
    for index, block in enumerate(blocks):
        script += block + "\n"
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env={
                "JAX_PLATFORMS": "cpu",
                "PATH": "/usr/bin:/bin",
                "HOME": str(pathlib.Path.home()),
            },
            timeout=900,
        )
        if completed.returncode != 0:
            tail = [ln for ln in completed.stderr.strip().splitlines() if ln.strip()]
            pytest.fail(
                f"{page.name} block {index} failed:\n  " + "\n  ".join(tail[-6:])
            )


# ---------------------------------------------------------------------------
# Reference
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("page,target", reference_targets())
def test_every_reference_target_resolves(page, target):
    """A ``:::`` directive naming something that does not exist renders empty.

    mkdocstrings does not fail the build for it, so nothing would otherwise
    catch a renamed or removed object.
    """
    module_path, _, attribute = target.rpartition(".")
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        importlib.import_module(target)  # target is itself a module
        return
    assert hasattr(module, attribute), f"{page}: {target} does not exist"


def test_the_site_builds():
    """The build is the real check, and it is faster than reimplementing it.

    mkdocstrings resolves targets through griffe, which reads the *source*
    rather than importing. A name re-exported from a third-party package
    therefore passes the runtime check above and still fails the build with
    ``AliasResolutionError`` -- ``probjax.inference.State`` did exactly that.
    Running the builder catches that and anything else the config gets wrong.
    """
    if shutil.which("zensical") is None:
        pytest.skip("zensical is not installed")

    # `zensical build` has no destination flag -- it writes to `site_dir` from
    # the config -- so this builds in place, exactly as the deployment does.
    completed = subprocess.run(
        ["zensical", "build", "--strict"],
        cwd=DOCS.parent,
        capture_output=True,
        text=True,
        timeout=900,
    )
    combined = completed.stdout + completed.stderr
    assert completed.returncode == 0, f"zensical build failed:\n{combined[-2000:]}"
    # griffe reports docstring problems on stderr without failing; only hard
    # resolution errors should ever reach here.
    assert "AliasResolutionError" not in combined


@pytest.mark.parametrize(
    "module_name",
    ["probjax.core", "probjax.stats", "probjax.inference", "probjax.utils"],
)
def test_public_api_is_documented_or_explicitly_excluded(module_name):
    """Every name in ``__all__`` is either in the reference or excluded by hand.

    ``probjax.nn`` is out of scope: it exports 235 names, most of them building
    blocks rather than API, and the reference curates the user-facing subset.
    """
    module = importlib.import_module(module_name)
    public = set(getattr(module, "__all__", []))
    if not public:
        pytest.skip(f"{module_name} declares no __all__")

    referenced = {target.rpartition(".")[2] for _, target in reference_targets()}
    excluded = UNDOCUMENTED.get(module_name, set())
    missing = sorted(public - referenced - excluded)
    assert not missing, (
        f"{module_name} exports {missing} with no reference entry. Add them to "
        f"docs/reference/, or to UNDOCUMENTED in this file with a reason."
    )


# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_no_sphinx_files_remain():
    """The migration is only finished when the old system is gone."""
    leftovers = sorted(
        p.name for p in DOCS.rglob("*") if p.suffix == ".rst" or p.name == "conf.py"
    )
    assert not leftovers, f"Sphinx leftovers in docs/: {leftovers}"


def test_every_nav_entry_exists():
    """A nav pointing at a missing page builds fine and 404s in the browser."""
    config = (DOCS.parent / "zensical.toml").read_text()
    for relative in re.findall(r'"((?:[\w./-]+)\.md)"', config):
        assert (DOCS / relative).is_file(), f"nav references missing page {relative}"
