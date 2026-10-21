"""Package layering, checked on the source rather than on ``sys.modules``.

Importing any ``probjax`` module runs ``probjax/__init__``, which imports every
subpackage, so a runtime check cannot tell who depends on whom. Reading the
import statements can, and it also sees imports inside functions, which is how
a dependency would otherwise be hidden.

The intended order is ``utils`` <- ``core`` <- ``stats`` <- ``nn``, with the
flax-free part of ``nn.training`` (batches, kernel, loop) alongside ``stats``.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1] / "probjax"

# (name, files it covers, import prefixes those files must not use, exceptions)
RULES = [
    (
        "stats is plain JAX: no flax, optax, nn or inference",
        "stats/",
        ("flax", "optax", "probjax.nn", "probjax.inference"),
        (),
    ),
    (
        "core sits below stats and nn",
        "core/",
        ("flax", "optax", "probjax.stats", "probjax.nn", "probjax.inference"),
        (),
    ),
    (
        "utils is the bottom layer",
        "utils/",
        ("probjax.core", "probjax.stats", "probjax.nn", "probjax.inference"),
        # Known back-edge: ODE inversion registers a custom inverse from core.
        # Move it above core, then drop this exception.
        ("utils/odeutil/inversion.py",),
    ),
    (
        "the training loop is flax-free and independent of the model libraries",
        ("nn/training/config.py", "nn/training/state.py", "nn/training/trainer.py"),
        (
            "flax",
            "probjax.core",
            "probjax.stats",
            "probjax.inference",
            "probjax.nn.generative",
            "probjax.nn.layers",
            "probjax.nn.nets",
        ),
        (),
    ),
    (
        "filtering leaves import nothing above utils",
        (
            "inference/filtering/base.py",
            "inference/filtering/state_space.py",
            "inference/filtering/observations.py",
            "inference/filtering/gaussian.py",
            "inference/filtering/smoothing.py",
        ),
        (
            "probjax.inference.filtering.kalman_filter",
            "probjax.inference.filtering.extended_kalman_filter",
            "probjax.inference.filtering.unscented_kalman_filter",
            "probjax.inference.filtering.square_root_kf",
            "probjax.inference.filtering.rank_reduced_kalman_filter",
            "probjax.inference.filtering.particle_filter",
            "probjax.inference.filtering.temporal",
            "probjax.inference.filtering.trajectory",
            "probjax.inference.filtering.joint",
            "probjax.inference.smc",
        ),
        (),
    ),
    (
        "concrete filters stay below orchestration",
        (
            "inference/filtering/kalman_filter.py",
            "inference/filtering/extended_kalman_filter.py",
            "inference/filtering/unscented_kalman_filter.py",
            "inference/filtering/square_root_kf.py",
            "inference/filtering/rank_reduced_kalman_filter.py",
            "inference/filtering/particle_filter.py",
        ),
        (
            "probjax.inference.filtering.temporal",
            "probjax.inference.filtering.trajectory",
            "probjax.inference.filtering.joint",
            "probjax.inference.smc.temporal",
        ),
        (),
    ),
    (
        "no filtering module reaches smc.temporal (the old cycle)",
        "inference/filtering/",
        ("probjax.inference.smc.temporal",),
        (),
    ),
    (
        "smc/adaptation edges stay acyclic and public",
        (
            "inference/adaptation.py",
            "inference/smc/tuning.py",
            "inference/smc/base.py",
            "inference/smc/path_smc.py",
        ),
        ("probjax.inference.smc.temporal", "probjax.inference.smc.ports"),
        (),
    ),
    (
        "kernels stay runner-free",
        "inference/mcmc/",
        ("probjax.inference.mcmc_runner", "probjax.inference.smc_runner"),
        (),
    ),
]


def _module_name(path: pathlib.Path) -> list[str]:
    parts = list(path.relative_to(ROOT.parent).with_suffix("").parts)
    return parts[:-1] if parts[-1] == "__init__" else parts


def _imports(path: pathlib.Path) -> set[str]:
    """Every module a file imports, at any depth, with relative imports resolved."""
    package = (
        _module_name(path)[:-1] if path.name != "__init__.py" else _module_name(path)
    )
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                prefix = package[: len(package) - node.level + 1]
                base = ".".join([*prefix, base] if base else prefix)
            found.add(base)
            # ``from probjax import stats`` imports the submodule ``probjax.stats``
            found.update(f"{base}.{alias.name}" for alias in node.names)
    return found


def _covered(scope) -> list[pathlib.Path]:
    scope = (scope,) if isinstance(scope, str) else scope
    files = []
    for entry in scope:
        target = ROOT / entry
        files += sorted(target.rglob("*.py")) if target.is_dir() else [target]
    return files


@pytest.mark.parametrize("rule", RULES, ids=[rule[0] for rule in RULES])
def test_layering(rule):
    _, scope, forbidden, exceptions = rule
    files = _covered(scope)
    assert files, "rule matches no files; has the layout changed?"
    allowed = {ROOT / name for name in exceptions}
    violations = [
        f"{path.relative_to(ROOT.parent)} imports {module}"
        for path in files
        if path not in allowed
        for module in sorted(_imports(path))
        if any(module == bad or module.startswith(bad + ".") for bad in forbidden)
    ]
    assert not violations, "\n".join(violations)
