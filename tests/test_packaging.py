"""Tests pinning the packaging metadata contract in ``pyproject.toml``.

These tests are intentionally dependency-free (stdlib :mod:`tomllib` only) so
they can run without importing ``probjax`` or JAX.
"""

import pathlib
import re
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent


def load_pyproject():
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def test_license_is_spdx_expression():
    project = load_pyproject()["project"]
    assert project["license"] == "MIT"


def test_license_files_declared():
    project = load_pyproject()["project"]
    assert project["license-files"] == ["LICENSE.txt"]
    assert (ROOT / "LICENSE.txt").is_file()


def test_no_license_classifier():
    classifiers = load_pyproject()["project"]["classifiers"]
    assert not any(c.startswith("License ::") for c in classifiers)


def test_development_status_classifier():
    classifiers = load_pyproject()["project"]["classifiers"]
    assert any(c.startswith("Development Status ::") for c in classifiers)


def test_topic_classifiers():
    classifiers = load_pyproject()["project"]["classifiers"]
    assert any(c.startswith("Topic ::") for c in classifiers)


def test_keywords_expanded():
    keywords = load_pyproject()["project"]["keywords"]
    assert len(keywords) >= 6
    assert {
        "probabilistic",
        "jax",
        "computation",
        "bayesian-inference",
        "probabilistic-programming",
    } <= set(keywords)


def test_known_first_party():
    data = load_pyproject()
    isort = data["tool"]["ruff"]["lint"]["isort"]
    assert isort["known-first-party"] == ["probjax"]
    assert "sbi" not in (ROOT / "pyproject.toml").read_text()


def test_build_requires_supports_pep639():
    requires = load_pyproject()["build-system"]["requires"]
    floors = [
        int(m.group(1))
        for req in requires
        if (m := re.match(r"setuptools>=(\d+)", req))
    ]
    assert floors
    assert max(floors) >= 77


def test_author_email_unchanged():
    project = load_pyproject()["project"]
    assert {
        "name": "Manuel Gloeckler",
        "email": "manuel.gloeckler@uni-tuebingen.de",
    } in project["authors"]
