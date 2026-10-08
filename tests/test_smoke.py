"""Smoke test: checks that the project is installed and importable."""

import importlib
import sys

import pytest

SUBPACKAGES = ["data", "poisoning", "retrieval", "models", "evaluation", "analysis"]


def test_python_version():
    """The project requires Python 3.11 or newer."""
    assert sys.version_info >= (3, 11)


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    """Every pipeline layer can be imported from the installed package."""
    importlib.import_module(f"inragwetrust.{name}")
