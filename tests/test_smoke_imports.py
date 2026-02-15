# tests/test_smoke_imports.py

"""
Test Suite: Smoke Import Tests — All Package ``__init__.py`` Modules
====================================================================

Production-ready smoke import test covering MILIA_Test_Recommendations.md §1.2:

    **What it tests**: All top-level package imports succeed without errors
    (catches circular imports, missing dependencies, broken ``__init__.py``
    files).

    **Scope**: Each import is a separate test case.  Asserts no
    ``ImportError``, no ``CircularImportError``.  Fast (< 5 seconds total).

This file is the §1.2 **CI/CD fast gate** — the first test pytest runs in the
smoke stage.  It does NOT duplicate the deep smoke + contract testing performed
by the 15 individual ``test__init__*.py`` files (which validate ``__all__``
completeness, return-type contracts, conditional availability flags, etc.).
Rather, it provides a single, lightweight, aggregated import-health check for
all 15 ``__init__.py`` modules that CI can run in < 5 seconds to decide
whether to proceed to heavier stages.

Relationship to ``test__init__*.py`` files:
    - ``test_smoke_imports.py`` (this file): thin import gate — "can we import
      each package at all?"
    - ``test__init__<module>.py`` (15 files): deep smoke + contract tests —
      "does each package expose the correct API surface, types, and contracts?"

Modules exercised (15 — matches §1.2 specification exactly):
    1.  ``milia_pipeline``
    2.  ``milia_pipeline.config``
    3.  ``milia_pipeline.molecules``
    4.  ``milia_pipeline.transformations``
    5.  ``milia_pipeline.datasets``
    6.  ``milia_pipeline.handlers``
    7.  ``milia_pipeline.preprocessing``
    8.  ``milia_pipeline.descriptors``
    9.  ``milia_pipeline.models``
    10. ``milia_pipeline.models.hpo``
    11. ``milia_pipeline.models.post_training``
    12. ``milia_pipeline.models.builders``
    13. ``milia_pipeline.models.acceleration``
    14. ``milia_pipeline.models.deployment``
    15. ``milia_pipeline.plugins``

Design:
    - **Dynamic**: The ``_ALL_PACKAGES`` tuple is the single source of truth.
      Adding a new subpackage requires only one tuple entry — all test cases,
      parametrize IDs, and assertions adapt automatically.
    - **Non-breaking**: Each import is wrapped in ``importlib.import_module``
      inside its own parametrized test case, so one failing package does not
      prevent the remaining packages from being tested.
    - **Future-proof**: Uses ``importlib.import_module`` (the standard library
      mechanism) — no hard-coded ``import`` statements that would need manual
      updates per package.
    - **Production-ready**: Validates not just importability but also that the
      result is a proper ``types.ModuleType`` with a ``__file__`` attribute
      (ruling out namespace packages or broken stubs).

Launch:
    From project root (/app/milia):
        pytest tests/test_smoke_imports.py -v --tb=short

Markers:
    smoke — Quick health-check tests (§1)

Author : MILIA Team
Version: 1.0.0
"""

from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup — ensure the project root is importable
# ---------------------------------------------------------------------------
# Defensive: conftest.py already sets this, but every test__init__*.py in the
# project includes its own copy for standalone-run safety.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Single source of truth: all packages specified in §1.2
# ---------------------------------------------------------------------------
# To add a new subpackage to the smoke gate, append ONE entry here.
# All parametrized test cases below adapt automatically.
_ALL_PACKAGES: tuple[str, ...] = (
    "milia_pipeline",
    "milia_pipeline.config",
    "milia_pipeline.molecules",
    "milia_pipeline.transformations",
    "milia_pipeline.datasets",
    "milia_pipeline.handlers",
    "milia_pipeline.preprocessing",
    "milia_pipeline.descriptors",
    "milia_pipeline.models",
    "milia_pipeline.models.hpo",
    "milia_pipeline.models.post_training",
    "milia_pipeline.models.builders",
    "milia_pipeline.models.acceleration",
    "milia_pipeline.models.deployment",
    "milia_pipeline.plugins",
)


# ===================================================================
# SMOKE TESTS — §1.2
# ===================================================================


class TestSmokeImports:
    """
    §1.2 — Verify every top-level package ``__init__.py`` imports without
    errors.

    Each parametrized case is an independent test so that a single broken
    package does not mask failures in the remaining packages.
    """

    @pytest.mark.smoke
    @pytest.mark.parametrize("package_name", _ALL_PACKAGES)
    def test_import_succeeds(self, package_name: str):
        """
        ``importlib.import_module(package_name)`` completes without raising
        ``ImportError`` or any other exception.
        """
        try:
            mod = importlib.import_module(package_name)
        except ImportError as exc:
            pytest.fail(f"ImportError when importing '{package_name}': {exc}")
        except Exception as exc:
            pytest.fail(f"Unexpected {type(exc).__name__} when importing '{package_name}': {exc}")
        assert mod is not None, f"importlib.import_module('{package_name}') returned None"

    @pytest.mark.smoke
    @pytest.mark.parametrize("package_name", _ALL_PACKAGES)
    def test_is_module_type(self, package_name: str):
        """
        The imported object is a proper ``types.ModuleType`` — not a
        namespace stub, a ``MagicMock``, or an accidental non-module object.
        """
        mod = importlib.import_module(package_name)
        assert isinstance(mod, types.ModuleType), (
            f"'{package_name}' imported as {type(mod).__name__}, expected types.ModuleType"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("package_name", _ALL_PACKAGES)
    def test_has_file_attribute(self, package_name: str):
        """
        Each package exposes a ``__file__`` attribute, confirming it is a
        regular package (backed by an ``__init__.py`` on disk) and not a
        PEP 420 namespace package or a virtual module.
        """
        mod = importlib.import_module(package_name)
        assert hasattr(mod, "__file__"), (
            f"'{package_name}' has no __file__ attribute — may be a "
            f"namespace package or virtual module"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("package_name", _ALL_PACKAGES)
    def test_name_matches(self, package_name: str):
        """
        The module's ``__name__`` attribute matches the requested import
        path, ruling out import-path aliasing or redirection bugs.
        """
        mod = importlib.import_module(package_name)
        assert mod.__name__ == package_name, (
            f"Expected __name__ == '{package_name}', got '{mod.__name__}'"
        )

    @pytest.mark.smoke
    @pytest.mark.parametrize("package_name", _ALL_PACKAGES)
    def test_in_sys_modules(self, package_name: str):
        """
        After import, the package is registered in ``sys.modules`` under
        its canonical name — confirming normal import machinery was used
        (no ``importlib`` hacks that bypass registration).
        """
        importlib.import_module(package_name)
        assert package_name in sys.modules, (
            f"'{package_name}' was imported but not found in sys.modules"
        )
