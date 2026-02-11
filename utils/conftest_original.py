"""Pytest configuration â€” custom markers and third-party warning filters.

This ``conftest.py`` lives in the ``tests/`` directory and is auto-discovered
by pytest at collection time (before any test runs).

It performs two tasks via the ``pytest_configure`` hook:

1. **Marker registration** â€” registers project-wide custom markers
   (``smoke``, ``contract``, ``e2e``, ``regression``, ``thread_safety``,
   ``slow``) so that ``--strict-markers`` mode does not raise
   ``PytestUnknownMarkWarning``.

2. **Third-party warning suppression** â€” filters out ``DeprecationWarning``
   messages emitted internally by matplotlib <3.10.7, which uses deprecated
   camelCase pyparsing APIs (``oneOf``, ``parseString``, ``resetCache``,
   ``enablePackrat``).  These warnings originate in matplotlib's own code
   (``_fontconfig_pattern.py``, ``_mathtext.py``), not in MILIA source.

   Upstream issue : https://github.com/matplotlib/matplotlib/issues/30617
   Fixed in       : matplotlib >=3.10.7
   Action         : Remove the filterwarnings lines after upgrading matplotlib.

References
----------
- Marker registration via ``pytest_configure``:
  https://docs.pytest.org/en/stable/example/markers.html
- ``config.addinivalue_line("filterwarnings", ...)`` pattern:
  https://docs.pytest.org/en/stable/how-to/capture-warnings.html
"""


def pytest_configure(config):
    """Register custom markers and third-party warning filters.

    This hook runs at plugin registration time (before collection),
    which is the earliest point where both marker registration and
    ``filterwarnings`` injection take effect.

    ``config.addinivalue_line`` appends to the same ini-option lists
    that ``pytest.ini`` / ``pyproject.toml`` would populate, so the
    semantics are identical.

    References
    ----------
    - Markers : https://docs.pytest.org/en/stable/example/markers.html
    - Warnings: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
    """
    # -----------------------------------------------------------------
    # Custom markers
    # -----------------------------------------------------------------
    config.addinivalue_line(
        "markers",
        "smoke: Smoke tests â€” fast, first gate in CI/CD pipeline",
    )
    config.addinivalue_line(
        "markers",
        "contract: Interface contract validation tests",
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end workflow tests",
    )
    config.addinivalue_line(
        "markers",
        "regression: Regression protection tests",
    )
    config.addinivalue_line(
        "markers",
        "thread_safety: Concurrent access tests",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests taking > 30 seconds",
    )

    # -----------------------------------------------------------------
    # Third-party warning filters
    # -----------------------------------------------------------------
    # matplotlib <3.10.7 internally uses deprecated pyparsing camelCase
    # APIs.  The warnings originate in matplotlib/_fontconfig_pattern.py
    # and matplotlib/_mathtext.py â€” not in MILIA source code.
    #
    # Tracked  : https://github.com/matplotlib/matplotlib/issues/30617
    # Fixed in : matplotlib >=3.10.7 (min pyparsing bumped to 3.0)
    # Action   : Remove these four lines after upgrading matplotlib.
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'oneOf' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'parseString' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'resetCache' deprecated:DeprecationWarning:matplotlib",
    )
    config.addinivalue_line(
        "filterwarnings",
        "ignore:'enablePackrat' deprecated:DeprecationWarning:matplotlib",
    )

