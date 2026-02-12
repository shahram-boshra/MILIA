# setup.py — Backward-compatibility shim
# ========================================
# All project metadata has been migrated to pyproject.toml (PEP 621/639).
# This file exists solely for compatibility with legacy tooling that
# invokes `python setup.py ...` directly.
#
# The single source of truth for all metadata is pyproject.toml.
# Do NOT add metadata here — it will be ignored by modern build frontends.
#
# See: https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html

from setuptools import setup

setup()
