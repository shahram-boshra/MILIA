#!/usr/bin/env python3
"""
Contract tests for the ``hpo-postgres`` extra (P2-4b, blueprint S4 / F3).

A study shared by several HPO workers needs an RDB server; ``uv.lock`` previously held no PostgreSQL
driver (F3), so every ``postgresql://`` URL failed at import. The extra declares psycopg 3 (SQLAlchemy
URL ``postgresql+psycopg://``) with a lower bound only (PyPA: exact pins belong to the lock file), and
every image installs it (Dockerfile, ``docker/verify_build.py`` check (e)). Tests that need the driver
skip where the extra is not installed (e.g. ``uv sync --extra cpu --extra dev``); the test image
installs it and its build fails without it, so they always run there.
"""

import importlib.metadata

import pytest
from packaging.requirements import Requirement
from sqlalchemy import create_engine
from sqlalchemy.engine import make_url

URL = "postgresql+psycopg://milia@db.example:5432/milia_hpo"


def _extra_requirements(extra):
    requirements = [Requirement(r) for r in importlib.metadata.requires("milia-py") or []]
    return [r for r in requirements if r.marker and r.marker.evaluate({"extra": extra})]


@pytest.mark.contract
class TestExtra:
    def test_extra_is_declared(self):
        assert "hpo-postgres" in importlib.metadata.metadata("milia-py").get_all("Provides-Extra")

    def test_extra_requires_psycopg_binary_with_floor_only(self):
        (requirement,) = [r for r in _extra_requirements("hpo-postgres") if r.name == "psycopg"]
        assert requirement.extras == {"binary"}
        assert str(requirement.specifier) == ">=3.3.6"

    def test_installed_driver_satisfies_the_extra(self):
        pytest.importorskip("psycopg", reason="hpo-postgres extra not installed")
        (requirement,) = [r for r in _extra_requirements("hpo-postgres") if r.name == "psycopg"]
        assert requirement.specifier.contains(importlib.metadata.version("psycopg"))


@pytest.mark.contract
class TestDialect:
    @pytest.fixture(autouse=True)
    def _driver(self):
        pytest.importorskip("psycopg", reason="hpo-postgres extra not installed")

    def test_psycopg_dialect_loads_its_driver(self):
        dialect = make_url(URL).get_dialect()
        driver = dialect.import_dbapi()
        assert (dialect.name, dialect.driver, driver.__name__) == (
            "postgresql",
            "psycopg",
            "psycopg",
        )

    def test_engine_is_created_without_connecting(self):
        engine = create_engine(URL)  # lazy: no connection until first use
        try:
            assert engine.dialect.driver == "psycopg"
            assert engine.url.render_as_string(hide_password=True) == URL
        finally:
            engine.dispose()
