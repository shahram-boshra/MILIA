"""Distributions ship every data file of the package, including plugin docs and templates.

The wheel published to PyPI is built from the source distribution: ``uv build``
builds the sdist first and then the wheel from that sdist. A data file must
therefore reach both. This test builds the sdist from a copy of the source
tree, builds the wheel from the unpacked sdist, and checks that every
non-Python file under ``milia_pipeline/`` is in both.

Each build hook runs in a subprocess, as PEP 517 frontends run them, so the
backend's state stays out of the test process. Everything is written under
``tmp_path``; the working tree is not touched.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import milia_pipeline

PACKAGE_DIR = Path(milia_pipeline.__file__).resolve().parent
SOURCE_ROOT = PACKAGE_DIR.parent

# What the build reads besides the package: project metadata ([project] readme and
# license-files), the sdist manifest, and the root module listed in py-modules.
BUILD_INPUTS = ("pyproject.toml", "setup.py", "MANIFEST.in", "README.md", "LICENSE", "main.py")

# Data file types the package ships. A non-Python file of any other type under
# milia_pipeline/ fails ``test_package_data_types_are_known``, so adding a new
# data type is a deliberate packaging decision.
DATA_SUFFIXES = {".md", ".template", ".yaml", ".yml"}

_HOOK = "import sys, setuptools.build_meta as backend; getattr(backend, sys.argv[1])(sys.argv[2])"

pytestmark = [
    pytest.mark.smoke,
    pytest.mark.skipif(
        not all((SOURCE_ROOT / name).is_file() for name in ("pyproject.toml", "MANIFEST.in")),
        reason="needs the source tree, not an installed package",
    ),
    pytest.mark.skipif(
        not hasattr(tarfile, "data_filter"),
        reason="needs tarfile extraction filters (PEP 706)",
    ),
]


def _run_hook(hook: str, source_dir: Path, out_dir: Path, pattern: str) -> Path:
    out_dir.mkdir()
    result = subprocess.run(
        [sys.executable, "-c", _HOOK, hook, str(out_dir)],
        cwd=source_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"{hook} failed:\n{result.stderr[-4000:]}")
    (artifact,) = out_dir.glob(pattern)
    return artifact


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    """Build the sdist from a copy of the source tree, then the wheel from that sdist."""
    root = tmp_path_factory.mktemp("packaging")
    source = root / "source"
    source.mkdir()
    for name in BUILD_INPUTS:
        shutil.copy2(SOURCE_ROOT / name, source / name)
    shutil.copytree(
        PACKAGE_DIR,
        source / PACKAGE_DIR.name,
        ignore=shutil.ignore_patterns("__pycache__", "*.py[cod]"),
    )

    sdist = _run_hook("build_sdist", source, root / "sdist", "*.tar.gz")
    with tarfile.open(sdist) as archive:
        sdist_files = {
            member.name.split("/", 1)[1] for member in archive.getmembers() if member.isfile()
        }
        archive.extractall(root / "unpacked", filter="data")
    (unpacked,) = (root / "unpacked").iterdir()

    wheel = _run_hook("build_wheel", unpacked, root / "wheel", "*.whl")
    with zipfile.ZipFile(wheel) as archive:
        wheel_files = set(archive.namelist())

    package_data = {
        path.relative_to(source).as_posix()
        for path in (source / PACKAGE_DIR.name).rglob("*")
        if path.is_file() and path.suffix != ".py"
    }
    return package_data, sdist_files, wheel_files


def test_package_data_types_are_known(distributions):
    package_data, _, _ = distributions

    assert {Path(name).suffix for name in package_data} <= DATA_SUFFIXES
    plugins = PACKAGE_DIR / "plugins"
    kinds = sorted(path.name for path in plugins.iterdir() if (path / "user_template").is_dir())
    assert kinds == ["descriptors", "models", "transformations"]
    for kind in kinds:
        prefix = f"{PACKAGE_DIR.name}/plugins/{kind}"
        assert f"{prefix}/README.md" in package_data
        assert f"{prefix}/user_template/README.md" in package_data
        assert f"{prefix}/user_template/plugin.yaml.template" in package_data


def test_sdist_contains_every_package_data_file(distributions):
    package_data, sdist_files, _ = distributions

    assert sorted(package_data - sdist_files) == []


def test_wheel_built_from_sdist_contains_every_package_data_file(distributions):
    package_data, _, wheel_files = distributions

    assert sorted(package_data - wheel_files) == []


def test_wheel_contains_the_example_model_plugin(distributions):
    """The first-party example plugin ships whole: manifest and module, in both artefacts."""
    _, sdist_files, wheel_files = distributions

    prefix = f"{PACKAGE_DIR.name}/plugins/models/example_models"
    for name in ("plugin.yaml", "model.py"):
        assert f"{prefix}/{name}" in sdist_files
        assert f"{prefix}/{name}" in wheel_files
