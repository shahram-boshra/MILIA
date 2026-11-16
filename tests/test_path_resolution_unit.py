"""
Input paths given on the command line resolve the way POSIX does first (F59).

A relative ``--test-path`` or ``--model-path`` is taken relative to the current directory, then
looked up in the configured locations (checkpoints directory by name, then working_root_dir).
The error lists exactly the locations that were checked. The CLI's pre-flight validation and
``main.handle_predict_mode`` share the resolver, so both pick the same file.
"""

import argparse
import logging
from unittest.mock import Mock, patch

import pytest

import main
from milia_pipeline.cli_manager import CLIManager, CLIValidationError
from milia_pipeline.path_resolution import (
    InputPathNotFoundError,
    input_path_candidates,
    resolve_input_path,
)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    """A current directory and a separate working_root_dir with a checkpoints directory."""
    cwd = tmp_path / "cwd"
    root = tmp_path / "root"
    (cwd / "test_data").mkdir(parents=True)
    (root / "checkpoints").mkdir(parents=True)
    monkeypatch.chdir(cwd)
    return cwd, root


def test_relative_path_is_found_in_the_current_directory_first(layout):
    cwd, root = layout
    (cwd / "test_data" / "molecules.csv").write_text("smiles,molecule_id\nCCO,a\n")
    (root / "test_data").mkdir()
    (root / "test_data" / "molecules.csv").write_text("smiles,molecule_id\nCC,b\n")
    assert (
        resolve_input_path("test_data/molecules.csv", root) == cwd / "test_data" / "molecules.csv"
    )


def test_relative_path_falls_back_to_working_root_dir(layout):
    _, root = layout
    (root / "inputs.csv").write_text("smiles,molecule_id\n")
    assert resolve_input_path("inputs.csv", root) == root / "inputs.csv"


def test_model_path_order_is_cwd_then_checkpoints_by_name_then_root(layout):
    cwd, root = layout
    checkpoints = (root / "checkpoints",)
    (root / "checkpoints" / "best.pt").write_bytes(b"x")
    assert resolve_input_path("./checkpoints/best.pt", root, name_dirs=checkpoints) == (
        root / "checkpoints" / "best.pt"
    )
    (cwd / "checkpoints").mkdir()
    (cwd / "checkpoints" / "best.pt").write_bytes(b"y")
    assert resolve_input_path("./checkpoints/best.pt", root, name_dirs=checkpoints) == (
        cwd / "checkpoints" / "best.pt"
    )


def test_absolute_and_home_paths_are_used_as_given(layout, monkeypatch):
    cwd, root = layout
    target = cwd / "abs.csv"
    target.write_text("x")
    assert input_path_candidates(str(target), root) == (input_path_candidates(target, root)[0],)
    assert resolve_input_path(str(target), root) == target
    monkeypatch.setenv("HOME", str(cwd))
    assert resolve_input_path("~/abs.csv", root) == target


def test_missing_path_lists_each_checked_location_once_in_order(layout):
    cwd, root = layout
    with pytest.raises(InputPathNotFoundError) as missing:
        resolve_input_path("best.pt", root, name_dirs=(root / "checkpoints",))
    error = missing.value
    assert isinstance(error, FileNotFoundError)
    assert [c.path for c in error.candidates] == [
        cwd / "best.pt",
        root / "checkpoints" / "best.pt",
        root / "best.pt",
    ]
    assert (
        error.searched.splitlines()[0]
        == f"  1. {cwd / 'best.pt'} (relative to the current directory)"
    )

    with pytest.raises(InputPathNotFoundError) as same_dir:
        resolve_input_path("x.csv", cwd)  # cwd == working_root_dir: one candidate, not two
    assert len(same_dir.value.candidates) == 1


@patch("milia_pipeline.cli_manager.load_config")
def test_cli_validation_accepts_the_readme_sample_relative_to_cwd(mock_load_config, layout):
    """README step 7: --test-path test_data/molecules.csv from the checkout or /app in Docker."""
    cwd, root = layout
    (cwd / "test_data" / "molecules.csv").write_text("smiles,molecule_id\nCCO,a\n")
    (root / "checkpoints" / "best.pt").write_bytes(b"x")
    mock_load_config.return_value = {"global_paths": {"working_root_dir": str(root)}}
    cli = CLIManager()
    args = cli.parser.parse_args(
        [
            "--predict",
            "--model-path",
            "./checkpoints/best.pt",
            "--test-path",
            "test_data/molecules.csv",
        ]
    )
    assert cli.load_and_merge_config(args) is not None


@patch("milia_pipeline.cli_manager.load_config")
def test_cli_validation_error_names_every_checked_location(mock_load_config, layout):
    cwd, root = layout
    (root / "checkpoints" / "best.pt").write_bytes(b"x")
    mock_load_config.return_value = {"global_paths": {"working_root_dir": str(root)}}
    cli = CLIManager()
    args = cli.parser.parse_args(
        ["--predict", "--model-path", "best.pt", "--test-path", "missing.csv"]
    )
    with pytest.raises(CLIValidationError) as error:
        cli.load_and_merge_config(args)
    message = str(error.value)
    assert "Test path not found: missing.csv" in message
    assert f"1. {cwd / 'missing.csv'} (relative to the current directory)" in message
    assert f"2. {root / 'missing.csv'} (relative to working_root_dir)" in message


def test_predict_mode_uses_the_same_resolution_and_reports_missing_inputs(layout):
    cwd, root = layout
    args = Mock(spec=argparse.Namespace)
    args.model_path = "best.pt"
    args.test_path = "missing.csv"
    args.preds_path = None
    logger = Mock(spec=logging.Logger)
    config = {"global_paths": {"working_root_dir": str(root)}, "models": {"prediction": {}}}
    (root / "checkpoints" / "best.pt").write_bytes(b"x")

    assert main.handle_predict_mode(args, logger, config) == 1
    logged = " ".join(str(call) for call in logger.error.call_args_list)
    assert "missing.csv not found" in logged and str(cwd / "missing.csv") in logged
