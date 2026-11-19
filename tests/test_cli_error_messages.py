"""CLI errors name what failed: the configuration file, or a command-line argument.

``CLIManager.load_and_merge_config`` wrapped every error, including the argument checks of its
override step ("Test path not found: ..."), as "Failed to load configuration: ...", and ``main``
logged that as "Configuration error: ...", so a missing input file read as a broken config.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from milia_pipeline.cli_manager import CLIManager, CLIValidationError


@pytest.fixture
def root(tmp_path, monkeypatch):
    (tmp_path / "root" / "checkpoints").mkdir(parents=True)
    (tmp_path / "root" / "checkpoints" / "best.pt").write_bytes(b"x")
    monkeypatch.chdir(tmp_path)
    return tmp_path / "root"


def _predict_args(cli, test_path="missing.csv"):
    return cli.parser.parse_args(["--predict", "--model-path", "best.pt", "--test-path", test_path])


@patch("milia_pipeline.cli_manager.load_config")
def test_argument_error_is_not_reported_as_a_configuration_failure(mock_load_config, root):
    mock_load_config.return_value = {"global_paths": {"working_root_dir": str(root)}}
    cli = CLIManager()
    with pytest.raises(CLIValidationError) as error:
        cli.load_and_merge_config(_predict_args(cli))
    message = str(error.value)
    assert message.startswith("Test path not found: missing.csv")
    assert "Failed to load configuration" not in message


@patch("milia_pipeline.cli_manager.load_config", side_effect=ValueError("bad YAML"))
def test_configuration_file_failure_names_the_file(mock_load_config, root):
    cli = CLIManager()
    args = _predict_args(cli)
    with pytest.raises(CLIValidationError) as error:
        cli.load_and_merge_config(args)
    assert str(error.value) == f"Failed to load configuration {args.config}: bad YAML"


@patch("milia_pipeline.cli_manager.load_config")
def test_unexpected_override_failure_names_the_arguments(mock_load_config, root):
    mock_load_config.return_value = {"global_paths": {"working_root_dir": str(root)}}
    cli = CLIManager()
    with (
        patch.object(CLIManager, "_apply_cli_overrides", side_effect=KeyError("seed")),
        pytest.raises(CLIValidationError, match="^Failed to apply command-line arguments: "),
    ):
        cli.load_and_merge_config(_predict_args(cli))


def test_main_logs_the_message_without_a_configuration_prefix(root, capfd):
    from pathlib import Path

    import main

    configs = str(Path(main.__file__).parent / "configs")
    argv = [
        "milia",
        "--config",
        configs,
        "--predict",
        "--model-path",
        "m.pt",
        "--test-path",
        "x.csv",
    ]
    with (
        patch.object(
            CLIManager,
            "load_and_merge_config",
            side_effect=CLIValidationError("Test path not found: missing.csv"),
        ),
        patch("sys.argv", argv),
        pytest.raises(SystemExit) as exit_info,
    ):
        main.main()
    assert exit_info.value.code == 1
    # main's logger writes through its own handlers (not propagated to caplog)
    output = "".join(capfd.readouterr())
    assert "ERROR - Test path not found: missing.csv" in output
    assert "Configuration error" not in output
