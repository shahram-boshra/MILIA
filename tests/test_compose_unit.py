#!/usr/bin/env python3
"""
Contract tests for the parallel-HPO Compose files (P2-4c, blueprint S4).

``compose.yaml`` (profile ``hpo``) chains the shared-study lifecycle of P2-3e/P2-3f — process, init,
workers, finalize — on PostgreSQL, with secrets as files and pinned images. These tests read the YAML
statically (no Docker needed); ``docker compose config`` and a real run are the operational gate.
"""

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
DIGEST = re.compile(r"@sha256:[0-9a-f]{64}$")
MILIA_SERVICES = ("hpo-process", "hpo-init", "hpo-worker-0", "hpo-worker-1", "hpo-finalize")


@pytest.fixture(scope="module")
def compose():
    return yaml.safe_load((ROOT / "compose.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def gpu():
    return yaml.safe_load((ROOT / "compose.gpu.yaml").read_text(encoding="utf-8"))


def _after(service, compose):
    return {
        name: spec["condition"]
        for name, spec in (compose["services"][service].get("depends_on") or {}).items()
    }


@pytest.mark.contract
class TestLifecycle:
    def test_steps_run_in_order(self, compose):
        assert _after("hpo-process", compose) == {}
        assert _after("hpo-init", compose) == {
            "postgres": "service_healthy",
            "hpo-process": "service_completed_successfully",
        }
        for worker in ("hpo-worker-0", "hpo-worker-1"):
            assert _after(worker, compose) == {"hpo-init": "service_completed_successfully"}
        assert _after("hpo-finalize", compose) == {
            "hpo-worker-0": "service_completed_successfully",
            "hpo-worker-1": "service_completed_successfully",
        }

    def test_commands(self, compose):
        services = compose["services"]
        assert services["hpo-process"]["command"] == ["--process"]
        assert services["hpo-init"]["command"] == ["--hpo-init"]
        assert services["hpo-finalize"]["command"] == ["--train", "--hpo", "--hpo-finalize"]

    def test_every_worker_has_its_own_index(self, compose):
        workers = [name for name in compose["services"] if name.startswith("hpo-worker-")]
        for name in workers:
            command = compose["services"][name]["command"]
            assert command[:3] == ["--train", "--hpo", "--hpo-worker"]
            assert command[3] == name.rsplit("-", 1)[1]
        assert sorted(int(n.rsplit("-", 1)[1]) for n in workers) == list(range(len(workers)))
        assert set(_after("hpo-finalize", compose)) == set(workers)


@pytest.mark.contract
class TestSecurity:
    def test_images_pinned_by_digest(self, compose):
        for name in ("postgres", "optuna-dashboard"):
            assert DIGEST.search(compose["services"][name]["image"]), name

    def test_secrets_are_files_under_one_directory(self, compose):
        for spec in compose["secrets"].values():
            assert set(spec) == {"file"}
            assert spec["file"].startswith("./secrets/")

    def test_secret_directory_is_ignored(self):
        # .gitignore / .dockerignore are themselves docker-ignored: present in a checkout, absent in the
        # test image, so this check runs where the files exist (checkout, sandbox), not in the image.
        ignore_files = (ROOT / ".gitignore", ROOT / ".dockerignore")
        if not all(path.is_file() for path in ignore_files):
            pytest.skip("ignore files are not part of the image build context")
        assert "/secrets/" in ignore_files[0].read_text(encoding="utf-8").splitlines()
        assert "secrets/" in ignore_files[1].read_text(encoding="utf-8").splitlines()

    def test_no_credentials_in_environment(self, compose):
        environment = compose["services"]["postgres"]["environment"]
        assert "POSTGRES_PASSWORD" not in environment
        assert environment["POSTGRES_PASSWORD_FILE"] == "/run/secrets/postgres_password"
        for name, service in compose["services"].items():
            for value in (service.get("environment") or {}).values():
                assert "://" not in str(value), name

    def test_milia_services_read_the_url_secret(self, compose):
        for name in MILIA_SERVICES:
            assert compose["services"][name]["secrets"] == ["hpo_storage_url"], name

    def test_every_service_runs_as_the_host_user(self, compose):
        for name, service in compose["services"].items():
            assert service["user"].startswith("${MILIA_UID:?"), name
            assert "${MILIA_GID:?" in service["user"], name

    def test_database_not_published_and_dashboard_on_loopback(self, compose):
        assert "ports" not in compose["services"]["postgres"]
        (port,) = compose["services"]["optuna-dashboard"]["ports"]
        assert port.startswith("127.0.0.1:")

    def test_dashboard_stops_when_its_secret_is_unreadable(self, compose):
        script = compose["services"]["optuna-dashboard"]["entrypoint"][2]
        assert 'url="$$(cat /run/secrets/dashboard_storage_url)" &&' in script
        assert script.rstrip().endswith('"$$url"')


@pytest.mark.contract
class TestGpuOverride:
    def test_one_distinct_gpu_per_worker(self, gpu):
        ids = {}
        for name, service in gpu["services"].items():
            (device,) = service["deploy"]["resources"]["reservations"]["devices"]
            assert device["driver"] == "nvidia" and device["capabilities"] == ["gpu"]
            assert "count" not in device  # mutually exclusive with device_ids
            ids[name] = device["device_ids"]
        workers = [ids[n] for n in ids if n.startswith("hpo-worker-")]
        assert len({tuple(i) for i in workers}) == len(workers)

    def test_overrides_only_existing_services(self, compose, gpu):
        assert set(gpu["services"]) <= set(compose["services"])
