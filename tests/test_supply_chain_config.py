"""Accepted OSV advisories (osv-scanner.toml, F54) stay explained, time-boxed and disclosed.

OSV-Scanner skips every ``[[IgnoredVulns]]`` entry until its ``effectiveUntil`` date. Each entry must say why it is
accepted, carry that re-triage date, and be listed in the CHANGELOG ``Security`` notes, so an ignore never hides
an advisory silently.
"""

import datetime
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIG = tomllib.loads((REPO / "osv-scanner.toml").read_text(encoding="utf-8"))
ENTRIES = CONFIG.get("IgnoredVulns", [])


def test_accepted_advisories_are_listed():
    assert ENTRIES, "osv-scanner.toml has no [[IgnoredVulns]] entries"
    ids = [entry["id"] for entry in ENTRIES]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("entry", ENTRIES, ids=lambda entry: entry["id"])
def test_accepted_advisory_has_reason_date_and_disclosure(entry):
    assert entry.get("reason", "").strip()
    assert isinstance(entry.get("effectiveUntil"), datetime.date)
    assert entry["id"] in (REPO / "CHANGELOG.md").read_text(encoding="utf-8")
