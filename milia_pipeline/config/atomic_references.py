"""Isolated-atom reference energies for atomization energies, with their provenance.

An atomization energy subtracts the energies of a molecule's isolated atoms from its total energy, so
both must come from the same level of theory (method, basis, program and its implementation of the
method, spin treatment). A dataset declares its references in its configuration section:

.. code-block:: yaml

    qm9_config:
      atomic_references:
        source: author-published            # author-published | milia-computed | user-supplied
        level_of_theory: B3LYP/6-31G(2df,p)
        program: Gaussian 09
        citation: Ramakrishnan et al., Sci. Data 1, 140022 (2014); atomref.txt
        energies_hartree:                   # one mapping per energy property they apply to
          U0: {1: -0.500273, 6: -37.846772}

References are used in this order: published by the dataset's authors; computed by MILIA at exactly
the dataset's level of theory (documented in ``docs/atomic_reference_energies.md`` with input scripts
and outputs); set by the user. A dataset without references does not compute its atomization energy.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from milia_pipeline.exceptions import ConfigurationError

__all__ = [
    "ATOMIC_REFERENCE_SOURCES",
    "AtomicReferences",
    "get_dataset_atomic_references",
]

#: Allowed values of ``atomic_references.source``.
ATOMIC_REFERENCE_SOURCES = frozenset({"author-published", "milia-computed", "user-supplied"})

_TEXT_FIELDS = ("source", "level_of_theory", "program", "citation")


@dataclass(frozen=True)
class AtomicReferences:
    """A dataset's isolated-atom reference energies (Hartree) and where they come from."""

    dataset_type: str
    source: str
    level_of_theory: str
    program: str
    citation: str
    energies_hartree: Mapping[str, Mapping[int, float]]

    def for_property(self, energy_key: str) -> Mapping[int, float] | None:
        """References for the energy property ``energy_key`` (e.g. ``U0``), or ``None``."""
        return self.energies_hartree.get(energy_key)

    def provenance(self, energy_key: str | None = None) -> dict[str, Any]:
        """JSON-serialisable record; with ``energy_key``, only that property's energies."""
        keys = [energy_key] if energy_key is not None else sorted(self.energies_hartree)
        return {
            "dataset_type": self.dataset_type,
            "source": self.source,
            "level_of_theory": self.level_of_theory,
            "program": self.program,
            "citation": self.citation,
            "energies_hartree": {
                key: {str(z): e for z, e in sorted(self.energies_hartree[key].items())}
                for key in keys
                if key in self.energies_hartree
            },
        }


def _invalid(section_key: str, detail: str) -> ConfigurationError:
    return ConfigurationError(f"'{section_key}.atomic_references': {detail}")


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _parse(dataset_type: str, section_key: str, raw: Any) -> AtomicReferences:
    if not isinstance(raw, Mapping):
        raise _invalid(section_key, f"must be a mapping, got {type(raw).__name__}")
    for field in _TEXT_FIELDS:
        value = raw.get(field)
        if not isinstance(value, str) or not value.strip():
            raise _invalid(section_key, f"'{field}' must be a non-empty string")
    if raw["source"] not in ATOMIC_REFERENCE_SOURCES:
        raise _invalid(
            section_key,
            f"'source' must be one of {sorted(ATOMIC_REFERENCE_SOURCES)}, got {raw['source']!r}",
        )
    energies = raw.get("energies_hartree")
    if not isinstance(energies, Mapping) or not energies:
        raise _invalid(section_key, "'energies_hartree' must map energy properties to references")
    parsed: dict[str, Mapping[int, float]] = {}
    for energy_key, table in energies.items():
        if (
            not isinstance(energy_key, str)
            or not isinstance(table, Mapping)
            or not table
            or not all(
                isinstance(z, int) and not isinstance(z, bool) and z > 0 and _is_number(e)
                for z, e in table.items()
            )
        ):
            raise _invalid(
                section_key,
                f"'energies_hartree.{energy_key}' must map atomic numbers (int > 0) to energies "
                f"in Hartree (float); got {table!r}",
            )
        parsed[energy_key] = MappingProxyType({z: float(e) for z, e in table.items()})
    return AtomicReferences(
        dataset_type=dataset_type,
        source=raw["source"],
        level_of_theory=raw["level_of_theory"].strip(),
        program=raw["program"].strip(),
        citation=raw["citation"].strip(),
        energies_hartree=MappingProxyType(parsed),
    )


def get_dataset_atomic_references(dataset_type: str) -> AtomicReferences | None:
    """The atomic references declared in ``<dataset_type>_config.atomic_references``, or ``None``.

    Raises:
        ConfigurationError: If the declaration is malformed.
    """
    from milia_pipeline.config.config_loader import load_config

    section_key = f"{dataset_type.lower()}_config"
    section = load_config().get(section_key) or {}
    raw = section.get("atomic_references")
    if raw is None:
        return None
    return _parse(dataset_type, section_key, raw)
