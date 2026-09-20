"""
Electronic / conceptual-DFT reactivity descriptors (extended Hückel) for MILIA
(descriptor programme, Pace 10 / v1.13.0).

First-party PLUGIN-NATIVE quantum-chemical reactivity descriptors computed with RDKit's
extended Hückel theory engine (``rdkit.Chem.rdEHTTools`` = YAeHMOP, Greg Landrum, BSD) on the
MILIA-core conformer (AddHs -> ETKDG EmbedMolecule(randomSeed=42) -> MMFFOptimizeMolecule).
RDKit-native, deterministic, bit-exact; validated against a frozen-conformer snapshot and a
live RDKit-EHT cross-check. Zero-core-modification plugin contract (``name`` == ``function_name``;
NaN on any failure, never raise).

**IMPORTANT — level of theory (read before use).** These are **extended-Hückel-level**
(qualitative, semi-empirical) reactivity descriptors. Extended Hückel theory (Hoffmann 1963)
gives interpretable frontier-orbital *trends*, NOT DFT-grade absolute energies. Every value is
**method- and geometry-dependent**: it is defined only within this fixed protocol (EHT single
point on the MMFF-optimised, seed-42 ETKDG conformer) and is **not comparable** to HOMO/LUMO
etc. from a different method (xTB, DFT) or geometry. Use them as relative QSAR/QSPR features,
not as physical observables. See the Blueprint §9 Pace-10 DECISION RECORD for why EHT (not
GFN2-xTB/DFT) was chosen for this in-pipeline, deterministic, dependency-free slot.

Family (25): frontier orbitals (HOMO/LUMO/gap), conceptual-DFT reactivity indices (IP, EA,
electronegativity, chemical potential, hardness, softness, electrophilicity, electro-
donating/accepting power; Parr-Pearson/Gazquez, via Koopmans' theorem), EHT total/Fermi
energy, EHT partial-charge descriptors, and condensed Fukui aggregates (f+/f-/f0 + dual
descriptor) from finite differences on the N+-1 species at fixed geometry (Yang-Mortier).

Cost note: EHT is a single-point in the same conformer-cost tier as the other 3D plugins
(`cpsa_geometric_3d`, `addcore_3d`); Fukui runs three EHT single-points. For latency-sensitive
deployments the whole plugin can be disabled via `configs/plugins.yaml` -> `disabled_plugins`.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdEHTTools

logger = logging.getLogger(__name__)
_NAN = float("nan")


def _run(mol):
    ok, res = rdEHTTools.RunMol(mol)
    if not ok:
        raise ValueError("extended Hückel calculation failed")
    return res


def _state_charges(mol, dcharge):
    """EHT atomic charges of the N+dcharge-electron species at the SAME geometry.

    The net charge is set as a formal charge (EHT uses only the total electron count, not
    where the charge sits — verified placement-invariant), enabling finite-difference
    condensed Fukui functions.
    """
    mm = Chem.Mol(mol)
    if dcharge:
        atom = mm.GetAtomWithIdx(0)
        atom.SetNoImplicit(True)
        atom.SetFormalCharge(atom.GetFormalCharge() + dcharge)
    return np.array(_run(mm).GetAtomicCharges())


@functools.lru_cache(maxsize=1)
def _eht_block(mol):
    """Compute all 25 EHT reactivity descriptors once per molecule (single-slot memoised).

    Requires a conformer (embedded by the MILIA core). Any failure -> all NaN.
    """
    out = {name: _NAN for name in ALL_DESCRIPTOR_NAMES}
    try:
        mol.GetConformer(-1)
        res = _run(mol)
    except Exception:
        return out

    oe = np.array(res.GetOrbitalEnergies())
    nocc = res.numElectrons // 2
    if nocc < 1 or nocc >= len(oe):
        return out
    homo, lumo = float(oe[nocc - 1]), float(oe[nocc])
    gap = lumo - homo
    ip, ea = -homo, -lumo
    chi = (ip + ea) / 2.0
    mu = -chi
    eta = (ip - ea) / 2.0
    denom = 16.0 * (ip - ea)
    q = np.array(res.GetAtomicCharges())
    aq = np.abs(q)

    out["E_HOMO"] = homo
    out["E_LUMO"] = lumo
    out["HOMO_LUMO_Gap"] = gap
    out["IonizationPotential"] = ip
    out["ElectronAffinity"] = ea
    out["Electronegativity"] = chi
    out["ChemicalPotential"] = mu
    out["ChemicalHardness"] = eta
    out["ChemicalSoftness"] = 1.0 / eta if eta != 0 else _NAN
    out["ElectrophilicityIndex"] = mu * mu / (2.0 * eta) if eta != 0 else _NAN
    out["ElectrodonatingPower"] = (3.0 * ip + ea) ** 2 / denom if denom != 0 else _NAN
    out["ElectroacceptingPower"] = (ip + 3.0 * ea) ** 2 / denom if denom != 0 else _NAN
    out["EHT_TotalEnergy"] = float(res.totalEnergy)
    out["EHT_FermiEnergy"] = float(res.fermiEnergy)
    out["MaxEHTCharge"] = float(q.max())
    out["MinEHTCharge"] = float(q.min())
    out["MaxAbsEHTCharge"] = float(aq.max())
    out["MinAbsEHTCharge"] = float(aq.min())
    out["SumAbsEHTCharge"] = float(aq.sum())
    out["MeanAbsEHTCharge"] = float(aq.mean())

    try:  # condensed Fukui (finite difference, fixed geometry)
        q_cat = _state_charges(mol, +1)
        q_an = _state_charges(mol, -1)
        f_plus = q - q_an
        f_minus = q_cat - q
        f_rad = (q_cat - q_an) / 2.0
        dual = f_plus - f_minus
        out["MaxFukuiNucleophilic"] = float(f_plus.max())
        out["MaxFukuiElectrophilic"] = float(f_minus.max())
        out["MaxFukuiRadical"] = float(f_rad.max())
        out["MaxDualDescriptor"] = float(dual.max())
        out["MinDualDescriptor"] = float(dual.min())
    except Exception:
        pass
    return out


def _make(name):
    def fn(mol):
        try:
            if mol is None:
                return _NAN
            return float(_eht_block(mol)[name])
        except Exception:
            return _NAN

    return fn


_E_HOMO = _make("E_HOMO")
_E_LUMO = _make("E_LUMO")
_HOMO_LUMO_Gap = _make("HOMO_LUMO_Gap")
_IonizationPotential = _make("IonizationPotential")
_ElectronAffinity = _make("ElectronAffinity")
_Electronegativity = _make("Electronegativity")
_ChemicalPotential = _make("ChemicalPotential")
_ChemicalHardness = _make("ChemicalHardness")
_ChemicalSoftness = _make("ChemicalSoftness")
_ElectrophilicityIndex = _make("ElectrophilicityIndex")
_ElectrodonatingPower = _make("ElectrodonatingPower")
_ElectroacceptingPower = _make("ElectroacceptingPower")
_EHT_TotalEnergy = _make("EHT_TotalEnergy")
_EHT_FermiEnergy = _make("EHT_FermiEnergy")
_MaxEHTCharge = _make("MaxEHTCharge")
_MinEHTCharge = _make("MinEHTCharge")
_MaxAbsEHTCharge = _make("MaxAbsEHTCharge")
_MinAbsEHTCharge = _make("MinAbsEHTCharge")
_SumAbsEHTCharge = _make("SumAbsEHTCharge")
_MeanAbsEHTCharge = _make("MeanAbsEHTCharge")
_MaxFukuiNucleophilic = _make("MaxFukuiNucleophilic")
_MaxFukuiElectrophilic = _make("MaxFukuiElectrophilic")
_MaxFukuiRadical = _make("MaxFukuiRadical")
_MaxDualDescriptor = _make("MaxDualDescriptor")
_MinDualDescriptor = _make("MinDualDescriptor")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def E_HOMO(mol):
    """EHT HOMO orbital energy (eV)."""
    return _E_HOMO(mol)


def E_LUMO(mol):
    """EHT LUMO orbital energy (eV)."""
    return _E_LUMO(mol)


def HOMO_LUMO_Gap(mol):
    """EHT HOMO-LUMO gap (eV)."""
    return _HOMO_LUMO_Gap(mol)


def IonizationPotential(mol):
    """ionization potential -E_HOMO (Koopmans, eV)."""
    return _IonizationPotential(mol)


def ElectronAffinity(mol):
    """electron affinity -E_LUMO (Koopmans, eV)."""
    return _ElectronAffinity(mol)


def Electronegativity(mol):
    """Mulliken electronegativity chi=(IP+EA)/2 (eV)."""
    return _Electronegativity(mol)


def ChemicalPotential(mol):
    """electronic chemical potential mu=-chi (eV)."""
    return _ChemicalPotential(mol)


def ChemicalHardness(mol):
    """chemical hardness eta=(IP-EA)/2 (eV)."""
    return _ChemicalHardness(mol)


def ChemicalSoftness(mol):
    """chemical softness S=1/eta (1/eV)."""
    return _ChemicalSoftness(mol)


def ElectrophilicityIndex(mol):
    """Parr electrophilicity index omega=mu^2/(2 eta) (eV)."""
    return _ElectrophilicityIndex(mol)


def ElectrodonatingPower(mol):
    """electrodonating power omega- (Gazquez, eV)."""
    return _ElectrodonatingPower(mol)


def ElectroacceptingPower(mol):
    """electroaccepting power omega+ (Gazquez, eV)."""
    return _ElectroacceptingPower(mol)


def EHT_TotalEnergy(mol):
    """EHT total electronic energy."""
    return _EHT_TotalEnergy(mol)


def EHT_FermiEnergy(mol):
    """EHT Fermi energy (eV)."""
    return _EHT_FermiEnergy(mol)


def MaxEHTCharge(mol):
    """maximum EHT (Mulliken) partial charge."""
    return _MaxEHTCharge(mol)


def MinEHTCharge(mol):
    """minimum EHT partial charge."""
    return _MinEHTCharge(mol)


def MaxAbsEHTCharge(mol):
    """maximum absolute EHT partial charge."""
    return _MaxAbsEHTCharge(mol)


def MinAbsEHTCharge(mol):
    """minimum absolute EHT partial charge."""
    return _MinAbsEHTCharge(mol)


def SumAbsEHTCharge(mol):
    """sum of absolute EHT partial charges."""
    return _SumAbsEHTCharge(mol)


def MeanAbsEHTCharge(mol):
    """mean absolute EHT partial charge."""
    return _MeanAbsEHTCharge(mol)


def MaxFukuiNucleophilic(mol):
    """max condensed Fukui f+ (nucleophilic attack site)."""
    return _MaxFukuiNucleophilic(mol)


def MaxFukuiElectrophilic(mol):
    """max condensed Fukui f- (electrophilic attack site)."""
    return _MaxFukuiElectrophilic(mol)


def MaxFukuiRadical(mol):
    """max condensed Fukui f0 (radical attack site)."""
    return _MaxFukuiRadical(mol)


def MaxDualDescriptor(mol):
    """max dual descriptor f+ - f- (electrophilic site)."""
    return _MaxDualDescriptor(mol)


def MinDualDescriptor(mol):
    """min dual descriptor f+ - f- (nucleophilic site)."""
    return _MinDualDescriptor(mol)


ALL_DESCRIPTOR_NAMES = (
    "E_HOMO",
    "E_LUMO",
    "HOMO_LUMO_Gap",
    "IonizationPotential",
    "ElectronAffinity",
    "Electronegativity",
    "ChemicalPotential",
    "ChemicalHardness",
    "ChemicalSoftness",
    "ElectrophilicityIndex",
    "ElectrodonatingPower",
    "ElectroacceptingPower",
    "EHT_TotalEnergy",
    "EHT_FermiEnergy",
    "MaxEHTCharge",
    "MinEHTCharge",
    "MaxAbsEHTCharge",
    "MinAbsEHTCharge",
    "SumAbsEHTCharge",
    "MeanAbsEHTCharge",
    "MaxFukuiNucleophilic",
    "MaxFukuiElectrophilic",
    "MaxFukuiRadical",
    "MaxDualDescriptor",
    "MinDualDescriptor",
)
