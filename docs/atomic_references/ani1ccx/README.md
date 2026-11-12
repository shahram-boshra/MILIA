# ANI-1ccx — atomic references: not reproducible from the published recipe

**Status:** not computed, no inputs. ANI-1ccx does not compute `ccsd_energy_ATOM`.

Supplementary Information Section S1.1 of the paper (chemRxiv doi:10.26434/chemrxiv.6744440.v2) defines
CCSD(T)\*/CBS as the MP2/CBS energy (HF and MP2 extrapolated from cc-pVTZ and cc-pVQZ with the Halkier and
Helgaker formulas, the "EP1" scheme of Hobza and coworkers) plus [DLPNO-CCSD(T) − MP2] with NormalPNO in a
smaller basis, plus ΔTightPNO (TightPNO − NormalPNO) in a double-zeta basis. It does not state the ORCA
version, auxiliary basis sets, frozen-core treatment or SCF/grid settings, which affect the DLPNO-CCSD(T)
and MP2 energies, so isolated-atom references at exactly this level cannot be computed (policy:
`docs/atomic_reference_energies.md`). They can be once the authors publish those settings or the
references themselves.

ANI-1ccx energies are CCSD(T)\*/CBS, a composite computed in ORCA (Smith et al., *Nat. Commun.* 10, 2903
(2019), doi:10.1038/s41467-019-10827-4; *Sci. Data* 7 (2020), doi:10.1038/s41597-020-0473-z). The data descriptor describes its terms:
HF/CBS extrapolated from cc-pVTZ and cc-pVQZ; MP2 correlation extrapolated to CBS from cc-pVTZ and cc-pVQZ
(α = 5.46, β = 3.05, after Halkier and Helgaker); DLPNO-CCSD(T)/cc-pVTZ with NormalPNO; a TightPNO −
NormalPNO correction with cc-pVDZ. The exact equation, frozen-core treatment, auxiliary basis sets, ORCA
version and SCF/grid settings are not published (see above). No atomic references are published; the
per-element values in SI Section S1.2.3 are fitted self-energy offsets for training the network, not
isolated-atom energies.
