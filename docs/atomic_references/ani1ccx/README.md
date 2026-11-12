# ANI-1ccx — atomic references: blocked on the published recipe

**Status:** no inputs yet. ANI-1ccx does not compute `ccsd_energy_ATOM`.

ANI-1ccx energies are CCSD(T)\*/CBS, a composite computed in ORCA (Smith et al., *Nat. Commun.* 10, 2903
(2019), doi:10.1038/s41467-019-10827-4; *Sci. Data* 7 (2020), doi:10.1038/s41597-020-0473-z). The data descriptor describes its terms:
HF/CBS extrapolated from cc-pVTZ and cc-pVQZ; MP2 correlation extrapolated to CBS from cc-pVTZ and cc-pVQZ
(α = 5.46, β = 3.05, after Halkier and Helgaker); DLPNO-CCSD(T)/cc-pVTZ with NormalPNO; a TightPNO −
NormalPNO correction with cc-pVDZ. The exact equation, frozen-core treatment, auxiliary basis sets, ORCA
version and SCF/grid settings are in the paper's Supplementary Information, Section S1.1, which MILIA has not
been able to read. No atomic references are published.

To unblock: add the Supplementary Information (Section S1.1) so the composite can be written as exact ORCA
inputs for H, C, N, O (open-shell atoms: confirm the ORCA version supports open-shell DLPNO-CCSD(T)),
then follow the rMD17 procedure.
