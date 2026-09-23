"""
CDK Laggner substructure-count descriptors for MILIA
(descriptor programme, Pace 11 / v1.14.0).

DEP-BOUND, opt-in plugin. Computes the 307 Laggner functional-group substructure COUNTS
(CDK `SubstructureFingerprinter.getCountFingerprint`; the Christian Laggner / InteLigand
SMARTS set) by calling the CANONICAL CDK engine directly via a persistent in-process JVM
(`jpype`), with the complete CDK jar sourced from the `CDK-pywrapper` dependency (its wheel,
offline-safe — NOT bundled in the MILIA repo, NOT fetched from Maven at runtime).

Why an engine wrapper, not an RDKit reimplementation: RDKit and CDK use different aromaticity/
SMARTS perception, so an RDKit reimplementation reproduces CDK on only ~98.86% of patterns
(~11/307 fused-heteroaromatic/tautomer patterns diverge). Using CDK itself makes the values
**canonical and bit-exact by construction**. Full rationale: Blueprint §9 Pace-11 STRATEGY
EVALUATION + DECISION RECORD.

Design (zero core modification):
    * Lazy, once-per-process JVM boot (~0.4 s) via jpype; the CDK jar path is discovered from
      the installed `CDK_pywrapper` package. Per-molecule compute ~7 ms.
    * Thread-safe: a module lock serialises CDK calls (MILIA's descriptor calculator is
      sequential by default — `configs/descriptors.yaml parallel_computation: false`).
    * Block-cache: one CDK call per molecule -> all 307 counts cached; 307 wrappers index it.
    * Input RDKit mol -> canonical SMILES -> CDK parse -> counts (the consistent integration
      point; deterministic).
    * If `CDK-pywrapper`/`jpype`/Java are unavailable, the plugin is auto-SKIPPED at discovery by
      MILIA's optional-dependency "skip-not-fail" mechanism (the declared `CDK-pywrapper`
      dependency), so the base install and the default descriptor count (3,026) are unaffected.
      Installing the `[descriptors-cdk]` extra is the opt-in that activates it (-> 3,333). If the
      dependency is present but a call still fails, descriptors return NaN.

Names: `SubFPC1..307` (CDK/PaDEL convention); the Laggner functional-group label of each is in
its description. `category: fragments`, `block: SubstructureCount`, `requires_3d: false`,
`dependencies: [CDK-pywrapper]`.

Reproducibility: pinned to the CDK version shipped by the installed `CDK-pywrapper`; the frozen
snapshot is CDK's own integer counts (self-consistent, cross-platform). Regenerate on a CDK bump.

Author: Asadollah (Shahram) Boshra
License: MIT
"""

from __future__ import annotations

import functools
import glob
import logging
import os
import threading

# Module-level imports of the optional engine deps. If the `[descriptors-cdk]` extra (and thus
# Python >= 3.11) is not installed, these raise ImportError at module load, and MILIA's plugin
# loader skip-not-fails this plugin (its 307 descriptors are NOT registered) — so the base install
# and the default descriptor count (3,026) are unaffected. The heavy JVM boot stays lazy (below).
import CDK_pywrapper  # noqa: F401  (imported for availability-gating + its bundled CDK jar path)
import jpype

logger = logging.getLogger(__name__)
_NAN = float("nan")
_LOCK = threading.Lock()
_CDK = {"ready": None}


def _init_cdk():
    """Lazily start one JVM and load CDK (from the CDK-pywrapper wheel). Idempotent, guarded."""
    if _CDK["ready"] is not None:
        return _CDK["ready"]
    with _LOCK:
        if _CDK["ready"] is not None:
            return _CDK["ready"]
        try:
            jar = glob.glob(
                os.path.join(os.path.dirname(CDK_pywrapper.__file__), "**", "*.jar"),
                recursive=True,
            )[0]
            if not jpype.isJVMStarted():
                jpype.startJVM(classpath=[jar], convertStrings=False)
            _CDK["SmilesParser"] = jpype.JClass("org.openscience.cdk.smiles.SmilesParser")
            _CDK["builder"] = jpype.JClass(
                "org.openscience.cdk.silent.SilentChemObjectBuilder"
            ).getInstance()
            _CDK["Fingerprinter"] = jpype.JClass(
                "org.openscience.cdk.fingerprint.SubstructureFingerprinter"
            )
            _CDK["ACM"] = jpype.JClass(
                "org.openscience.cdk.tools.manipulator.AtomContainerManipulator"
            )
            _CDK["ready"] = True
        except Exception as exc:  # jpype / CDK-pywrapper / Java absent
            logger.warning("cdk_substructure: CDK engine unavailable (%s); descriptors -> NaN", exc)
            _CDK["ready"] = False
    return _CDK["ready"]


@functools.lru_cache(maxsize=1)
def _cdk_block(mol):
    """Compute all 307 Laggner substructure counts once per molecule (single-slot memoised)."""
    out = {name: _NAN for name in ALL_DESCRIPTOR_NAMES}
    if mol is None or not _init_cdk():
        return out
    try:
        from rdkit import Chem

        smiles = Chem.MolToSmiles(mol)
        if not smiles:
            return out
        counts = [0] * 307
        with _LOCK:
            parser = _CDK["SmilesParser"](_CDK["builder"])
            ac = parser.parseSmiles(smiles)
            _CDK["ACM"].percieveAtomTypesAndConfigureAtoms(ac)
            cf = _CDK["Fingerprinter"]().getCountFingerprint(ac)
            for i in range(cf.numOfPopulatedbins()):
                counts[int(cf.getHash(i))] = int(cf.getCount(i))
        for i in range(307):
            out[f"SubFPC{i + 1}"] = float(counts[i])
    except Exception:
        return {name: _NAN for name in ALL_DESCRIPTOR_NAMES}
    return out


def _make(name):
    def fn(mol):
        try:
            return float(_cdk_block(mol)[name])
        except Exception:
            return _NAN

    return fn


_SubFPC1 = _make("SubFPC1")
_SubFPC2 = _make("SubFPC2")
_SubFPC3 = _make("SubFPC3")
_SubFPC4 = _make("SubFPC4")
_SubFPC5 = _make("SubFPC5")
_SubFPC6 = _make("SubFPC6")
_SubFPC7 = _make("SubFPC7")
_SubFPC8 = _make("SubFPC8")
_SubFPC9 = _make("SubFPC9")
_SubFPC10 = _make("SubFPC10")
_SubFPC11 = _make("SubFPC11")
_SubFPC12 = _make("SubFPC12")
_SubFPC13 = _make("SubFPC13")
_SubFPC14 = _make("SubFPC14")
_SubFPC15 = _make("SubFPC15")
_SubFPC16 = _make("SubFPC16")
_SubFPC17 = _make("SubFPC17")
_SubFPC18 = _make("SubFPC18")
_SubFPC19 = _make("SubFPC19")
_SubFPC20 = _make("SubFPC20")
_SubFPC21 = _make("SubFPC21")
_SubFPC22 = _make("SubFPC22")
_SubFPC23 = _make("SubFPC23")
_SubFPC24 = _make("SubFPC24")
_SubFPC25 = _make("SubFPC25")
_SubFPC26 = _make("SubFPC26")
_SubFPC27 = _make("SubFPC27")
_SubFPC28 = _make("SubFPC28")
_SubFPC29 = _make("SubFPC29")
_SubFPC30 = _make("SubFPC30")
_SubFPC31 = _make("SubFPC31")
_SubFPC32 = _make("SubFPC32")
_SubFPC33 = _make("SubFPC33")
_SubFPC34 = _make("SubFPC34")
_SubFPC35 = _make("SubFPC35")
_SubFPC36 = _make("SubFPC36")
_SubFPC37 = _make("SubFPC37")
_SubFPC38 = _make("SubFPC38")
_SubFPC39 = _make("SubFPC39")
_SubFPC40 = _make("SubFPC40")
_SubFPC41 = _make("SubFPC41")
_SubFPC42 = _make("SubFPC42")
_SubFPC43 = _make("SubFPC43")
_SubFPC44 = _make("SubFPC44")
_SubFPC45 = _make("SubFPC45")
_SubFPC46 = _make("SubFPC46")
_SubFPC47 = _make("SubFPC47")
_SubFPC48 = _make("SubFPC48")
_SubFPC49 = _make("SubFPC49")
_SubFPC50 = _make("SubFPC50")
_SubFPC51 = _make("SubFPC51")
_SubFPC52 = _make("SubFPC52")
_SubFPC53 = _make("SubFPC53")
_SubFPC54 = _make("SubFPC54")
_SubFPC55 = _make("SubFPC55")
_SubFPC56 = _make("SubFPC56")
_SubFPC57 = _make("SubFPC57")
_SubFPC58 = _make("SubFPC58")
_SubFPC59 = _make("SubFPC59")
_SubFPC60 = _make("SubFPC60")
_SubFPC61 = _make("SubFPC61")
_SubFPC62 = _make("SubFPC62")
_SubFPC63 = _make("SubFPC63")
_SubFPC64 = _make("SubFPC64")
_SubFPC65 = _make("SubFPC65")
_SubFPC66 = _make("SubFPC66")
_SubFPC67 = _make("SubFPC67")
_SubFPC68 = _make("SubFPC68")
_SubFPC69 = _make("SubFPC69")
_SubFPC70 = _make("SubFPC70")
_SubFPC71 = _make("SubFPC71")
_SubFPC72 = _make("SubFPC72")
_SubFPC73 = _make("SubFPC73")
_SubFPC74 = _make("SubFPC74")
_SubFPC75 = _make("SubFPC75")
_SubFPC76 = _make("SubFPC76")
_SubFPC77 = _make("SubFPC77")
_SubFPC78 = _make("SubFPC78")
_SubFPC79 = _make("SubFPC79")
_SubFPC80 = _make("SubFPC80")
_SubFPC81 = _make("SubFPC81")
_SubFPC82 = _make("SubFPC82")
_SubFPC83 = _make("SubFPC83")
_SubFPC84 = _make("SubFPC84")
_SubFPC85 = _make("SubFPC85")
_SubFPC86 = _make("SubFPC86")
_SubFPC87 = _make("SubFPC87")
_SubFPC88 = _make("SubFPC88")
_SubFPC89 = _make("SubFPC89")
_SubFPC90 = _make("SubFPC90")
_SubFPC91 = _make("SubFPC91")
_SubFPC92 = _make("SubFPC92")
_SubFPC93 = _make("SubFPC93")
_SubFPC94 = _make("SubFPC94")
_SubFPC95 = _make("SubFPC95")
_SubFPC96 = _make("SubFPC96")
_SubFPC97 = _make("SubFPC97")
_SubFPC98 = _make("SubFPC98")
_SubFPC99 = _make("SubFPC99")
_SubFPC100 = _make("SubFPC100")
_SubFPC101 = _make("SubFPC101")
_SubFPC102 = _make("SubFPC102")
_SubFPC103 = _make("SubFPC103")
_SubFPC104 = _make("SubFPC104")
_SubFPC105 = _make("SubFPC105")
_SubFPC106 = _make("SubFPC106")
_SubFPC107 = _make("SubFPC107")
_SubFPC108 = _make("SubFPC108")
_SubFPC109 = _make("SubFPC109")
_SubFPC110 = _make("SubFPC110")
_SubFPC111 = _make("SubFPC111")
_SubFPC112 = _make("SubFPC112")
_SubFPC113 = _make("SubFPC113")
_SubFPC114 = _make("SubFPC114")
_SubFPC115 = _make("SubFPC115")
_SubFPC116 = _make("SubFPC116")
_SubFPC117 = _make("SubFPC117")
_SubFPC118 = _make("SubFPC118")
_SubFPC119 = _make("SubFPC119")
_SubFPC120 = _make("SubFPC120")
_SubFPC121 = _make("SubFPC121")
_SubFPC122 = _make("SubFPC122")
_SubFPC123 = _make("SubFPC123")
_SubFPC124 = _make("SubFPC124")
_SubFPC125 = _make("SubFPC125")
_SubFPC126 = _make("SubFPC126")
_SubFPC127 = _make("SubFPC127")
_SubFPC128 = _make("SubFPC128")
_SubFPC129 = _make("SubFPC129")
_SubFPC130 = _make("SubFPC130")
_SubFPC131 = _make("SubFPC131")
_SubFPC132 = _make("SubFPC132")
_SubFPC133 = _make("SubFPC133")
_SubFPC134 = _make("SubFPC134")
_SubFPC135 = _make("SubFPC135")
_SubFPC136 = _make("SubFPC136")
_SubFPC137 = _make("SubFPC137")
_SubFPC138 = _make("SubFPC138")
_SubFPC139 = _make("SubFPC139")
_SubFPC140 = _make("SubFPC140")
_SubFPC141 = _make("SubFPC141")
_SubFPC142 = _make("SubFPC142")
_SubFPC143 = _make("SubFPC143")
_SubFPC144 = _make("SubFPC144")
_SubFPC145 = _make("SubFPC145")
_SubFPC146 = _make("SubFPC146")
_SubFPC147 = _make("SubFPC147")
_SubFPC148 = _make("SubFPC148")
_SubFPC149 = _make("SubFPC149")
_SubFPC150 = _make("SubFPC150")
_SubFPC151 = _make("SubFPC151")
_SubFPC152 = _make("SubFPC152")
_SubFPC153 = _make("SubFPC153")
_SubFPC154 = _make("SubFPC154")
_SubFPC155 = _make("SubFPC155")
_SubFPC156 = _make("SubFPC156")
_SubFPC157 = _make("SubFPC157")
_SubFPC158 = _make("SubFPC158")
_SubFPC159 = _make("SubFPC159")
_SubFPC160 = _make("SubFPC160")
_SubFPC161 = _make("SubFPC161")
_SubFPC162 = _make("SubFPC162")
_SubFPC163 = _make("SubFPC163")
_SubFPC164 = _make("SubFPC164")
_SubFPC165 = _make("SubFPC165")
_SubFPC166 = _make("SubFPC166")
_SubFPC167 = _make("SubFPC167")
_SubFPC168 = _make("SubFPC168")
_SubFPC169 = _make("SubFPC169")
_SubFPC170 = _make("SubFPC170")
_SubFPC171 = _make("SubFPC171")
_SubFPC172 = _make("SubFPC172")
_SubFPC173 = _make("SubFPC173")
_SubFPC174 = _make("SubFPC174")
_SubFPC175 = _make("SubFPC175")
_SubFPC176 = _make("SubFPC176")
_SubFPC177 = _make("SubFPC177")
_SubFPC178 = _make("SubFPC178")
_SubFPC179 = _make("SubFPC179")
_SubFPC180 = _make("SubFPC180")
_SubFPC181 = _make("SubFPC181")
_SubFPC182 = _make("SubFPC182")
_SubFPC183 = _make("SubFPC183")
_SubFPC184 = _make("SubFPC184")
_SubFPC185 = _make("SubFPC185")
_SubFPC186 = _make("SubFPC186")
_SubFPC187 = _make("SubFPC187")
_SubFPC188 = _make("SubFPC188")
_SubFPC189 = _make("SubFPC189")
_SubFPC190 = _make("SubFPC190")
_SubFPC191 = _make("SubFPC191")
_SubFPC192 = _make("SubFPC192")
_SubFPC193 = _make("SubFPC193")
_SubFPC194 = _make("SubFPC194")
_SubFPC195 = _make("SubFPC195")
_SubFPC196 = _make("SubFPC196")
_SubFPC197 = _make("SubFPC197")
_SubFPC198 = _make("SubFPC198")
_SubFPC199 = _make("SubFPC199")
_SubFPC200 = _make("SubFPC200")
_SubFPC201 = _make("SubFPC201")
_SubFPC202 = _make("SubFPC202")
_SubFPC203 = _make("SubFPC203")
_SubFPC204 = _make("SubFPC204")
_SubFPC205 = _make("SubFPC205")
_SubFPC206 = _make("SubFPC206")
_SubFPC207 = _make("SubFPC207")
_SubFPC208 = _make("SubFPC208")
_SubFPC209 = _make("SubFPC209")
_SubFPC210 = _make("SubFPC210")
_SubFPC211 = _make("SubFPC211")
_SubFPC212 = _make("SubFPC212")
_SubFPC213 = _make("SubFPC213")
_SubFPC214 = _make("SubFPC214")
_SubFPC215 = _make("SubFPC215")
_SubFPC216 = _make("SubFPC216")
_SubFPC217 = _make("SubFPC217")
_SubFPC218 = _make("SubFPC218")
_SubFPC219 = _make("SubFPC219")
_SubFPC220 = _make("SubFPC220")
_SubFPC221 = _make("SubFPC221")
_SubFPC222 = _make("SubFPC222")
_SubFPC223 = _make("SubFPC223")
_SubFPC224 = _make("SubFPC224")
_SubFPC225 = _make("SubFPC225")
_SubFPC226 = _make("SubFPC226")
_SubFPC227 = _make("SubFPC227")
_SubFPC228 = _make("SubFPC228")
_SubFPC229 = _make("SubFPC229")
_SubFPC230 = _make("SubFPC230")
_SubFPC231 = _make("SubFPC231")
_SubFPC232 = _make("SubFPC232")
_SubFPC233 = _make("SubFPC233")
_SubFPC234 = _make("SubFPC234")
_SubFPC235 = _make("SubFPC235")
_SubFPC236 = _make("SubFPC236")
_SubFPC237 = _make("SubFPC237")
_SubFPC238 = _make("SubFPC238")
_SubFPC239 = _make("SubFPC239")
_SubFPC240 = _make("SubFPC240")
_SubFPC241 = _make("SubFPC241")
_SubFPC242 = _make("SubFPC242")
_SubFPC243 = _make("SubFPC243")
_SubFPC244 = _make("SubFPC244")
_SubFPC245 = _make("SubFPC245")
_SubFPC246 = _make("SubFPC246")
_SubFPC247 = _make("SubFPC247")
_SubFPC248 = _make("SubFPC248")
_SubFPC249 = _make("SubFPC249")
_SubFPC250 = _make("SubFPC250")
_SubFPC251 = _make("SubFPC251")
_SubFPC252 = _make("SubFPC252")
_SubFPC253 = _make("SubFPC253")
_SubFPC254 = _make("SubFPC254")
_SubFPC255 = _make("SubFPC255")
_SubFPC256 = _make("SubFPC256")
_SubFPC257 = _make("SubFPC257")
_SubFPC258 = _make("SubFPC258")
_SubFPC259 = _make("SubFPC259")
_SubFPC260 = _make("SubFPC260")
_SubFPC261 = _make("SubFPC261")
_SubFPC262 = _make("SubFPC262")
_SubFPC263 = _make("SubFPC263")
_SubFPC264 = _make("SubFPC264")
_SubFPC265 = _make("SubFPC265")
_SubFPC266 = _make("SubFPC266")
_SubFPC267 = _make("SubFPC267")
_SubFPC268 = _make("SubFPC268")
_SubFPC269 = _make("SubFPC269")
_SubFPC270 = _make("SubFPC270")
_SubFPC271 = _make("SubFPC271")
_SubFPC272 = _make("SubFPC272")
_SubFPC273 = _make("SubFPC273")
_SubFPC274 = _make("SubFPC274")
_SubFPC275 = _make("SubFPC275")
_SubFPC276 = _make("SubFPC276")
_SubFPC277 = _make("SubFPC277")
_SubFPC278 = _make("SubFPC278")
_SubFPC279 = _make("SubFPC279")
_SubFPC280 = _make("SubFPC280")
_SubFPC281 = _make("SubFPC281")
_SubFPC282 = _make("SubFPC282")
_SubFPC283 = _make("SubFPC283")
_SubFPC284 = _make("SubFPC284")
_SubFPC285 = _make("SubFPC285")
_SubFPC286 = _make("SubFPC286")
_SubFPC287 = _make("SubFPC287")
_SubFPC288 = _make("SubFPC288")
_SubFPC289 = _make("SubFPC289")
_SubFPC290 = _make("SubFPC290")
_SubFPC291 = _make("SubFPC291")
_SubFPC292 = _make("SubFPC292")
_SubFPC293 = _make("SubFPC293")
_SubFPC294 = _make("SubFPC294")
_SubFPC295 = _make("SubFPC295")
_SubFPC296 = _make("SubFPC296")
_SubFPC297 = _make("SubFPC297")
_SubFPC298 = _make("SubFPC298")
_SubFPC299 = _make("SubFPC299")
_SubFPC300 = _make("SubFPC300")
_SubFPC301 = _make("SubFPC301")
_SubFPC302 = _make("SubFPC302")
_SubFPC303 = _make("SubFPC303")
_SubFPC304 = _make("SubFPC304")
_SubFPC305 = _make("SubFPC305")
_SubFPC306 = _make("SubFPC306")
_SubFPC307 = _make("SubFPC307")


# ---- public wrappers (name == function_name; declared -> 0 bonus) ----


def SubFPC1(mol):
    """Laggner substructure count: Primary_carbon."""
    return _SubFPC1(mol)


def SubFPC2(mol):
    """Laggner substructure count: Secondary_carbon."""
    return _SubFPC2(mol)


def SubFPC3(mol):
    """Laggner substructure count: Tertiary_carbon."""
    return _SubFPC3(mol)


def SubFPC4(mol):
    """Laggner substructure count: Quaternary_carbon."""
    return _SubFPC4(mol)


def SubFPC5(mol):
    """Laggner substructure count: Alkene."""
    return _SubFPC5(mol)


def SubFPC6(mol):
    """Laggner substructure count: Alkyne."""
    return _SubFPC6(mol)


def SubFPC7(mol):
    """Laggner substructure count: Allene."""
    return _SubFPC7(mol)


def SubFPC8(mol):
    """Laggner substructure count: Alkylchloride."""
    return _SubFPC8(mol)


def SubFPC9(mol):
    """Laggner substructure count: Alkylfluoride."""
    return _SubFPC9(mol)


def SubFPC10(mol):
    """Laggner substructure count: Alkylbromide."""
    return _SubFPC10(mol)


def SubFPC11(mol):
    """Laggner substructure count: Alkyliodide."""
    return _SubFPC11(mol)


def SubFPC12(mol):
    """Laggner substructure count: Alcohol."""
    return _SubFPC12(mol)


def SubFPC13(mol):
    """Laggner substructure count: Primary_alcohol."""
    return _SubFPC13(mol)


def SubFPC14(mol):
    """Laggner substructure count: Secondary_alcohol."""
    return _SubFPC14(mol)


def SubFPC15(mol):
    """Laggner substructure count: Tertiary_alcohol."""
    return _SubFPC15(mol)


def SubFPC16(mol):
    """Laggner substructure count: Dialkylether."""
    return _SubFPC16(mol)


def SubFPC17(mol):
    """Laggner substructure count: Dialkylthioether."""
    return _SubFPC17(mol)


def SubFPC18(mol):
    """Laggner substructure count: Alkylarylether."""
    return _SubFPC18(mol)


def SubFPC19(mol):
    """Laggner substructure count: Diarylether."""
    return _SubFPC19(mol)


def SubFPC20(mol):
    """Laggner substructure count: Alkylarylthioether."""
    return _SubFPC20(mol)


def SubFPC21(mol):
    """Laggner substructure count: Diarylthioether."""
    return _SubFPC21(mol)


def SubFPC22(mol):
    """Laggner substructure count: Oxonium."""
    return _SubFPC22(mol)


def SubFPC23(mol):
    """Laggner substructure count: Amine."""
    return _SubFPC23(mol)


def SubFPC24(mol):
    """Laggner substructure count: Primary_aliph_amine."""
    return _SubFPC24(mol)


def SubFPC25(mol):
    """Laggner substructure count: Secondary_aliph_amine."""
    return _SubFPC25(mol)


def SubFPC26(mol):
    """Laggner substructure count: Tertiary_aliph_amine."""
    return _SubFPC26(mol)


def SubFPC27(mol):
    """Laggner substructure count: Quaternary_aliph_ammonium."""
    return _SubFPC27(mol)


def SubFPC28(mol):
    """Laggner substructure count: Primary_arom_amine."""
    return _SubFPC28(mol)


def SubFPC29(mol):
    """Laggner substructure count: Secondary_arom_amine."""
    return _SubFPC29(mol)


def SubFPC30(mol):
    """Laggner substructure count: Tertiary_arom_amine."""
    return _SubFPC30(mol)


def SubFPC31(mol):
    """Laggner substructure count: Quaternary_arom_ammonium."""
    return _SubFPC31(mol)


def SubFPC32(mol):
    """Laggner substructure count: Secondary_mixed_amine."""
    return _SubFPC32(mol)


def SubFPC33(mol):
    """Laggner substructure count: Tertiary_mixed_amine."""
    return _SubFPC33(mol)


def SubFPC34(mol):
    """Laggner substructure count: Quaternary_mixed_ammonium."""
    return _SubFPC34(mol)


def SubFPC35(mol):
    """Laggner substructure count: Ammonium."""
    return _SubFPC35(mol)


def SubFPC36(mol):
    """Laggner substructure count: Alkylthiol."""
    return _SubFPC36(mol)


def SubFPC37(mol):
    """Laggner substructure count: Dialkylthioether."""
    return _SubFPC37(mol)


def SubFPC38(mol):
    """Laggner substructure count: Alkylarylthioether."""
    return _SubFPC38(mol)


def SubFPC39(mol):
    """Laggner substructure count: Disulfide."""
    return _SubFPC39(mol)


def SubFPC40(mol):
    """Laggner substructure count: 1,2-Aminoalcohol."""
    return _SubFPC40(mol)


def SubFPC41(mol):
    """Laggner substructure count: 1,2-Diol."""
    return _SubFPC41(mol)


def SubFPC42(mol):
    """Laggner substructure count: 1,1-Diol."""
    return _SubFPC42(mol)


def SubFPC43(mol):
    """Laggner substructure count: Hydroperoxide."""
    return _SubFPC43(mol)


def SubFPC44(mol):
    """Laggner substructure count: Peroxo."""
    return _SubFPC44(mol)


def SubFPC45(mol):
    """Laggner substructure count: Organolithium_compounds."""
    return _SubFPC45(mol)


def SubFPC46(mol):
    """Laggner substructure count: Organomagnesium_compounds."""
    return _SubFPC46(mol)


def SubFPC47(mol):
    """Laggner substructure count: Organometallic_compounds."""
    return _SubFPC47(mol)


def SubFPC48(mol):
    """Laggner substructure count: Aldehyde."""
    return _SubFPC48(mol)


def SubFPC49(mol):
    """Laggner substructure count: Ketone."""
    return _SubFPC49(mol)


def SubFPC50(mol):
    """Laggner substructure count: Thioaldehyde."""
    return _SubFPC50(mol)


def SubFPC51(mol):
    """Laggner substructure count: Thioketone."""
    return _SubFPC51(mol)


def SubFPC52(mol):
    """Laggner substructure count: Imine."""
    return _SubFPC52(mol)


def SubFPC53(mol):
    """Laggner substructure count: Immonium."""
    return _SubFPC53(mol)


def SubFPC54(mol):
    """Laggner substructure count: Oxime."""
    return _SubFPC54(mol)


def SubFPC55(mol):
    """Laggner substructure count: Oximether."""
    return _SubFPC55(mol)


def SubFPC56(mol):
    """Laggner substructure count: Acetal."""
    return _SubFPC56(mol)


def SubFPC57(mol):
    """Laggner substructure count: Hemiacetal."""
    return _SubFPC57(mol)


def SubFPC58(mol):
    """Laggner substructure count: Aminal."""
    return _SubFPC58(mol)


def SubFPC59(mol):
    """Laggner substructure count: Hemiaminal."""
    return _SubFPC59(mol)


def SubFPC60(mol):
    """Laggner substructure count: Thioacetal."""
    return _SubFPC60(mol)


def SubFPC61(mol):
    """Laggner substructure count: Thiohemiacetal."""
    return _SubFPC61(mol)


def SubFPC62(mol):
    """Laggner substructure count: Halogen_acetal_like."""
    return _SubFPC62(mol)


def SubFPC63(mol):
    """Laggner substructure count: Acetal_like."""
    return _SubFPC63(mol)


def SubFPC64(mol):
    """Laggner substructure count: Halogenmethylen_ester_and_similar."""
    return _SubFPC64(mol)


def SubFPC65(mol):
    """Laggner substructure count: NOS_methylen_ester_and_similar."""
    return _SubFPC65(mol)


def SubFPC66(mol):
    """Laggner substructure count: Hetero_methylen_ester_and_similar."""
    return _SubFPC66(mol)


def SubFPC67(mol):
    """Laggner substructure count: Cyanhydrine."""
    return _SubFPC67(mol)


def SubFPC68(mol):
    """Laggner substructure count: Chloroalkene."""
    return _SubFPC68(mol)


def SubFPC69(mol):
    """Laggner substructure count: Fluoroalkene."""
    return _SubFPC69(mol)


def SubFPC70(mol):
    """Laggner substructure count: Bromoalkene."""
    return _SubFPC70(mol)


def SubFPC71(mol):
    """Laggner substructure count: Iodoalkene."""
    return _SubFPC71(mol)


def SubFPC72(mol):
    """Laggner substructure count: Enol."""
    return _SubFPC72(mol)


def SubFPC73(mol):
    """Laggner substructure count: Endiol."""
    return _SubFPC73(mol)


def SubFPC74(mol):
    """Laggner substructure count: Enolether."""
    return _SubFPC74(mol)


def SubFPC75(mol):
    """Laggner substructure count: Enolester."""
    return _SubFPC75(mol)


def SubFPC76(mol):
    """Laggner substructure count: Enamine."""
    return _SubFPC76(mol)


def SubFPC77(mol):
    """Laggner substructure count: Thioenol."""
    return _SubFPC77(mol)


def SubFPC78(mol):
    """Laggner substructure count: Thioenolether."""
    return _SubFPC78(mol)


def SubFPC79(mol):
    """Laggner substructure count: Acylchloride."""
    return _SubFPC79(mol)


def SubFPC80(mol):
    """Laggner substructure count: Acylfluoride."""
    return _SubFPC80(mol)


def SubFPC81(mol):
    """Laggner substructure count: Acylbromide."""
    return _SubFPC81(mol)


def SubFPC82(mol):
    """Laggner substructure count: Acyliodide."""
    return _SubFPC82(mol)


def SubFPC83(mol):
    """Laggner substructure count: Acylhalide."""
    return _SubFPC83(mol)


def SubFPC84(mol):
    """Laggner substructure count: Carboxylic_acid."""
    return _SubFPC84(mol)


def SubFPC85(mol):
    """Laggner substructure count: Carboxylic_ester."""
    return _SubFPC85(mol)


def SubFPC86(mol):
    """Laggner substructure count: Lactone."""
    return _SubFPC86(mol)


def SubFPC87(mol):
    """Laggner substructure count: Carboxylic_anhydride."""
    return _SubFPC87(mol)


def SubFPC88(mol):
    """Laggner substructure count: Carboxylic_acid_derivative."""
    return _SubFPC88(mol)


def SubFPC89(mol):
    """Laggner substructure count: Carbothioic_acid."""
    return _SubFPC89(mol)


def SubFPC90(mol):
    """Laggner substructure count: Carbothioic_S_ester."""
    return _SubFPC90(mol)


def SubFPC91(mol):
    """Laggner substructure count: Carbothioic_S_lactone."""
    return _SubFPC91(mol)


def SubFPC92(mol):
    """Laggner substructure count: Carbothioic_O_ester."""
    return _SubFPC92(mol)


def SubFPC93(mol):
    """Laggner substructure count: Carbothioic_O_lactone."""
    return _SubFPC93(mol)


def SubFPC94(mol):
    """Laggner substructure count: Carbothioic_halide."""
    return _SubFPC94(mol)


def SubFPC95(mol):
    """Laggner substructure count: Carbodithioic_acid."""
    return _SubFPC95(mol)


def SubFPC96(mol):
    """Laggner substructure count: Carbodithioic_ester."""
    return _SubFPC96(mol)


def SubFPC97(mol):
    """Laggner substructure count: Carbodithiolactone."""
    return _SubFPC97(mol)


def SubFPC98(mol):
    """Laggner substructure count: Amide."""
    return _SubFPC98(mol)


def SubFPC99(mol):
    """Laggner substructure count: Primary_amide."""
    return _SubFPC99(mol)


def SubFPC100(mol):
    """Laggner substructure count: Secondary_amide."""
    return _SubFPC100(mol)


def SubFPC101(mol):
    """Laggner substructure count: Tertiary_amide."""
    return _SubFPC101(mol)


def SubFPC102(mol):
    """Laggner substructure count: Lactam."""
    return _SubFPC102(mol)


def SubFPC103(mol):
    """Laggner substructure count: Alkyl_imide."""
    return _SubFPC103(mol)


def SubFPC104(mol):
    """Laggner substructure count: N_hetero_imide."""
    return _SubFPC104(mol)


def SubFPC105(mol):
    """Laggner substructure count: Imide_acidic."""
    return _SubFPC105(mol)


def SubFPC106(mol):
    """Laggner substructure count: Thioamide."""
    return _SubFPC106(mol)


def SubFPC107(mol):
    """Laggner substructure count: Thiolactam."""
    return _SubFPC107(mol)


def SubFPC108(mol):
    """Laggner substructure count: Oximester."""
    return _SubFPC108(mol)


def SubFPC109(mol):
    """Laggner substructure count: Amidine."""
    return _SubFPC109(mol)


def SubFPC110(mol):
    """Laggner substructure count: Hydroxamic_acid."""
    return _SubFPC110(mol)


def SubFPC111(mol):
    """Laggner substructure count: Hydroxamic_acid_ester."""
    return _SubFPC111(mol)


def SubFPC112(mol):
    """Laggner substructure count: Imidoacid."""
    return _SubFPC112(mol)


def SubFPC113(mol):
    """Laggner substructure count: Imidoacid_cyclic."""
    return _SubFPC113(mol)


def SubFPC114(mol):
    """Laggner substructure count: Imidoester."""
    return _SubFPC114(mol)


def SubFPC115(mol):
    """Laggner substructure count: Imidolactone."""
    return _SubFPC115(mol)


def SubFPC116(mol):
    """Laggner substructure count: Imidothioacid."""
    return _SubFPC116(mol)


def SubFPC117(mol):
    """Laggner substructure count: Imidothioacid_cyclic."""
    return _SubFPC117(mol)


def SubFPC118(mol):
    """Laggner substructure count: Imidothioester."""
    return _SubFPC118(mol)


def SubFPC119(mol):
    """Laggner substructure count: Imidothiolactone."""
    return _SubFPC119(mol)


def SubFPC120(mol):
    """Laggner substructure count: Amidine."""
    return _SubFPC120(mol)


def SubFPC121(mol):
    """Laggner substructure count: Imidolactam."""
    return _SubFPC121(mol)


def SubFPC122(mol):
    """Laggner substructure count: Imidoylhalide."""
    return _SubFPC122(mol)


def SubFPC123(mol):
    """Laggner substructure count: Imidoylhalide_cyclic."""
    return _SubFPC123(mol)


def SubFPC124(mol):
    """Laggner substructure count: Amidrazone."""
    return _SubFPC124(mol)


def SubFPC125(mol):
    """Laggner substructure count: Alpha_aminoacid."""
    return _SubFPC125(mol)


def SubFPC126(mol):
    """Laggner substructure count: Alpha_hydroxyacid."""
    return _SubFPC126(mol)


def SubFPC127(mol):
    """Laggner substructure count: Peptide_middle."""
    return _SubFPC127(mol)


def SubFPC128(mol):
    """Laggner substructure count: Peptide_C_term."""
    return _SubFPC128(mol)


def SubFPC129(mol):
    """Laggner substructure count: Peptide_N_term."""
    return _SubFPC129(mol)


def SubFPC130(mol):
    """Laggner substructure count: Carboxylic_orthoester."""
    return _SubFPC130(mol)


def SubFPC131(mol):
    """Laggner substructure count: Ketene."""
    return _SubFPC131(mol)


def SubFPC132(mol):
    """Laggner substructure count: Ketenacetal."""
    return _SubFPC132(mol)


def SubFPC133(mol):
    """Laggner substructure count: Nitrile."""
    return _SubFPC133(mol)


def SubFPC134(mol):
    """Laggner substructure count: Isonitrile."""
    return _SubFPC134(mol)


def SubFPC135(mol):
    """Laggner substructure count: Vinylogous_carbonyl_or_carboxyl_derivative."""
    return _SubFPC135(mol)


def SubFPC136(mol):
    """Laggner substructure count: Vinylogous_acid."""
    return _SubFPC136(mol)


def SubFPC137(mol):
    """Laggner substructure count: Vinylogous_ester."""
    return _SubFPC137(mol)


def SubFPC138(mol):
    """Laggner substructure count: Vinylogous_amide."""
    return _SubFPC138(mol)


def SubFPC139(mol):
    """Laggner substructure count: Vinylogous_halide."""
    return _SubFPC139(mol)


def SubFPC140(mol):
    """Laggner substructure count: Carbonic_acid_dieester."""
    return _SubFPC140(mol)


def SubFPC141(mol):
    """Laggner substructure count: Carbonic_acid_esterhalide."""
    return _SubFPC141(mol)


def SubFPC142(mol):
    """Laggner substructure count: Carbonic_acid_monoester."""
    return _SubFPC142(mol)


def SubFPC143(mol):
    """Laggner substructure count: Carbonic_acid_derivatives."""
    return _SubFPC143(mol)


def SubFPC144(mol):
    """Laggner substructure count: Thiocarbonic_acid_dieester."""
    return _SubFPC144(mol)


def SubFPC145(mol):
    """Laggner substructure count: Thiocarbonic_acid_esterhalide."""
    return _SubFPC145(mol)


def SubFPC146(mol):
    """Laggner substructure count: Thiocarbonic_acid_monoester."""
    return _SubFPC146(mol)


def SubFPC147(mol):
    """Laggner substructure count: Urea."""
    return _SubFPC147(mol)


def SubFPC148(mol):
    """Laggner substructure count: Thiourea."""
    return _SubFPC148(mol)


def SubFPC149(mol):
    """Laggner substructure count: Isourea."""
    return _SubFPC149(mol)


def SubFPC150(mol):
    """Laggner substructure count: Isothiourea."""
    return _SubFPC150(mol)


def SubFPC151(mol):
    """Laggner substructure count: Guanidine."""
    return _SubFPC151(mol)


def SubFPC152(mol):
    """Laggner substructure count: Carbaminic_acid."""
    return _SubFPC152(mol)


def SubFPC153(mol):
    """Laggner substructure count: Urethan."""
    return _SubFPC153(mol)


def SubFPC154(mol):
    """Laggner substructure count: Biuret."""
    return _SubFPC154(mol)


def SubFPC155(mol):
    """Laggner substructure count: Semicarbazide."""
    return _SubFPC155(mol)


def SubFPC156(mol):
    """Laggner substructure count: Carbazide."""
    return _SubFPC156(mol)


def SubFPC157(mol):
    """Laggner substructure count: Semicarbazone."""
    return _SubFPC157(mol)


def SubFPC158(mol):
    """Laggner substructure count: Carbazone."""
    return _SubFPC158(mol)


def SubFPC159(mol):
    """Laggner substructure count: Thiosemicarbazide."""
    return _SubFPC159(mol)


def SubFPC160(mol):
    """Laggner substructure count: Thiocarbazide."""
    return _SubFPC160(mol)


def SubFPC161(mol):
    """Laggner substructure count: Thiosemicarbazone."""
    return _SubFPC161(mol)


def SubFPC162(mol):
    """Laggner substructure count: Thiocarbazone."""
    return _SubFPC162(mol)


def SubFPC163(mol):
    """Laggner substructure count: Isocyanate."""
    return _SubFPC163(mol)


def SubFPC164(mol):
    """Laggner substructure count: Cyanate."""
    return _SubFPC164(mol)


def SubFPC165(mol):
    """Laggner substructure count: Isothiocyanate."""
    return _SubFPC165(mol)


def SubFPC166(mol):
    """Laggner substructure count: Thiocyanate."""
    return _SubFPC166(mol)


def SubFPC167(mol):
    """Laggner substructure count: Carbodiimide."""
    return _SubFPC167(mol)


def SubFPC168(mol):
    """Laggner substructure count: Orthocarbonic_derivatives."""
    return _SubFPC168(mol)


def SubFPC169(mol):
    """Laggner substructure count: Phenol."""
    return _SubFPC169(mol)


def SubFPC170(mol):
    """Laggner substructure count: 1,2-Diphenol."""
    return _SubFPC170(mol)


def SubFPC171(mol):
    """Laggner substructure count: Arylchloride."""
    return _SubFPC171(mol)


def SubFPC172(mol):
    """Laggner substructure count: Arylfluoride."""
    return _SubFPC172(mol)


def SubFPC173(mol):
    """Laggner substructure count: Arylbromide."""
    return _SubFPC173(mol)


def SubFPC174(mol):
    """Laggner substructure count: Aryliodide."""
    return _SubFPC174(mol)


def SubFPC175(mol):
    """Laggner substructure count: Arylthiol."""
    return _SubFPC175(mol)


def SubFPC176(mol):
    """Laggner substructure count: Iminoarene."""
    return _SubFPC176(mol)


def SubFPC177(mol):
    """Laggner substructure count: Oxoarene."""
    return _SubFPC177(mol)


def SubFPC178(mol):
    """Laggner substructure count: Thioarene."""
    return _SubFPC178(mol)


def SubFPC179(mol):
    """Laggner substructure count: Hetero_N_basic_H."""
    return _SubFPC179(mol)


def SubFPC180(mol):
    """Laggner substructure count: Hetero_N_basic_no_H."""
    return _SubFPC180(mol)


def SubFPC181(mol):
    """Laggner substructure count: Hetero_N_nonbasic."""
    return _SubFPC181(mol)


def SubFPC182(mol):
    """Laggner substructure count: Hetero_O."""
    return _SubFPC182(mol)


def SubFPC183(mol):
    """Laggner substructure count: Hetero_S."""
    return _SubFPC183(mol)


def SubFPC184(mol):
    """Laggner substructure count: Heteroaromatic."""
    return _SubFPC184(mol)


def SubFPC185(mol):
    """Laggner substructure count: Nitrite."""
    return _SubFPC185(mol)


def SubFPC186(mol):
    """Laggner substructure count: Thionitrite."""
    return _SubFPC186(mol)


def SubFPC187(mol):
    """Laggner substructure count: Nitrate."""
    return _SubFPC187(mol)


def SubFPC188(mol):
    """Laggner substructure count: Nitro."""
    return _SubFPC188(mol)


def SubFPC189(mol):
    """Laggner substructure count: Nitroso."""
    return _SubFPC189(mol)


def SubFPC190(mol):
    """Laggner substructure count: Azide."""
    return _SubFPC190(mol)


def SubFPC191(mol):
    """Laggner substructure count: Acylazide."""
    return _SubFPC191(mol)


def SubFPC192(mol):
    """Laggner substructure count: Diazo."""
    return _SubFPC192(mol)


def SubFPC193(mol):
    """Laggner substructure count: Diazonium."""
    return _SubFPC193(mol)


def SubFPC194(mol):
    """Laggner substructure count: Nitrosamine."""
    return _SubFPC194(mol)


def SubFPC195(mol):
    """Laggner substructure count: Nitrosamide."""
    return _SubFPC195(mol)


def SubFPC196(mol):
    """Laggner substructure count: N-Oxide."""
    return _SubFPC196(mol)


def SubFPC197(mol):
    """Laggner substructure count: Hydrazine."""
    return _SubFPC197(mol)


def SubFPC198(mol):
    """Laggner substructure count: Hydrazone."""
    return _SubFPC198(mol)


def SubFPC199(mol):
    """Laggner substructure count: Hydroxylamine."""
    return _SubFPC199(mol)


def SubFPC200(mol):
    """Laggner substructure count: Sulfon."""
    return _SubFPC200(mol)


def SubFPC201(mol):
    """Laggner substructure count: Sulfoxide."""
    return _SubFPC201(mol)


def SubFPC202(mol):
    """Laggner substructure count: Sulfonium."""
    return _SubFPC202(mol)


def SubFPC203(mol):
    """Laggner substructure count: Sulfuric_acid."""
    return _SubFPC203(mol)


def SubFPC204(mol):
    """Laggner substructure count: Sulfuric_monoester."""
    return _SubFPC204(mol)


def SubFPC205(mol):
    """Laggner substructure count: Sulfuric_diester."""
    return _SubFPC205(mol)


def SubFPC206(mol):
    """Laggner substructure count: Sulfuric_monoamide."""
    return _SubFPC206(mol)


def SubFPC207(mol):
    """Laggner substructure count: Sulfuric_diamide."""
    return _SubFPC207(mol)


def SubFPC208(mol):
    """Laggner substructure count: Sulfuric_esteramide."""
    return _SubFPC208(mol)


def SubFPC209(mol):
    """Laggner substructure count: Sulfuric_derivative."""
    return _SubFPC209(mol)


def SubFPC210(mol):
    """Laggner substructure count: Sulfonic_acid."""
    return _SubFPC210(mol)


def SubFPC211(mol):
    """Laggner substructure count: Sulfonamide."""
    return _SubFPC211(mol)


def SubFPC212(mol):
    """Laggner substructure count: Sulfonic_ester."""
    return _SubFPC212(mol)


def SubFPC213(mol):
    """Laggner substructure count: Sulfonic_halide."""
    return _SubFPC213(mol)


def SubFPC214(mol):
    """Laggner substructure count: Sulfonic_derivative."""
    return _SubFPC214(mol)


def SubFPC215(mol):
    """Laggner substructure count: Sulfinic_acid."""
    return _SubFPC215(mol)


def SubFPC216(mol):
    """Laggner substructure count: Sulfinic_amide."""
    return _SubFPC216(mol)


def SubFPC217(mol):
    """Laggner substructure count: Sulfinic_ester."""
    return _SubFPC217(mol)


def SubFPC218(mol):
    """Laggner substructure count: Sulfinic_halide."""
    return _SubFPC218(mol)


def SubFPC219(mol):
    """Laggner substructure count: Sulfinic_derivative."""
    return _SubFPC219(mol)


def SubFPC220(mol):
    """Laggner substructure count: Sulfenic_acid."""
    return _SubFPC220(mol)


def SubFPC221(mol):
    """Laggner substructure count: Sulfenic_amide."""
    return _SubFPC221(mol)


def SubFPC222(mol):
    """Laggner substructure count: Sulfenic_ester."""
    return _SubFPC222(mol)


def SubFPC223(mol):
    """Laggner substructure count: Sulfenic_halide."""
    return _SubFPC223(mol)


def SubFPC224(mol):
    """Laggner substructure count: Sulfenic_derivative."""
    return _SubFPC224(mol)


def SubFPC225(mol):
    """Laggner substructure count: Phosphine."""
    return _SubFPC225(mol)


def SubFPC226(mol):
    """Laggner substructure count: Phosphine_oxide."""
    return _SubFPC226(mol)


def SubFPC227(mol):
    """Laggner substructure count: Phosphonium."""
    return _SubFPC227(mol)


def SubFPC228(mol):
    """Laggner substructure count: Phosphorylen."""
    return _SubFPC228(mol)


def SubFPC229(mol):
    """Laggner substructure count: Phosphonic_acid."""
    return _SubFPC229(mol)


def SubFPC230(mol):
    """Laggner substructure count: Phosphonic_monoester."""
    return _SubFPC230(mol)


def SubFPC231(mol):
    """Laggner substructure count: Phosphonic_diester."""
    return _SubFPC231(mol)


def SubFPC232(mol):
    """Laggner substructure count: Phosphonic_monoamide."""
    return _SubFPC232(mol)


def SubFPC233(mol):
    """Laggner substructure count: Phosphonic_diamide."""
    return _SubFPC233(mol)


def SubFPC234(mol):
    """Laggner substructure count: Phosphonic_esteramide."""
    return _SubFPC234(mol)


def SubFPC235(mol):
    """Laggner substructure count: Phosphonic_acid_derivative."""
    return _SubFPC235(mol)


def SubFPC236(mol):
    """Laggner substructure count: Phosphoric_acid."""
    return _SubFPC236(mol)


def SubFPC237(mol):
    """Laggner substructure count: Phosphoric_monoester."""
    return _SubFPC237(mol)


def SubFPC238(mol):
    """Laggner substructure count: Phosphoric_diester."""
    return _SubFPC238(mol)


def SubFPC239(mol):
    """Laggner substructure count: Phosphoric_triester."""
    return _SubFPC239(mol)


def SubFPC240(mol):
    """Laggner substructure count: Phosphoric_monoamide."""
    return _SubFPC240(mol)


def SubFPC241(mol):
    """Laggner substructure count: Phosphoric_diamide."""
    return _SubFPC241(mol)


def SubFPC242(mol):
    """Laggner substructure count: Phosphoric_triamide."""
    return _SubFPC242(mol)


def SubFPC243(mol):
    """Laggner substructure count: Phosphoric_monoestermonoamide."""
    return _SubFPC243(mol)


def SubFPC244(mol):
    """Laggner substructure count: Phosphoric_diestermonoamide."""
    return _SubFPC244(mol)


def SubFPC245(mol):
    """Laggner substructure count: Phosphoric_monoesterdiamide."""
    return _SubFPC245(mol)


def SubFPC246(mol):
    """Laggner substructure count: Phosphoric_acid_derivative."""
    return _SubFPC246(mol)


def SubFPC247(mol):
    """Laggner substructure count: Phosphinic_acid."""
    return _SubFPC247(mol)


def SubFPC248(mol):
    """Laggner substructure count: Phosphinic_ester."""
    return _SubFPC248(mol)


def SubFPC249(mol):
    """Laggner substructure count: Phosphinic_amide."""
    return _SubFPC249(mol)


def SubFPC250(mol):
    """Laggner substructure count: Phosphinic_acid_derivative."""
    return _SubFPC250(mol)


def SubFPC251(mol):
    """Laggner substructure count: Phosphonous_acid."""
    return _SubFPC251(mol)


def SubFPC252(mol):
    """Laggner substructure count: Phosphonous_monoester."""
    return _SubFPC252(mol)


def SubFPC253(mol):
    """Laggner substructure count: Phosphonous_diester."""
    return _SubFPC253(mol)


def SubFPC254(mol):
    """Laggner substructure count: Phosphonous_monoamide."""
    return _SubFPC254(mol)


def SubFPC255(mol):
    """Laggner substructure count: Phosphonous_diamide."""
    return _SubFPC255(mol)


def SubFPC256(mol):
    """Laggner substructure count: Phosphonous_esteramide."""
    return _SubFPC256(mol)


def SubFPC257(mol):
    """Laggner substructure count: Phosphonous_derivatives."""
    return _SubFPC257(mol)


def SubFPC258(mol):
    """Laggner substructure count: Phosphinous_acid."""
    return _SubFPC258(mol)


def SubFPC259(mol):
    """Laggner substructure count: Phosphinous_ester."""
    return _SubFPC259(mol)


def SubFPC260(mol):
    """Laggner substructure count: Phosphinous_amide."""
    return _SubFPC260(mol)


def SubFPC261(mol):
    """Laggner substructure count: Phosphinous_derivatives."""
    return _SubFPC261(mol)


def SubFPC262(mol):
    """Laggner substructure count: Quart_silane."""
    return _SubFPC262(mol)


def SubFPC263(mol):
    """Laggner substructure count: Non-quart_silane."""
    return _SubFPC263(mol)


def SubFPC264(mol):
    """Laggner substructure count: Silylmonohalide."""
    return _SubFPC264(mol)


def SubFPC265(mol):
    """Laggner substructure count: Het_trialkylsilane."""
    return _SubFPC265(mol)


def SubFPC266(mol):
    """Laggner substructure count: Dihet_dialkylsilane."""
    return _SubFPC266(mol)


def SubFPC267(mol):
    """Laggner substructure count: Trihet_alkylsilane."""
    return _SubFPC267(mol)


def SubFPC268(mol):
    """Laggner substructure count: Silicic_acid_derivative."""
    return _SubFPC268(mol)


def SubFPC269(mol):
    """Laggner substructure count: Trialkylborane."""
    return _SubFPC269(mol)


def SubFPC270(mol):
    """Laggner substructure count: Boric_acid_derivatives."""
    return _SubFPC270(mol)


def SubFPC271(mol):
    """Laggner substructure count: Boronic_acid_derivative."""
    return _SubFPC271(mol)


def SubFPC272(mol):
    """Laggner substructure count: Borohydride."""
    return _SubFPC272(mol)


def SubFPC273(mol):
    """Laggner substructure count: Quaternary_boron."""
    return _SubFPC273(mol)


def SubFPC274(mol):
    """Laggner substructure count: Aromatic."""
    return _SubFPC274(mol)


def SubFPC275(mol):
    """Laggner substructure count: Heterocyclic."""
    return _SubFPC275(mol)


def SubFPC276(mol):
    """Laggner substructure count: Epoxide."""
    return _SubFPC276(mol)


def SubFPC277(mol):
    """Laggner substructure count: NH_aziridine."""
    return _SubFPC277(mol)


def SubFPC278(mol):
    """Laggner substructure count: Spiro."""
    return _SubFPC278(mol)


def SubFPC279(mol):
    """Laggner substructure count: Annelated_rings."""
    return _SubFPC279(mol)


def SubFPC280(mol):
    """Laggner substructure count: Bridged_rings."""
    return _SubFPC280(mol)


def SubFPC281(mol):
    """Laggner substructure count: Sugar_pattern_1."""
    return _SubFPC281(mol)


def SubFPC282(mol):
    """Laggner substructure count: Sugar_pattern_2."""
    return _SubFPC282(mol)


def SubFPC283(mol):
    """Laggner substructure count: Sugar_pattern_combi."""
    return _SubFPC283(mol)


def SubFPC284(mol):
    """Laggner substructure count: Sugar_pattern_2_reducing."""
    return _SubFPC284(mol)


def SubFPC285(mol):
    """Laggner substructure count: Sugar_pattern_2_alpha."""
    return _SubFPC285(mol)


def SubFPC286(mol):
    """Laggner substructure count: Sugar_pattern_2_beta."""
    return _SubFPC286(mol)


def SubFPC287(mol):
    """Laggner substructure count: Conjugated_double_bond."""
    return _SubFPC287(mol)


def SubFPC288(mol):
    """Laggner substructure count: Conjugated_tripple_bond."""
    return _SubFPC288(mol)


def SubFPC289(mol):
    """Laggner substructure count: Cis_double_bond."""
    return _SubFPC289(mol)


def SubFPC290(mol):
    """Laggner substructure count: Trans_double_bond."""
    return _SubFPC290(mol)


def SubFPC291(mol):
    """Laggner substructure count: Mixed_anhydrides."""
    return _SubFPC291(mol)


def SubFPC292(mol):
    """Laggner substructure count: Halogen_on_hetero."""
    return _SubFPC292(mol)


def SubFPC293(mol):
    """Laggner substructure count: Halogen_multi_subst."""
    return _SubFPC293(mol)


def SubFPC294(mol):
    """Laggner substructure count: Trifluoromethyl."""
    return _SubFPC294(mol)


def SubFPC295(mol):
    """Laggner substructure count: C_ONS_bond."""
    return _SubFPC295(mol)


def SubFPC296(mol):
    """Laggner substructure count: Charged."""
    return _SubFPC296(mol)


def SubFPC297(mol):
    """Laggner substructure count: Anion."""
    return _SubFPC297(mol)


def SubFPC298(mol):
    """Laggner substructure count: Kation."""
    return _SubFPC298(mol)


def SubFPC299(mol):
    """Laggner substructure count: Salt."""
    return _SubFPC299(mol)


def SubFPC300(mol):
    """Laggner substructure count: 1,3-Tautomerizable."""
    return _SubFPC300(mol)


def SubFPC301(mol):
    """Laggner substructure count: 1,5-Tautomerizable."""
    return _SubFPC301(mol)


def SubFPC302(mol):
    """Laggner substructure count: Rotatable_bond."""
    return _SubFPC302(mol)


def SubFPC303(mol):
    """Laggner substructure count: Michael_acceptor."""
    return _SubFPC303(mol)


def SubFPC304(mol):
    """Laggner substructure count: Dicarbodiazene."""
    return _SubFPC304(mol)


def SubFPC305(mol):
    """Laggner substructure count: CH-acidic."""
    return _SubFPC305(mol)


def SubFPC306(mol):
    """Laggner substructure count: CH-acidic_strong."""
    return _SubFPC306(mol)


def SubFPC307(mol):
    """Laggner substructure count: Chiral_center_specified."""
    return _SubFPC307(mol)


ALL_DESCRIPTOR_NAMES = (
    "SubFPC1",
    "SubFPC2",
    "SubFPC3",
    "SubFPC4",
    "SubFPC5",
    "SubFPC6",
    "SubFPC7",
    "SubFPC8",
    "SubFPC9",
    "SubFPC10",
    "SubFPC11",
    "SubFPC12",
    "SubFPC13",
    "SubFPC14",
    "SubFPC15",
    "SubFPC16",
    "SubFPC17",
    "SubFPC18",
    "SubFPC19",
    "SubFPC20",
    "SubFPC21",
    "SubFPC22",
    "SubFPC23",
    "SubFPC24",
    "SubFPC25",
    "SubFPC26",
    "SubFPC27",
    "SubFPC28",
    "SubFPC29",
    "SubFPC30",
    "SubFPC31",
    "SubFPC32",
    "SubFPC33",
    "SubFPC34",
    "SubFPC35",
    "SubFPC36",
    "SubFPC37",
    "SubFPC38",
    "SubFPC39",
    "SubFPC40",
    "SubFPC41",
    "SubFPC42",
    "SubFPC43",
    "SubFPC44",
    "SubFPC45",
    "SubFPC46",
    "SubFPC47",
    "SubFPC48",
    "SubFPC49",
    "SubFPC50",
    "SubFPC51",
    "SubFPC52",
    "SubFPC53",
    "SubFPC54",
    "SubFPC55",
    "SubFPC56",
    "SubFPC57",
    "SubFPC58",
    "SubFPC59",
    "SubFPC60",
    "SubFPC61",
    "SubFPC62",
    "SubFPC63",
    "SubFPC64",
    "SubFPC65",
    "SubFPC66",
    "SubFPC67",
    "SubFPC68",
    "SubFPC69",
    "SubFPC70",
    "SubFPC71",
    "SubFPC72",
    "SubFPC73",
    "SubFPC74",
    "SubFPC75",
    "SubFPC76",
    "SubFPC77",
    "SubFPC78",
    "SubFPC79",
    "SubFPC80",
    "SubFPC81",
    "SubFPC82",
    "SubFPC83",
    "SubFPC84",
    "SubFPC85",
    "SubFPC86",
    "SubFPC87",
    "SubFPC88",
    "SubFPC89",
    "SubFPC90",
    "SubFPC91",
    "SubFPC92",
    "SubFPC93",
    "SubFPC94",
    "SubFPC95",
    "SubFPC96",
    "SubFPC97",
    "SubFPC98",
    "SubFPC99",
    "SubFPC100",
    "SubFPC101",
    "SubFPC102",
    "SubFPC103",
    "SubFPC104",
    "SubFPC105",
    "SubFPC106",
    "SubFPC107",
    "SubFPC108",
    "SubFPC109",
    "SubFPC110",
    "SubFPC111",
    "SubFPC112",
    "SubFPC113",
    "SubFPC114",
    "SubFPC115",
    "SubFPC116",
    "SubFPC117",
    "SubFPC118",
    "SubFPC119",
    "SubFPC120",
    "SubFPC121",
    "SubFPC122",
    "SubFPC123",
    "SubFPC124",
    "SubFPC125",
    "SubFPC126",
    "SubFPC127",
    "SubFPC128",
    "SubFPC129",
    "SubFPC130",
    "SubFPC131",
    "SubFPC132",
    "SubFPC133",
    "SubFPC134",
    "SubFPC135",
    "SubFPC136",
    "SubFPC137",
    "SubFPC138",
    "SubFPC139",
    "SubFPC140",
    "SubFPC141",
    "SubFPC142",
    "SubFPC143",
    "SubFPC144",
    "SubFPC145",
    "SubFPC146",
    "SubFPC147",
    "SubFPC148",
    "SubFPC149",
    "SubFPC150",
    "SubFPC151",
    "SubFPC152",
    "SubFPC153",
    "SubFPC154",
    "SubFPC155",
    "SubFPC156",
    "SubFPC157",
    "SubFPC158",
    "SubFPC159",
    "SubFPC160",
    "SubFPC161",
    "SubFPC162",
    "SubFPC163",
    "SubFPC164",
    "SubFPC165",
    "SubFPC166",
    "SubFPC167",
    "SubFPC168",
    "SubFPC169",
    "SubFPC170",
    "SubFPC171",
    "SubFPC172",
    "SubFPC173",
    "SubFPC174",
    "SubFPC175",
    "SubFPC176",
    "SubFPC177",
    "SubFPC178",
    "SubFPC179",
    "SubFPC180",
    "SubFPC181",
    "SubFPC182",
    "SubFPC183",
    "SubFPC184",
    "SubFPC185",
    "SubFPC186",
    "SubFPC187",
    "SubFPC188",
    "SubFPC189",
    "SubFPC190",
    "SubFPC191",
    "SubFPC192",
    "SubFPC193",
    "SubFPC194",
    "SubFPC195",
    "SubFPC196",
    "SubFPC197",
    "SubFPC198",
    "SubFPC199",
    "SubFPC200",
    "SubFPC201",
    "SubFPC202",
    "SubFPC203",
    "SubFPC204",
    "SubFPC205",
    "SubFPC206",
    "SubFPC207",
    "SubFPC208",
    "SubFPC209",
    "SubFPC210",
    "SubFPC211",
    "SubFPC212",
    "SubFPC213",
    "SubFPC214",
    "SubFPC215",
    "SubFPC216",
    "SubFPC217",
    "SubFPC218",
    "SubFPC219",
    "SubFPC220",
    "SubFPC221",
    "SubFPC222",
    "SubFPC223",
    "SubFPC224",
    "SubFPC225",
    "SubFPC226",
    "SubFPC227",
    "SubFPC228",
    "SubFPC229",
    "SubFPC230",
    "SubFPC231",
    "SubFPC232",
    "SubFPC233",
    "SubFPC234",
    "SubFPC235",
    "SubFPC236",
    "SubFPC237",
    "SubFPC238",
    "SubFPC239",
    "SubFPC240",
    "SubFPC241",
    "SubFPC242",
    "SubFPC243",
    "SubFPC244",
    "SubFPC245",
    "SubFPC246",
    "SubFPC247",
    "SubFPC248",
    "SubFPC249",
    "SubFPC250",
    "SubFPC251",
    "SubFPC252",
    "SubFPC253",
    "SubFPC254",
    "SubFPC255",
    "SubFPC256",
    "SubFPC257",
    "SubFPC258",
    "SubFPC259",
    "SubFPC260",
    "SubFPC261",
    "SubFPC262",
    "SubFPC263",
    "SubFPC264",
    "SubFPC265",
    "SubFPC266",
    "SubFPC267",
    "SubFPC268",
    "SubFPC269",
    "SubFPC270",
    "SubFPC271",
    "SubFPC272",
    "SubFPC273",
    "SubFPC274",
    "SubFPC275",
    "SubFPC276",
    "SubFPC277",
    "SubFPC278",
    "SubFPC279",
    "SubFPC280",
    "SubFPC281",
    "SubFPC282",
    "SubFPC283",
    "SubFPC284",
    "SubFPC285",
    "SubFPC286",
    "SubFPC287",
    "SubFPC288",
    "SubFPC289",
    "SubFPC290",
    "SubFPC291",
    "SubFPC292",
    "SubFPC293",
    "SubFPC294",
    "SubFPC295",
    "SubFPC296",
    "SubFPC297",
    "SubFPC298",
    "SubFPC299",
    "SubFPC300",
    "SubFPC301",
    "SubFPC302",
    "SubFPC303",
    "SubFPC304",
    "SubFPC305",
    "SubFPC306",
    "SubFPC307",
)
