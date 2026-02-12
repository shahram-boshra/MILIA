# MILIA Pipeline — Production Release File Audit

**Purpose**: Evidence-based audit of production-ready files at the repository root, based on PyPA, pyOpenSci, Scientific Python Development Guide, Citation File Format, and GitHub community standards.

**Scope**: Repository-level infrastructure, metadata, documentation, and tooling files (outside `milia_pipeline/` source).

**Version**: 1.1.0 | **Date**: February 2026

---

## 1. Current State of the Repository Root

**Implementation methodology**: Start from the root, directory by directory, file by file. For each item, determine with evidence: should this directory/file and its contents be uploaded to GitHub as part of the MILIA software, or should it be relocated/excluded because it is irrelevant to the production release?

Based on line-by-line verification of `find .` output. Each directory's **actual contents** were examined.

| Status | File / Directory | Verified Contents |
|--------|-----------------|-------------------|
| ✅ | `main.py` | Entry point (~5,280 lines) |
| ✅ | `setup.py` | Legacy packaging file (see §5) |
| ❌ Removed | `.gitignore` | Stale file deleted by maintainer. A fresh `.gitignore` must be created from GitHub's Python template before `git init` (§4.8) |
| ✅ | `research_experiments.yaml` | Research experiment configuration |
| ✅ | `configs/` | 7 root YAMLs + `datasets/` with 10 dataset-specific YAMLs |
| ✅ | `milia_pipeline/` | 11 submodules, ~100+ `.py` files |
| ✅ | `tests/` | 127 test files + `conftest.py` + `fixtures/` + `data/` |
| ⚠️ Dev-only | `test_data/` | 13 test fixture files (`.pt`, `.csv`, `.tar.gz`, `.xyz`, `.md`). No test references root `test_data/` — all use `/tmp/test_data/` with mocks. **Exclude from GitHub via `.gitignore`** |
| ⚠️ Dev-only | `scripts/` | 9 utility scripts (NPZ checking, doc generation, import analysis). Developer tools only, 2 still named VQM24. **Exclude from GitHub via `.gitignore`** |
| ⚠️ Outdated | `docs/` | 21 files (18 .md + 2 .py + 1 subdir). 13+ files still reference old VQM24 name. Needs revision before GitHub — **exclude via `.gitignore` for now** |
| ⚠️ Outdated | `examples/` | 3 files in `preprocessing/` — all reference VQM24. Delete contents, keep directory as placeholder with `.gitkeep` |
| ✅ Functional | `experiments/` | Empty but **functional target** for Research API (§4.2) |
| ❌ Deleted | `utils/` | Contained only deprecated tests + old conftest. No active usage. Directory removed entirely. |
| ⚠️ Internal | `archive/` | Development blueprints, VQM24 notes, testing guides — **exclude from GitHub via `.gitignore`** (§4.4) |
| ⚠️ Build artifact | `milia_pipeline.egg-info/` | Setuptools cache — needed locally, exclude from GitHub via `.gitignore` (§4.1) |

**Also found in source tree** (all removed):
- ~~2 log files: `milia_pipeline/transformations/plugin_system.log`, `milia_pipeline/logging_config.log`~~ — **DELETED**
- ~~2 deprecated files: `handlers/dataset_handlers.py.DEPRECATED`, `handlers/dataset_handler_integration.py.backup`~~ — **DELETED**
- ~~2 deprecated root files: `config.yaml.DEPRECATED`, `migrate_config.py.DEPRECATED`~~ — **DELETED**

### 1.1 GitHub Upload Decision Tracker

Each root-level item reviewed directory-by-directory. Decision and evidence recorded here.

| Item | Decision | Reason |
|------|----------|--------|
| `archive/` | ❌ Exclude from GitHub (`.gitignore`) | Internal dev docs (blueprints, VQM24 notes). No value to end users. Git docs: `.gitignore` for files all cloners should not receive. |
| `configs/` | ✅ Upload to GitHub as-is | Functional runtime configuration (7 root YAMLs + 10 dataset YAMLs). Software depends on these. |
| `docs/` | ❌ Exclude from GitHub (`.gitignore`) for now | 21 files, 13+ still reference old VQM24 name. Contains internal dev docs (blueprints, bug analyses) and deferred Phase 8 draft code. No current user-facing value. Revise VQM24→MILIA and curate before uploading. |
| `examples/` | ✅ Upload to GitHub as empty placeholder (`.gitkeep`) | All 3 existing files reference VQM24 — outdated, delete. Keep directory with `.gitkeep` for future MILIA examples. |
| `experiments/` | ✅ Upload to GitHub as empty placeholder (`.gitkeep`) | Functional target for end-user experimental extensions (transformations, descriptors, etc.). Currently empty. |
| `milia_pipeline/` | ✅ Upload to GitHub as-is | Core installable package (11 submodules, ~100+ .py files). Two runtime `.log` files deleted — `.gitignore` will prevent future tracking. |
| `milia_pipeline.egg-info/` | ❌ Exclude from GitHub (`.gitignore`) | Setuptools cache artifact. Generated locally by `pip install -e .`. Each clone generates its own. |
| `scripts/` | ❌ Exclude from GitHub (`.gitignore`) | 9 developer-only utility scripts. No end-user value. 2 still reference VQM24. Kept locally for developer reference. |
| `test_data/` | ❌ Exclude from GitHub (`.gitignore`) | Tests use `/tmp/test_data/` with mocks, not this directory. Dev-only fixture files. Delete `checkpoint_prediction_tracker.md`. |
| `tests/` | ✅ Upload to GitHub as-is | 127+ test files + `conftest.py` + `fixtures/` + `data/`. Essential for CI/CD and contributor validation. |
| `utils/` | ❌ Deleted entirely | Only deprecated tests + old conftest. No active usage. Directory removed. |
| `main.py` | ✅ Upload to GitHub as-is | Entry point orchestration script for MILIA pipeline. |
| `setup.py` | ✅ Upload to GitHub (review after `pyproject.toml` created, §2.1) | Legacy packaging file. May be reduced/removed after migrating metadata to `pyproject.toml`. |
| `research_experiments.yaml` | ✅ Upload to GitHub as-is | Research experiments configuration for MILIA pipeline. |

---

## 2. Files That MUST Be Created (P0)

### 2.1 `pyproject.toml` — CRITICAL

**Source**: PyPA, PEP 517/518/621. PyPA: *"The `[build-system]` table should always be present"*; *"For new projects, use the `[project]` table"*.

**Required contents**: `[build-system]` (setuptools backend), `[project]` (name, version, description, readme, license, requires-python, authors, keywords, classifiers, dependencies, urls), `[project.optional-dependencies]`, `[project.scripts]` (CLI entry points from `cli_manager.py`), tool configs (`pytest`, `ruff`).

**Re: existing `setup.py`**: If it contains only static metadata → migrate to `pyproject.toml`, remove `setup.py`. If it has dynamic logic (version computation, C extensions) → retain alongside `pyproject.toml` with those fields marked `dynamic`.

---

### 2.2 `LICENSE` — CRITICAL

**Source**: PyPA: *"Every package should include a license file... packages without an explicit license can not be legally used or distributed."* pyOpenSci: *"Your LICENSE file should be stored at root."*

**Action**: Create with full license text (BSD-3-Clause, MIT, or Apache-2.0). Declare SPDX identifier in `pyproject.toml` (PEP 639).

---

### 2.3 Root `README.md` — CRITICAL

**Source**: PyPA (`readme` field in `pyproject.toml` must point to a file), pyOpenSci, GitHub Community Standards.

**Contents**: Project name, badges, one-paragraph description, key features, installation, quick-start example, links to docs/contributing, citation pointer, license summary.

**Note**: Distinct from existing `docs/README.md` (internal documentation navigation).

---

### 2.4 `CHANGELOG.md`

**Source**: pyOpenSci, Keep a Changelog (`keepachangelog.com`), Semantic Versioning.

**Format**: `## [Version] - Date` sections with `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security` subsections. `[Unreleased]` section at top.

---

### 2.5 `CONTRIBUTING.md`

**Source**: pyOpenSci, GitHub Community Standards. GitHub auto-links this from Community tab and PR/issue creation.

**Contents**: Dev environment setup, running tests, code style, how to submit changes (adapted to actual workflow), review process.

---

### 2.6 `CODE_OF_CONDUCT.md`

**Source**: pyOpenSci, GitHub Community Standards. Use Contributor Covenant (`contributor-covenant.org`) with project-specific contact info.

---

### 2.7 `CITATION.cff`

**Source**: Citation File Format standard. GitHub natively renders it; Zenodo uses it for DOI publication.

**Contents**:
```yaml
cff-version: 1.2.0
message: "If you use this software, please cite it using the metadata from this file."
title: "MILIA Pipeline"
type: software
authors:
  - family-names: <Last Name>
    given-names: <First Name>
    orcid: "https://orcid.org/XXXX-XXXX-XXXX-XXXX"
version: <current version>
date-released: <YYYY-MM-DD>
license: <SPDX identifier>
repository-code: <GitHub URL>
keywords:
  - molecular machine learning
  - quantum chemistry
  - graph neural networks
```

---

## 3. Files That SHOULD Be Created (P1–P3)

### 3.1 `.pre-commit-config.yaml` *(requires Git)*

**Source**: pre-commit.com, Scientific Python Development Guide. Hooks for `ruff`, `check-yaml`, `check-toml`, `end-of-file-fixer`, `trailing-whitespace-fixer`, `detect-secrets`.

### 3.2 `Makefile`

**Source**: Common in scientific Python (NumPy, SciPy, scikit-learn). Targets: `install`, `install-dev`, `test`, `lint`, `format`, `docs`, `clean`, `build`, `help`.

### 3.3 `SECURITY.md`

**Source**: GitHub Security Advisories, OpenSSF. Contents: supported versions, how to report vulnerabilities (email, not public issue), response timeline.

### 3.4 `.github/` Directory *(requires Git + GitHub)*

| File | Purpose |
|------|---------|
| `workflows/ci.yml` | Test on push/PR (multi-Python matrix) |
| `workflows/release.yml` | Publish to PyPI on tag |
| `ISSUE_TEMPLATE/bug_report.md` | Structured bug reports |
| `ISSUE_TEMPLATE/feature_request.md` | Feature requests |
| `PULL_REQUEST_TEMPLATE.md` | PR checklist |
| `dependabot.yml` | Automated dependency updates |

### 3.5 `MANIFEST.in`

**Source**: PyPA, setuptools docs. Controls sdist file inclusion when using setuptools. Not needed if migrating to `hatchling`/`flit`.

```
include LICENSE README.md CHANGELOG.md CITATION.cff pyproject.toml
recursive-include configs *.yaml
recursive-include milia_pipeline *.py *.yaml
recursive-exclude tests *
recursive-exclude test_data *
prune _legacy
prune experiments
prune milia_pipeline.egg-info
```

### 3.6 `requirements-dev.txt` or `[project.optional-dependencies]`

**Source**: PyPA. Dev dependencies (pytest, ruff, pre-commit, sphinx) separate from runtime. Modern approach: `[project.optional-dependencies]` in `pyproject.toml`.

### 3.7 `RELEASE_CHECKLIST.md`

**Source**: py-pkgs.org: *"It's useful to list files that need to be updated every release as part of a release checklist."* Steps: version bump, CHANGELOG, build sdist/wheel, TestPyPI, PyPI, docs deploy.

### 3.8 `tox.ini` or `noxfile.py`

**Source**: Scientific Python Development Guide. Reproducible multi-Python-version test environments.

### 3.9 Documentation Build System

**Source**: Sphinx, Read the Docs, pyOpenSci. Add `docs/conf.py` (Sphinx) or `mkdocs.yml`, `docs/requirements.txt`, API reference auto-generation.

---

## 4. Files That Should NOT Be in the Repository

### 4.1 `milia_pipeline.egg-info/` — EXCLUDE FROM GITHUB VIA `.gitignore`

**Do NOT delete locally.** This is a setuptools cache artifact needed for the installed package to function (pypa/setuptools #3348: *"a cache artifact in your file system"*; #4197: *"editable installs... the metadata should still exist somewhere"*).

You do **not** need two different versions of the project. You work in one directory. `.gitignore` tells Git which files to skip — the files remain on your disk (Git docs: *"ensure that certain files not tracked by Git remain untracked"*). Per #4197: *"you shouldn't commit it into Git. So a common practice is to exclude it via .gitignore."*

```
Local machine:                     GitHub:
├── milia_pipeline/         ✅     ├── milia_pipeline/         ✅
├── milia_pipeline.egg-info ✅     │   (skipped by .gitignore)
├── __pycache__/            ✅     │   (skipped by .gitignore)
├── tests/                  ✅     ├── tests/                  ✅
└── .gitignore              ✅     └── .gitignore              ✅
```

**Action**: Add `*.egg-info/` to `.gitignore` (included in GitHub's standard Python template). Anyone who clones generates their own via `pip install -e .`.

---

### 4.2 `experiments/` — RETAIN

Currently empty but **functionally required** by the Research API:
- Structural doc line 44: listed as `experiments/  # Experimental code/configs`
- `PluginMetadata.plugin_type` includes `'user_experimental'` (line 2716)
- CLI commands: `run-experiment`, `validate-experiment`, `list-experiments` (line 241)
- `ExperimentRunner(config, output_dir)` (line 2802)
- `ExperimentConfiguration` with `save_to_yaml`/`load_from_yaml` (lines 2763–2770)

**Action**: Keep. Add `experiments/README.md` explaining purpose. Add `prune experiments` to `MANIFEST.in`.

**Unverified**: Whether `ExperimentRunner.output_dir` defaults to `experiments/` — requires source code inspection.

---

### 4.3 `utils/` — RELOCATE TO `_legacy/`, THEN DELETE

Contains **only** deprecated content: `DEPRECATED_TESTs/` (7 test files) + `conftest_original.py`.

**Critical — no Git**: Without VCS, deletion = permanent loss. Evidence for archiving:
- Eric Ma's data science project guide: `archive/` subdirectory for `no-longer-useful.py`
- IN-COM Data Systems: transition management when removing deprecated code
- SemVer: deprecation cycle before removal
- setuptools: `prune` excludes directories from sdist

**Action**:
1. Create `_legacy/deprecated_tests/`, move all 8 files there
2. Delete empty `utils/`
3. Add `_legacy/README.md` (what's here, why, policy: do not import, delete when Git adopted)
4. Add `prune _legacy` to `MANIFEST.in`

---

### 4.4 `archive/` — KEEP LOCALLY, EXCLUDE FROM GITHUB VIA `.gitignore`

Internal development documentation (blueprints, VQM24 investigation notes, testing guides, decision records). Contains 16 files across 2 subdirectories. No value to end users or external contributors — purely maintainer reference.

**Decision**: Exclude from GitHub entirely. Per Git official docs (git-scm.com/docs/gitignore): patterns for files that all developers/cloners should not receive go into `.gitignore`.

**Action**: Add `archive/` to `.gitignore`. No `MANIFEST.in` entry needed (`.gitignore` already prevents it from reaching Git, so it will never be in an sdist).

---

### 4.5 `*.DEPRECATED` and `*.backup` in `milia_pipeline/handlers/` — RELOCATE

Two files inside the **installable source package**: `dataset_handlers.py.DEPRECATED`, `dataset_handler_integration.py.backup`.

**Action**: Move to `_legacy/deprecated_handlers/`, rename to original `.py` names.

---

### 4.6–4.7 `*.log` files — DELETE

`plugin_system.log` and `logging_config.log` are runtime artifacts. Add `*.log` to `.gitignore`.

---

### 4.8 `.gitignore` — REPLACE WITH FRESH VERSION

Essential for GitHub upload — excludes `.egg-info/`, `__pycache__/`, `*.log`, etc. (§4.1). Current contents are stale. Per GitHub Docs, start from maintained templates.

**Action**:
```bash
curl -o .gitignore https://raw.githubusercontent.com/github/gitignore/main/Python.gitignore
```
Append MILIA-specific patterns:
```
# ===== MILIA-specific =====
*.log
# experiments/  # Uncomment if experiments/ generates large output files
```

---

## 5. Existing Files to Review

| File | Action |
|------|--------|
| `setup.py` | After creating `pyproject.toml`: migrate static metadata, reduce or remove |
| `docs/README.md` | Keep — internal documentation navigation (distinct from root README) |
| `docs/INDEX.md` | Adapt as entry point if Sphinx/MkDocs adopted |
| `examples/` | Expand: add training, HPO, prediction, plugin development examples |

---

## 6. Complete Production-Ready File Tree

```
milia/
├── .github/                             # ⬜ CREATE *(requires Git + GitHub)*
│   ├── workflows/ci.yml
│   ├── workflows/release.yml
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── .gitignore                           # ⚠️ REPLACE with fresh Python template
├── .pre-commit-config.yaml              # ⬜ CREATE *(requires Git)*
├── CHANGELOG.md                         # ⬜ CREATE
├── CITATION.cff                         # ⬜ CREATE
├── CODE_OF_CONDUCT.md                   # ⬜ CREATE
├── CONTRIBUTING.md                      # ⬜ CREATE
├── LICENSE                              # ⬜ CREATE
├── Makefile                             # ⬜ CREATE
├── MANIFEST.in                          # ⬜ CREATE (if using setuptools)
├── README.md                            # ⬜ CREATE (root-level)
├── RELEASE_CHECKLIST.md                 # ⬜ CREATE
├── SECURITY.md                          # ⬜ CREATE
├── pyproject.toml                       # ⬜ CREATE — CRITICAL
├── setup.py                             # ✅ EXISTS — review/reduce
├── main.py                              # ✅ EXISTS
├── research_experiments.yaml            # ✅ EXISTS
├── configs/                             # ✅ EXISTS
├── milia_pipeline/                      # ✅ EXISTS
├── tests/                               # ✅ EXISTS
├── test_data/                           # ⚠️ LOCAL ONLY — .gitignore excludes from GitHub
├── scripts/                             # ⚠️ LOCAL ONLY — .gitignore excludes from GitHub
├── docs/                                # ⚠️ LOCAL ONLY — .gitignore excludes from GitHub until revised (VQM24→MILIA)
├── examples/                            # ✅ PLACEHOLDER — `.gitkeep`, populate with MILIA examples later
├── experiments/                         # ✅ PLACEHOLDER — `.gitkeep`, end-user experimental extensions
├── archive/                             # ⚠️ LOCAL ONLY — .gitignore excludes from GitHub
├── _legacy/                             # ⬜ CREATE — deprecated code archive
│   ├── README.md
│   ├── deprecated_tests/                #   from utils/DEPRECATED_TESTs/
│   └── deprecated_handlers/             #   from handlers/*.DEPRECATED, *.backup
├── ⚠️ milia_pipeline.egg-info/          # LOCAL ONLY — .gitignore excludes from GitHub
├── ❌ utils/                            # DELETE after relocating to _legacy/
├── ↗️ *.DEPRECATED, *.backup            # RELOCATE to _legacy/
└── ❌ *.log in source tree              # DELETE (runtime artifacts)
```

---

## 7. Priority Order

| Priority | Action | Reason |
|----------|--------|--------|
| **P0** | `pyproject.toml` | Blocks proper installation |
| **P0** | `LICENSE` | Legal requirement |
| **P0** | Root `README.md` | PyPI long description / project front page |
| **P0** | Replace `.gitignore` with fresh Python template | Essential for GitHub upload |
| **P0** | Create `_legacy/`, relocate deprecated files | Preserve without Git |
| **P0** | Delete `utils/` (after relocation) | Repository hygiene |
| **P0** | Delete `*.log` files from source tree | Runtime artifacts |
| **P0** | Add `experiments/README.md` | Prevents mistaken removal |
| **P1** | `CHANGELOG.md` | Release communication |
| **P1** | `CITATION.cff` | Scientific citation standard |
| **P1** | `CONTRIBUTING.md` | Contributor onboarding |
| **P1** | `CODE_OF_CONDUCT.md` | Community standards |
| **P2** | `Makefile` | Developer experience |
| **P2** | `MANIFEST.in` | Correct sdist packaging |
| **P2** | `SECURITY.md` | Vulnerability reporting |
| **P2** | `.pre-commit-config.yaml` | Code quality *(requires Git)* |
| **P2** | `.github/workflows/ci.yml` | Automated testing *(requires Git + GitHub)* |
| **P3** | `RELEASE_CHECKLIST.md` | Release process docs |
| **P3** | `.github/` templates | Issue/PR quality *(requires Git + GitHub)* |
| **P3** | `tox.ini` / `noxfile.py` | Multi-env testing |
| **P3** | Documentation build system | Hosted docs |
| **P3** | Expanded `examples/` | User onboarding |

---

## 7A. Note: Discrepancy

The structural document lists `config.yaml` and `migrate_config.py` at root — neither appears in `find .` output. Verify with maintainer.

---

## 7B. Advisory: Adopt Git

Every source cited here assumes VCS. Without it: no change tracking, no recovery, no CI/CD, no collaboration. Adopting Git eliminates `_legacy/`, unlocks P2/P3 items, and is the single highest-impact infrastructure improvement.

---

## 8. References

1. **PyPA** — `packaging.python.org` (pyproject.toml, MANIFEST.in, modernize setup.py, versioning)
2. **pyOpenSci** — `pyopensci.org/python-package-guide/` (structure, LICENSE, CODE_OF_CONDUCT)
3. **Scientific Python Development Guide** — `learn.scientific-python.org/development/`
4. **Citation File Format** — `citation-file-format.github.io/`; GitHub CITATION files docs
5. **PEP 517, 518, 621, 639** — Build system and metadata standards
6. **setuptools docs** — `setuptools.pypa.io/en/latest/`
7. **py-pkgs.org** — Releasing and versioning: `py-pkgs.org/07-releasing-versioning.html`
8. **Keep a Changelog** — `keepachangelog.com/`
9. **IN-COM Data Systems** — Managing deprecated code: `in-com.com/blog/managing-deprecated-code-in-software-development/`
10. **Eric Ma** — Data science project structure (GitHub Gist)
11. **Semantic Versioning 2.0.0** — `semver.org/`
12. **pypa/setuptools #3348** — .egg-info as cache artifact
13. **pypa/setuptools #4197** — Why not auto-delete .egg-info
14. **pypa/setuptools #4198** — Installing generates unnecessary files
15. **pypa/setuptools #4658** — Auto-ignore generated directories
16. **Git official docs** — `git-scm.com/docs/gitignore`
