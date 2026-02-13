# MILIA — Production Release File Audit

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
| ✅ Created | `.gitignore` | Fresh file created from GitHub's Python template + MILIA-specific exclusions (§2.4) |
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

**✅ IMPLEMENTATION PROGRESS**: §1 (Current State) and §1.1 (GitHub Upload Decision Tracker) fully completed — all root-level directories and files reviewed, decisions recorded, deletions done. §2.1 (`pyproject.toml`) DONE + reviewed. §2.2 (`LICENSE`) DONE. §2.3 (Root `README.md`) DONE + extensively reviewed (scope, terminology, end-user perspective corrections). §2.4 (`.gitignore`) DONE — fresh GitHub Python template + MILIA-specific exclusions (`archive/`, `docs/`, `scripts/`, `test_data/`). §2.5 (`CHANGELOG.md`) DONE — Keep a Changelog 1.1.0 format, initial release `[1.1.0] - 2026-02-12`. §2.6 (`CONTRIBUTING.md`) DONE — pyOpenSci/GitHub Community Standards compliant; covers bug reporting, fork-and-PR workflow, dev setup (conda + `pip install -e ".[dev]"`), test suite (127 tests, 8 markers, shared fixtures from `conftest.py`), Ruff code style (py310, line-length 100, E/W/F/I/UP/B/SIM rules), Keep a Changelog 1.1.0 format, PR guidelines, MIT license. All content evidence-based from `pyproject.toml`, `conftest.py`, `__init__.py`, `CHANGELOG.md`, `LICENSE`. §2.7 (`CODE_OF_CONDUCT.md`) DONE — Contributor Covenant 3.0 (latest version, released 2025-07-28). Canonical Markdown from `contributor-covenant.org/version/3/0/code_of_conduct/code_of_conduct.md`. `[NOTE]` reporting placeholder filled with maintainer email (`a.boshra@gmail.com` from `pyproject.toml` authors). `[NOTE]` enforcement advisory removed (default ladder kept as guidelines). Licensed CC BY-SA 4.0. §2.8 (`CITATION.cff`) DONE — CFF 1.2.0 (current latest). Validated with `cffconvert --validate`. Two authors with ORCIDs (`0009-0004-8925-2868`, `0009-0003-8540-1662`). Alias field for parenthetical name per CFF person schema. All metadata sourced from `pyproject.toml`, `__init__.py` `__version__`, `CHANGELOG.md`. APA/BibTeX outputs verified. **§2 (P0 files) FULLY COMPLETE.** §3.1 (`.pre-commit-config.yaml`) DONE — pre-commit.com + Scientific Python Dev Guide. 3 repos: `pre-commit-hooks` v6.0.0 (7 hooks), `ruff-pre-commit` v0.15.0 (lint+format, reads `pyproject.toml`), `detect-secrets` v1.5.0. Validated with `pre-commit validate-config`. File is a static YAML config — activates after `git init` + `pre-commit install`. §3.2 (`Makefile`) DONE — 20 phony targets in 7 sections (help, install, test, code quality, pre-commit, build, cleanup, info). Self-documenting `help` default. Bash strict mode. All targets validated with `make --dry-run`. Every target evidence-sourced from `pyproject.toml`, `CONTRIBUTING.md`, `.pre-commit-config.yaml`, `.gitignore`, `__init__.py`. §3.3 (`SECURITY.md`) DONE — GitHub Docs + OpenSSF OSPS Baseline. 7 sections: Supported Versions (1.1.x, Python 3.10–3.12), Reporting (GitHub Private Vulnerability Reporting + email `a.boshra@gmail.com`), Response Timeline (48h ack, 7d assessment), Coordinated Disclosure, Scope (6 in-scope categories: plugin abuse, YAML injection, dependency chain, path traversal, code execution, info exposure), Security-Related Configuration. All content evidence-based. §3.4 (`.github/` directory) DONE — 7 files created (ci.yml, release.yml, bug_report.yml, feature_request.yml, config.yml, PULL_REQUEST_TEMPLATE.md, dependabot.yml). YAML issue forms (modern GitHub standard) over legacy Markdown. Trusted Publishers for PyPI release. All YAML validated. §3.5 (`MANIFEST.in`) DONE — 76 lines. Explicit includes: root-level entry point (`main.py` — **critical fix** over audit draft, required by `project.scripts` but outside `milia_pipeline/`), metadata files (`LICENSE`, `README.md`, `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`, `setup.py`), runtime config (`configs/*.yaml`, `research_experiments.yaml`), package source (`milia_pipeline/ *.py *.yaml *.yml`). Excludes: `tests/`, `test_data/`, `_legacy/`, `experiments/`, `.egg-info/`. Global safety: `*.py[codz]`, `__pycache__`, `*.log`. Sources: PyPA MANIFEST.in guide, setuptools 82.0.0 docs. §3.6 (`[project.optional-dependencies]`) ALREADY SATISFIED in §2.1 — `pyproject.toml` declares `dev = ["pytest>=8.0", "pytest-mock>=3.14", "ruff"]`. No separate file needed. **Post-§3.5 consistency audit**: `pyproject.toml` `[tool.pytest.ini_options].markers` had only 3 of 8 markers (`slow`, `integration`, `gpu`). `conftest.py` `pytest_configure` registers 6 (`smoke`, `contract`, `e2e`, `regression`, `thread_safety`, `slow`). Per pytest docs, both mechanisms are additive — no functional breakage under `--strict-markers`. However, `pyproject.toml` is the canonical config and should declare all 8. **FIX APPLIED**: Added 5 missing markers (`smoke`, `contract`, `e2e`, `regression`, `thread_safety`) to `pyproject.toml`. Descriptions sourced from `conftest.py` lines 522–545. Cross-verified: `CONTRIBUTING.md` (8-marker table correct), `Makefile` (8-marker comment + 3 marker targets correct), `.github/workflows/ci.yml` (`smoke` marker reference correct). All downstream files already had complete marker information — only `pyproject.toml` was incomplete. **§3.7 (`RELEASE_CHECKLIST.md`) DONE** — py-pkgs.org §7.3 + pythonpackaging.info §7. 14-step workflow: version bump (`__init__.py` line 252 — single source of truth via `pyproject.toml` `[tool.setuptools.dynamic]`), `CHANGELOG.md` `[Unreleased]` promotion (Keep a Changelog 1.1.0), `CITATION.cff` `version` + `date-released` update (CFF 1.2.0, validated with `cffconvert --validate`), `SECURITY.md` Supported Versions table (minor/major only), quality checks (`ruff check .` + `ruff format --check .` from `[tool.ruff]`, `pytest -v --tb=short` from `[tool.pytest.ini_options]`, `make check` composite), local build verification (`python -m build` from `[build-system]`), tag (`git tag -a vX.Y.Z` matching `release.yml` `v*.*.*` pattern), push triggers `release.yml` → PyPI Trusted Publishers (OIDC), GitHub Release creation, post-release PyPI verification (`get_package_info()` from `__init__.py` line 623). Quick-reference table: 4 files updated per release. One-time prerequisites section (Trusted Publisher, `pypi` environment, Private Vulnerability Reporting). **§3 (P1–P3 files) FULLY COMPLETE.**

---

## 2. Files That MUST Be Created (P0)

### 2.1 `pyproject.toml` — ✅ IMPLEMENTED

**Source**: PyPA, PEP 517/518/621. PyPA: *"The `[build-system]` table should always be present"*; *"For new projects, use the `[project]` table"*.

**Required contents**: `[build-system]` (setuptools backend), `[project]` (name, version, description, readme, license, requires-python, authors, keywords, classifiers, dependencies, urls), `[project.optional-dependencies]`, `[project.scripts]` (CLI entry points from `cli_manager.py`), tool configs (`pytest`, `ruff`).

**Re: existing `setup.py`**: If it contains only static metadata → migrate to `pyproject.toml`, remove `setup.py`. If it has dynamic logic (version computation, C extensions) → retain alongside `pyproject.toml` with those fields marked `dynamic`.

**✅ DONE**: `setup.py` contained 100% static metadata — fully migrated to `pyproject.toml`. `setup.py` reduced to backward-compatibility shim. Version resolved via `dynamic = ["version"]` with `{attr = "milia_pipeline.__version__"}` (single source of truth: `__init__.py`). PEP 639 SPDX license (`license = "MIT"`, `setuptools>=77`). Fixed stale `python_requires` in `__init__.py` `get_package_info()` (`">=3.8"` → `">=3.10"`).

**Revisions applied in review**:
- Package name: `milia-pipeline` → `milia` (brand alignment, simpler `pip install milia`)
- URLs: updated to `github.com/shahram-boshra/MILIA`
- Description: "pipeline" → "framework"; scope broadened from "quantum chemistry" → "molecular sciences" (covers computational physical chemistry, chemical physics, materials science, computational medicinal chemistry, and beyond)

---

### 2.2 `LICENSE` — ✅ IMPLEMENTED

**Source**: PyPA: *"Every package should include a license file... packages without an explicit license can not be legally used or distributed."* pyOpenSci: *"Your LICENSE file should be stored at root."*

**Action**: Create with full license text (BSD-3-Clause, MIT, or Apache-2.0). Declare SPDX identifier in `pyproject.toml` (PEP 639).

**✅ DONE**: Created `LICENSE` with canonical MIT text (choosealicense.com). Copyright `2026-present Asadollah Boshra`. SPDX already declared in `pyproject.toml` (`license = "MIT"`, `license-files = ["LICENSE"]`).

---

### 2.3 Root `README.md` — ✅ IMPLEMENTED

**Source**: PyPA (`readme` field in `pyproject.toml` must point to a file), pyOpenSci, GitHub Community Standards.

**Contents**: Project name, badges, one-paragraph description, key features, installation, quick-start example, links to docs/contributing, citation pointer, license summary.

**Note**: Distinct from existing `docs/README.md` (internal documentation navigation).

**✅ DONE**: Created root `README.md` per pyOpenSci checklist. Includes: name with acronym expansion (Machine Intelligent Learning Inference Assistant), badges (Python/License/Ruff), 2-paragraph description with ecosystem context, key features sections, conda+pip installation, CLI quick-start examples, 11-module architecture table, testing section (127 tests), contributing/citation/license pointers, project links.

**Revisions applied in review**:
- Scope: "quantum chemical datasets" → "molecular sciences" (covers computational physical chemistry, chemical physics, materials science, computational medicinal chemistry, and beyond)
- No-code scope: "without writing model-level code" → full-stack no-code at every pipeline level (datasets, transformations, descriptors, models, training, deployment), extensible via plugins
- Terminology: "molecular transformations" → "molecular graph transformations" (evidence: `graph_transforms.py`, `get_graph_transforms()`, PyG transform categories)
- Terminology: "descriptor computation" → "molecular descriptor computation" (evidence: "Molecular descriptor system" L66, "Molecular descriptor calculation" L3383)
- Model access: reframed from "dynamic introspection" (developer concept) to "simply by naming them in configuration — no model-level code" (end-user experience)
- Architecture Builder / Model Composer: reframed from user-facing modules to configuration-driven experience ("Define custom architectures from 10 built-in templates... all through YAML configuration, no code required")
- Removed "Zero-Modification Dataset Extension" section (developer concern, not end-user)
- Removed `DeviceManager` reference (internal module); replaced with "automatic device detection or explicit selection through configuration"
- Molecular Descriptors section: retitled from "Comprehensive Molecular Processing", added QSAR/QSPR/in-silico drug discovery significance, plugin extension "for advanced research needs"
- HPO: removed Ray Tune (evidence: "complete, inactive" throughout codebase); Optuna is the only active backend
- "Extensible Transformation System" → "Extensible Graph Transformation System"
- Dataset handlers: "ships with" → "currently ships with" (signals extensibility)

---

### 2.4 `.gitignore` — ✅ IMPLEMENTED

Essential for GitHub upload — excludes `.egg-info/`, `__pycache__/`, `*.log`, etc. (§4.1). Current contents are stale. Per GitHub Docs, start from maintained templates. **Must be in place before the initial commit** — `.gitignore` is a repository gatekeeper that determines what enters Git history (Git best practices: "Commit a .gitignore file as early as possible in your project's lifecycle — ideally, as part of your initial commit").

**✅ DONE**: Created fresh `.gitignore` from GitHub's official Python template (`github/gitignore/main/Python.gitignore`, retrieved 2026-02-12). Appended MILIA-specific exclusion patterns per §1.1 Decision Tracker: `archive/*` (§4.4 — internal dev docs), `docs/` (§1.1 — VQM24 references, exclude until revised), `scripts/` (§1.1 — developer-only utilities), `test_data/` (§1.1 — dev-only fixtures). `*.egg-info/` and `*.log` already covered by GitHub template. Commented `experiments/` placeholder included. Used `archive/*` (not `archive/`) with `!archive/MILIA_Production_Release_File_Audit.md` negation so the active audit file is Git-tracked per `git-scm.com/docs/gitignore` directory-vs-contents pattern.

---

### 2.5 `CHANGELOG.md` — ✅ IMPLEMENTED

**Source**: pyOpenSci, Keep a Changelog (`keepachangelog.com`), Semantic Versioning.

**Format**: `## [Version] - Date` sections with `Added`, `Changed`, `Deprecated`, `Removed`, `Fixed`, `Security` subsections. `[Unreleased]` section at top.

**✅ DONE**: Created `CHANGELOG.md` in Keep a Changelog 1.1.0 format with SemVer 2.0.0 adherence. Contains `[Unreleased]` section at top, `[1.1.0] - 2026-02-12` initial release entry with `### Added` listing all shipped features (from `__init__.py` module docstring). Footer links use `releases/tag/v1.1.0` for first version (no prior tag for comparison) per keepachangelog.com canonical format. ISO 8601 date format (`YYYY-MM-DD`). `pyproject.toml` already declares Changelog URL pointing to `CHANGELOG.md` on `main` branch.

---

### 2.6 `CONTRIBUTING.md` — ✅ IMPLEMENTED

**Source**: pyOpenSci, GitHub Community Standards. GitHub auto-links this from Community tab and PR/issue creation.

**Contents**: Dev environment setup, running tests, code style, how to submit changes (adapted to actual workflow), review process.

**✅ DONE**: Created `CONTRIBUTING.md` at repository root per pyOpenSci Contributing File guide and GitHub Community Standards. Welcoming tone with accessible language per pyOpenSci recommendation. Contains: bug reporting (with `get_package_info()`/`check_dependencies()` diagnostic instructions), enhancement suggestion process (YAML configuration examples encouraged), fork-and-pull-request workflow (6-step with branch naming), development setup (conda + `pip install -e ".[dev]"` matching `pyproject.toml` `[project.optional-dependencies].dev`), test suite section (127 tests, `pytest` commands, 8 registered markers from `conftest.py` `pytest_configure` + `pyproject.toml`, shared fixtures list, test writing conventions), Ruff code style (py310, line-length 100, E/W/F/I/UP/B/SIM rules, E501 ignored — all from `[tool.ruff]`), changelog contribution (Keep a Changelog 1.1.0 + SemVer 2.0.0, `[Unreleased]` section), PR guidelines (focused PRs, tests, checks, changelog, description), project structure overview, link to `CODE_OF_CONDUCT.md`, link to `LICENSE` (MIT). All content evidence-based from actual project files.

---

### 2.7 `CODE_OF_CONDUCT.md` — ✅ IMPLEMENTED

**Source**: pyOpenSci, GitHub Community Standards. Use Contributor Covenant (`contributor-covenant.org`) with project-specific contact info.

**✅ DONE**: Created `CODE_OF_CONDUCT.md` at repository root using **Contributor Covenant 3.0** (latest version, released 2025-07-28 by Organization for Ethical Source). Canonical Markdown text sourced verbatim from `https://www.contributor-covenant.org/version/3/0/code_of_conduct/code_of_conduct.md`. Two `[NOTE]` placeholders customized: (1) Reporting contact filled with maintainer email `a.boshra@gmail.com` (from `pyproject.toml` `[[project.authors]]`); (2) Enforcement advisory note removed — default enforcement ladder retained as guidelines per Contributor Covenant FAQ ("Communities are encouraged to customize the 'Addressing and Repairing Harm' section"). Contributor Covenant 3.0 licensed CC BY-SA 4.0; attribution section preserved verbatim. Version 3.0 chosen over 2.1 per evidence: official `/adopt` page lists English (3.0) as primary, and OES announcement (2025-07-28) states it is the current recommended version.

---

### 2.8 `CITATION.cff` — ✅ IMPLEMENTED

**Source**: Citation File Format standard (`citation-file-format.github.io`). GitHub natively renders it as a "Cite this repository" sidebar widget; Zenodo uses it for DOI publication via GitHub–Zenodo integration; Zotero imports references directly.

**Spec version**: CFF 1.2.0 (current latest as of February 2026; confirmed via `citation-file-format.github.io` and GitHub `citation-file-format/citation-file-format` repository). Schema guide: `github.com/citation-file-format/citation-file-format/blob/1.2.0/schema-guide.md`.

**✅ DONE**: Created `CITATION.cff` at repository root per CFF 1.2.0 schema. Validated with `cffconvert --validate` → "Citation metadata are valid according to schema version 1.2.0." APA and BibTeX outputs verified correct.

**Metadata sourcing** (all evidence-based from actual project files):
- `title`: `"MILIA"` — from `pyproject.toml` `name = "milia"`, project brand name is uppercase MILIA
- `type`: `software` — CFF spec default for software projects
- `authors[0]`: `family-names: "Boshra"`, `given-names: "Asadollah"`, `alias: "Shahram"` — from `pyproject.toml` `authors` field (`"Asadollah (Shahram) Boshra"`); parenthetical name mapped to CFF `alias` field per CFF person schema (`cff_schema_definitions_person()`: `alias` is a valid person key)
- `authors[0].email`: `a.boshra@gmail.com` — from `pyproject.toml` `authors`
- `authors[0].orcid`: `https://orcid.org/0009-0004-8925-2868` — maintainer-provided; format verified per ORCID structure spec (`support.orcid.org/hc/en-us/articles/360006897674`: `0009-xxxx` range is valid, stored as full `https://orcid.org/` URI)
- `authors[1]`: `family-names: "Boshra"`, `given-names: "Ilia"` — from `pyproject.toml` `authors`
- `authors[1].email`: `ilia.boshra@gmail.com` — from `pyproject.toml` `authors`
- `authors[1].orcid`: `https://orcid.org/0009-0003-8540-1662` — maintainer-provided
- `version`: `"1.1.0"` — from `milia_pipeline/__init__.py` line 252 (`__version__ = "1.1.0"`)
- `date-released`: `"2026-02-12"` — from `CHANGELOG.md` initial release `[1.1.0] - 2026-02-12` (ISO 8601)
- `license`: `"MIT"` — from `pyproject.toml` `license = "MIT"` (SPDX identifier)
- `repository-code`: `"https://github.com/shahram-boshra/MILIA"` — from `pyproject.toml` `[project.urls].Repository`
- `url`: `"https://github.com/shahram-boshra/MILIA"` — from `pyproject.toml` `[project.urls].Homepage`
- `abstract`: exact `description` string from `pyproject.toml`
- `keywords`: exact 7-item list from `pyproject.toml` `keywords`

**Rendered citations** (verified via `cffconvert`):
- APA: `Boshra A., Boshra I. (2026). MILIA (version 1.1.0). URL: https://github.com/shahram-boshra/MILIA`
- BibTeX: `@misc{..., author = {Boshra, Asadollah and Boshra, Ilia}, title = {MILIA}, year = {2026}}`

---

## 3. Files That SHOULD Be Created (P1–P3)

### 3.1 `.pre-commit-config.yaml` — ✅ IMPLEMENTED *(activates when Git is initialized)*

**Source**: pre-commit.com, Scientific Python Development Guide (`learn.scientific-python.org/development/guides/style/`). Configuration file created at repository root — becomes active after `git init` + `pre-commit install`.

**Evidence for creating now (before Git)**: The `.pre-commit-config.yaml` is a static YAML configuration file independent of Git. Per pre-commit.com: "If a repository doesn't have a `.pre-commit-config.yaml` file, pre-commit will simply do nothing." The file defines hooks; `pre-commit install` activates them inside `.git/hooks/`. File creation and hook installation are separate steps (pre-commit.com setup guide: "create a configuration file" → "install the hooks"). Creating now ensures the config is ready the moment `git init` is run.

**✅ DONE**: Created `.pre-commit-config.yaml` at repository root. Validated with `pre-commit validate-config` (exit code 0, no errors) and Python `yaml.safe_load()` (valid YAML).

**Hook repos and versions** (all latest releases verified via web search):
- `pre-commit/pre-commit-hooks` `v6.0.0` (PyPI: Aug 9, 2025) — 7 hooks: `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-added-large-files`, `check-merge-conflict`, `debug-statements`
- `astral-sh/ruff-pre-commit` `v0.15.0` (Ruff 0.15.0 released 2026-02-03; official README + PyPI + Ruff docs all reference `v0.15.0`) — 2 hooks: `ruff-check` (with `--fix --exit-non-zero-on-fix --show-fixes`), `ruff-format`. Lint before format per Ruff docs: "Ruff's fix behavior can output code changes that require reformatting"
- `Yelp/detect-secrets` `v1.5.0` (official GitHub README) — 1 hook: `detect-secrets`

**Configuration alignment**: Ruff hooks read from `pyproject.toml` `[tool.ruff]` (py310, line-length 100, select E/W/F/I/UP/B/SIM, ignore E501, isort known-first-party `milia_pipeline`) — no `args` needed in hook config. Per Ruff best practice: "move tool-specific arguments from `.pre-commit-config.yaml` to `pyproject.toml`... the tools will automatically discover and use the settings."

**Activation steps** (documented in file header):
```bash
pip install pre-commit        # or: conda install pre-commit
pre-commit install            # install hooks into .git/hooks/
pre-commit run --all-files    # run all hooks on entire codebase
pre-commit autoupdate         # update hook versions
```

### 3.2 `Makefile` — ✅ IMPLEMENTED

**Source**: Common in scientific Python (NumPy, SciPy, scikit-learn). pyOpenSci Python Package Guide — Task Runners: Make is "widely known and available on most systems, making it a familiar choice for many developers." Scientific Python Development Guide — Task Runners: task runners "make specialized developer tasks easy" and "help new contributors get productive quickly." scikit-learn uses a root `Makefile` with `dev`, `clean`, `help` targets. KDnuggets (2025): "Think of a Makefile as a single place where you define shortcuts for all the things you do repeatedly."

**✅ DONE**: Created `Makefile` at repository root with 20 phony targets organized into 7 sections. Self-documenting `help` as default target (parses `## ` comments via `grep`/`awk`). Validated with `make --dry-run` for all targets (0 syntax errors). Bash with strict error handling (`-eu -o pipefail`). All variables overridable from command line (`PYTHON ?= python3`, `PIP ?= pip`).

**Targets and evidence sources** (all evidence-based from actual project files):
- **Installation** (2 targets): `install` (`pip install -e .`), `install-dev` (`pip install -e ".[dev]"`) — from `pyproject.toml` `[project.optional-dependencies].dev` (`pytest>=8.0`, `pytest-mock>=3.14`, `ruff`) and CONTRIBUTING.md "Development Setup"
- **Testing** (4 targets): `test` (full suite), `test-smoke` (`-m smoke`), `test-integration` (`-m integration`), `test-fast` (`-m "not slow"`) — from `pyproject.toml` `[tool.pytest.ini_options]` (`testpaths = ["tests"]`, `addopts = ["-v", "--tb=short", "--strict-markers"]`, markers: `slow`, `integration`, `gpu`, `smoke`, `contract`, `e2e`, `regression`, `thread_safety`) and CONTRIBUTING.md "Running Tests"
- **Code Quality** (5 targets): `lint` (`ruff check .`), `lint-fix` (`ruff check --fix --show-fixes .`), `format` (`ruff format .`), `format-check` (`ruff format --check .`), `check` (composite: lint + format-check + test) — from `pyproject.toml` `[tool.ruff]` (py310, line-length 100, select E/W/F/I/UP/B/SIM, ignore E501) and CONTRIBUTING.md "Running Ruff" and `.pre-commit-config.yaml` ruff-check args
- **Pre-commit** (3 targets): `pre-commit` (`pre-commit run --all-files`), `pre-commit-install` (`pre-commit install`), `pre-commit-update` (`pre-commit autoupdate`) — from `.pre-commit-config.yaml` header (3 repos: pre-commit-hooks v6.0.0, ruff-pre-commit v0.15.0, detect-secrets v1.5.0)
- **Build** (1 target): `build` (`python -m build`) — from `pyproject.toml` `[build-system]` (`setuptools>=77`, `setuptools.build_meta`)
- **Cleanup** (2 targets): `clean` (safe — removes build/, dist/, `__pycache__`, `.pytest_cache`, `.ruff_cache`, htmlcov, coverage files, `*.py[codz]`, `MANIFEST`; does NOT delete `.egg-info` in project root per audit §4.1), `clean-all` (full reset — also removes `$(PACKAGE_NAME).egg-info/`) — patterns from `.gitignore` GitHub Python template
- **Information** (2 targets): `version` (prints `__version__` from `milia_pipeline/__init__.py`), `info` (calls `get_package_info()` + `check_dependencies()` from `milia_pipeline/__init__.py`)

### 3.3 `SECURITY.md` — ✅ IMPLEMENTED

**Source**: GitHub Docs ("Adding a security policy to your repository", "Securing your repository"), OpenSSF OSPS Baseline (Vulnerability Management), GitHub Docs ("Privately reporting a security vulnerability", "Configuring private vulnerability reporting for a repository").

**Required contents** (per GitHub Docs "Securing your repository"): supported versions, how to report vulnerabilities, response timeline, responsible disclosure guidelines.

**✅ DONE**: Created `SECURITY.md` at repository root per GitHub Docs and OpenSSF OSPS Baseline. Contains 7 sections: Supported Versions (1.1.x on Python 3.10/3.11/3.12 — from `pyproject.toml` classifiers + `__init__.py` `__version__`), Reporting a Vulnerability (two private channels: GitHub Private Vulnerability Reporting preferred + email `a.boshra@gmail.com` from `pyproject.toml` maintainers; explicit "do not use public issues" to distinguish from `CONTRIBUTING.md` bug reporting), What to Include (with `get_package_info()`/`check_dependencies()` diagnostics consistent with `CONTRIBUTING.md`), Response Timeline (48-hour acknowledgment, 7-day initial assessment — per community standards), Disclosure Policy (coordinated vulnerability disclosure per OpenSSF; GitHub Security Advisory publication; `Security` changelog entry per Keep a Changelog format), Scope (6 in-scope categories tailored to MILIA's attack surface: plugin system abuse, YAML config injection, dependency chain, path traversal, arbitrary code execution, sensitive info exposure; 3 out-of-scope exclusions), Security-Related Configuration (plugin trust, config validation, file path review). All content evidence-based from `pyproject.toml`, `__init__.py`, `CHANGELOG.md`, `CONTRIBUTING.md`. **Post-push action**: enable Private Vulnerability Reporting in GitHub repository settings (Settings → Security → Advanced Security → Enable).

### 3.4 `.github/` Directory — ✅ IMPLEMENTED *(activates when pushed to GitHub)*

**Source**: GitHub Docs (Actions workflows, issue/PR templates, Dependabot). All files are static YAML/Markdown created at repository root under `.github/` — same creation-before-activation principle as §3.1 (`.pre-commit-config.yaml`): files are created now, become functional after first push to GitHub.

**Evidence for creating now (before Git/GitHub)**: These are static configuration and template files with no runtime dependency on Git or GitHub for *creation*. Per §3.1 precedent: `.pre-commit-config.yaml` was created as a static YAML config that "activates after `git init` + `pre-commit install`." By the same logic, `.github/` files are static configs that activate after push to GitHub. Creating now ensures CI/CD, templates, and dependency management are ready the moment the repository is pushed.

**✅ DONE**: Created `.github/` directory with 7 files. All YAML files validated with `yaml.safe_load()` (0 errors).

- **`workflows/ci.yml`**: Lint job (Ruff check + format check on Python 3.12) → Test matrix (Python 3.10/3.11/3.12, smoke tests as first CI gate per `CONTRIBUTING.md`). Full conda-based test job commented as placeholder. Actions: `actions/checkout@v6`, `actions/setup-python@v6` (current latest Feb 2026). Concurrency group cancels in-progress runs. Evidence: `pyproject.toml` classifiers (3.10/3.11/3.12), `[tool.pytest]`, `[tool.ruff]`, `[project.optional-dependencies].dev`; `CONTRIBUTING.md` smoke marker.
- **`workflows/release.yml`**: Tag-triggered (`v*.*.*`) build → publish via PyPI Trusted Publishers (OIDC). Two jobs: `build` (`python -m build`, stores artifacts via `actions/upload-artifact@v5`) → `publish-to-pypi` (`pypa/gh-action-pypi-publish@release/v1`, `id-token: write`, `environment: pypi`). Evidence: `pyproject.toml` `[build-system]` (setuptools>=77), `name = "milia"`; PyPA Publishing Guide; PyPI Trusted Publishers docs.
- **`ISSUE_TEMPLATE/bug_report.yml`**: YAML issue form with: prerequisites checkboxes (existing issue search + security ≠ bug per `SECURITY.md`), description, steps to reproduce, expected/actual behavior, Python version dropdown (3.10/3.11/3.12), environment details with `get_package_info()`/`check_dependencies()` diagnostics, configuration (YAML render), additional context. Evidence: `CONTRIBUTING.md` bug reporting section, `SECURITY.md`, `pyproject.toml` classifiers.
- **`ISSUE_TEMPLATE/feature_request.yml`**: YAML issue form with: prerequisite checkbox, problem description, proposed solution with YAML config example, affected area multi-select dropdown (11 modules + docs + other), alternatives, additional context. Evidence: `CONTRIBUTING.md` enhancement suggestions, project structure (11 submodules).
- **`ISSUE_TEMPLATE/config.yml`**: Template chooser config. `blank_issues_enabled: false`. Security contact link directs to `SECURITY.md` per §3.3. Evidence: GitHub Docs ("Configuring issue templates").
- **`PULL_REQUEST_TEMPLATE.md`**: Description, related issue, type-of-change checkboxes (bug fix/feature/breaking/docs/refactor/tests), checklist aligned with `CONTRIBUTING.md` PR Guidelines (focused PRs, tests, pytest, ruff check, ruff format, changelog, self-review). Evidence: `CONTRIBUTING.md` PR Guidelines section.
- **`dependabot.yml`**: Version 2. Two ecosystems: `pip` (weekly, dev deps in `pyproject.toml`) + `github-actions` (weekly, Actions versions in `.github/workflows/`). Commit message prefixes (`build(deps)`, `ci(deps)`). Evidence: GitHub Docs (Dependabot config), `pyproject.toml` `[project.optional-dependencies].dev`.

**Issue template format**: YAML issue forms (`.yml`) chosen over legacy Markdown templates (`.md`). Per GitHub Docs ("Syntax for issue forms"): YAML forms provide structured web form fields with input types, required-field validation, dropdowns, and checkboxes — ensuring contributors submit complete, consistently formatted reports. GitHub Docs ("About issue and pull request templates"): "Issue templates created with issue forms need a `.yml` extension." The Markdown format is now considered legacy. `config.yml` added per GitHub Docs ("Configuring issue templates for your repository"): controls the template chooser, disables blank issues, and adds a security reporting contact link (directing to `SECURITY.md` per §3.3).

**Post-push setup required** (one-time, after first push to GitHub):
1. **PyPI Trusted Publisher** (for `release.yml`): Configure at `pypi.org/manage/project/milia/settings/publishing/` — add GitHub Actions publisher (owner: `shahram-boshra`, repo: `MILIA`, workflow: `release.yml`, environment: `pypi`).
2. **GitHub Environment** (for `release.yml`): Create `pypi` environment in repo Settings → Environments (optionally require manual approval).
3. **Private Vulnerability Reporting** (per §7C): Enable in repo Settings → Security → Code security and analysis.

### 3.5 `MANIFEST.in` — ✅ IMPLEMENTED

**Source**: PyPA — "Including files in source distributions with MANIFEST.in" (`packaging.python.org/guides/using-manifest-in/`), setuptools 82.0.0 — "Controlling files in the distribution" (`setuptools.pypa.io/en/latest/userguide/miscellaneous.html`). Controls sdist file inclusion when using setuptools. Not needed if migrating to `hatchling`/`flit`.

**✅ DONE**: Created `MANIFEST.in` at repository root (76 lines). Explicitly includes root-level metadata (`LICENSE`, `README.md`, `CHANGELOG.md`, `CITATION.cff`, `pyproject.toml`, `setup.py`), root-level entry point and config (`main.py`, `research_experiments.yaml`), runtime configuration directory (`recursive-include configs *.yaml`), and package source (`recursive-include milia_pipeline *.py` / `*.yaml *.yml`). Excludes `tests/`, `test_data/`, `_legacy/`, `experiments/`, `milia_pipeline.egg-info/` via `recursive-exclude` and `prune`. Global safety net: `global-exclude *.py[codz]`, `global-exclude __pycache__`, `global-exclude *.log`. Command order follows setuptools docs: include → recursive-include → recursive-exclude → prune → global-exclude.

**Critical fix over audit draft**: Original draft (above) was **missing `main.py`**. This root-level file is the CLI entry point (`pyproject.toml`: `milia = "main:main"`) but resides outside `milia_pipeline/`, so it is NOT auto-discovered by `[tool.setuptools.packages.find]` (`include = ["milia_pipeline*"]`). Without `include main.py`, an sdist built from this project would be broken — the console script entry point would fail to resolve. Also added: `include setup.py`, `include research_experiments.yaml`, `global-exclude` safety patterns. Evidence: setuptools 82.0.0 docs — default sdist inclusion rules ("pure Python module files implied by the `py-modules` and `packages` configuration").

### 3.6 `requirements-dev.txt` or `[project.optional-dependencies]` — ✅ ALREADY SATISFIED

**Source**: PyPA. Dev dependencies (pytest, ruff, pre-commit, sphinx) separate from runtime. Modern approach: `[project.optional-dependencies]` in `pyproject.toml`.

**✅ ALREADY DONE in §2.1**: `pyproject.toml` `[project.optional-dependencies].dev` declares `pytest>=8.0`, `pytest-mock>=3.14`, `ruff`. Install via `pip install -e ".[dev]"`. No separate `requirements-dev.txt` file needed — the modern `pyproject.toml` approach is already in place. Referenced by `CONTRIBUTING.md` "Development Setup", `Makefile` `install-dev` target, `.github/workflows/ci.yml` install step.

### 3.7 `RELEASE_CHECKLIST.md` — ✅ IMPLEMENTED

**Source**: py-pkgs.org §7.3: *"Checklist for releasing a new package version"* — steps: make changes, document in changelog, bump version, run tests and build, tag with version control, build and release to PyPI. pythonpackaging.info §7: *"It's useful to list files that need to be updated every release as part of a release checklist (which you can add as a RELEASE_CHECKLIST.md file in your repository)."*

**✅ DONE**: Created `RELEASE_CHECKLIST.md` at repository root. 14-step workflow organized into Pre-Release (steps 1–8), Release (steps 9–11), and Post-Release (steps 12–14). Every step references the exact file, field, and line number from the actual project.

**Files updated per release** (evidence-based quick-reference table in checklist):

| File | Field(s) | When |
|------|----------|------|
| `milia_pipeline/__init__.py` | `__version__` (line 252) | Every release |
| `CHANGELOG.md` | `[Unreleased]` → `[X.Y.Z]`, footer links | Every release |
| `CITATION.cff` | `version` (line 31), `date-released` (line 32) | Every release |
| `SECURITY.md` | Supported Versions table | Minor/major releases |

**Key evidence mappings**:
- Version single source of truth: `pyproject.toml` `[tool.setuptools.dynamic]` → `{attr = "milia_pipeline.__version__"}`
- Tag format: `release.yml` `tags: ["v*.*.*"]` triggers build → PyPI Trusted Publishers
- Quality gates: `ruff check .` + `ruff format --check .` (from `[tool.ruff]`), `pytest -v --tb=short` (from `[tool.pytest.ini_options]`)
- Build: `python -m build` (from `[build-system]` `setuptools>=77`)
- Post-release verification: `get_package_info()` (from `__init__.py` line 623)
- One-time prerequisites: PyPI Trusted Publisher, GitHub `pypi` environment, Private Vulnerability Reporting (from `release.yml` header, `SECURITY.md` §7C)

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

## 5. Existing Files to Review

| File | Action |
|------|--------|
| `setup.py` | ✅ DONE — migrated to `pyproject.toml`, reduced to shim |
| `docs/README.md` | Keep — internal documentation navigation (distinct from root README) |
| `docs/INDEX.md` | Adapt as entry point if Sphinx/MkDocs adopted |
| `examples/` | Expand: add training, HPO, prediction, plugin development examples |

---

## 6. Complete Production-Ready File Tree

```
milia/
├── .github/                             # ✅ CREATED — CI/CD, YAML issue forms, PR template, Dependabot *(activates when pushed to GitHub)*
│   ├── workflows/ci.yml
│   ├── workflows/release.yml
│   ├── ISSUE_TEMPLATE/bug_report.yml
│   ├── ISSUE_TEMPLATE/feature_request.yml
│   ├── ISSUE_TEMPLATE/config.yml
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── dependabot.yml
├── .gitignore                           # ✅ CREATED — GitHub Python template + MILIA-specific exclusions
├── .pre-commit-config.yaml              # ✅ CREATED — activates after git init + pre-commit install
├── CHANGELOG.md                         # ✅ CREATED
├── CITATION.cff                         # ✅ CREATED — CFF 1.2.0, validated with cffconvert
├── CODE_OF_CONDUCT.md                   # ✅ CREATED — Contributor Covenant 3.0 (CC BY-SA 4.0)
├── CONTRIBUTING.md                      # ✅ CREATED — pyOpenSci/GitHub Community Standards compliant
├── LICENSE                              # ✅ CREATED — MIT (SPDX: MIT)
├── Makefile                             # ✅ CREATED — 20 targets, self-documenting help
├── MANIFEST.in                          # ✅ CREATED — sdist file control (setuptools, 76 lines)
├── README.md                            # ✅ CREATED — project front page (pyOpenSci compliant) (root-level)
├── RELEASE_CHECKLIST.md                 # ✅ CREATED — 14-step release workflow (py-pkgs.org §7.3)
├── SECURITY.md                          # ✅ CREATED — GitHub Docs + OpenSSF compliant, coordinated disclosure
├── pyproject.toml                       # ✅ CREATED — canonical metadata (PEP 621/639)
├── setup.py                             # ✅ REDUCED — backward-compatibility shim
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
| **P0** ✅ | `pyproject.toml` | Blocks proper installation |
| **P0** ✅ | `LICENSE` | Legal requirement |
| **P0** ✅ | Root `README.md` | PyPI long description / project front page |
| **P0** ✅ | Replace `.gitignore` with fresh Python template | Essential for GitHub upload |
| **P0** ✅ | Create `_legacy/`, relocate deprecated files | Preserve without Git |
| **P0** ✅ | Delete `utils/` (after relocation) | Repository hygiene |
| **P0** ✅ | Delete `*.log` files from source tree | Runtime artifacts |
| **P0** ✅ | Add `experiments/README.md` | Prevents mistaken removal |
| **P1** | `CHANGELOG.md` | Release communication | ✅ |
| **P1** | `CITATION.cff` | Scientific citation standard | ✅ |
| **P1** | `CONTRIBUTING.md` | Contributor onboarding | ✅ |
| **P1** | `CODE_OF_CONDUCT.md` | Community standards | ✅ |
| **P2** | `Makefile` | Developer experience | ✅ |
| **P2** | `MANIFEST.in` | Correct sdist packaging | ✅ |
| **P2** | `SECURITY.md` | Vulnerability reporting *(post-push: enable Private Vulnerability Reporting — §7C)* | ✅ |
| **P2** | `.pre-commit-config.yaml` | Code quality *(activates after Git init)* | ✅ |
| **P2** | `.github/workflows/ci.yml` | Automated testing *(activates when pushed to GitHub)* | ✅ |
| **P3** | `RELEASE_CHECKLIST.md` | Release process docs | ✅ |
| **P3** | `.github/` templates | Issue/PR quality — YAML issue forms + config.yml + PR template *(activates when pushed to GitHub)* | ✅ |
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

## 7C. Post-Push: Enable GitHub Private Vulnerability Reporting

**Source**: GitHub Docs — [Configuring private vulnerability reporting for a repository](https://docs.github.com/en/code-security/security-advisories/working-with-repository-security-advisories/configuring-private-vulnerability-reporting-for-a-repository).

`SECURITY.md` (§3.3) directs reporters to the GitHub Security Advisories page as the **preferred** channel. This requires **Private Vulnerability Reporting** to be enabled in the repository settings — otherwise the "Report a vulnerability" button will not appear and reporters will have only the email fallback.

**Action** (one-time, after first push to GitHub):

1. Navigate to `https://github.com/shahram-boshra/MILIA` → **Settings**.
2. In the left sidebar under **Security**, click **Code security and analysis**.
3. Under **Private vulnerability reporting**, click **Enable**.

**Verification**: Visit `https://github.com/shahram-boshra/MILIA/security` — the **"Report a vulnerability"** button should be visible to any GitHub user.

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
