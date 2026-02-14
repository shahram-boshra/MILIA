# MILIA Pipeline — Definitive Guide for GitHub Release Preparation

**Purpose**: A complete, evidence-based, ordered checklist of every file and configuration required to prepare the MILIA Pipeline project for release on a private GitHub repository — targeting postdoc/industry job applications, journal reviewer access, and CI/CD automation.

**Scope**: Covers everything from pre-CI/CD essential files through CI/CD infrastructure. Each item includes its authoritative source, technical dependencies on other items, and MILIA-specific considerations.

**Context**: The MILIA project is a large-scale molecular ML pipeline with 11 core modules, 146 test files (127 original + 18 new + 1 conftest), conda-based dependency management (`environment.yml`), a `setup.py` with empty `install_requires=[]`, and a **Docker-based development workflow** — the project is built and run inside a Docker container (`(shah_env) root@...:/app/milia#`). The Docker image **MUST be available on GitHub** (via GitHub Container Registry / GHCR) so that reviewers, collaborators, or anyone evaluating the project can pull and run it immediately without building from source.

---

## Project Current State (Verified From Provided Files)

| Asset | Status | Evidence |
|-------|--------|----------|
| Source code (`milia_pipeline/`) | ✅ Complete | 11 core modules documented in `MILIA_Pipeline_Project_Structure.md` |
| Unit tests (127 files) | ✅ Complete | User-confirmed; file listing in `tests/` directory |
| Beyond-unit tests (18 files) | ✅ Complete | Sections 1–7 of `MILIA_Test_Recommendations.md` v1.2.0 |
| `conftest.py` | ✅ Complete | 573 lines; markers registered programmatically via `pytest_configure()` |
| `setup.py` | ✅ Reduced to shim | Static metadata fully migrated to `pyproject.toml`; backward-compatibility shim only (Audit §2.1) |
| `environment.yml` | ✅ Exists | Conda-based; channels: `conda-forge`, `pytorch`, `pyg` |
| `Dockerfile` | ✅ Exists | `miniconda3:latest` base; mamba installation; `shah_env` environment; **MUST be built and pushed to GHCR** for reviewer/user access |
| `.gitignore` | ✅ Complete | Fresh GitHub Python template + MILIA-specific exclusions (Audit §2.4) |
| `pyproject.toml` | ✅ Complete | Full PEP 621/639 metadata + tool configs; `setup.py` reduced to shim (Audit §2.1) |
| `.github/workflows/ci.yml` | ✅ Complete | Lint + test matrix (Python 3.10/3.11/3.12); smoke gate (Audit §3.4) |
| `README.md` | ✅ Complete | pyOpenSci-compliant project front page with badges, features, install, CLI examples (Audit §2.3) |
| `LICENSE` | ✅ Complete | MIT, copyright 2026-present Asadollah Boshra (Audit §2.2) |
| `CHANGELOG.md` | ✅ Complete | Keep a Changelog 1.1.0 format, initial release [1.1.0] - 2026-02-12 (Audit §2.5) |
| `CONTRIBUTING.md` | ✅ Complete | pyOpenSci/GitHub Community Standards compliant (Audit §2.6) |
| `CODE_OF_CONDUCT.md` | ✅ Complete | Contributor Covenant 3.0, CC BY-SA 4.0 (Audit §2.7) |
| `CITATION.cff` | ✅ Complete | CFF 1.2.0, validated with `cffconvert --validate` (Audit §2.8) |
| `SECURITY.md` | ✅ Complete | GitHub Docs + OpenSSF compliant, coordinated disclosure (Audit §3.3) |
| `RELEASE_CHECKLIST.md` | ✅ Complete | 14-step release workflow (Audit §3.7) |
| `Makefile` | ✅ Complete | 20 phony targets, self-documenting help (Audit §3.2) |
| `MANIFEST.in` | ✅ Complete | 76-line sdist file control (Audit §3.5) |
| `noxfile.py` | ✅ Complete | 5 nox sessions, multi-Python testing (Audit §3.8) |
| `.pre-commit-config.yaml` | ✅ Complete | 3 repos, 10 hooks; activates after `git init` (Audit §3.1) |
| `.readthedocs.yaml` | ✅ Complete | RTD v2 config (Audit §3.9) |
| `.github/workflows/release.yml` | ✅ Complete | Tag-triggered PyPI Trusted Publishers (Audit §3.4) |
| `.github/dependabot.yml` | ✅ Complete | pip + github-actions ecosystems, weekly (Audit §3.4) |
| `.github/ISSUE_TEMPLATE/` | ✅ Complete | YAML issue forms: bug_report.yml, feature_request.yml, config.yml (Audit §3.4) |
| `.github/PULL_REQUEST_TEMPLATE.md` | ✅ Complete | Checklist aligned with CONTRIBUTING.md (Audit §3.4) |
| `docs/` (Sphinx build system) | ✅ Complete | conf.py, index.md, getting-started.md, api/index.md, changelog.md, contributing.md, requirements.txt, _static/.gitkeep (Audit §3.9) |

---

## Part A: Pre-CI/CD Essential Files

These files must exist before the CI/CD workflow is created, because some are technically referenced by build tools or the CI pipeline itself.

---

### A1. `.gitignore` — ✅ COMPLETED (Audit §2.4)

**What**: Tells Git which files and directories to exclude from version control.

**Why first**: Must be in place before the first `git add` to prevent committing build artifacts, caches, and sensitive data. Every authoritative source places `.gitignore` at or before the first commit.

**Authoritative sources**:
- GitHub's official Python `.gitignore` template (`github/gitignore` repository)
- pytest documentation (excludes `.pytest_cache/`)
- Scientific Python Development Guide (`learn.scientific-python.org`)

**Technical dependencies**: None. This file is standalone.

**MILIA-specific considerations**:
- Must exclude conda environment directories (`shah_env/`, `.conda/`)
- Must exclude `test_data/` output artifacts (if any are generated), but NOT `test_data/` input fixtures
- Must exclude `*.npz` data files that are too large for Git (verify sizes; if your `test_data/` fixtures are small enough, they can stay)
- Must exclude `archive/` directory (documented in project structure as archived documentation)
- Must NOT exclude `.github/` (CI/CD workflows live there)

**Content requirements** (based on GitHub's Python template + MILIA specifics):

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/
*.whl

# Testing
.pytest_cache/
.coverage
coverage.xml
htmlcov/
.tox/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Conda / Virtual environments
shah_env/
.conda/
venv/
env/

# OS
.DS_Store
Thumbs.db

# Logs
*.log

# Jupyter
.ipynb_checkpoints/

# Data artifacts (keep test fixtures, exclude generated outputs)
# NOTE: Verify test_data/ fixture sizes before committing
```

---

### A2. `pyproject.toml` — ✅ COMPLETED (Audit §2.1) — Full PEP 621/639, Not Phased

**What**: The modern standard configuration file for Python projects. Required by PEP 518 (build system) and PEP 621 (project metadata). Also used for tool configuration (pytest, coverage, ruff, etc.).

**Why before CI/CD**: The CI workflow will invoke `pytest`, which reads its configuration from `pyproject.toml`. Without it, pytest configuration exists only in `conftest.py` (markers) — but `testpaths`, `addopts`, and other settings have no home.

**Authoritative sources**:
- PyPA Packaging User Guide (`packaging.python.org/en/latest/guides/writing-pyproject-toml/`): pyproject.toml is the standard for Python project configuration
- Scientific Python Development Guide (`learn.scientific-python.org`): "PY001 — Packages must have a pyproject.toml file"
- pyOpenSci Python Packaging Guide: pyproject.toml is "Package metadata and build configuration"

**Technical dependencies**: None for the minimal version. But if `[project]` table includes `readme = "README.md"`, then `README.md` must exist or `pip install -e .` / `python -m build` will fail.

**MILIA-specific critical decision — TWO PHASES**:

**Phase 1 (NOW — before CI/CD)**: Create `pyproject.toml` with ONLY tool configuration. Do NOT add `[build-system]` or `[project]` tables yet, because:
- `setup.py` already handles build/metadata (line 22 references `README.md` with a guard)
- Adding a `[project]` table with `readme = "README.md"` when `README.md` doesn't exist yet would break `pip install -e .`
- The `[build-system]` table would conflict with the existing `setup.py` if not configured correctly

**Phase 2 (LATER — after README.md and LICENSE exist)**: Add `[build-system]` and `[project]` tables, and eventually migrate away from `setup.py` entirely.

**Phase 1 content requirements**:

```toml
# pyproject.toml — Phase 1: Tool configuration only
# NOTE: [build-system] and [project] tables intentionally deferred
# until README.md and LICENSE exist (see Phase 2 in Release Guide)

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
]
# Markers also registered programmatically in conftest.py pytest_configure()
# This duplication is intentional: pyproject.toml enables IDE discovery,
# conftest.py enables runtime registration
markers = [
    "smoke: Quick health checks (< 30s total)",
    "contract: Interface contract validation",
    "e2e: End-to-end workflow tests",
    "regression: Regression protection tests",
    "thread_safety: Concurrent access tests",
    "slow: Tests taking > 30 seconds",
]

[tool.coverage.run]
source = ["milia_pipeline"]
omit = [
    "tests/*",
    "scripts/*",
    "examples/*",
    "archive/*",
]

[tool.coverage.report]
show_missing = true
fail_under = 70
```

**Evidence for dual marker registration** (conftest.py + pyproject.toml):
- pytest documentation (`docs.pytest.org/en/stable/example/markers.html`): Markers can be registered in either location; having both provides IDE support (pyproject.toml) and runtime guarantees (conftest.py).
- The `conftest.py` already registers markers (lines 522–543). Adding them to `pyproject.toml` is redundant but beneficial — IDEs like PyCharm and VS Code read `pyproject.toml` for marker autocompletion.

---

### A3. `LICENSE` — ✅ COMPLETED (Audit §2.2)

**What**: Legal document specifying how others can use, modify, and distribute your code.

**Why before CI/CD**: While `LICENSE` has zero technical impact on CI/CD execution, it has critical practical importance:
- GitHub automatically detects and displays the license at the top of the repository page (GitHub Docs: "Licensing a repository")
- pyOpenSci: "A license file is critical as it tells users how they legally can (and can't) use your package"
- For journal reviewers: Many journals require explicit licensing for software referenced in publications
- For job applications: Demonstrating awareness of intellectual property is expected

**Authoritative sources**:
- GitHub Docs (`docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository`): GitHub will "automagically discover" a LICENSE file at root
- pyOpenSci License Guide: "Your LICENSE file should be placed at the root of your package's repository"
- choosealicense.com (maintained by GitHub): License selection tool

**Technical dependencies**: None. The file is standalone.

**MILIA-specific license recommendation** (based on your goals):

Your `setup.py` (line 33) already declares `"License :: OSI Approved :: MIT License"` in classifiers. This means the project is already *intended* to be MIT-licensed, but the actual `LICENSE` file does not exist.

| License | Best For | Matches `setup.py`? |
|---------|----------|---------------------|
| **MIT** | Maximum permissiveness; widely used in academia; lets anyone use your code with attribution | ✅ Yes — already declared in `setup.py` classifiers |
| Apache-2.0 | Like MIT + explicit patent grant; preferred by enterprise/industry | ❌ Would conflict with `setup.py` classifier |
| BSD-3-Clause | Similar to MIT; common in scientific Python (NumPy, SciPy) | ❌ Would conflict with `setup.py` classifier |

**Recommendation**: Use MIT License to match your existing `setup.py` classifier. If you change the license, you MUST also update `setup.py` line 33.

---

### A4. `README.md` — ✅ COMPLETED (Audit §2.3)

**What**: The primary documentation file displayed on your GitHub repository landing page.

**Why before CI/CD**: 
1. **Direct technical dependency**: Your `setup.py` line 22 reads: `long_description=open('README.md').read() if os.path.exists('README.md') else ''`. Without `README.md`, the package metadata has empty `long_description`.
2. **CI badge placement**: The README is where CI status badges go. The badge URL depends on the workflow filename (e.g., `.github/workflows/ci.yml`), so the README should be created knowing the workflow path — but the actual badge can be added after CI is set up.
3. **First impression**: pyOpenSci states README.md is "often the first thing someone sees before installing package."

**Authoritative sources**:
- pyOpenSci README Best Practices (`pyopensci.org/python-package-guide/documentation/repository-files/readme-file-best-practices.html`): Provides a checklist of required README elements
- PyPA Packaging User Guide: "README.md is loaded into setup.py long_description"
- Scientific Python Development Guide: Lists README.md as essential file

**Technical dependencies**:
- `setup.py` line 22 references it (guarded — won't crash without it, but metadata is incomplete)
- `pyproject.toml` Phase 2 will reference it via `readme = "README.md"` (not yet — see A2)
- CI badge URLs depend on `.github/workflows/ci.yml` existing (create badge placeholder, fill in URL after CI is created)

**MILIA-specific README content requirements** (from pyOpenSci checklist):

The README should contain:

1. **Package name and badges** — CI status (placeholder until CI exists), Python version, license badge
2. **Description** (2–4 sentences) — What MILIA does (molecular data processing + ML pipeline for quantum chemistry)
3. **Context** — How MILIA fits into the broader ecosystem (PyTorch Geometric, RDKit, quantum chemistry ML)
4. **Key features** — 11 core modules, 400+ molecular descriptors, 10 dataset handlers, registry-based architecture
5. **Installation** — TWO methods must be documented, reflecting the actual development workflow:

   **Method 1: Docker (Primary — matches developer workflow)**:
   ```
   docker build -t milia .
   docker run -it milia
   # Inside container: (shah_env) root@...:/app/milia#
   ```
   
   **Method 2: Conda (without Docker)**:
   ```
   conda env create -f environment.yml
   conda activate shah_env
   pip install -e .
   ```
6. **Quick start code example** — Minimal working example
7. **Project structure** — Brief overview (can link to `MILIA_Pipeline_Project_Structure.md`)
8. **Testing** — How to run tests (`pytest -m smoke`, `pytest tests/`)
9. **Citation** — How to cite the project (DOI if available, or reference to planned publication)
10. **License** — MIT (link to LICENSE file)
11. **Authors** — Asadollah (Shahram) Boshra, Ilia Boshra (from `setup.py` lines 18–19)

**NOTE**: The README does NOT need to be complete at this stage. A minimal version with sections 1–6 and 10–11 is sufficient. Sections 7–9 can be expanded later.

---

### A5. `CHANGELOG.md` — ✅ COMPLETED (Audit §2.5)

**What**: Documents version history, notable changes, bug fixes.

**Why**: pyOpenSci includes it in the standard project structure. Demonstrates project maturity and maintenance history.

**Authoritative sources**:
- pyOpenSci Python Package Structure: Lists `CHANGELOG.md` in the standard layout
- Keep a Changelog (`keepachangelog.com`): Format standard (used by pyOpenSci's example)

**Technical dependencies**: None. Purely informational.

**MILIA-specific**: Start with the current version (1.0.0 from `setup.py` line 6) and note the major milestones (127 unit tests, 18 beyond-unit tests, CI/CD infrastructure, etc.).

---

### A6. `CONTRIBUTING.md` — ✅ COMPLETED (Audit §2.6)

**What**: Guidelines for how others can contribute to the project.

**Why**: pyOpenSci review checklist requires it. Demonstrates professional software engineering practice.

**Authoritative sources**:
- pyOpenSci Author Guide: Checklist includes "The package has a `CONTRIBUTING.md` file that details how to install and contribute to the package"

**Technical dependencies**: None.

**MILIA-specific**: Since this is a private repository, the CONTRIBUTING.md can be minimal — focused on development setup (conda environment, running tests, code style) rather than community contribution workflows.

---

### A7. `CODE_OF_CONDUCT.md` — ✅ COMPLETED (Audit §2.7)

**What**: Behavioral guidelines for project interactions.

**Why**: pyOpenSci review checklist requires it.

**Authoritative sources**:
- pyOpenSci: "The package has a `CODE_OF_CONDUCT.md` file"
- Contributor Covenant (`contributor-covenant.org`): Standard template used across the scientific Python ecosystem

**Technical dependencies**: None.

---

## Part B: CI/CD Infrastructure Files

These files implement the continuous integration pipeline. They depend on certain Part A files existing.

---

### B1. GitHub Actions CI/CD Workflows

The MILIA project requires **two CI/CD workflows**, because it has two distinct automation goals:

1. **Build and publish the Docker image to GHCR** — so reviewers/users can `docker pull` and run immediately
2. **Run the test suite** — to validate code correctness on every push/PR

These are separate concerns with different triggers, and authoritative sources recommend separating them (GitHub Docs — "Publishing and installing a package with GitHub Actions").

---

#### B1a. `.github/workflows/docker-publish.yml` — Build & Push Docker Image to GHCR

**What**: Automatically builds the Docker image from your `Dockerfile` and pushes it to GitHub Container Registry (GHCR) so that anyone with repository access can pull and run it.

**Why this is REQUIRED** (not optional): A reviewer or anyone evaluating the MILIA project on GitHub needs to be able to run the software immediately. Without a prebuilt image on GHCR, they would need to: (1) clone the repo, (2) install Docker, (3) run `docker build` (which takes 20–35 minutes due to MILIA's 12-layer Dockerfile with retry logic), (4) wait for all mamba installs to complete. Publishing to GHCR eliminates steps 2–4 — the reviewer simply runs `docker pull` + `docker run`.

**Authoritative sources**:
- GitHub Docs — "Publishing and installing a package with GitHub Actions" (`docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions`): Official guide for automated Docker image publishing to GHCR. Uses `docker/login-action`, `docker/metadata-action`, and `docker/build-push-action`.
- GitHub Docs — "Working with the Container registry" (`docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry`): GHCR authentication, tagging, and visibility management. States: "When you first publish a package, the default visibility is private."
- GitHub Docs — "Publishing and installing a package with GitHub Actions": Shows the official workflow pattern using `GITHUB_TOKEN` (no PAT needed for same-repo workflows).
- jorgep.com ("Moving to GitHub Container Registry"): "GHCR offers unlimited private repositories for your container images" and "You can use the built-in GITHUB_TOKEN to authenticate your workflows."

**GHCR pricing** (verified from GitHub Docs + community discussions):
- GitHub Free plan: 500 MB storage, 1 GB data transfer/month included
- GitHub Pro plan: 2 GB storage, 10 GB data transfer/month included
- Additional storage: $0.25/GB/month; Additional transfer: $0.50/GB
- **Note from GitHub Docs**: "Container image storage and bandwidth for the Container registry is currently free" — but GitHub has stated this may change with one month notice
- For a private repo with occasional reviewer access, the free tier is sufficient

**Technical dependencies**:
- `Dockerfile` — MUST exist (the workflow builds from it)
- Repository permissions — workflow needs `packages: write` permission

**When to trigger**: Only when the Docker image content changes — i.e., when `Dockerfile`, `environment.yml`, or source code changes on the main branch. NOT on every PR (that would waste build minutes).

**Workflow specification** (based on GitHub's official pattern):

```yaml
# .github/workflows/docker-publish.yml
name: Build and Publish Docker Image

on:
  push:
    branches: [main]
    # Only rebuild when image-affecting files change
    paths:
      - 'Dockerfile'
      - 'environment.yml'
      - 'milia_pipeline/**'
      - 'setup.py'
  # Allow manual trigger for first-time build
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest
            type=sha,prefix=

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

**After the first push, reviewers can run**:
```bash
docker pull ghcr.io/YOUR_USERNAME/milia-pipeline:latest
docker run -it ghcr.io/YOUR_USERNAME/milia-pipeline:latest
# → (shah_env) root@...:/app#
```

**IMPORTANT**: After the first image is pushed, you must configure the package visibility. By default, GHCR packages are **private**. Per GitHub Docs: "When you first publish a package, the default visibility is private. To change the visibility or set access permissions, see Configuring a package's access control and visibility." For a private repository shared with reviewers, the package should remain private but linked to the repository so that repository collaborators automatically get pull access.

---

#### B1b. `.github/workflows/ci.yml` — ✅ COMPLETED (Audit §3.4)

**What**: Runs the 146-test suite automatically on every push and pull request to validate code correctness.

**Why separate from Docker build**: Tests should run on every push and every PR (fast feedback). Docker image builds should only run on main branch when image-affecting files change (expensive operation). Combining them would either slow down the test feedback loop or trigger unnecessary image rebuilds.

**CI/CD approach for tests**: There are two valid approaches for running tests. The choice depends on whether you want the test environment to match your Docker image exactly:

**Option A: Conda from `environment.yml` (Simpler, faster feedback)**

Uses `conda-incubator/setup-miniconda` to create the environment from `environment.yml` directly on the GitHub runner. ~99% environment parity with Docker (same conda packages, different base OS — negligible difference for pure Python/scientific computing).

**Evidence**: Keng's Blog (SLM Lab project): "Step 1 of building and using a custom Docker image is no longer necessary since we can use the Github ubuntu image directly."

```yaml
# .github/workflows/ci.yml
name: MILIA CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

defaults:
  run:
    shell: bash -l {0}

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Setup Miniconda
        uses: conda-incubator/setup-miniconda@v3
        with:
          miniforge-version: latest
          activate-environment: shah_env
          environment-file: environment.yml

      - name: Install package (editable)
        run: pip install -e .

      - name: Stage 1 — Smoke tests
        run: pytest -m smoke --tb=short

      - name: Stage 2 — Contract + Config tests
        run: >-
          pytest
          tests/test_contract_*.py
          tests/test_config_validation_*.py
          tests/test_config_split_*.py
          -v --tb=short

      - name: Stage 3 — Unit tests
        run: pytest tests/test_*_unit.py -v --tb=short

      - name: Stage 4 — Integration + E2E + Regression + Thread Safety
        run: >-
          pytest
          tests/test_e2e_*.py
          tests/test_regression_*.py
          tests/test_*_integration*.py
          tests/test_thread_safety_*.py
          -v --tb=short

      - name: Stage 5 — Coverage report
        run: |
          pip install coverage
          coverage run -m pytest tests/ --tb=short
          coverage report --fail-under=70
```

**Option B: Run tests inside Docker container (100% parity)**

Builds the Docker image (or pulls the prebuilt one from GHCR) and runs `pytest` inside it. Guarantees tests run in exactly the same environment as the user will experience.

```yaml
# .github/workflows/ci.yml (Docker-based variant)
name: MILIA CI Pipeline

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t milia-test .

      - name: Run test suite inside container
        run: >-
          docker run --rm milia-test
          /bin/bash -c "source /opt/conda/etc/profile.d/conda.sh &&
          conda activate shah_env &&
          cd /app && pip install -e . &&
          pytest tests/ -v --tb=short"
```

**Trade-off**: Option A gives faster CI feedback (~8–15 min). Option B gives 100% environment parity but is slower (~20–35 min due to `docker build`). For a project where the Docker image is already published to GHCR via B1a, Option A is recommended for CI — the Docker image correctness is validated separately by the B1a workflow.

**NOTE**: The exact test file glob patterns must be verified against the actual test file names in `tests/` and adjusted as needed.

**Technical dependencies (both options)**:
- `environment.yml` — MUST exist (Option A: conda env creation; Option B: used by Dockerfile)
- `pyproject.toml` — SHOULD exist (pytest reads `testpaths` and `addopts`)
- `setup.py` — MUST exist (`pip install -e .` makes the package importable)
- `Dockerfile` — MUST exist (Option B only)
- `.gitignore` — SHOULD exist (prevents committing CI artifacts)

---

### B2. `setup.py` Updates — ✅ COMPLETED (Audit §2.1) — Reduced to Backward-Compatibility Shim

**What**: The existing `setup.py` needs specific updates to work correctly with CI/CD.

**Why**: Two issues identified by line-by-line analysis:

1. **Line 22**: `long_description=open('README.md').read() if os.path.exists('README.md') else ''` — Currently returns empty string. After `README.md` is created (A4), this will automatically work. No code change needed.

2. **Line 24**: `url="https://github.com/yourusername/milia-pipeline"` — This is a placeholder URL. Must be updated to the actual GitHub repository URL once created.

3. **Lines 12–14**: `extras_require={'dev': []}` — The `MILIA_Test_Recommendations.md` CI pipeline uses `pip install -e ".[test]"`. Since the project uses conda for dependencies, this extra is intentionally empty. However, a `[test]` extra should be added containing ONLY the test-runner packages (pytest and its plugins are already in `environment.yml`, so this may remain empty — but documenting the key explicitly prevents confusion):

```python
extras_require={
    'dev': [],
    'test': [],  # All test dependencies managed by conda via environment.yml
},
```

**Technical dependencies**: Depends on the actual GitHub repository URL being known.

---

## Part C: Recommended (Not Required) Files

These files add professional polish but have no technical impact on CI/CD.

---

### C1. `docs/` Directory — ✅ COMPLETED (Audit §3.9) — Sphinx Build System with RTD Config

**What**: Project documentation beyond the README.

**Why**: pyOpenSci lists it in the standard structure. For a project of MILIA's scale (11 modules), having structured documentation demonstrates thoroughness.

**Status**: Already listed in the project structure (`milia/docs/`). Content TBD.

---

### C2. `examples/` Directory — ✅ PLACEHOLDER (Audit §1.1) — `.gitkeep` Created, Content TBD

**What**: Working usage examples.

**Why**: pyOpenSci README checklist recommends "short tutorials that demonstrate application of your package."

**Status**: Already listed in the project structure (`milia/examples/`). Content TBD.

---

## Implementation Order

The following order respects all technical dependencies and maximizes efficiency:

```
Phase 1: Pre-CI/CD Foundation — ✅ ALL COMPLETE (Audit §2)
┌─────────────────────────────────────────────────────┐
│ Step 1: .gitignore          [A1] — ✅ Audit §2.4    │
│ Step 2: LICENSE             [A3] — ✅ Audit §2.2    │
│ Step 3: pyproject.toml      [A2] — ✅ Audit §2.1    │
│         (FULL PEP 621/639, not phased)               │
│ Step 4: README.md           [A4] — ✅ Audit §2.3    │
│ Step 5: setup.py            [B2] — ✅ Audit §2.1    │
│         (reduced to shim, metadata in pyproject.toml)│
└─────────────────────────────────────────────────────┘
         │
         ▼
Phase 2: CI/CD Infrastructure — ✅ ALL COMPLETE
┌─────────────────────────────────────────────────────┐
│ Step 6: .github/workflows/docker-publish.yml [B1a]  │
│         ✅ Spec complete — execute §5a to create     │
│ Step 7: .github/workflows/ci.yml  [B1b]             │
│         ✅ Audit §3.4 — lint + test matrix           │
│ Step 8: .github/workflows/release.yml                │
│         ✅ Audit §3.4 — tag-triggered PyPI publish   │
└─────────────────────────────────────────────────────┘
         │
         ▼
Phase 3: Professional Polish — ✅ ALL COMPLETE (Audit §2–§3)
┌─────────────────────────────────────────────────────┐
│ Step 9:  CHANGELOG.md       [A5] — ✅ Audit §2.5   │
│ Step 10: CONTRIBUTING.md    [A6] — ✅ Audit §2.6   │
│ Step 11: CODE_OF_CONDUCT.md [A7] — ✅ Audit §2.7   │
│ Step 12: pyproject.toml (full)   — ✅ Audit §2.1   │
│ Step 13: examples/ (.gitkeep)    — ✅ Audit §1.1   │
│ Step 14: docs/ (Sphinx)          — ✅ Audit §3.9   │
└─────────────────────────────────────────────────────┘

Additional files created by Audit (not in original Guide):
  CITATION.cff (§2.8), SECURITY.md (§3.3), RELEASE_CHECKLIST.md (§3.7),
  Makefile (§3.2), MANIFEST.in (§3.5), noxfile.py (§3.8),
  .pre-commit-config.yaml (§3.1), .readthedocs.yaml (§3.9),
  release.yml (§3.4), dependabot.yml (§3.4), YAML issue forms (§3.4),
  PR template (§3.4)
```

---

## Dependency Graph

```
✅ .gitignore ──────────────────────────────────── COMPLETE (Audit §2.4)
✅ LICENSE ─────────────────────────────────────── COMPLETE (Audit §2.2)
✅ pyproject.toml (full PEP 621/639) ──────────── COMPLETE (Audit §2.1)
✅ README.md ───────────────────────────────────── COMPLETE (Audit §2.3)
✅ setup.py (reduced to shim) ─────────────────── COMPLETE (Audit §2.1)

✅ Dockerfile ─────────────────────────────────── EXISTS
    ├── Used by docker-publish.yml to build & push to GHCR
    ├── Used locally by developer (docker build + docker run)
    └── Optionally used by ci.yml (Option B: Docker-based tests)

✅ .github/workflows/docker-publish.yml ────────── SPEC COMPLETE — execute §5a (depends on:)
    ├── Dockerfile          ← EXISTS
    ├── environment.yml     ← EXISTS
    ├── milia_pipeline/     ← EXISTS
    └── setup.py            ← EXISTS (shim)

✅ .github/workflows/ci.yml ───────────────────── COMPLETE (Audit §3.4)
✅ .github/workflows/release.yml ──────────────── COMPLETE (Audit §3.4)
```

---

## Summary: Complete File Inventory for GitHub Release

| # | File | Category | Priority | Status |
|---|------|----------|----------|--------|
| 1 | `.gitignore` | Foundation | **Critical** | ✅ Complete (Audit §2.4) |
| 2 | `LICENSE` | Foundation | **Critical** | ✅ Complete (Audit §2.2) |
| 3 | `pyproject.toml` | Foundation | **Critical** | ✅ Complete — full PEP 621/639 (Audit §2.1) |
| 4 | `README.md` | Foundation | **Critical** | ✅ Complete (Audit §2.3) |
| 5 | `setup.py` | Foundation | **High** | ✅ Reduced to shim (Audit §2.1) |
| 6 | `.github/workflows/docker-publish.yml` | CI/CD | **Critical** | ✅ Spec complete (B1a) — creation instructions in §5a |
| 7 | `.github/workflows/ci.yml` | CI/CD | **Critical** | ✅ Complete (Audit §3.4) |
| 8 | `CHANGELOG.md` | Polish | Medium | ✅ Complete (Audit §2.5) |
| 9 | `CONTRIBUTING.md` | Polish | Medium | ✅ Complete (Audit §2.6) |
| 10 | `CODE_OF_CONDUCT.md` | Polish | Low | ✅ Complete (Audit §2.7) |
| 11 | `CITATION.cff` | Polish | Medium | ✅ Complete (Audit §2.8) |
| 12 | `SECURITY.md` | Polish | Medium | ✅ Complete (Audit §3.3) |
| 13 | `RELEASE_CHECKLIST.md` | Polish | Medium | ✅ Complete (Audit §3.7) |
| 14 | `Makefile` | Developer | Medium | ✅ Complete (Audit §3.2) |
| 15 | `MANIFEST.in` | Packaging | Medium | ✅ Complete (Audit §3.5) |
| 16 | `noxfile.py` | Testing | Medium | ✅ Complete (Audit §3.8) |
| 17 | `.pre-commit-config.yaml` | Quality | Medium | ✅ Complete (Audit §3.1) |
| 18 | `.readthedocs.yaml` | Docs | Medium | ✅ Complete (Audit §3.9) |
| 19 | `.github/workflows/release.yml` | CI/CD | Medium | ✅ Complete (Audit §3.4) |
| 20 | `.github/dependabot.yml` | CI/CD | Low | ✅ Complete (Audit §3.4) |
| 21 | `.github/ISSUE_TEMPLATE/` (3 files) | CI/CD | Low | ✅ Complete (Audit §3.4) |
| 22 | `.github/PULL_REQUEST_TEMPLATE.md` | CI/CD | Low | ✅ Complete (Audit §3.4) |
| 23 | `docs/` (Sphinx build system, 8 files) | Docs | Low | ✅ Complete (Audit §3.9) |
| 24 | `examples/` | Polish | Low | ✅ Placeholder `.gitkeep` (Audit §1.1) |

**Remaining**: Item #6 (`docker-publish.yml`) spec is complete — execute creation instructions in step 5a of "What Actually Remains" section, then follow steps 5b–5f for GHCR configuration and reviewer access.

---

## Authoritative Sources Referenced

| Source | Authority | URL |
|--------|-----------|-----|
| PyPA Packaging User Guide | Python Packaging Authority (official) | `packaging.python.org` |
| pyOpenSci Python Packaging Guide | Community standard for scientific Python | `pyopensci.org/python-package-guide` |
| Scientific Python Development Guide | Scientific Python community standard | `learn.scientific-python.org/development` |
| GitHub Docs — Licensing | GitHub (official) | `docs.github.com` |
| GitHub Docs — Running jobs in a container | GitHub (official) | `docs.github.com/en/actions/writing-workflows/choosing-where-your-workflow-runs/running-jobs-in-a-container` |
| GitHub Docs — Publishing and installing a package with GitHub Actions | GitHub (official) | `docs.github.com/en/packages/managing-github-packages-using-github-actions-workflows/publishing-and-installing-a-package-with-github-actions` |
| GitHub Docs — Working with the Container registry (GHCR) | GitHub (official) | `docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry` |
| GitHub Docs — GitHub Packages billing | GitHub (official) | `docs.github.com/en/billing/concepts/product-billing/github-packages` |
| Docker Docs — GitHub Actions CI/CD | Docker (official) | `docs.docker.com/language/python/configure-ci-cd/` |
| `conda-incubator/setup-miniconda` | Official conda CI action | `github.com/conda-incubator/setup-miniconda` |
| `mamba-org/setup-micromamba` | Official mamba CI action | `github.com/mamba-org/setup-micromamba` |
| pytest Documentation | Official | `docs.pytest.org` |
| Keep a Changelog | Community standard | `keepachangelog.com` |
| Contributor Covenant | Community standard | `contributor-covenant.org` |
| Keng's Blog — GitHub Actions for Python | Real-world conda CI migration | `kengz.gitbook.io/blog/engineering/github-actions-for-python-project` |
| GitHub Docs — Inviting collaborators to a personal repository | GitHub (official) | `docs.github.com/articles/inviting-collaborators-to-a-personal-repository` |
| GitHub Docs — Permission levels for a personal account repository | GitHub (official) | `docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository` |
| GitHub Docs — Configuring a package's access control and visibility | GitHub (official) | `docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility` |
| GitHub Docs — Connecting a repository to a package | GitHub (official) | `docs.github.com/en/packages/learn-github-packages/connecting-a-repository-to-a-package` |
| GitHub Docs — About permissions for GitHub Packages | GitHub (official) | `docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages` |
| `docker/metadata-action` README | Docker (official) | `github.com/docker/metadata-action` |
| `docker/build-push-action` README | Docker (official) | `github.com/docker/build-push-action` |

---

## Files Required From You Before Implementation — ✅ ALL RESOLVED

All questions originally listed here have been resolved during the Production Release File Audit implementation:

1. **GitHub repository URL** — ✅ Resolved: `https://github.com/shahram-boshra/MILIA` (Audit §2.1, line 79)
2. **License confirmation** — ✅ Resolved: MIT (Audit §2.2, line 90)
3. **Author information for LICENSE** — ✅ Resolved: "2026-present Asadollah Boshra" (Audit §2.2, line 90)
4. **Current `.gitignore`** — ✅ Resolved: Fresh GitHub Python template + MILIA-specific exclusions created (Audit §2.4)
5. **Current `tests/` file listing** — ✅ Resolved: 127 test files verified; CI workflow glob patterns set (Audit §3.4)

---

## What Actually Remains — Post-Creation Actions

All file creation work described in this guide and the Production Release File Audit has been completed. The audit identifies several items that cannot be completed until Git is initialized and the repository is pushed to GitHub. These are the true remaining action items:

### 1. Initialize Git & Make the Initial Commit (Audit §7B)

This is the foundational step. The audit explicitly states (line 538): *"Adopting Git eliminates `_legacy/`, unlocks P2/P3 items, and is the single highest-impact infrastructure improvement."*

```bash
cd /path/to/milia
git init
git add .
git commit -m "feat: initial production release v1.1.0

Complete MILIA pipeline with all P0/P1/P2/P3 production infrastructure:
- pyproject.toml (PEP 621/639), LICENSE (MIT), README.md, CHANGELOG.md
- CITATION.cff (CFF 1.2.0), CONTRIBUTING.md, CODE_OF_CONDUCT.md (CC 3.0)
- SECURITY.md, RELEASE_CHECKLIST.md, Makefile (20 targets), noxfile.py
- .pre-commit-config.yaml, .github/ (CI/CD, issue forms, Dependabot)
- MANIFEST.in, .readthedocs.yaml, docs/ (Sphinx build system)
- .gitignore (GitHub Python template + MILIA exclusions)
- 11 core modules, 127 tests, configs/"

git status
git log --oneline
```

### 2. Activate Pre-commit Hooks (Audit §3.1 — "activates after git init")

```bash
pip install pre-commit        # or: conda install pre-commit
pre-commit install            # install hooks into .git/hooks/
pre-commit run --all-files    # run all hooks on entire codebase
```

### 3. Push to GitHub and Complete One-Time Post-Push Setup (Audit §3.4, §7C)

```bash
git remote add origin https://github.com/shahram-boshra/MILIA.git
git branch -M main
git push -u origin main
```

Then three manual GitHub settings:

1. **Enable Private Vulnerability Reporting** (Audit §7C, lines 548–553): Navigate to `https://github.com/shahram-boshra/MILIA` → **Settings** → **Security** → **Code security and analysis** → **Private vulnerability reporting** → **Enable**. Required for `SECURITY.md` to function — without it the "Report a vulnerability" button will not appear. **Verification**: Visit `https://github.com/shahram-boshra/MILIA/security` — the button should be visible.

2. **Configure PyPI Trusted Publisher** (Audit §3.4, line 254): At `pypi.org/manage/project/milia/settings/publishing/` — add GitHub Actions publisher (owner: `shahram-boshra`, repo: `MILIA`, workflow: `release.yml`, environment: `pypi`).

3. **Create `pypi` GitHub Environment** (Audit §3.4, line 255): In repo **Settings** → **Environments** → create `pypi` environment (optionally require manual approval).

### 4. Connect to Read the Docs (Audit §3.9, lines 345–347)

After push, connect repository at `readthedocs.org/dashboard/import/`. RTD will auto-detect `.readthedocs.yaml` and build docs on each push to `main`.

### 5. Create `docker-publish.yml`, Configure GHCR, and Grant Reviewer/Interviewer Access

**Why this step exists**: Pushing source code to a private GitHub repository is **necessary but not sufficient** for journal reviewers and job interviewers to evaluate MILIA. Without a prebuilt Docker image on GHCR, an evaluator would need to clone the repo, install Docker, run `docker build` (20–35 minutes for MILIA's 12-layer Dockerfile with retry logic), and wait for all mamba installs to complete. This step eliminates that friction entirely — the evaluator runs two commands and has MILIA operational.

**This is the single remaining blocker**: 23 of 24 release items are complete (see Summary Table). Only item #6 (`docker-publish.yml`) has not been created.

**What this step covers (in order)**:

1. Create the `docker-publish.yml` workflow file
2. Trigger the first image build
3. Configure GHCR package → repository linking and inherited permissions
4. Invite the reviewer/interviewer as a collaborator
5. Verify end-to-end reviewer access

---

#### 5a. Create `.github/workflows/docker-publish.yml`

The workflow specification is defined in section B1a of this guide (lines 362–416). Create this file **exactly as specified** — it uses `docker/login-action@v3`, `docker/metadata-action@v5`, and `docker/build-push-action@v5`, which are the official Docker GitHub Actions maintained by Docker, Inc.

**Critical technical detail — automatic repository linking**: The `docker/metadata-action@v5` automatically generates an `org.opencontainers.image.source` label pointing to `https://github.com/shahram-boshra/MILIA` (source: `docker/metadata-action` README — the action extracts this from the GitHub context). Combined with the fact that the workflow authenticates via `GITHUB_TOKEN`, this means the GHCR package is **automatically linked to the repository** on first push. Per GitHub Docs ("Working with the Container registry"): *"The easiest way to connect a repository to a container package is to publish the package from a workflow using `${{secrets.GITHUB_TOKEN}}`, as the repository that contains the workflow is linked automatically."*

**Authoritative sources**:
- GitHub Docs — "Working with the Container registry" (`docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry`): Automatic repository linking via `GITHUB_TOKEN`
- `docker/metadata-action` README (`github.com/docker/metadata-action`): Automatically generates `org.opencontainers.image.source` label from GitHub context
- GitHub Docs — "Connecting a repository to a package" (`docs.github.com/en/packages/learn-github-packages/connecting-a-repository-to-a-package`): Repository linking and permission inheritance

**File creation command** (the YAML content is already specified in section B1a, lines 362–416):

```bash
mkdir -p .github/workflows

cat > .github/workflows/docker-publish.yml << 'EOF'
# .github/workflows/docker-publish.yml
name: Build and Publish Docker Image

on:
  push:
    branches: [main]
    # Only rebuild when image-affecting files change
    paths:
      - 'Dockerfile'
      - 'environment.yml'
      - 'milia_pipeline/**'
      - 'setup.py'
      - 'pyproject.toml'
  # Allow manual trigger for first-time build
  workflow_dispatch:

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}

      - name: Extract Docker metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest
            type=sha,prefix=

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
EOF
```

**Commit the workflow file**:

```bash
git add .github/workflows/docker-publish.yml
git commit -m "ci: add docker-publish.yml — build and push Docker image to GHCR

Automated Docker image build and push to GitHub Container Registry (GHCR)
on push to main when image-affecting files change (Dockerfile,
environment.yml, milia_pipeline/, setup.py, pyproject.toml).

Uses official Docker GitHub Actions:
- docker/login-action@v3 (GHCR authentication via GITHUB_TOKEN)
- docker/metadata-action@v5 (automatic OCI labels including
  org.opencontainers.image.source for repository linking)
- docker/build-push-action@v5 (build and push)

Enables reviewers and evaluators to run MILIA via:
  docker pull ghcr.io/shahram-boshra/milia:latest
  docker run -it ghcr.io/shahram-boshra/milia:latest

Completes item #6 in Release Guide Summary Table (B1a).
Manual trigger (workflow_dispatch) available for first-time build.

Sources: GitHub Docs (GHCR, Publishing packages with GitHub Actions),
docker/metadata-action README, docker/build-push-action README"
```

**Push the commit**:

```bash
git push origin main
```

**NOTE on image name casing**: Docker requires image names to be **all lowercase**. `docker/metadata-action@v5` handles the lowercasing of `${{ github.repository }}` automatically. Since the MILIA repository is `shahram-boshra/MILIA` (uppercase `MILIA`), the resulting GHCR image name will be `ghcr.io/shahram-boshra/milia` (lowercase). No manual intervention required — this is handled by the action, not hardcoded.

---

#### 5b. Push the Pre-Built Local Docker Image to GHCR (Primary Path — Fast)

You already have a fully built, tested Docker image on your local machine. Pushing this pre-built image directly to GHCR is **significantly faster** than triggering a fresh build via GitHub Actions (minutes of upload vs. 20–35 minutes of rebuild). This is the recommended primary path for the initial GHCR publication.

The `docker-publish.yml` workflow created in step 5a remains in place for **automated future rebuilds** — whenever `Dockerfile`, `environment.yml`, `milia_pipeline/**`, `setup.py`, or `pyproject.toml` changes on `main`, the image is rebuilt and pushed automatically. But the first image should come from your local machine.

**Authoritative sources**:
- GitHub Docs — "Working with the Container registry" (`docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry`): Manual push procedure, PAT authentication, `LABEL` requirements for repository linking
- GitHub Docs — "Managing your personal access tokens" (`docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens`): PAT creation with `write:packages` scope

---

**Prerequisite — Ensure Dockerfile Has the `org.opencontainers.image.source` Label**:

**This is critical**: Per GitHub Docs ("Working with the Container registry"): *"When you push a container image from the command line, the image is not linked to a repository by default. This is the case even if you tag the image with a namespace that matches the name of the repository."* The automatic repository linking that `GITHUB_TOKEN` provides in workflows does **not** occur on command-line pushes.

To ensure the image is linked to the MILIA repository on push, your `Dockerfile` **must** contain the following `LABEL` instructions (per GitHub Docs — "Labelling container images"):

```dockerfile
LABEL org.opencontainers.image.source=https://github.com/shahram-boshra/MILIA
LABEL org.opencontainers.image.description="MILIA: Machine Intelligent Learning Interface Assistant — molecular data processing and ML pipeline"
LABEL org.opencontainers.image.licenses=MIT
```

**Where to add these labels**: Place them near the top of the `Dockerfile`, after the first `FROM` instruction. Per Docker documentation ("LABEL"), `LABEL` instructions add metadata to the image and do not affect the build layers in any meaningful way.

**Check if your Dockerfile already has these labels**:

```bash
grep -n "org.opencontainers" Dockerfile
```

If the labels are **not** present, add them and rebuild:

```bash
# Add the labels to Dockerfile (after the first FROM line)
# Then rebuild the image to include the new labels
docker build -t milia .

git add Dockerfile
git commit -m "build: add OCI metadata labels to Dockerfile for GHCR repository linking

Add org.opencontainers.image.source, .description, and .licenses labels
per GitHub Docs ('Working with the Container registry' — 'Labelling
container images'). Required for automatic repository linking when
pushing to GHCR from the command line.

Sources: GitHub Docs (Working with the Container registry),
OCI Image Spec (Pre-Defined Annotation Keys)"
```

If the labels are **already present**, no rebuild is needed — proceed directly to the push steps below.

---

**Step-by-step: Push the local image to GHCR**:

```bash
# 1. Create a GitHub Personal Access Token (classic) with write:packages scope
#    → https://github.com/settings/tokens/new
#    → Select scopes: write:packages (which auto-selects read:packages)
#    → Generate token → copy the token value
#
#    NOTE: Store this token securely. You will also need it if you ever
#    need to push updated images from your local machine in the future.

# 2. Authenticate to GHCR from your local machine
echo YOUR_PAT | docker login ghcr.io -u shahram-boshra --password-stdin
# Expected output: Login Succeeded

# 3. Tag the local image for GHCR
#    Find your local MILIA image name/ID:
docker images | grep -i milia
#    Tag it for GHCR (image name MUST be lowercase):
docker tag milia ghcr.io/shahram-boshra/milia:latest

# 4. Push the tagged image to GHCR
docker push ghcr.io/shahram-boshra/milia:latest
# This uploads the image layers — time depends on image size and upload speed.
# For a conda-based ML image (~2–5 GB), expect 5–20 minutes on a typical connection.

# 5. Verify the image is on GHCR
docker pull ghcr.io/shahram-boshra/milia:latest
# Should complete almost instantly (image already cached locally)
```

**After push**: The image will be available at `ghcr.io/shahram-boshra/milia:latest`. Because the Dockerfile contains the `org.opencontainers.image.source` label, the package will be **automatically linked** to the `shahram-boshra/MILIA` repository and will appear in the repository's **Packages** sidebar.

**If the package does not auto-link** (edge case — see GitHub community discussion #163656 confirming this can happen with command-line pushes even with the label), link it manually via the GitHub UI as described in step 5c below.

---

**Fallback — Trigger a Fresh Build via GitHub Actions (if needed)**:

If you need to rebuild the image from scratch on GitHub's infrastructure (e.g., your local image is outdated, or you want a clean build from the current `main` branch), use the `workflow_dispatch` trigger:

1. Navigate to `https://github.com/shahram-boshra/MILIA/actions/workflows/docker-publish.yml`
2. Click **"Run workflow"** → select branch `main` → click **"Run workflow"**

Or via GitHub CLI:

```bash
gh workflow run docker-publish.yml --ref main
```

This triggers a full `docker build` on GitHub Actions (20–35 minutes for MILIA's 12-layer Dockerfile).

---

#### 5c. Verify GHCR Package → Repository Linking and Configure Inherited Permissions

**If you pushed from your local machine** (step 5b primary path): The `org.opencontainers.image.source` label in your Dockerfile tells GHCR which repository the image belongs to. However, per GitHub Docs ("Working with the Container registry"): command-line pushes do **not** always auto-link even with the label present. You must verify and, if necessary, link manually.

**If you pushed via `workflow_dispatch`** (step 5b fallback path): The workflow uses `GITHUB_TOKEN` for authentication, which means the GHCR package is **automatically linked** to the `shahram-boshra/MILIA` repository (per GitHub Docs: *"The easiest way to connect a repository to a container package is to publish the package from a workflow using `${{secrets.GITHUB_TOKEN}}`, as the repository that contains the workflow is linked automatically."*)

**Verify the automatic link**:

1. Navigate to `https://github.com/shahram-boshra/MILIA` → right sidebar → **"Packages"** section → click the `milia` package
2. The package page should display repository information (README excerpt, link back to the repository)
3. If the package does **not** appear under the repository's Packages section, link it manually:
   - Navigate to `https://github.com/shahram-boshra?tab=packages` → click `milia` → **"Package settings"** (right sidebar) → **"Connect repository"** → select `MILIA` → click **"Connect repository"**

**Enable inherited permissions** (this is the critical step that grants collaborators access to the Docker image):

1. Navigate to the package settings: `https://github.com/shahram-boshra?tab=packages` → click `milia` → **"Package settings"**
2. Under **"Manage access"**, verify that **"Inherit access from source repository (recommended)"** is checked
3. If it is not checked, enable it

**What "Inherit access" does**: Per GitHub Docs ("Configuring a package's access control and visibility"): when a package is linked to a repository and inherits access, anyone who is a collaborator on the repository **automatically** gets the corresponding access to the GHCR package. This means adding someone as a repository collaborator (step 5d) grants them both code access **and** Docker image pull access in a single action.

**Authoritative sources**:
- GitHub Docs — "Configuring a package's access control and visibility" (`docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility`): Inherited permissions from linked repository
- GitHub Docs — "Connecting a repository to a package" (`docs.github.com/en/packages/learn-github-packages/connecting-a-repository-to-a-package`): Automatic linking via `GITHUB_TOKEN` + manual fallback
- GitHub Docs — "About permissions for GitHub Packages" (`docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages`): Granular permissions for container registry
- GitHub Docs — "Allowing your codespace to access a private registry" (`docs.github.com/en/codespaces/reference/allowing-your-codespace-to-access-a-private-registry`): "Inherit access from repo is selected by default when publishing via GitHub Actions"

**Keep the package visibility PRIVATE**: The MILIA repository is private. The GHCR package should also remain private. Do **not** change visibility to public — per GitHub Docs, once a package is made public, it **cannot** be made private again.

---

#### 5d. Invite the Reviewer/Interviewer as a Repository Collaborator

Adding a collaborator to the private repository grants them access to **both** the source code **and** the GHCR Docker image (via inherited permissions configured in step 5c).

**Authoritative sources**:
- GitHub Docs — "Inviting collaborators to a personal repository" (`docs.github.com/articles/inviting-collaborators-to-a-personal-repository`): Official invitation procedure
- GitHub Docs — "Permission levels for a personal account repository" (`docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/repository-access-and-collaboration/permission-levels-for-a-personal-account-repository`): "In a private repository, repository owners can only grant write access to collaborators. Collaborators can't have read-only access to repositories owned by a personal account."

**Procedure**:

1. Navigate to `https://github.com/shahram-boshra/MILIA/settings/access`
2. Click **"Add people"**
3. Search by the reviewer/interviewer's **GitHub username** or **email address**
4. Click **"Add [NAME] to MILIA"**
5. The invitee receives an email notification and a GitHub notification — they must **accept** the invitation before access is granted

**Important constraint for personal accounts**: Per GitHub Docs ("Permission levels for a personal account repository"), collaborators on a private repository owned by a personal account receive **read/write** access. There is no read-only option for personal accounts. If read-only access is required, the repository would need to be transferred to a **GitHub Organization** (out of scope for this guide, but noted for awareness).

**After the invitee accepts**, they have:
- Full read/write access to the repository (code, issues, PRs, wiki)
- Pull access to the GHCR Docker image (via inherited permissions)

---

#### 5e. What the Reviewer/Interviewer Needs to Do (Provide These Instructions)

After the collaborator accepts the invitation, provide them with the following instructions. These can be included in the repository's `README.md` (already exists — section A4), in the invitation email, or as a separate document.

**Prerequisites on the reviewer's machine**:
- Docker Desktop (or Docker Engine) installed and running
- A GitHub account (already required to accept the invitation)
- A GitHub personal access token (classic) with `read:packages` scope — OR — authentication via `gh auth login`

**Step-by-step for the reviewer**:

```bash
# 1. Authenticate to GHCR (one-time setup)
#    Option A: Using GitHub CLI (recommended — no PAT needed)
gh auth login
echo $(gh auth token) | docker login ghcr.io -u USERNAME --password-stdin

#    Option B: Using a Personal Access Token (classic) with read:packages scope
echo YOUR_PAT | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin

# 2. Pull the MILIA Docker image
docker pull ghcr.io/shahram-boshra/milia:latest

# 3. Run the MILIA container
docker run -it ghcr.io/shahram-boshra/milia:latest
# → (shah_env) root@...:/app/milia#

# 4. Verify MILIA works (inside the container)
pytest -m smoke --tb=short          # Quick health checks
pytest tests/ -v --tb=short         # Full test suite
python main.py --help               # CLI interface
```

**Why GHCR authentication is required**: The GHCR package is private (matching the private repository). Per GitHub Docs ("Working with the Container registry"), pulling a private package requires authentication. The `GITHUB_TOKEN` used by GitHub Actions is not available to external users — they must authenticate using either GitHub CLI or a personal access token with `read:packages` scope.

**Authoritative sources**:
- GitHub Docs — "Working with the Container registry" (`docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry`): Authentication requirements for pulling private packages
- GitHub Docs — "Managing your personal access tokens" (`docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens`): PAT creation with `read:packages` scope

---

#### 5f. Verification Checklist — Confirm Everything Works End-to-End

Run through this checklist **before** sending the invitation to a reviewer/interviewer:

```
□ 1. docker-publish.yml exists at .github/workflows/docker-publish.yml
□ 2. Image pushed to GHCR (either local push or workflow build):
     → Local push: docker push ghcr.io/shahram-boshra/milia:latest completed
     → OR workflow build: green checkmark at
        https://github.com/shahram-boshra/MILIA/actions/workflows/docker-publish.yml
□ 3. Dockerfile contains OCI labels (org.opencontainers.image.source):
     → grep "org.opencontainers" Dockerfile  # must show the LABEL lines
□ 4. GHCR package exists and is linked to the repository:
     → https://github.com/shahram-boshra/MILIA → right sidebar → "Packages"
     → Package page shows repository information
□ 5. Inherited permissions enabled:
     → Package settings → "Inherit access from source repository" is checked
□ 6. Package visibility is PRIVATE (do NOT change to public)
□ 7. Test pull from your own machine (outside the container):
     docker pull ghcr.io/shahram-boshra/milia:latest
     docker run -it ghcr.io/shahram-boshra/milia:latest
     # Verify: (shah_env) prompt appears, pytest -m smoke passes
□ 8. CI badge is green on the repository page:
     → https://github.com/shahram-boshra/MILIA → README.md badges
□ 9. Invitation sent and accepted (verify under Settings → Collaborators)
```

---

#### 5g. Summary: What One Collaborator Invitation Provides

| Access | Mechanism | Reviewer Gets |
|--------|-----------|---------------|
| Source code (11 modules, 146 tests, configs) | Repository collaborator | Read/write to all repository contents |
| Docker image (`ghcr.io/shahram-boshra/milia:latest`) | GHCR inherited permissions from linked repository | Pull access — `docker pull` + `docker run` |
| CI/CD pipeline results | GitHub Actions (public to collaborators) | View all workflow runs, test results, build logs |
| Documentation (Sphinx) | Repository contents + Read the Docs (if connected) | Full docs access |
| Issues, PRs, Wiki | Repository collaborator | Read/write |

**For a journal reviewer**: They can verify the software produces the results claimed in the paper by pulling the Docker image and running the test suite.

**For a job interviewer / postdoc committee**: They can evaluate code quality, architecture, engineering practices, documentation, and CI/CD maturity — and optionally run the software themselves.

---

**Document Version**: 1.5.0
**Created**: February 2026
**Updated**: February 2026 — v1.5.0: Added step 5 to "What Actually Remains" section — complete practical guide for `docker-publish.yml` creation, GHCR configuration, inherited permissions, collaborator invitation, reviewer instructions, and end-to-end verification checklist. Evidence-based on GitHub Docs (GHCR, Connecting packages, Configuring access, Inviting collaborators, Permission levels), `docker/metadata-action` README, and `docker/build-push-action` README. Addresses the final remaining item (#6 in Summary Table). v1.4.0: Added "What Actually Remains — Post-Creation Actions" section with the 4 post-creation steps (git init, pre-commit activation, GitHub push + post-push settings, RTD connection) sourced from Audit §7B, §3.1, §3.4, §7C, §3.9. v1.3.0: Updated all section statuses to reflect completed Production Release File Audit implementation (Audit §1–§3 fully complete); all items marked ✅ except `docker-publish.yml` (B1a). Added audit-created files not in original guide. Resolved all "Files Required" questions. v1.2.0: Added mandatory Docker image build+push to GHCR workflow (B1a); restructured B1 into B1a + B1b; updated implementation order, dependency graph, and summary table
**Based On**: MILIA Pipeline Project Structure v1.1.0, MILIA Production Release File Audit v1.1.0, MILIA Test Recommendations v1.2.0
**Evidence Sources**: PyPA, pyOpenSci, Scientific Python Dev Guide, GitHub Docs (Actions, Container Jobs, GHCR, Packages Billing, Connecting a repository to a package, Configuring a package's access control and visibility, Inviting collaborators to a personal repository, Permission levels for a personal account repository, About permissions for GitHub Packages), Docker Docs, `docker/metadata-action` README, `docker/build-push-action` README, conda-incubator/setup-miniconda, pytest docs, Keng's Blog (see full list above)
