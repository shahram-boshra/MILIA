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
Phase 2: CI/CD Infrastructure — ✅ PARTIALLY COMPLETE
┌─────────────────────────────────────────────────────┐
│ Step 6: .github/workflows/docker-publish.yml [B1a]  │
│         ❌ NOT YET CREATED — build & push to GHCR    │
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

❌ .github/workflows/docker-publish.yml ────────── NOT YET CREATED (depends on:)
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
| 6 | `.github/workflows/docker-publish.yml` | CI/CD | **Critical** | ❌ Not yet created |
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

**Remaining**: Only item #6 (`docker-publish.yml`) is not yet created.

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

---

## Files Required From You Before Implementation — ✅ ALL RESOLVED

All questions originally listed here have been resolved during the Production Release File Audit implementation:

1. **GitHub repository URL** — ✅ Resolved: `https://github.com/shahram-boshra/MILIA` (Audit §2.1, line 79)
2. **License confirmation** — ✅ Resolved: MIT (Audit §2.2, line 90)
3. **Author information for LICENSE** — ✅ Resolved: "2026-present Asadollah Boshra" (Audit §2.2, line 90)
4. **Current `.gitignore`** — ✅ Resolved: Fresh GitHub Python template + MILIA-specific exclusions created (Audit §2.4)
5. **Current `tests/` file listing** — ✅ Resolved: 127 test files verified; CI workflow glob patterns set (Audit §3.4)

---

**Document Version**: 1.3.0
**Created**: February 2026
**Updated**: February 2026 — v1.3.0: Updated all section statuses to reflect completed Production Release File Audit implementation (Audit §1–§3 fully complete); all items marked ✅ except `docker-publish.yml` (B1a). Added audit-created files not in original guide (CITATION.cff, SECURITY.md, RELEASE_CHECKLIST.md, Makefile, MANIFEST.in, noxfile.py, .pre-commit-config.yaml, .readthedocs.yaml, release.yml, dependabot.yml, YAML issue forms, PR template, docs/ Sphinx build system). Resolved all "Files Required" questions. v1.2.0: Added mandatory Docker image build+push to GHCR workflow (B1a) for reviewer/user access; restructured B1 into B1a (Docker publish) + B1b (test suite); updated implementation order, dependency graph, and summary table
**Based On**: MILIA Pipeline Project Structure v1.1.0, MILIA Production Release File Audit v1.1.0, MILIA Test Recommendations v1.2.0
**Evidence Sources**: PyPA, pyOpenSci, Scientific Python Dev Guide, GitHub Docs (Actions, Container Jobs, GHCR, Packages Billing), Docker Docs, conda-incubator/setup-miniconda, pytest docs, Keng's Blog (see full list above)
