# Release Checklist

Step-by-step checklist for releasing a new version of MILIA.
Follow every step in order. Each step references the exact file and field to update.

This checklist follows [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
and the [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/) format.

---

## Pre-Release

### Step 1 — Decide the Version Number

Determine the new version `X.Y.Z` per [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **Patch** (1.1.0 → 1.1.1): backward-compatible bug fixes.
- **Minor** (1.1.0 → 1.2.0): backward-compatible new features.
- **Major** (1.1.0 → 2.0.0): breaking changes.

### Step 2 — Update `__version__` (Single Source of Truth)

File: `milia_pipeline/__init__.py`, line containing `__version__`.

```python
# Before
__version__ = "1.1.0"

# After (example: patch release)
__version__ = "1.1.1"
```

This is the **single source of truth** for the project version.
`pyproject.toml` reads it dynamically via:

```toml
[tool.setuptools.dynamic]
version = {attr = "milia_pipeline.__version__"}
```

No other version field needs to be set in `pyproject.toml`.

### Step 3 — Update `CHANGELOG.md`

File: `CHANGELOG.md`

1. Move content from `## [Unreleased]` into a new version section.
2. Add a fresh empty `## [Unreleased]` section above the new version.
3. Update the footer comparison links.

```markdown
## [Unreleased]

## [X.Y.Z] - YYYY-MM-DD

### Added
- ...

### Changed
- ...

### Fixed
- ...
```

Update the footer links at the bottom of the file:

```markdown
[unreleased]: https://github.com/shahram-boshra/MILIA/compare/vX.Y.Z...HEAD
[X.Y.Z]: https://github.com/shahram-boshra/MILIA/compare/vPREVIOUS...vX.Y.Z
```

Use only the subsection headers that apply (`Added`, `Changed`, `Deprecated`,
`Removed`, `Fixed`, `Security`). Omit empty subsections.

### Step 4 — Update `CITATION.cff`

File: `CITATION.cff`

Update exactly two fields:

```yaml
version: "X.Y.Z"
date-released: "YYYY-MM-DD"
```

Validate after editing:

```bash
cffconvert --validate
```

### Step 5 — Update `SECURITY.md` (Minor/Major Releases Only)

File: `SECURITY.md`, Supported Versions table.

For **patch** releases within the same minor version, no update is needed
(the `1.1.x` pattern already covers patches). For **minor** or **major**
releases, update the table:

```markdown
| Version | Python | Supported |
|---------|--------|-----------|
| X.Y.x   | 3.10, 3.11, 3.12 | ✅ Yes — security updates provided |
| 1.1.x   | 3.10, 3.11, 3.12 | ❌ No — upgrade to X.Y.x |
```

### Step 6 — Run Quality Checks

```bash
# Lint and format check (from pyproject.toml [tool.ruff])
ruff check .
ruff format --check .

# Run full test suite (from pyproject.toml [tool.pytest.ini_options])
pytest -v --tb=short

# Or use the Makefile composite target (lint + format-check + test)
make check
```

All checks must pass before proceeding.

### Step 7 — Verify the Build Locally

```bash
# Clean previous build artifacts
make clean

# Build sdist and wheel (from pyproject.toml [build-system]: setuptools>=77)
python -m build

# Verify the built version matches the intended release
python -c "from milia_pipeline import __version__; print(__version__)"
```

Inspect the sdist contents to confirm `main.py`, `configs/`, and metadata
files are included (per `MANIFEST.in`):

```bash
tar tzf dist/milia-X.Y.Z.tar.gz | head -30
```

### Step 8 — Commit the Release

```bash
git add milia_pipeline/__init__.py CHANGELOG.md CITATION.cff
# Include SECURITY.md only if it was updated (minor/major releases):
# git add SECURITY.md

git commit -m "release: prepare vX.Y.Z

- Bump __version__ to X.Y.Z in milia_pipeline/__init__.py
- Promote [Unreleased] to [X.Y.Z] in CHANGELOG.md
- Update version and date-released in CITATION.cff"
```

---

## Release

### Step 9 — Tag the Release

The tag format **must** match the pattern in `.github/workflows/release.yml`
(`v*.*.*`):

```bash
git tag -a vX.Y.Z -m "vX.Y.Z"
```

### Step 10 — Push the Commit and Tag

```bash
git push origin main
git push origin vX.Y.Z
```

Pushing the tag triggers the `.github/workflows/release.yml` workflow, which:

1. Builds the sdist and wheel (`python -m build`).
2. Publishes to PyPI via Trusted Publishers (OIDC — no API token needed).

### Step 11 — Create a GitHub Release

1. Go to `https://github.com/shahram-boshra/MILIA/releases/new`.
2. Select the `vX.Y.Z` tag.
3. Set the release title to `vX.Y.Z`.
4. Copy the `CHANGELOG.md` entry for this version into the release description.
5. Click **Publish release**.

---

## Post-Release

### Step 12 — Verify PyPI

1. Check `https://pypi.org/project/milia/X.Y.Z/` for the published release.
2. Verify the README renders correctly on the PyPI page.
3. Test installation in a clean environment:

```bash
pip install milia==X.Y.Z
python -c "from milia_pipeline import get_package_info; print(get_package_info())"
```

### Step 13 — Verify CI

Check that the `release.yml` workflow completed successfully:
`https://github.com/shahram-boshra/MILIA/actions/workflows/release.yml`

### Step 14 — Prepare for Next Development Cycle

Ensure `CHANGELOG.md` has an empty `## [Unreleased]` section ready for new
changes. If the release commit already includes it (Step 3), no action is
needed.

---

## Quick Reference — Files Updated Per Release

| File | Field(s) | When |
|------|----------|------|
| `milia_pipeline/__init__.py` | `__version__` | Every release |
| `CHANGELOG.md` | `[Unreleased]` → `[X.Y.Z]`, footer links | Every release |
| `CITATION.cff` | `version`, `date-released` | Every release |
| `SECURITY.md` | Supported Versions table | Minor/major releases |

---

## Prerequisites (One-Time Setup)

These must be configured before the **first** release. They are documented
here for reference and do not need to be repeated.

1. **PyPI Trusted Publisher**: Configure at
   `https://pypi.org/manage/project/milia/settings/publishing/` — add a
   GitHub Actions publisher with owner `shahram-boshra`, repository `MILIA`,
   workflow `release.yml`, environment `pypi`.

2. **GitHub Environment**: Create a `pypi` environment in the repository
   settings (Settings → Environments). Optionally require manual approval
   for release protection.

3. **Private Vulnerability Reporting**: Enable in repository settings
   (Settings → Security → Code security and analysis → Private vulnerability
   reporting → Enable). Required for `SECURITY.md` to function as documented.

---

## References

- [py-pkgs.org — Releasing and versioning](https://py-pkgs.org/07-releasing-versioning.html)
- [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html)
- [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- [PyPA — Publishing package distribution releases using GitHub Actions](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
- [PyPI Trusted Publishers](https://docs.pypi.org/trusted-publishers/)
- [Citation File Format 1.2.0](https://citation-file-format.github.io/)
- [cffconvert — Validate and convert CITATION.cff](https://github.com/citation-file-format/cffconvert)
