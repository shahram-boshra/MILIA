# syntax=docker/dockerfile:1.7
# =============================================================================
# MILIA — uv-based multi-stage build (replaces the conda/mamba image).
# ONE Dockerfile, parameterized by accelerator. Build the variant you need:
#   docker build -t milia:cpu   .                        # default (ARG ACCEL=cpu)
#   docker build --build-arg ACCEL=cu124 -t milia:cu124 .
# A single image contains exactly ONE torch build (cpu | cu118 | cu121 | cu124):
# CPU+GPU cannot coexist in one environment. "All accelerators" = a CI build
# matrix over ACCEL publishing one GHCR tag per variant (see docker-publish.yml).
# TPU is NOT a target: PyG's compiled kernels have no XLA backend and MILIA has no
# torch_xla code, so a TPU image would be non-functional for the GNN path.
# CUDA runtime libs ship INSIDE the torch/PyG wheels, so a slim base suffices; GPU
# use at runtime needs the host NVIDIA driver + `--gpus all` (nvidia-container-toolkit).
# Refs: Astral uv — Docker guide; PyPA; PyTorch/PyG wheels bundle the CUDA runtime.
# =============================================================================

# Global build args (available to FROM lines; re-declared inside stages for RUN use).
ARG PYTHON_VERSION=3.10        # matches the current image; local dev also validated on 3.12
ARG UV_VERSION=0.12.0          # pin uv for reproducible builds (the version used to validate the lock)
ARG ACCEL=cpu                  # cpu | cu118 | cu121 | cu124

# uv binary — aliased via FROM so the version ARG expands. Using ${ARG} directly in a
# `COPY --from=<image>:${ARG}` reference is an unresolved BuildKit limitation
# (moby/buildkit#1167, docker/cli#3356): it yields `uv:` → "invalid reference format".
# A FROM instruction CAN expand a global ARG, so we alias here and COPY by stage name.
FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

# ---- Stage 1: deps — third-party dependencies from the committed lock (no project source) ----
# Shared base of `builder` (→ runtime) and `test-deps` (→ test). Cached until pyproject.toml/uv.lock
# change, so source edits never re-run dependency installation in either chain.
FROM python:${PYTHON_VERSION}-slim AS deps
COPY --from=uv /uv /uvx /usr/local/bin/
ARG ACCEL
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# Dependencies only (no project) — this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra ${ACCEL}

# ---- Stage 2: builder — project source + MILIA itself (feeds `runtime`) ----
FROM deps AS builder
ARG ACCEL
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ${ACCEL}

# Build-time verification (fails the build early): dist metadata + import, and the compiled PyG
# kernels run against this torch build (ABI/R1). Defined once in docker/verify_build.py and shared
# with the `test` stage. uv installs the project EDITABLE by default (PEP 660), so `milia_pipeline`
# legitimately resolves to the baked /app source — the single-copy state we want.
RUN /app/.venv/bin/python /app/docker/verify_build.py

# ---- Stage 3: test-deps — JRE + dev/test dependencies, cached independently of project source ----
FROM deps AS test-deps
ARG ACCEL
# Java (headless JRE) is needed to gate the opt-in `cdk_substructure` plugin's tests (Pace 11 /
# v1.14.0), which call the CDK engine via jpype. Use the distro DEFAULT headless JRE
# (`default-jre-headless`) rather than a pinned major version: the python:3.x-slim base tracks
# Debian stable (now Trixie → Java 21; openjdk-17 was dropped), so pinning a JRE version breaks on
# base bumps. CDK SubFPC values are integer SMARTS counts, independent of the JRE version.
# Without Java these tests importorskip (skip).
RUN apt-get update && apt-get install -y --no-install-recommends default-jre-headless && \
    rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --extra ${ACCEL} --extra dev --extra descriptors-cdk

# ---- Stage 4: test — test-deps + project source, for in-image test runs (not published) ----
# Built with `--target test`; BuildKit builds only this chain (deps → test-deps → test), never
# `builder`/`runtime`. A source change re-runs only COPY + project install + verification.
FROM test-deps AS test
ARG ACCEL
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ${ACCEL} --extra dev --extra descriptors-cdk
RUN /app/.venv/bin/python /app/docker/verify_build.py
ENV PATH="/app/.venv/bin:$PATH" \
    MILIA_LOG_DIR=/tmp
# e.g. docker run --rm <test-image> pytest -m smoke -q tests/

# ---- Stage 5: runtime — minimal, non-root, production (DEFAULT build target) ----
FROM python:${PYTHON_VERSION}-slim AS runtime
# libgomp1: OpenMP runtime required by torch / scikit-learn at import.
# (If a runtime ImportError reports another missing .so — e.g. libXrender for some RDKit
#  drawing paths — add the minimal lib here; keep the set tight.)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Non-root user — safe now that logging writes to $MILIA_LOG_DIR, not the package dir.
RUN useradd --create-home --uid 10001 milia

# Copy the fully-built app (source + /app/.venv) from the builder, owned by the app user.
COPY --from=builder --chown=milia:milia /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    MILIA_LOG_DIR=/tmp \
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app
USER milia

# Re-declare so the build-arg value reaches this stage's provenance label.
ARG ACCEL
LABEL org.opencontainers.image.source="https://github.com/shahram-boshra/MILIA" \
      org.opencontainers.image.title="MILIA" \
      org.opencontainers.image.description="Molecular graph ML/DL pipeline (uv build, accelerator=${ACCEL})"

# `milia` is on PATH via /app/.venv/bin. Run: `docker run --rm milia:cpu --root-dir /data`
# Override for other entrypoints, e.g.: `docker run --rm --entrypoint python milia:cpu /app/main.py --help`
ENTRYPOINT ["milia"]
CMD ["--help"]
