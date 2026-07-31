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

# ---- Stage 1: builder — resolve + install into /app/.venv from the committed lock ----
FROM python:${PYTHON_VERSION}-slim AS builder
COPY --from=uv /uv /uvx /usr/local/bin/
ARG ACCEL
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /app

# 1) Dependencies only (no project) — this layer is cached until pyproject.toml/uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --extra ${ACCEL}

# 2) Project source, then install MILIA itself against the already-locked deps.
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --extra ${ACCEL}

# Build-time verification (fails the build early). uv installs the project EDITABLE by
# default (PEP 660), so `milia_pipeline` legitimately resolves to the baked /app source —
# that is the single-copy state we want (a --no-editable copy would leave two trees). We
# therefore verify what actually matters, not the physical location:
#   (a) the dist is installed (metadata resolves) and the package imports;
#   (b) the compiled PyG companion kernels load AND run against this torch build (ABI/R1).
RUN /app/.venv/bin/python -c "import importlib.metadata as md, milia_pipeline; print('milia', md.version('milia'), 'importable from', milia_pipeline.__file__)" && \
    /app/.venv/bin/python -c "import torch, torch_scatter; src=torch.tensor([1.,1.,1.,1.]); idx=torch.tensor([0,0,1,1]); assert torch_scatter.scatter_add(src, idx, dim=0).tolist()==[2.,2.]; print('OK torch', torch.__version__, '| torch_scatter', torch_scatter.__version__)"

# ---- Stage 2: test — builder + dev tools, for CI in-image smoke tests (not published) ----
# CI builds this with `--target test` to run `pytest -m smoke`; the published image is `runtime`.
FROM builder AS test
ARG ACCEL
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra ${ACCEL} --extra dev
ENV PATH="/app/.venv/bin:$PATH" \
    MILIA_LOG_DIR=/tmp
# e.g. docker run --rm <test-image> pytest -m smoke -q tests/

# ---- Stage 3: runtime — minimal, non-root, production (DEFAULT build target) ----
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
    MILIA_LOG_DIR=/tmp
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
