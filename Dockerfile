# Dockerfile for shah_env ML/Chemistry environment
# Base: condaforge/miniforge3 — the Mamba project's recommended distribution.
# Miniforge3 ships with mamba pre-installed and uses only the conda-forge channel.
# This avoids the Anaconda-defaults/conda-forge channel incompatibility that causes
# broken environments (ref: mamba.readthedocs.io/en/latest/user_guide/troubleshooting.html).
# Pin to versioned tag per Docker best practices for reproducibility.
# To upgrade: check https://github.com/conda-forge/miniforge/releases for new tags,
# update the tag below, and optionally pin by digest from `docker pull` output.
FROM condaforge/miniforge3:25.11.0-1

# OCI metadata labels — required for GHCR repository linking and package discoverability
# Source: GitHub Docs ("Working with the Container registry" — "Labelling container images")
# Source: OCI Image Spec — Pre-Defined Annotation Keys (opencontainers/image-spec)
LABEL org.opencontainers.image.source=https://github.com/shahram-boshra/MILIA
LABEL org.opencontainers.image.description="MILIA: Machine Intelligent Learning Interface Assistant — molecular data processing and ML pipeline for quantum chemistry"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /app

# Install minimal system tooling required for in-container git workflows.
# - git: enables 'git pull / git status' inside running containers (without this,
#   reviewers cannot fetch updates from origin/main without rebuilding the image)
# - openssh-client: required by git when remote URLs use the git@github.com:... form
#   (without this, 'git pull' fails with 'error: cannot run ssh: No such file or directory')
# - ca-certificates: HTTPS verification for git/curl
#
# Pattern follows Docker official best practices (docs.docker.com/build/building/best-practices):
# - apt-get update && install in a single RUN layer so the cache is current at install time
# - --no-install-recommends to skip optional dependencies (smaller image)
# - rm -rf /var/lib/apt/lists/* in the SAME layer to avoid persisting cache in lower layers
# - DEBIAN_FRONTEND=noninteractive set per-RUN (NOT as ENV) per Docker FAQ — avoids leaking
#   into runtime where it would suppress legitimate prompts
# Retry loop matches the resilience pattern used by the conda layers below.
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing system packages (git, openssh-client, ca-certificates)..."; \
        DEBIAN_FRONTEND=noninteractive apt-get update && \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            ca-certificates \
            git \
            openssh-client \
        && rm -rf /var/lib/apt/lists/* \
        && echo "System packages installed!" && _success=1 && break || sleep 10; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install system packages after all retries"; exit 1; fi

# Configure conda/mamba: conda-forge only, strict priority, generous timeouts.
# Miniforge3 ships mamba pre-installed — no separate install step needed.
# channel_priority=strict per Mamba docs: prevents mixing incompatible channels.
RUN conda config --set remote_connect_timeout_secs 180.0 && \
    conda config --set remote_read_timeout_secs 900.0 && \
    conda config --set channel_priority strict && \
    conda config --remove channels defaults 2>/dev/null || true && \
    conda config --add channels conda-forge && \
    conda clean --all -y

# Create base environment with retry logic
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Creating base environment..."; \
        rm -rf /opt/conda/envs/shah_env; \
        mamba create -n shah_env python=3.10 pip -y && \
        test -f /opt/conda/envs/shah_env/bin/python && \
        echo "Base environment created!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to create base environment after all retries"; exit 1; fi && \
    conda clean --all -y

# Install numpy, scipy, and small packages
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing core numerical libraries..."; \
        mamba install -n shah_env -c conda-forge \
            numpy=1.26.4 \
            scipy=1.15.2 \
            pyyaml=6.0.2 \
            attrs \
            h5py \
            pandas=2.3.1 \
            -y && echo "Core libraries installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install core libraries after all retries"; exit 1; fi && \
    conda clean --all -y

# Install PyTorch (large package - most likely to fail)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 5); do \
        echo "Attempt $i/5: Installing PyTorch..."; \
        mamba install -n shah_env -c pytorch -c conda-forge \
            --no-channel-priority \
            pytorch::pytorch=2.4.0 \
            pytorch::cpuonly=2.0 \
            -y && echo "PyTorch installed!" && _success=1 && break || \
        (echo "PyTorch install failed, retrying..." && sleep 20); \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install PyTorch after all retries"; exit 1; fi && \
    conda clean --all -y

# Install torch-geometric
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing torch-geometric..."; \
        mamba install -n shah_env -c pyg -c conda-forge \
            --no-channel-priority \
            torch-geometric \
            -y && echo "torch-geometric installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install torch-geometric after all retries"; exit 1; fi && \
    conda clean --all -y

# Install PyG optional packages via pip (required for SchNet, DimeNet, PointNet++, etc.)
# Using pip wheels as recommended by PyG documentation for PyTorch 2.4.0
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing PyG optional packages (torch-cluster, torch-scatter, torch-sparse, torch-spline-conv)..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            torch-cluster \
            torch-scatter \
            torch-sparse \
            torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.4.0+cpu.html && \
        echo "PyG optional packages installed!" && _success=1 && break || sleep 10; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install PyG optional packages after all retries"; exit 1; fi

# Install RDKit
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing RDKit..."; \
        mamba install -n shah_env -c conda-forge \
            rdkit=2025.03.5 \
            -y && echo "RDKit installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install RDKit after all retries"; exit 1; fi && \
    conda clean --all -y

# Install matplotlib and testing packages
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing remaining packages..."; \
        mamba install -n shah_env -c conda-forge \
            matplotlib=3.10.3 \
            pytest=8.4.1 \
            pytest-mock=3.14.1 \
            pydantic-settings=2.10.1 \
            -y && echo "Remaining packages installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install remaining packages after all retries"; exit 1; fi && \
    conda clean --all -y

# Install ASE (Atomic Simulation Environment) for XYZ file support (Tests 67, 71)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing ASE..."; \
        mamba install -n shah_env -c conda-forge \
            ase \
            -y && echo "ASE installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install ASE after all retries"; exit 1; fi && \
    conda clean --all -y

# Install torchmetrics (NEW - Required for MetricsRegistry)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing torchmetrics..."; \
        mamba install -n shah_env -c conda-forge \
            "torchmetrics>=1.0.0" \
            -y && echo "torchmetrics installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install torchmetrics after all retries"; exit 1; fi && \
    conda clean --all -y

# Install Hydra (Configuration Management Framework for ML)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing hydra-core..."; \
        mamba install -n shah_env -c conda-forge \
            "hydra-core>=1.3.0" \
            -y && echo "hydra-core installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install hydra-core after all retries"; exit 1; fi && \
    conda clean --all -y

# Install HPO conda packages (Optuna, visualization)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing HPO packages (Optuna, visualization)..."; \
        mamba install -n shah_env -c conda-forge \
            "optuna>=3.0.0" \
            "plotly>=5.0.0" \
            python-kaleido \
            -y && echo "HPO conda packages installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install HPO conda packages after all retries"; exit 1; fi && \
    conda clean --all -y

# Install scikit-learn separately to ensure it installs
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing scikit-learn..."; \
        mamba install -n shah_env -c conda-forge \
            scikit-learn=1.5.2 \
            -y && echo "scikit-learn installed!" && _success=1 && break || sleep 15; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install scikit-learn after all retries"; exit 1; fi && \
    conda clean --all -y

# Install pip packages with retry
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing pip packages..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            memory-profiler \
            qc-iodata && \
        echo "Pip packages installed!" && _success=1 && break || sleep 10; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install pip packages after all retries"; exit 1; fi

# Install HPO optional pip packages (Ray Tune, Optuna Dashboard)
RUN set -e && \
    _success=0 && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing HPO pip packages (Ray Tune, Optuna Dashboard)..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            "ray[tune]>=2.0.0" \
            optuna-dashboard && \
        echo "HPO pip packages installed!" && _success=1 && break || sleep 10; \
    done && \
    if [ "$_success" -ne 1 ]; then echo "FATAL: Failed to install HPO pip packages after all retries"; exit 1; fi

# Verify critical packages
RUN /opt/conda/envs/shah_env/bin/python -c "import sys; print(f'Python: {sys.version}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import numpy; print(f'NumPy: {numpy.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import rdkit; print(f'RDKit: {rdkit.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "from iodata import load_one; print('IOData: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import optuna; print(f'Optuna: {optuna.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import ray; print(f'Ray: {ray.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import kaleido; print('Kaleido: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch_geometric; print(f'PyG: {torch_geometric.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch_cluster; print('torch-cluster: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch_scatter; print('torch-scatter: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch_sparse; print('torch-sparse: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torch_spline_conv; print('torch-spline-conv: OK')" && \
    /opt/conda/envs/shah_env/bin/python -c "import torchmetrics; print(f'torchmetrics: {torchmetrics.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import hydra; print(f'Hydra: {hydra.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import omegaconf; print(f'OmegaConf: {omegaconf.__version__}')" && \
    /opt/conda/envs/shah_env/bin/python -c "import ase; print(f'ASE: {ase.__version__}')" && \
    echo "========================================" && \
    echo "SUCCESS: All packages verified!" && \
    echo "========================================"

# Setup conda/mamba activation for interactive shells
SHELL ["/bin/bash", "-c"]
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo ". /opt/conda/etc/profile.d/mamba.sh" >> ~/.bashrc && \
    echo "conda activate shah_env" >> ~/.bashrc

# Ensure shah_env is active for all invocation modes (docker run, exec, CMD)
# .bashrc alone only covers interactive non-login shells
ENV PATH="/opt/conda/envs/shah_env/bin:${PATH}"
ENV CONDA_DEFAULT_ENV=shah_env
ENV CONDA_PREFIX=/opt/conda/envs/shah_env
# Required by mamba >=2.0 to locate the installation root (ref: mamba-org/mamba#756)
ENV MAMBA_ROOT_PREFIX=/opt/conda

# Two physical copies of milia_pipeline/main exist in the image: the source tree
# at /app (COPY below) and the non-editable install in site-packages (pip install
# below). On this Python 3.10 base, import resolution is launch-path dependent —
# the `milia` console script resolves to site-packages, while `python /app/main.py`
# and any `python -c` run from WORKDIR /app resolve to the /app source (CWD is
# prepended to sys.path). PYTHONSAFEPATH/-P, which would suppress that prepend,
# only exists on Python >=3.11 (added there per the CPython docs), so it is inert
# here and is deliberately NOT set — it would also break the /app-based pytest and
# `python main.py` invocations in the smoke-test job if the base were upgraded.
#
# What actually prevents the "stale results until reinstall" bug is IMMUTABILITY:
# `pip install .` copies the package from the same /app source in the same build,
# so both copies are byte-identical, and the image is never mutated at runtime
# (no in-container `git pull`; updates ship as a rebuilt, retagged image). Which
# copy an import resolves to is therefore immaterial — they can never diverge.
# The structural fix that would collapse this to a single import root on every
# Python version is the src/ layout (package under src/milia_pipeline/, so the
# project root holds no importable milia_pipeline/); tracked as a future refactor.
# Ref: PyPA Packaging User Guide — "src layout vs flat layout"; Docker Docs —
#      Immutable infrastructure.

# Copy application code
COPY . /app

# Install MILIA as a Python package (production, non-editable)
# This step serves two purposes:
# 1. Registers the 'milia' CLI entry point defined in pyproject.toml [project.scripts]
#    (milia = "main:main") — setuptools generates a wrapper script in shah_env's bin/
# 2. Installs milia_pipeline into shah_env's site-packages for formal package imports
#
# --no-deps: All runtime dependencies are managed by conda/mamba (see install layers above).
#   pyproject.toml intentionally declares dependencies=[] to avoid pip/conda conflicts
#   with binary packages (PyTorch, RDKit, PyG). This flag is a safety measure.
# --no-cache-dir: Minimizes image layer size (no pip cache to clean up).
#
# Two byte-identical copies exist (site-packages install + /app source); on this
# Python 3.10 base neither is forced to win at import time, but immutability makes
# that moot (see the note above the COPY step). The /app source tree is retained
# only for reference/config assets and the smoke-test's `python /app/main.py` and
# `pytest tests/` runs — do NOT mutate it at runtime expecting code changes to take
# effect. Images are immutable: to ship a code update, rebuild and publish a new tag
# (the CI/CD short-SHA tag), then redeploy — never `git pull && pip install` inside a
# running container. That in-place mutation is exactly what caused prior "results didn't
# update until we reinstalled" incidents, and it breaks reproducibility.
# Ref: Docker Docs — Immutable infrastructure ("rebuild and redeploy rather than patch
#      running containers"); non-editable install per PyDevTools ("do not use editable
#      installs in production").
#
# Source: PyPA Packaging User Guide — "Writing your pyproject.toml" (console scripts)
# Source: pyOpenSci — "Declare Dependencies" (conda-managed + pip install pattern)
RUN /opt/conda/envs/shah_env/bin/pip install --no-deps --no-cache-dir . && \
    echo "MILIA package installed:" && \
    /opt/conda/envs/shah_env/bin/pip show milia && \
    echo "CLI entry point registered:" && \
    which milia && \
    echo "Verifying the non-editable install placed milia_pipeline in site-packages:" && \
    /opt/conda/envs/shah_env/bin/python -I -c "import importlib.util as u; s = u.find_spec('milia_pipeline'); origin = s.origin if s else None; assert origin and 'site-packages' in origin, f'FATAL: pip install . did not package milia_pipeline into site-packages (find_spec origin={origin}); the install would silently fall back to the /app source copy'; print('milia_pipeline installed at', origin)" && \
    echo "========================================"

CMD ["/bin/bash", "--login"]
