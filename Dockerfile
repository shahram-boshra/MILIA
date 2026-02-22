# Dockerfile for shah_env ML/Chemistry environment
# Pin to miniconda3 v25.11.1-1 (Python 3.13, conda 25.11.1) — matches working container base.
# Digest verified from docker build log (2026-02-22). Pin by digest per Docker best practices.
# To upgrade: pull new miniconda3:latest, note its sha256, update digest + comment here.
FROM docker.io/continuumio/miniconda3@sha256:5df7c31c16e90e4ea370836770feed507a1cf51c6e8aad835c65fb26b9eca941

# OCI metadata labels — required for GHCR repository linking and package discoverability
# Source: GitHub Docs ("Working with the Container registry" — "Labelling container images")
# Source: OCI Image Spec — Pre-Defined Annotation Keys (opencontainers/image-spec)
LABEL org.opencontainers.image.source=https://github.com/shahram-boshra/MILIA
LABEL org.opencontainers.image.description="MILIA: Machine Intelligent Learning Interface Assistant — molecular data processing and ML pipeline for quantum chemistry"
LABEL org.opencontainers.image.licenses=MIT

WORKDIR /app

# Install mamba with aggressive retry logic
RUN set -e && \
    for i in $(seq 1 3); do \
        conda install -n base -c conda-forge mamba=2.4.0 -y && break || sleep 10; \
    done && \
    conda config --set remote_connect_timeout_secs 180.0 && \
    conda config --set remote_read_timeout_secs 900.0 && \
    conda config --set channel_priority flexible && \
    conda clean --all -y

# Create base environment with retry logic
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Creating base environment..."; \
        rm -rf /opt/conda/envs/shah_env; \
        mamba create -n shah_env python=3.10 pip -y && \
        test -f /opt/conda/envs/shah_env/bin/python && \
        echo "Base environment created!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install numpy, scipy, and small packages
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing core numerical libraries..."; \
        mamba install -n shah_env -c conda-forge \
            numpy=1.26.4 \
            scipy=1.15.2 \
            pyyaml=6.0.2 \
            attrs \
            h5py \
            pandas=2.3.1 \
            -y && echo "Core libraries installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install PyTorch (large package - most likely to fail)
RUN set -e && \
    for i in $(seq 1 5); do \
        echo "Attempt $i/5: Installing PyTorch..."; \
        mamba install -n shah_env -c pytorch -c conda-forge \
            pytorch=2.4.0 \
            cpuonly=2.0 \
            -y && echo "PyTorch installed!" && break || \
        (echo "PyTorch install failed, retrying..." && sleep 20); \
    done && \
    conda clean --all -y

# Install torch-geometric
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing torch-geometric..."; \
        mamba install -n shah_env -c pyg -c conda-forge \
            torch-geometric \
            -y && echo "torch-geometric installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install PyG optional packages via pip (required for SchNet, DimeNet, PointNet++, etc.)
# Using pip wheels as recommended by PyG documentation for PyTorch 2.4.0
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing PyG optional packages (torch-cluster, torch-scatter, torch-sparse, torch-spline-conv)..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            torch-cluster \
            torch-scatter \
            torch-sparse \
            torch-spline-conv \
            -f https://data.pyg.org/whl/torch-2.4.0+cpu.html && \
        echo "PyG optional packages installed!" && break || sleep 10; \
    done

# Install RDKit
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing RDKit..."; \
        mamba install -n shah_env -c conda-forge \
            rdkit=2025.03.5 \
            -y && echo "RDKit installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install matplotlib and testing packages
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing remaining packages..."; \
        mamba install -n shah_env -c conda-forge \
            matplotlib=3.10.3 \
            pytest=8.4.1 \
            pytest-mock=3.14.1 \
            pydantic-settings=2.10.1 \
            -y && echo "Remaining packages installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install ASE (Atomic Simulation Environment) for XYZ file support (Tests 67, 71)
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing ASE..."; \
        mamba install -n shah_env -c conda-forge \
            ase \
            -y && echo "ASE installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install torchmetrics (NEW - Required for MetricsRegistry)
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing torchmetrics..."; \
        mamba install -n shah_env -c conda-forge \
            "torchmetrics>=1.0.0" \
            -y && echo "torchmetrics installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install Hydra (Configuration Management Framework for ML)
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing hydra-core..."; \
        mamba install -n shah_env -c conda-forge \
            "hydra-core>=1.3.0" \
            -y && echo "hydra-core installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install HPO conda packages (Optuna, visualization)
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing HPO packages (Optuna, visualization)..."; \
        mamba install -n shah_env -c conda-forge \
            "optuna>=3.0.0" \
            "plotly>=5.0.0" \
            python-kaleido \
            -y && echo "HPO conda packages installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install scikit-learn separately to ensure it installs
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing scikit-learn..."; \
        mamba install -n shah_env -c conda-forge \
            scikit-learn=1.5.2 \
            -y && echo "scikit-learn installed!" && break || sleep 15; \
    done && \
    conda clean --all -y

# Install pip packages with retry
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing pip packages..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            memory-profiler \
            qc-iodata && \
        echo "Pip packages installed!" && break || sleep 10; \
    done

# Install HPO optional pip packages (Ray Tune, Optuna Dashboard)
RUN set -e && \
    for i in $(seq 1 3); do \
        echo "Attempt $i/3: Installing HPO pip packages (Ray Tune, Optuna Dashboard)..."; \
        /opt/conda/envs/shah_env/bin/pip install --no-cache-dir \
            "ray[tune]>=2.0.0" \
            optuna-dashboard && \
        echo "HPO pip packages installed!" && break || sleep 10; \
    done

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

# Setup conda activation
SHELL ["/bin/bash", "-c"]
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate shah_env" >> ~/.bashrc

# Copy application code
COPY . /app

CMD ["/bin/bash", "--login"]
