#!/usr/bin/env python3
"""
Host resources seen by one worker of a shared study (P2-6, blueprint S3 / F21, F50).

The workers of a study run side by side on a host without knowing of each other (D3: no declared
worker count). The launcher sizes each one: it pins one GPU per worker (``CUDA_VISIBLE_DEVICES`` or
Compose ``device_ids``) and sets its CPU thread budget through ``OMP_NUM_THREADS``, which PyTorch reads
when it starts — torchrun sets 1 per process when it starts several, and PyTorch's multiprocessing
notes bound each of M processes on N vCPUs to ``floor(N / M)`` threads. A worker reports what it was
given and warns about what oversubscribes the host; it stops early only on a certain failure:

- Memory. The processed dataset is loaded whole (``InMemoryDataset``), so a processed file larger
  than the memory available now cannot be loaded without swapping or an out-of-memory kill. Workers
  that start later see the memory that earlier workers already hold, so the check covers N workers
  without knowing N. Available memory is the smaller of the host's (``psutil``: memory that "can be
  given instantly to processes without the system going into swap") and the headroom of this
  process's cgroup v2 limits (a container memory limit), where usage excludes reclaimable cache
  (``inactive_file``) as ``docker stats`` computes it. cgroup v1 limits are not read.
- CPU and GPU: warnings only.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import psutil
import torch

from milia_pipeline.exceptions import HPOError

CGROUP_ROOT = Path("/sys/fs/cgroup")
PROC_SELF_CGROUP = Path("/proc/self/cgroup")


@dataclass(frozen=True)
class WorkerResources:
    """CPU and GPU resources of this process, as the launcher configured them."""

    usable_cpus: int
    torch_threads: int
    omp_num_threads: str | None
    cuda_visible_devices: str | None
    cuda_devices: int

    def describe(self) -> str:
        return (
            f"CUDA_VISIBLE_DEVICES={self.cuda_visible_devices or '<unset>'}, "
            f"visible CUDA devices: {self.cuda_devices}, "
            f"OMP_NUM_THREADS={self.omp_num_threads or '<unset>'}, "
            f"PyTorch threads: {self.torch_threads}, usable CPUs: {self.usable_cpus}"
        )

    def warnings(self) -> list[str]:
        """Settings that oversubscribe the host when several workers share it."""
        found = []
        if self.usable_cpus > 1 and self.torch_threads >= self.usable_cpus:
            found.append(
                f"PyTorch uses {self.torch_threads} threads on {self.usable_cpus} usable CPUs; with "
                f"several workers on this host each one competes for every CPU. Give each worker "
                f"OMP_NUM_THREADS=floor({self.usable_cpus} / number of workers) "
                f"(Compose: MILIA_WORKER_THREADS)"
            )
        if self.cuda_devices > 1:
            found.append(
                f"{self.cuda_devices} CUDA devices are visible; give each worker one GPU "
                f"(CUDA_VISIBLE_DEVICES or Compose device_ids)"
            )
        return found


def usable_cpu_count() -> int:
    """CPUs this process may run on (its affinity mask where the platform has one)."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


def describe_worker_resources() -> WorkerResources:
    return WorkerResources(
        usable_cpus=usable_cpu_count(),
        torch_threads=torch.get_num_threads(),
        omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cuda_devices=torch.cuda.device_count(),
    )


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError):
        return None


def _inactive_file(directory: Path) -> int:
    for line in (_read_text(directory / "memory.stat") or "").splitlines():
        key, _, value = line.partition(" ")
        if key == "inactive_file" and value.isdigit():
            return int(value)
    return 0


def cgroup_memory_headroom(
    cgroup_root: Path = CGROUP_ROOT, proc_self_cgroup: Path = PROC_SELF_CGROUP
) -> int | None:
    """Smallest ``memory.max - (memory.current - inactive_file)`` over this process's cgroup v2 and
    its ancestors; None when no level sets a limit (no cgroup v2, or ``memory.max`` is ``max``)."""
    entry = next(
        (
            line[3:]
            for line in (_read_text(proc_self_cgroup) or "").splitlines()
            if line.startswith("0::")
        ),
        None,
    )
    if entry is None:
        return None
    root = cgroup_root.resolve()
    directory = (root / entry.lstrip("/")).resolve()
    if directory != root and root not in directory.parents:
        return None
    headroom = None
    while True:
        limit = _read_text(directory / "memory.max")
        current = _read_text(directory / "memory.current")
        if limit and limit.isdigit() and current and current.isdigit():
            used = max(int(current) - _inactive_file(directory), 0)
            free = max(int(limit) - used, 0)
            headroom = free if headroom is None else min(headroom, free)
        if directory == root:
            return headroom
        directory = directory.parent


def available_memory_bytes() -> int:
    """Memory this process can take now without swapping: host and container limits."""
    available = psutil.virtual_memory().available
    headroom = cgroup_memory_headroom()
    return available if headroom is None else min(available, headroom)


def check_worker_memory(dataset_path: Path) -> tuple[int, int]:
    """Return (processed dataset bytes, available bytes).

    Raises:
        HPOError: The processed dataset is larger than the available memory
    """
    dataset_bytes = Path(dataset_path).stat().st_size
    available = available_memory_bytes()
    if dataset_bytes > available:
        raise HPOError(
            f"HPO worker: the processed dataset ({_gib(dataset_bytes)}) is larger than the memory "
            f"available now ({_gib(available)}); loading it would swap or be killed",
            details=(
                "Start fewer workers on this host, give the container more memory, or use a host "
                "with more memory. Workers started earlier hold their own copy of the dataset"
            ),
        )
    return dataset_bytes, available


def _gib(size: int) -> str:
    return f"{size / 2**30:.2f} GiB"
