"""
Processed datasets and processing chunks load with ``torch.load(weights_only=True)`` only (F58).

MILIA writes both in a layout of plain dictionaries plus a PyG data class, so no file it reads
needs full unpickling. An older-layout file (a pickled ``Data`` object) or a crafted one is
reported as stale without its payload running, and ``miliaDataset`` rebuilds it once from the
raw data instead of falling back to full unpickling as PyG's ``InMemoryDataset.load`` does.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
import torch
from torch_geometric.data import Data, InMemoryDataset

from milia_pipeline.config.config_containers import DatasetConfig, FilterConfig, ProcessingConfig
from milia_pipeline.datasets.milia_dataset import miliaDataset
from milia_pipeline.datasets.processed_io import (
    StaleProcessedFileError,
    load_chunk,
    load_processed,
    save_chunk,
    save_processed,
)
from milia_pipeline.exceptions import DataProcessingError


class _Payload:
    """Pickles to a call of ``Path.touch(marker)``: unpickling it creates the marker file."""

    def __init__(self, marker: Path):
        self.marker = marker

    def __reduce__(self):
        return (Path.touch, (self.marker,))


def _molecules(*, descriptor_names: bool = True) -> list[Data]:
    """Two graphs with the attribute kinds MILIA stores: tensors, a str and a list of str.

    ``descriptor_names`` is off for the miliaDataset path: its collate keeps descriptor names
    only alongside descriptor vectors, which is unrelated to how the file is stored.
    """
    extra = {"descriptor_names": ["mw", "logp"]} if descriptor_names else {}
    return [
        Data(
            x=torch.randn(n, 4),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            y=torch.tensor([[float(n)]]),
            inchi=f"InChI=1S/test{n}",
            **extra,
        )
        for n in (2, 3)
    ]


def _assert_same(loaded: list[Data], expected: list[Data]) -> None:
    assert len(loaded) == len(expected)
    for got, want in zip(loaded, expected, strict=True):
        assert torch.equal(got.x, want.x)
        assert torch.equal(got.edge_index, want.edge_index)
        assert got.inchi == want.inchi
        if "descriptor_names" in want:
            assert got.descriptor_names == want.descriptor_names


def _write_processed(path: Path, data_list: list[Data]) -> None:
    data, slices = InMemoryDataset.collate(data_list)
    save_processed(data, slices, path)


def _unbatch(data, slices) -> list[Data]:
    holder = InMemoryDataset()
    holder.data, holder.slices = data, slices
    return [holder.get(i) for i in range(len(holder))]


def test_chunk_round_trip(tmp_path):
    path = tmp_path / "chunk_0.pt"
    molecules = _molecules()
    save_chunk(molecules, path)
    _assert_same(load_chunk(path), molecules)


def test_chunk_rejects_mixed_data_classes(tmp_path):
    class OtherData(Data):
        pass

    with pytest.raises(ValueError, match="one data class"):
        save_chunk([Data(x=torch.zeros(1, 1)), OtherData(x=torch.zeros(1, 1))], tmp_path / "c.pt")


def test_processed_round_trip_loads_without_pyg_fallback(tmp_path, recwarn):
    path = tmp_path / "processed.pt"
    molecules = _molecules()
    _write_processed(path, molecules)
    data, slices = load_processed(path)
    _assert_same(_unbatch(data, slices), molecules)
    assert not [w for w in recwarn if "Weights only load failed" in str(w.message)]


def test_older_layout_is_stale(tmp_path):
    """A pickled Data object (the previous processed and chunk layout) needs PyG storage globals."""
    data, slices = InMemoryDataset.collate(_molecules())
    legacy = tmp_path / "legacy.pt"
    torch.save((data, slices), legacy)
    with pytest.raises(StaleProcessedFileError) as stale:
        load_processed(legacy)
    message = str(stale.value)
    assert "rejected global torch_geometric." in message
    assert "weights_only=False" not in message and "safe_globals" not in message
    legacy_chunk = tmp_path / "legacy_chunk.pt"
    torch.save(_molecules(), legacy_chunk)
    with pytest.raises(StaleProcessedFileError):
        load_chunk(legacy_chunk)


@pytest.mark.parametrize("loader", [load_processed, load_chunk])
def test_crafted_file_is_stale_and_payload_never_runs(tmp_path, loader):
    marker = tmp_path / "payload-ran"
    path = tmp_path / "crafted.pt"
    torch.save(([{"x": _Payload(marker)}], {}, Data), path)
    with pytest.raises(StaleProcessedFileError):
        loader(path)
    assert not marker.exists()


@pytest.mark.parametrize("loader", [load_processed, load_chunk])
def test_corrupt_truncated_or_wrong_shape_file_is_stale(tmp_path, loader):
    whole = tmp_path / "whole.pt"
    _write_processed(whole, _molecules())
    damaged = {
        "garbage": b"not a torch file",
        "short": b"junk",  # torch raises struct.error here
        "truncated": whole.read_bytes()[: whole.stat().st_size // 2],
    }
    for name, content in damaged.items():
        corrupt = tmp_path / f"{name}.pt"
        corrupt.write_bytes(content)
        with pytest.raises(StaleProcessedFileError):
            loader(corrupt)
    wrong = tmp_path / "wrong.pt"
    torch.save({"x": torch.zeros(1)}, wrong)
    with pytest.raises(StaleProcessedFileError):
        loader(wrong)


@pytest.mark.parametrize("loader", [load_processed, load_chunk])
def test_missing_file_is_an_io_error_not_stale(tmp_path, loader):
    with pytest.raises(FileNotFoundError):
        loader(tmp_path / "absent.pt")


def _dataset(root: Path) -> miliaDataset:
    with (
        patch.object(miliaDataset, "_download", return_value=None),
        patch.object(miliaDataset, "_process", return_value=None),
    ):
        return miliaDataset(
            root=str(root),
            dataset_config=DatasetConfig(dataset_type="DFT"),
            filter_config=FilterConfig(),
            processing_config=ProcessingConfig(scalar_graph_targets=[]),
        )


def _legacy_processed_file(tmp_path) -> Path:
    processed_path = Path(_dataset(tmp_path).processed_paths[0])
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    data, slices = InMemoryDataset.collate(_molecules())
    torch.save((data, slices), processed_path)
    return processed_path


def test_processing_writes_chunks_and_processed_file_that_load_strictly(tmp_path, recwarn):
    """Real _save_chunk -> _consolidate_chunks -> _finalize_dataset path, then a strict reopen."""
    dataset = _dataset(tmp_path)
    dataset.processed_chunk_dir.mkdir(parents=True, exist_ok=True)
    molecules = _molecules(descriptor_names=False)
    chunks = [dataset._save_chunk(molecules[:1], 0), dataset._save_chunk(molecules[1:], 1)]
    _assert_same(load_chunk(chunks[0]) + load_chunk(chunks[1]), molecules)

    dataset._finalize_dataset(chunks)
    load_processed(dataset.processed_paths[0])  # strict read; miliaDataset.get un-collates it

    with patch.object(miliaDataset, "process", autospec=True) as process:
        reopened = _dataset(tmp_path)
    process.assert_not_called()
    _assert_same([reopened.get(i) for i in range(len(reopened))], molecules)
    assert not [w for w in recwarn if "Weights only load failed" in str(w.message)]


def test_dataset_rebuilds_an_older_layout_file_once(tmp_path):
    processed_path = _legacy_processed_file(tmp_path)
    molecules = _molecules()

    def rebuild(self):
        _write_processed(processed_path, molecules)

    with patch.object(miliaDataset, "process", autospec=True, side_effect=rebuild) as process:
        dataset = _dataset(tmp_path)
    process.assert_called_once()
    _assert_same([dataset.get(i) for i in range(len(dataset))], molecules)

    with patch.object(miliaDataset, "process", autospec=True) as process:
        _dataset(tmp_path)
    process.assert_not_called()


def test_dataset_raises_when_the_rebuilt_file_is_still_unreadable(tmp_path):
    _legacy_processed_file(tmp_path)
    with (
        patch.object(miliaDataset, "process", autospec=True) as process,
        pytest.raises(DataProcessingError),
    ):
        _dataset(tmp_path)
    process.assert_called_once()
