"""
Resolution of the input paths a user passes on the command line (``--model-path``, ``--test-path``).

A relative path is first taken as given, which means relative to the current working directory,
as POSIX path resolution and every other command-line tool do. If no file exists there, the path
is looked up in the configured locations: by file name in ``name_dirs`` (for example the
checkpoints directory), then relative to ``working_root_dir``. An absolute path is used as is.
When nothing exists, :class:`InputPathNotFoundError` lists every location that was checked, in
order. ``main.py`` and the CLI's pre-flight validation use this one function, so they always
agree on which file is read.

Output paths such as ``--preds-path`` are not resolved here: every output MILIA writes goes
under ``working_root_dir``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["InputPathNotFoundError", "PathCandidate", "input_path_candidates", "resolve_input_path"]


@dataclass(frozen=True)
class PathCandidate:
    """One location checked for an input path, with how it was derived."""

    path: Path
    origin: str


class InputPathNotFoundError(FileNotFoundError):
    """No candidate location of an input path exists."""

    def __init__(self, raw: str, candidates: Sequence[PathCandidate]):
        self.raw = raw
        self.candidates = tuple(candidates)
        searched = "\n".join(
            f"  {number}. {candidate.path} ({candidate.origin})"
            for number, candidate in enumerate(self.candidates, start=1)
        )
        super().__init__(f"{raw} not found. Searched locations:\n{searched}")

    @property
    def searched(self) -> str:
        """The numbered list of checked locations (the message without its first line)."""
        return str(self).split("\n", 1)[1]


def input_path_candidates(
    raw: str | Path, working_root_dir: Path, *, name_dirs: Sequence[Path] = ()
) -> tuple[PathCandidate, ...]:
    """Every location checked for ``raw``, in order, without duplicates."""
    given = Path(raw).expanduser()
    if given.is_absolute():
        return (PathCandidate(given, "as provided"),)

    ordered = [PathCandidate(given.absolute(), "relative to the current directory")]
    ordered += [
        PathCandidate(directory / given.name, f"by name in {directory}") for directory in name_dirs
    ]
    ordered.append(PathCandidate(working_root_dir / given, "relative to working_root_dir"))

    unique: list[PathCandidate] = []
    seen: set[Path] = set()
    for candidate in ordered:
        key = candidate.path.absolute()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return tuple(unique)


def resolve_input_path(
    raw: str | Path, working_root_dir: Path, *, name_dirs: Sequence[Path] = ()
) -> Path:
    """Return the first existing candidate location of ``raw``.

    Raises:
        InputPathNotFoundError: None of the candidate locations exists.
    """
    candidates = input_path_candidates(raw, working_root_dir, name_dirs=name_dirs)
    for candidate in candidates:
        if candidate.path.exists():
            return candidate.path
    raise InputPathNotFoundError(str(raw), candidates)
