"""Data layer: patch index with patient integrity, distribution, deterministic cache.

Spec: PRD FR-1, LLD §4. Pure numpy/tifffile — no torch here (PR-5 wraps it).
Filename convention (NCT-CRC-HE-100K): <CLASS>-<PATIENT>.tif under <CLASS>/ dirs.
(Zenodo direct download; the HF mirror uses unsupported legacy scripts.)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile

from histogate.schemas.report import TissueClass

CLASS_ORDER: tuple[str, ...] = tuple(c.value for c in TissueClass)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_ORDER)}

_FNAME_RE = re.compile(r"^(?P<label>[A-Z]+)-(?P<patient>.+)\.(?:tif|tiff|png|jpg)$", re.I)


class PatientLeakError(Exception):
    """Raised when the same patient appears in more than one split."""


@dataclass(frozen=True)
class PatchEntry:
    path: Path
    patient_id: str
    label: str


class PatchIndex:
    def __init__(self, entries: list[PatchEntry]) -> None:
        self.entries = sorted(entries, key=lambda e: str(e.path))

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def patients(self) -> set[str]:
        return {e.patient_id for e in self.entries}


def build_index(split_dir: Path | str) -> PatchIndex:
    """Index <split_dir>/<CLASS>/<CLASS>-<PATIENT>.<ext>; dir/file label mismatch rejects."""
    split_dir = Path(split_dir)
    if not split_dir.is_dir():
        raise FileNotFoundError(f"split dir not found: {split_dir}")
    entries: list[PatchEntry] = []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        label = class_dir.name.upper()
        if label not in CLASS_TO_IDX:
            raise ValueError(f"unknown class dir '{class_dir.name}' in {split_dir}")
        for f in sorted(class_dir.iterdir()):
            m = _FNAME_RE.match(f.name)
            if m is None:
                raise ValueError(f"unparseable patch filename: {f}")
            if m.group("label").upper() != label:
                raise ValueError(
                    f"filename label {m.group('label')} != directory label {label}: {f}"
                )
            entries.append(PatchEntry(path=f, patient_id=m.group("patient"), label=label))
    if not entries:
        raise ValueError(f"no patches found under {split_dir}")
    return PatchIndex(entries)


def assert_no_patient_overlap(indexes: dict[str, PatchIndex]) -> None:
    """The leakage guard: any patient in two splits is a hard error (FR-1)."""
    names = list(indexes)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = indexes[names[i]].patients & indexes[names[j]].patients
            if shared:
                raise PatientLeakError(
                    f"patient leakage between '{names[i]}' and '{names[j]}': "
                    f"{sorted(shared)[:10]} ({len(shared)} total)"
                )


def class_distribution(index: PatchIndex) -> dict[str, int]:
    dist = {c: 0 for c in CLASS_ORDER}
    for e in index.entries:
        dist[e.label] += 1
    return dist


class CachedPatches:
    """Memory-mapped uint8 cache of a patch index. Deterministic build order."""

    def __init__(self, memmap_path: Path | str, index: PatchIndex, size: int) -> None:
        self.path = Path(memmap_path)
        self.index = index
        self.size = size
        self._mm = np.memmap(self.path, dtype=np.uint8, mode="r",
                             shape=(len(index), size, size, 3))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> tuple[np.ndarray, int]:
        return np.asarray(self._mm[i]), CLASS_TO_IDX[self.index.entries[i].label]


def build_cache(index: PatchIndex, out_path: Path | str, size: int = 224) -> CachedPatches:
    """Pack patches into one uint8 memmap, sorted-path order => byte-deterministic."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mm = np.memmap(out_path, dtype=np.uint8, mode="w+",
                   shape=(len(index), size, size, 3))
    for i, e in enumerate(index.entries):
        img = tifffile.imread(e.path)
        if img.shape != (size, size, 3):
            raise ValueError(f"unexpected patch shape {img.shape} (expected {(size, size, 3)}): {e.path}")
        mm[i] = img
    mm.flush()
    del mm
    return CachedPatches(out_path, index, size)
