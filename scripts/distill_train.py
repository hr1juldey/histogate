"""Stage B (plan C3): distill cached teacher embeddings into the 6M student.

Loss = CE(y) + w_kl * KL(teacher_soft || student_soft; T=2)
                 + w_mse * MSE(proj(student_feat) -> teacher_emb)
with a learned 768->teacher_dim projection on the student's backbone features
(the SAME feature path the KNN gate uses — model.features).

Teacher inputs come from an embedding cache built by scripts/extract_teacher.py
(any HF vision encoder; the license gate lives THERE, not here). Teacher soft
targets come from a logistic head fitted on the cache (minutes), so no teacher
forward ever runs during training.

Run (after the teacher cache exists):
  uv run python scripts/distill_train.py --teacher-cache runs/distill/<model> \
      --run-name distill-v1 --init-from runs/pr16-mixed/best.ckpt.pt
Smoke (no teacher needed, no license touched):
  uv run python scripts/distill_train.py --synthetic-smoke
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from histogate.config import load_settings
from histogate.data import (
    CachedPatches,
    PatchIndex,
    build_cache,
    build_index,
)
from histogate.distill_cache import extract_embeddings
from histogate.model import N_CLASSES, PatchClassifier

ROOT = Path(__file__).resolve().parents[1]


class _TeacherSide:
    """Row lookup for the teacher cache + a logistic soft-target head."""

    def __init__(self, cache_dir: Path, seed: int = 42) -> None:
        idx = json.loads((cache_dir / "files.json").read_text())
        self.emb = np.load(cache_dir / "emb.npy", mmap_mode="r")
        self.row = {f: i for i, f in enumerate(idx["files"])}
        self.dim = int(idx["dim"])
        y = np.asarray(idx["labels"])
        rng = np.random.default_rng(seed)
        fit = rng.permutation(len(y))[:60000]
        from sklearn.linear_model import LogisticRegression

        self.head = LogisticRegression(max_iter=1000, n_jobs=8, C=1.0)
        self.head.fit(
            np.asarray(self.emb[fit], dtype=np.float32), y[fit])
        hold = fit[:5000]
        acc = float(
            (self.head.predict(np.asarray(self.emb[hold], dtype=np.float32))
             == y[hold]).mean())
        print(f"[distill] teacher head fit (5k train-hold acc {acc:.4f}, "
              f"dim {self.dim})")

    def _rows(self, files: list[str]) -> list[int]:
        return [self.row[f] for f in files]

    def soft(self, files: list[str]) -> torch.Tensor:
        """Teacher LOG-probabilities (temperature applies once, Hinton-law)."""
        rows = self._rows(files)
        x = np.asarray(self.emb[rows], dtype=np.float32)
        return torch.from_numpy(
            np.log(np.clip(self.head.predict_proba(x), 1e-9, 1))).float()

    def embed(self, files: list[str]) -> torch.Tensor:
        return torch.from_numpy(
            np.asarray(self.emb[self._rows(files)], dtype=np.float32))


class _DistillDS(Dataset):
    """Images (uint8 cache) + aligned teacher soft targets / embeddings."""

    def __init__(self, patches: CachedPatches, rel_files: list[str],
                 teacher: _TeacherSide) -> None:
        self.patches = patches
        self.rel_files = rel_files
        self.teacher = teacher

    def __len__(self) -> int:
        return len(self.rel_files)

    def __getitem__(self, i: int):
        img, y = self.patches[i]  # (H,W,3) uint8, int label
        return (torch.from_numpy(img.copy()).permute(2, 0, 1).float() / 255,
                int(y),
                self.teacher.soft([self.rel_files[i]])[0],
                self.teacher.embed([self.rel_files[i]])[0])


def _subset(index: PatchIndex, n: int) -> PatchIndex:
    """Same deterministic class-balanced round-robin as cli._balanced_subset."""
    from histogate.data import CLASS_ORDER

    by: dict[str, list] = {c: [] for c in CLASS_ORDER}
    for e in index.entries:
        by[e.label].append(e)
    cap = (n + len(CLASS_ORDER) - 1) // len(CLASS_ORDER)
    picked = []
    for i in range(cap):
        for c in CLASS_ORDER:
            if i < len(by[c]):
                picked.append(by[c][i])
    return PatchIndex(picked[:n])


def _rel(e_path: Path) -> str:
    p = Path(e_path)
    return str(p.relative_to(ROOT)) if p.is_absolute() else str(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-cache", type=Path, default=None)
    ap.add_argument("--run-name", default="distill-v1")
    ap.add_argument("--init-from", type=Path,
                    default=ROOT / "runs/pr16-mixed/best.ckpt.pt")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--w-mse", type=float, default=0.5)
    ap.add_argument("--w-kl", type=float, default=1.0)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--synthetic-smoke", action="store_true",
                    help="tiny synthetic teacher cache keyed to 64 real tiles, "
                         "1 epoch — proves loop mechanics, no license touched")
    args = ap.parse_args()

    settings = load_settings(None)
    raw = settings.paths.data_dir / "raw"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    if args.synthetic_smoke:
        args.subset = args.subset or 64
        args.epochs = 1

    train_idx = build_index(raw / "NCT-CRC-HE-100K")
    if args.subset:
        train_idx = _subset(train_idx, args.subset)

    if args.synthetic_smoke:
        cache = ROOT / "runs/distill/smoke-teacher/nct"
        rel_files = [_rel(e.path) for e in train_idx.entries[: args.subset]]
        extract_embeddings(
            rel_files,
            lambda b: rng.standard_normal((len(b), 8)).astype(np.float32),
            cache, dim=8, bs=16)
        idx = json.loads((cache / "files.json").read_text())
        from histogate.data import CLASS_TO_IDX

        idx["labels"] = [
            int(CLASS_TO_IDX[train_idx.entries[i].label])
            for i in range(len(rel_files))]
        idx["classes"] = None  # smoke only; labels carry the info
        (cache / "files.json").write_text(json.dumps(idx))
        args.teacher_cache = cache.parent

    assert args.teacher_cache is not None
    teacher = _TeacherSide(args.teacher_cache / "nct")

    rel_files = [_rel(e.path) for e in train_idx.entries]
    missing = [f for f in rel_files if f not in teacher.row]
    if missing:
        raise SystemExit(
            f"{len(missing)} train files absent from the teacher cache "
            f"(first: {missing[0]}) — extract the teacher cache first")
    cache_bin = settings.paths.data_dir / "cache" / (
        "distill_smoke.bin" if args.synthetic_smoke else "distill_train.bin")
    if not cache_bin.exists():
        build_cache(train_idx, cache_bin)
    patches = CachedPatches(cache_bin, train_idx, 224)
    ds = _DistillDS(patches, rel_files, teacher)
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=4,
                    drop_last=True)

    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    if args.init_from and Path(args.init_from).exists():
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(src["state_dict"])
        print(f"[distill] warm-start from {args.init_from}")
    proj = nn.Linear(model.backbone.num_features, teacher.dim)
    model.to(dev).train()
    proj.to(dev)
    opt = torch.optim.AdamW(
        list(model.parameters()) + list(proj.parameters()), lr=args.lr)

    run_dir = ROOT / "runs" / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    from histogate.train_utils import distill_loss

    step = 0
    for epoch in range(args.epochs):
        for img, y, t_soft, t_emb in dl:
            img, y = img.to(dev), y.to(dev)
            t_soft, t_emb = t_soft.to(dev), t_emb.to(dev)
            with torch.autocast(dev if dev == "cuda" else "cpu",
                                dtype=torch.bfloat16):
                feats = model.features(img)
                logits = model.head(feats)
                loss, parts = distill_loss(
                    logits.float(), y, t_soft, t_emb, feats.float(), proj,
                    w_kl=args.w_kl, w_mse=args.w_mse)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0:
                print(f"e{epoch} s{step} loss {loss.item():.3f} {parts}",
                      flush=True)
            step += 1
    torch.save(
        {"state_dict": model.state_dict(), "epoch": args.epochs,
         "teacher_cache": str(args.teacher_cache), "args": vars(args)},
        run_dir / "last.ckpt.pt")
    print(f"[distill] saved {run_dir / 'last.ckpt.pt'}")


if __name__ == "__main__":
    main()
