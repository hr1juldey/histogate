"""Console entry points (see pyproject [project.scripts]). Thin wrappers over
modules — no business logic here (LLD §15). PR-5 implements train/eval; the
later-PR entry points remain loud stubs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from histogate.config import TrainSettings, load_settings
from histogate.data import (
    CLASS_ORDER,
    CachedPatches,
    PatchEntry,
    PatchIndex,
    build_cache,
    build_index,
)
from histogate.model import N_CLASSES, PatchClassifier, PatchDataset, Trainer, evaluate
from histogate.tracking import MlflowTracker

TRAIN_DATASET = "NCT-CRC-HE-100K"
VAL_DATASET = "CRC-VAL-HE-7K"


def _balanced_subset(index: PatchIndex, n: int) -> PatchIndex:
    """Deterministic class-balanced round-robin subset (smoke runs, tests)."""
    by_class: dict[str, list[PatchEntry]] = {c: [] for c in CLASS_ORDER}
    cap = (n + len(CLASS_ORDER) - 1) // len(CLASS_ORDER)
    for e in index.entries:
        if len(by_class[e.label]) < cap:
            by_class[e.label].append(e)
    picked: list[PatchEntry] = []
    for i in range(cap):
        for c in CLASS_ORDER:
            if len(by_class[c]) > i:
                picked.append(by_class[c][i])
    return PatchIndex(picked[:n])


def _cache_for(index: PatchIndex, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    build_cache(index, path)
    return path


def train(argv: list[str] | None = None) -> None:
    """`histogate-train [--config X] [--ssl]` (+ smoke-run conveniences)."""
    # Route DataLoader worker IPC through /tmp (NVMe) instead of /dev/shm:
    # concurrent GPU jobs on this box exhausted shm and dead workers leaked
    # 154MB torch_* segments there (parent-diagnosed 2026-11-20).
    torch.multiprocessing.set_sharing_strategy("file_system")
    p = argparse.ArgumentParser(prog="histogate-train")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--ssl", action="store_true",
                   help="SSL-pretrained backbone init (PR-16, not yet available)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--subset", type=int, default=None,
                   help="class-balanced subset of training patches (smoke runs)")
    p.add_argument("--val-subset", type=int, default=1024)
    p.add_argument("--shuffle-labels", action="store_true",
                   help="chance-control negative run (PLAN PR-5 RED-a)")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--grad-checkpoint", action="store_true",
                   help="activation checkpointing (batch-256 convnext_tiny on 12GB)")
    p.add_argument("--name", default="train")
    args = p.parse_args(argv)
    if args.ssl:
        raise SystemExit("--ssl: SSL backbone arrives in PR-16")

    settings = load_settings(args.config)
    if args.epochs is not None:
        settings.train.epochs = args.epochs
    cfg: TrainSettings = settings.train

    raw = settings.paths.data_dir / "raw"
    train_idx = build_index(raw / TRAIN_DATASET)
    if args.subset is not None:
        train_idx = _balanced_subset(train_idx, args.subset)
    val_idx = _balanced_subset(build_index(raw / VAL_DATASET), args.val_subset)

    tag = f"nct_{len(train_idx)}" if args.subset is not None else "nct_full"
    cache_root = settings.paths.data_dir / "cache"
    train_cache = _cache_for(train_idx, cache_root / f"{tag}.bin")
    val_cache = _cache_for(val_idx, cache_root / f"crcval_{len(val_idx)}.bin")
    train_ds = PatchDataset(CachedPatches(train_cache, train_idx, 224),
                            train=True, shuffle_labels=args.shuffle_labels, seed=cfg.seed)
    val_ds = PatchDataset(CachedPatches(val_cache, val_idx, 224), train=False)

    model = PatchClassifier(backbone=cfg.backbone, pretrained=not args.no_pretrained,
                            grad_checkpoint=args.grad_checkpoint)
    tracker = MlflowTracker(
        experiment=settings.mlflow.experiment,
        tracking_uri=str((Path.cwd() / settings.mlflow.tracking_uri).resolve())
        if not settings.mlflow.tracking_uri.startswith(("./", "/", "file:", "http:", "sqlite:"))
        else settings.mlflow.tracking_uri,
        enabled=settings.mlflow.enabled,
        system_metrics=settings.mlflow.system_metrics,
    )
    trainer = Trainer(cfg, run_dir=settings.paths.runs_dir / args.name, run_name=args.name,
                      tracker=tracker)
    metrics = trainer.fit(model, train_ds, val_ds)
    print(json.dumps({"final": metrics["final"], "peak_vram_mb": metrics["peak_vram_mb"],
                      "metrics_path": str(trainer.metrics_path)}, indent=2))


def eval(argv: list[str] | None = None) -> None:
    """`histogate-eval`: score a PR-5 checkpoint on CRC-VAL-HE-7K."""
    p = argparse.ArgumentParser(prog="histogate-eval")
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--subset", type=int, default=2048)
    args = p.parse_args(argv)

    settings = load_settings(args.config)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = PatchClassifier(pretrained=False, num_classes=N_CLASSES)
    model.load_state_dict(ckpt["state_dict"])

    val_idx = _balanced_subset(
        build_index(settings.paths.data_dir / "raw" / VAL_DATASET), args.subset
    )
    cache_path = _cache_for(val_idx, settings.paths.data_dir / "cache" / f"crcval_{len(val_idx)}.bin")
    val_ds = PatchDataset(CachedPatches(cache_path, val_idx, 224), train=False)
    results = evaluate(model, val_ds, settings.train)
    out = {
        "checkpoint": str(args.checkpoint),
        "config_hash": ckpt.get("config_hash"),
        "git_sha": ckpt.get("git_sha"),
        "n_val": len(val_ds),
        **results,
    }
    out_path = settings.paths.runs_dir / f"eval_{args.checkpoint.stem}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, sort_keys=True, indent=2) + "\n")
    print(json.dumps(out, indent=2))


def export_bench(argv: list[str] | None = None) -> None:
    raise SystemExit("histogate-export arrives in PR-10 (ONNX export + bench)")


def gate(argv: list[str] | None = None) -> None:
    """`histogate-gate`: blocking regression gate (exit code preserved)."""
    from histogate.gates.eval_gate import main as gate_main

    raise SystemExit(gate_main(argv))


def serve(argv: list[str] | None = None) -> None:
    raise SystemExit("histogate-serve arrives in PR-13/14 (API + MCP)")
