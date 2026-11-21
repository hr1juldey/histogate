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
from histogate.stainaug import StainAugmenter, StainParams
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


def _stain_augmenter(args: argparse.Namespace, cfg) -> StainAugmenter | None:
    """Build the calibrated stain augmenter when --stain-augment is set."""
    if not getattr(args, "stain_augment", False):
        return None
    calib = Path("configs/stain_calib.json")
    return StainAugmenter(StainParams.from_json(json.loads(calib.read_text())),
                          seed=cfg.seed)


def _cache_for(index: PatchIndex, path: Path) -> Path:
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    build_cache(index, path)
    return path


def build_train_parser() -> argparse.ArgumentParser:
    """Standalone train parser (P16-S3: testable CLI surface)."""
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
    p.add_argument("--resume-if-present", action="store_true",
                   help="continue from run state.json if it exists (overnight safety)")
    p.add_argument("--train-dataset", default=TRAIN_DATASET,
                   help="training data dir name under raw/ (e.g. NCT-CRC-HE-100K-NONORM)")
    p.add_argument("--train-dataset2", default=None,
                   help="optional SECOND training data dir (P16 mixed run: "
                        "indexes are concatenated, one shared cache)")
    p.add_argument("--init-from", default=None, type=Path,
                   help="warm-start weights from a checkpoint (domain fine-tune)")
    p.add_argument("--lr", default=None, type=float,
                   help="override train.lr (use ~1e-4 for warm-start fine-tunes)")
    p.add_argument("--stain-augment", action="store_true",
                   help="physically-grounded stain jitter calibrated on real "
                        "scanners (configs/stain_calib.json, PR-16)")
    p.add_argument("--name", default="train")
    return p


def train(argv: list[str] | None = None) -> None:
    """`histogate-train [--config X] [--ssl]` (+ smoke-run conveniences)."""
    # Route DataLoader worker IPC through /tmp (NVMe) instead of /dev/shm:
    # concurrent GPU jobs on this box exhausted shm and dead workers leaked
    # 154MB torch_* segments there (parent-diagnosed 2026-11-20).
    torch.multiprocessing.set_sharing_strategy("file_system")
    args = build_train_parser().parse_args(argv)
    if args.ssl:
        raise SystemExit("--ssl: SSL backbone arrives in PR-16")

    settings = load_settings(args.config)
    if args.epochs is not None:
        settings.train.epochs = args.epochs
    if args.lr is not None:
        settings.train.lr = args.lr
    cfg: TrainSettings = settings.train

    raw = settings.paths.data_dir / "raw"
    train_idx = build_index(raw / args.train_dataset)
    if args.train_dataset2 is not None:
        idx2 = build_index(raw / args.train_dataset2)
        train_idx = PatchIndex(train_idx.entries + idx2.entries)
        print(f"[train] mixed index: {len(train_idx)} patches "
              f"({args.train_dataset} + {args.train_dataset2})")
    if args.subset is not None:
        train_idx = _balanced_subset(train_idx, args.subset)
    val_idx = _balanced_subset(build_index(raw / VAL_DATASET), args.val_subset)

    if args.train_dataset2 is not None:
        tag = (f"{args.train_dataset.lower()[:8]}+{args.train_dataset2.lower()[:8]}"
               f"_{len(train_idx) if args.subset is not None else 'full'}")
    else:
        tag = (f"nct_{len(train_idx)}" if args.subset is not None else "nct_full") \
            if args.train_dataset == TRAIN_DATASET else \
            f"{args.train_dataset.lower()[:8]}_{len(train_idx) if args.subset is not None else 'full'}"
    cache_root = settings.paths.data_dir / "cache"
    train_cache = _cache_for(train_idx, cache_root / f"{tag}.bin")
    val_cache = _cache_for(val_idx, cache_root / f"crcval_{len(val_idx)}.bin")
    train_ds = PatchDataset(CachedPatches(train_cache, train_idx, 224),
                            train=True, shuffle_labels=args.shuffle_labels, seed=cfg.seed,
                            stain_augment=_stain_augmenter(args, cfg))
    val_ds = PatchDataset(CachedPatches(val_cache, val_idx, 224), train=False)

    model = PatchClassifier(backbone=cfg.backbone, pretrained=not args.no_pretrained,
                            grad_checkpoint=args.grad_checkpoint)
    if args.init_from is not None:
        src = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(src["state_dict"])
        print(f"[train] warm-start from {args.init_from} "
              f"(epoch {src.get('epoch', '?')})")
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
    resume = args.resume_if_present and (settings.paths.runs_dir / args.name / "state.json").exists()
    metrics = trainer.fit(model, train_ds, val_ds, resume=resume)
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
    # evaluate() routes inputs to cuda+bfloat16 autocast when a GPU is
    # present; the model must ride along (Trainer does this, the CLI didn't)
    model.to("cuda" if torch.cuda.is_available() else "cpu")

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
    """`histogate-serve [--api|--mcp]`: FastAPI (uvicorn) or FastMCP (stdio)."""
    import argparse as _ap

    p = _ap.ArgumentParser(prog="histogate-serve")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--api", action="store_true", help="FastAPI on serve.host:port")
    mode.add_argument("--mcp", action="store_true", help="FastMCP over stdio")
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args(argv)
    settings = load_settings(args.config)

    import torch

    from histogate.serve.inference import build_analyze

    ckpt = Path("runs/pr16-mixed/best.ckpt.pt")
    calib = Path("runs/pr16-mixed/calibration.json")
    analyze = build_analyze(ckpt, calib,
                            device="cuda" if torch.cuda.is_available() else "cpu")

    if args.mcp:
        from histogate.serve.mcp import build_mcp

        build_mcp(analyze=analyze).run()
        return
    if args.api:
        import uvicorn

        from histogate.serve.api import create_app
        from histogate.serve.engine import LockedEngine
        from histogate.serve.patch_demo import PatchAnalyzer

        app = create_app(engine=LockedEngine(analyze=analyze),
                         slides_root=settings.paths.data_dir,
                         patch_analyzer=PatchAnalyzer(ckpt, calib),
                         legacy_analyzer=PatchAnalyzer(
                             Path("runs/overnight-30ep/best.ckpt.pt"),
                             Path("runs/overnight-30ep/calibration.json")))
        uvicorn.run(app, host=settings.serve.host, port=settings.serve.port)
        return
    raise SystemExit("histogate-serve: pass --api or --mcp")
