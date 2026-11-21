"""timm backbone + supervised 9-class training loop (PR-5, LLD §6).

`PatchClassifier`: convnext_tiny (timm) trunk -> 768-d features -> 9 logits
(the features are the interface later consumed by AttentionMIL, PR-9).

`Trainer`: bf16 autocast + channels_last + EMA + AdamW/cosine +
label smoothing, all hyperparams from `TrainSettings` (configs/default.yaml).
Determinism is part of the contract (PLAN PR-5 RED-d): seeded runs produce
byte-identical metrics JSON — no timestamps in artifacts. VRAM peak is logged
into every metrics JSON. Checkpoints follow LLD §6: no orphan fields.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # cublas determinism (must precede matmul handle init)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # 12GB card fragmentation

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from histogate.config import TrainSettings
from histogate.data import CLASS_ORDER, CLASS_TO_IDX, CachedPatches
from histogate.metrics import ece, macro_f1, quadratic_kappa
from histogate.tracking import MlflowTracker

if TYPE_CHECKING:
    from histogate.stainaug import StainAugmenter

N_CLASSES = len(CLASS_ORDER)
# ImageNet normalization == timm's resolved data config for convnext_tiny pretrained
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

_AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def _git_sha() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


def config_hash(cfg: TrainSettings) -> str:
    return hashlib.sha256(
        json.dumps(cfg.model_dump(), sort_keys=True).encode()
    ).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # ops without a deterministic CUDA kernel: fall back and let the
        # byte-identity determinism TEST (PR-5 RED-d) be the arbiter
        torch.use_deterministic_algorithms(True, warn_only=True)


class PatchClassifier(nn.Module):
    """convnext_tiny trunk -> 9 logits; `.features()` returns the 768-d pooled
    embedding (input to AttentionMIL, PR-9)."""

    def __init__(self, backbone: str = "convnext_tiny", pretrained: bool = True,
                 num_classes: int = N_CLASSES, grad_checkpoint: bool = False) -> None:
        super().__init__()
        import timm

        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        self.head = nn.Linear(self.backbone.num_features, num_classes)
        # activation checkpointing: keeps batch-256 convnext_tiny training inside
        # the RTX 3060 12GB budget (~10GB peak un-checkpointed + co-tenant GPU
        # processes); no effect on values, ~30% step-time cost, off by default.
        self.grad_checkpoint = grad_checkpoint

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if self.grad_checkpoint and self.training and torch.is_grad_enabled():
            import torch.utils.checkpoint as cp

            b = self.backbone
            x = b.stem(x)
            for stage in b.stages:
                x = cp.checkpoint(stage, x, use_reentrant=False)
            return b.head(x)
        return self.backbone(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class PatchDataset(Dataset):
    """torch view over the PR-4 uint8 memmap cache.

    Augmentation is derived per (seed, epoch, index) so results are identical
    regardless of worker count/scheduling. `shuffle_labels` is the PR-5
    chance-control negative switch (deterministic permutation).
    """

    def __init__(self, cached: CachedPatches, train: bool = True, *,
                 shuffle_labels: bool = False, seed: int = 42, augment: bool = True,
                 stain_augment: StainAugmenter | None = None) -> None:
        self.cached = cached
        self.train = train
        self.augment = augment and train
        self.stain_augment = stain_augment if train else None
        self.seed = seed
        self.epoch = 0
        labels = np.array([CLASS_TO_IDX[e.label] for e in cached.index.entries], dtype=np.int64)
        if shuffle_labels:
            labels = np.random.default_rng(seed).permutation(labels)
        self.labels = labels

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.cached)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        img, _label = self.cached[int(i)]
        img = np.asarray(img)
        rng = np.random.default_rng((self.seed % 1_000_003) * 4_194_319 + self.epoch * 1_000_003 + int(i))
        if self.augment:
            img = np.rot90(img, int(rng.integers(0, 4)))
            if rng.random() < 0.5:
                img = img[:, ::-1]
        if self.stain_augment is not None:
            # AFTER geometric augs (uint8 HWC), BEFORE normalize; rng is
            # per-(seed,epoch,index) => worker-count invariant. Explicit
            # kwarg applies independently of the geometric `augment` flag.
            img = self.stain_augment(np.ascontiguousarray(img), rng=rng)
        x = torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)
        x = (x - _MEAN) / _STD
        return x, int(self.labels[i])


class EMA:
    """Exponential moving average of floating-point state (non-float buffers
    are excluded from averaging but still swapped so eval sees full state)."""

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            v.mul_(self.decay).add_(sd[k].detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def swap(self, model: nn.Module) -> None:
        sd = model.state_dict()
        for k, v in self.shadow.items():
            tmp = sd[k].detach().clone()
            sd[k].copy_(v)
            v.copy_(tmp)


def _seed_worker(worker_id: int) -> None:
    w = torch.initial_seed() % 2**32
    np.random.seed(w)
    random.seed(w)


def _shutdown_workers(loader: DataLoader) -> None:
    """Explicitly join DataLoader workers: a crash mid-epoch otherwise leaks
    their shared-memory segments (parent-diagnosed on this box, 2026-11-20)."""
    it = getattr(loader, "_iterator", None)
    if it is not None:
        shutdown = getattr(it, "_shutdown_workers", None)
        if callable(shutdown):
            shutdown()


def evaluate(model: nn.Module, ds: Dataset, cfg: TrainSettings,
             device: torch.device | str | None = None) -> dict[str, float]:
    """Val metrics (EMA weights expected to already be swapped into `model`)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model.eval()
    # Validation runs AFTER a training epoch whose worker pool is still
    # winding down; spawning a second worker IPC pool here dead-locked
    # shm on this box (parent-diagnosed 2026-11-20: ENOMEM in
    # _new_using_filename_cpu during val collate, size-dependent).
    # Val is 4 batches at smoke scale — worker IPC buys nothing.
    loader = DataLoader(ds, batch_size=cfg.batch, shuffle=False,
                        num_workers=0, pin_memory=device.type == "cuda")
    zs, ys = [], []
    amp = _AMP[cfg.amp]
    try:
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device).to(memory_format=torch.channels_last)
                with torch.autocast(device_type=device.type, dtype=amp, enabled=amp != torch.float32):
                    out = model(x)
                zs.append(out.float().cpu())
                ys.append(y)
    finally:
        _shutdown_workers(loader)
    z = torch.cat(zs).numpy()
    y = torch.cat(ys).numpy()
    pred = z.argmax(1)
    return {
        "val_acc": float((pred == y).mean()),
        "val_macro_f1": macro_f1(y, pred),
        "val_kappa": quadratic_kappa(y, pred),
        "val_ece": ece(z, y),
    }


class Trainer:
    def __init__(self, cfg: TrainSettings, run_dir: Path | str, run_name: str = "run",
                 device: str | None = None,
                 tracker: MlflowTracker | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_name = run_name
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.metrics_path = self.run_dir / "metrics.json"
        self.ckpt_path = self.run_dir / f"{run_name}.ckpt.pt"
        self.config_hash = config_hash(cfg)
        self.git_sha = _git_sha()
        self.last_metrics: dict | None = None
        self.tracker = tracker or MlflowTracker(
            experiment="histogate", tracking_uri="mlruns", enabled=False
        )

    def _loader(self, ds: Dataset, shuffle: bool) -> DataLoader:
        g = torch.Generator()
        g.manual_seed(self.cfg.seed)
        return DataLoader(ds, batch_size=self.cfg.batch, shuffle=shuffle,
                          num_workers=self.cfg.workers, pin_memory=self.device.type == "cuda",
                          generator=g if shuffle else None, worker_init_fn=_seed_worker,
                          persistent_workers=False)

    def fit(self, model: nn.Module, train_ds: Dataset, val_ds: Dataset,
            resume: bool = False) -> dict:
        """Train with MLflow as the primary tracker (runs/ JSON stays secondary).

        resume=True continues from run_dir/state.json + last.ckpt.pt; completed
        epochs are skipped (fail-closed if state is missing)."""
        with self.tracker.run(self.run_name):
            self.tracker.log_params({
                **self.cfg.model_dump(mode="json"),
                "git_sha": self.git_sha,
                "config_hash": self.config_hash,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
                "resume": resume,
            })
            return self._fit_inner(model, train_ds, val_ds, resume=resume)

    def _fit_inner(self, model: nn.Module, train_ds: Dataset, val_ds: Dataset,
                   resume: bool = False) -> dict:
        cfg = self.cfg
        seed_everything(cfg.seed)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        model.to(self.device)
        model.to(memory_format=torch.channels_last)
        amp = _AMP[cfg.amp]
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.05)
        crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
        # lr 1e-3 (config law) destabilizes ConvNeXt-T without warmup: measured
        # collapse to uniform outputs within 5 steps. Linear warmup of at least
        # 20 steps (5% of long runs) + cosine over the remainder is stable and
        # lets the PR-5 one-batch overfit probe run at SPEC lr/batch.
        steps_per_epoch = math.ceil(len(train_ds) / cfg.batch)
        total_steps = cfg.epochs * steps_per_epoch
        warmup_steps = max(20, int(0.05 * total_steps))

        def _lr_at(step: int) -> float:
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_at)
        ema = EMA(model, cfg.ema)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        # ---- resume state (fail-closed) ---------------------------------
        state_path = self.run_dir / "state.json"
        last_path = self.run_dir / "last.ckpt.pt"
        best_path = self.run_dir / "best.ckpt.pt"
        start_epoch, global_step, best_f1 = 0, 0, -1.0
        if resume:
            if not state_path.exists():
                raise FileNotFoundError(
                    f"resume requested but no state file: {state_path} (fail-closed)"
                )
            ck = torch.load(last_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            ema.shadow = {k: v.to(self.device) for k, v in ck["ema"].items()}
            start_epoch = int(ck["epoch"]) + 1
            global_step = int(ck["global_step"])
            best_f1 = float(ck["best_f1"])
            print(f"[resume] continuing at epoch {start_epoch} "
                  f"(best_f1={best_f1:.4f}, step={global_step})")

        loader = self._loader(train_ds, shuffle=True)
        epochs_out: list[dict] = []
        step_losses: list[float] = []
        LOG_EVERY = 50  # steps — dense TensorBoard-style curve in mlflow
        try:
            for epoch in range(start_epoch, cfg.epochs):
                inner = train_ds.dataset if isinstance(train_ds, Subset) else train_ds
                if hasattr(inner, "set_epoch"):
                    inner.set_epoch(epoch)
                model.train()
                losses: list[float] = []
                for x, y in loader:
                    x = x.to(self.device, non_blocking=True).to(memory_format=torch.channels_last)
                    y = y.to(self.device, non_blocking=True)
                    with torch.autocast(device_type=self.device.type, dtype=amp,
                                        enabled=amp != torch.float32):
                        loss = crit(model(x), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    global_step += 1
                    step_losses.append(float(loss.item()))
                    sched.step()
                    ema.update(model)
                    losses.append(float(loss.item()))
                    if global_step % LOG_EVERY == 0 and step_losses:
                        self.tracker.log_step(
                            global_step, {"train_loss": float(np.mean(step_losses))}
                        )
                        step_losses = []
                # (per-step warmup+cosine schedule stepped above)

                # drain train worker IPC BEFORE validation (parent-diagnosed
                # shm collision: two concurrent worker pools, 2026-11-20)
                _shutdown_workers(loader)
                ema.swap(model)  # validate EMA weights
                val = evaluate(model, val_ds, cfg, self.device)
                # best checkpoint captures the EMA weights WHILE they are
                # loaded in the model (before the swap back)
                if val["val_macro_f1"] > best_f1:
                    best_f1 = val["val_macro_f1"]
                    torch.save(
                        {
                            "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                            "config_hash": self.config_hash,
                            "git_sha": self.git_sha,
                            "metrics": {"final": {"epoch": epoch, **val}},
                            "kind": "best-ema",
                        },
                        best_path,
                    )
                ema.swap(model)
                # per-epoch resume state: model + optimizer + scheduler + EMA
                torch.save(
                    {
                        "model": model.state_dict(),
                        "opt": opt.state_dict(),
                        "sched": sched.state_dict(),
                        "ema": ema.shadow,
                        "epoch": epoch,
                        "global_step": global_step,
                        "best_f1": best_f1,
                        "config_hash": self.config_hash,
                        "git_sha": self.git_sha,
                    },
                    last_path,
                )
                state_path.write_text(json.dumps(
                    {"epoch": epoch, "global_step": global_step,
                     "best_f1": best_f1, "git_sha": self.git_sha,
                     "config_hash": self.config_hash}
                ) + "\n")
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()  # return eval blocks before next epoch

                epochs_out.append({
                    "epoch": epoch,
                    "train_loss": float(np.mean(losses)),
                    "lr": float(sched.get_last_lr()[0]),
                    **val,
                })
                # epoch-indexed series: val_* + lr only (train_loss lives on the
                # dense step-indexed series — mixing both in one metric corrupts the chart)
                self.tracker.log_epoch(
                    epoch, {k: v for k, v in epochs_out[-1].items() if k != "train_loss"}
                )
        finally:
            _shutdown_workers(loader)

        peak_mb = (
            torch.cuda.max_memory_allocated() / 1024**2
            if self.device.type == "cuda" else 0.0
        )
        metrics = {
            "run": self.run_name,
            "config": cfg.model_dump(mode="json"),
            "config_hash": self.config_hash,
            "git_sha": self.git_sha,
            "seed": cfg.seed,
            "n_train": len(train_ds),
            "n_val": len(val_ds),
            "epochs": epochs_out,
            "final": dict(epochs_out[-1]),
            "peak_vram_mb": round(peak_mb, 1),
        }
        self.last_metrics = metrics
        self.tracker.annotate(metrics)  # cross-reference: where this run lives in mlflow
        self.tracker.cur_metrics_path = str(self.metrics_path)
        self.metrics_path.write_text(json.dumps(metrics, sort_keys=True, indent=2) + "\n")
        torch.save(
            {
                "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "config_hash": self.config_hash,
                "git_sha": self.git_sha,
                "metrics": metrics,
            },
            self.ckpt_path,
        )
        self.tracker.log_final(metrics)
        self.tracker.log_artifacts({"metrics": self.metrics_path, "ckpt": self.ckpt_path})
        return metrics


class AttentionMIL(nn.Module):
    """Slide-level head over patch embeddings (gated attention pooling, Ilse 2018).

    O(N) in instances — softmax over a per-instance score, never NxN — sized
    for <=2000 patches/slide (PLAN PR-9 perf law). The PatchClassifier's
    feature extractor feeds this; slide logits + attention weights come out.
    """

    def __init__(self, in_dim: int, n_classes: int, attn_dim: int = 128) -> None:
        super().__init__()
        self.V = nn.Linear(in_dim, attn_dim)
        self.U = nn.Linear(in_dim, attn_dim)
        self.w = nn.Linear(attn_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, 256),
            nn.GELU(),
            nn.Linear(256, n_classes),
        )

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) -> (B, N) attention weights (sum to 1 over N)."""
        a = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)
        return torch.softmax(a, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, N, D) instance features -> (B, n_classes) slide logits."""
        z = torch.einsum("bn,bnd->bd", self.attention(x), x)
        return self.head(z)
