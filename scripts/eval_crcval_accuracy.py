"""CRC-VAL-HE-7K plain accuracy + macro-F1 for the shipped model.

Apples-to-apples with published baselines (Kather VGG19 94.3%, EffNet-B0 97.7%).
Writes runs/pr16-mixed/crcval_accuracy.json — the only source for any number
that reaches the docs or the site.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from histogate.serve.patch_demo import PatchAnalyzer

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "data/raw/CRC-VAL-HE-7K"
CKPT = ROOT / "runs/pr16-mixed/best.ckpt.pt"
CALIB = ROOT / "runs/pr16-mixed/calibration.json"
OUT = ROOT / "runs/pr16-mixed/crcval_accuracy.json"
BATCH = 64


class ValSet(Dataset):
    def __init__(self, files: list[Path], labels: list[int], analyzer: PatchAnalyzer) -> None:
        self.files, self.labels, self.an = files, labels, analyzer

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = Image.open(self.files[i]).convert("RGB")
        x = self.an._pre(img).squeeze(0)
        return x, self.labels[i]


def main() -> None:
    an = PatchAnalyzer(CKPT, CALIB)
    classes = sorted(p.name for p in VAL.iterdir() if p.is_dir())
    assert classes and classes[0] == "ADI", classes
    files, labels = [], []
    for ci, c in enumerate(classes):
        for f in sorted((VAL / c).glob("*.tif")):
            files.append(f)
            labels.append(ci)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    an.model.to(dev).eval()
    loader = DataLoader(ValSet(files, labels, an), batch_size=BATCH, num_workers=8)
    correct, n = 0, 0
    confusion = [[0] * len(classes) for _ in classes]
    with torch.no_grad():
        for x, y in tqdm(loader, desc="crcval", unit="batch"):
            z = an.model(x.to(dev))
            pred = z.argmax(1).cpu()
            correct += int((pred == y).sum())
            n += len(y)
            for p_, t_ in zip(pred.tolist(), y.tolist(), strict=True):
                confusion[t_][p_] += 1
    acc = correct / n
    f1s = []
    for i in range(len(classes)):
        tp = confusion[i][i]
        fp = sum(confusion[j][i] for j in range(len(classes))) - tp
        fn = sum(confusion[i]) - tp
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    macro_f1 = sum(f1s) / len(f1s)
    OUT.write_text(json.dumps({
        "n": n, "accuracy": round(acc, 4), "macro_f1": round(macro_f1, 4),
        "per_class_f1": {c: round(f, 4) for c, f in zip(classes, f1s, strict=True)},
        "confusion": confusion, "classes": classes,
        "ckpt": CKPT.name,
    }, indent=2))
    print(f"n={n} accuracy={acc:.4f} macro_f1={macro_f1:.4f} -> {OUT}")


if __name__ == "__main__":
    main()
