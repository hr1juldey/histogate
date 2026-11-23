"""Verify teacher-cache/dataset alignment: cached embedding vs live teacher
forward on the same tile indices. cosine > 0.99 everywhere = aligned."""
import sys
import torch

sys.path.insert(0, "src")
from histogate.distill_cache import load_cache  # noqa: E402

CACHE = "runs/distill/hibou-l"
IDX = [0, 1_000, 50_000]

train, val = load_cache(CACHE)
print("cache:", train["emb"].shape, val["emb"].shape if hasattr(val, "shape") else type(val))

# live teacher on the same indices requires the dataset + model; minimal:
from transformers import AutoModel  # noqa: E402
import torch.nn.functional as F  # noqa: E402

model = AutoModel.from_pretrained("histai/hibou-l", trust_remote_code=True)
model.eval().cuda() if torch.cuda.is_available() else model.eval()

from histogate.data import PatchDataset  # noqa: E402

ds = PatchDataset(root="data/raw/NCT-CRC-HE-100K", train=True, cache_tag="nct_full")
print("dataset N:", len(ds))
for i in IDX:
    img, label = ds[i]
    x = img.unsqueeze(0)
    if torch.cuda.is_available():
        x = x.cuda()
    with torch.no_grad():
        f = model(x).pooler_output if hasattr(model(x), "pooler_output") else model(x).last_hidden_state.mean(1)
    f = F.normalize(f.flatten(1).cpu(), dim=1)[0]
    c = F.normalize(torch.from_numpy(train["emb"][i]), dim=0)
    print(i, "label", label, "cosine", round(float(f @ c), 4))
