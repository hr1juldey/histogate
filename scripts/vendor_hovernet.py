"""Vendor TIAToolbox's HoVer-Net module WITHOUT the toolkit (P0b, plan B1).

Provenance: tiatoolbox==2.1.3 wheel (`pip download tiatoolbox --no-deps`),
file tiatoolbox/models/architecture/hovernet.py — BSD-3-Clause (verified in
wheel METADATA/LICENSE). This script performs minimal, recorded surgery:

  1. prepend vendor + license header
  2. drop `import dask.array as da` (numpy-only paths)
  3. inline-vendor UpSample2x / centre_crop / centre_crop_to_shape
     (from models/architecture/utils.py) and get_bounding_box / pad_contours
     (from utils/misc.py)
  4. drop ModelABC base -> plain nn.Module
  5. dead-code the dask branches in postproc / _inst_dict_for_dask_processing

Everything else is byte-identical to upstream. Run:
  uv run python scripts/vendor_hovernet.py   (expects the wheel unpacked at
  /tmp/tia/wheel — see docs/BENCHMARKS.md P0b section for the reproduction
  commands).
"""
from __future__ import annotations

import re
from pathlib import Path

WHEEL = Path("/tmp/tia/wheel/tiatoolbox")
OUT = Path(__file__).resolve().parents[1] / "src/histogate/vendor/hovernet.py"

HEADER = '''\
# Vendored from TIAToolbox (https://github.com/TissueImageAnalytics/tiatoolbox)
# tiatoolbox==2.1.3, file: tiatoolbox/models/architecture/hovernet.py
# License: BSD-3-Clause (TIA Centre, University of Warwick) — full text in
#   src/histogate/vendor/LICENSE.tiatoolbox
# Weights (not redistributed here): TIACentre/TIAToolbox_pretrained_weights
#   file hovernet_fast-pannuke.pth (144.1 MiB), fetched at runtime from the
#   HuggingFace Hub. PanNuke-trained — research/non-commercial use framing,
#   see docs/BENCHMARKS.md P0b.
# Local modifications are marked with `# vendored:` comments; see
#   scripts/vendor_hovernet.py for the exact transformation.
# fmt: off
'''

UTILS_SNIPPET = '''
# ---- vendored from tiatoolbox/models/architecture/utils.py (BSD-3-Clause) ----
class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2 (Kronecker, not interpolation)."""

    def __init__(self: "UpSample2x") -> None:
        super().__init__()
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )

    def forward(self: "UpSample2x", x: torch.Tensor) -> torch.Tensor:
        input_shape = list(x.shape)
        x = x.unsqueeze(-1)
        mat = self.unpool_mat.unsqueeze(0)
        ret = torch.tensordot(x, mat, dims=1)
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        return ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))


def centre_crop(
    img: np.ndarray | torch.Tensor,
    crop_shape: np.ndarray | torch.Tensor | tuple[int, int],
    data_format: str = "NCHW",
) -> np.ndarray | torch.Tensor:
    """Centre crop by subtracting crop_shape from each spatial dim."""
    crop_t: int = int(crop_shape[0] // 2)
    crop_b: int = int(crop_shape[0] - crop_t)
    crop_l: int = int(crop_shape[1] // 2)
    crop_r: int = int(crop_shape[1] - crop_l)
    if data_format == "NCHW":
        return img[:, :, crop_t:-crop_b, crop_l:-crop_r]
    return img[:, crop_t:-crop_b, crop_l:-crop_r, :]


def centre_crop_to_shape(
    x: np.ndarray | torch.Tensor,
    y: np.ndarray | torch.Tensor,
    data_format: str = "NCHW",
) -> np.ndarray | torch.Tensor:
    """Centre crop `x` to the shape of `y`."""
    if data_format == "NCHW":
        _, _, h1, w1 = x.shape
        _, _, h2, w2 = y.shape
    else:
        _, h1, w1, _ = x.shape
        _, h2, w2, _ = y.shape
    return centre_crop(x, (h1 - h2, w1 - w2), data_format)


# ---- vendored from tiatoolbox/utils/misc.py (BSD-3-Clause) ----
def get_bounding_box(img: np.ndarray) -> np.ndarray:
    """Minimal box containing all non-zero values: [c_min, r_min, c_max, r_max]."""
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    return np.array([c_min, r_min, c_max + 1, r_max + 1])


def pad_contours(
    contours: list[np.ndarray], pad_value: int | None = None
) -> np.ndarray:
    """Convert inhomogeneous contour lists to a rectangular array."""
    if pad_value is None:
        pad_value = int(np.iinfo(contours[0].dtype).min)
    max_len = max(c.shape[0] for c in contours)
    return np.stack(
        [
            np.vstack(
                [c, np.full((max_len - c.shape[0], 2), pad_value, dtype=c.dtype)]
            )
            for c in contours
        ]
    )
# ---- end vendored helpers ----
'''


def main() -> None:
    src = (WHEEL / "models/architecture/hovernet.py").read_text()

    # 1. drop the toolkit imports
    src = src.replace(
        "from tiatoolbox.models.architecture.utils import (\n"
        "    UpSample2x,\n"
        "    centre_crop,\n"
        "    centre_crop_to_shape,\n"
        ")\n"
        "from tiatoolbox.models.models_abc import ModelABC\n"
        "from tiatoolbox.utils.misc import get_bounding_box, pad_contours\n",
        "# vendored: toolkit imports removed — helpers inlined below\n",
    )
    assert "tiatoolbox" not in src.split('"""', 2)[-1] or True
    assert "from tiatoolbox" not in src, "toolkit import survived"

    # 2. dask out (numpy-only)
    src = src.replace("import dask.array as da\n", "# vendored: dask import removed (numpy-only paths)\n")
    src = src.replace(
        "        is_dask = isinstance(raw_maps[0], da.Array)\n",
        "        is_dask = False  # vendored: numpy-only\n",
    )
    src = src.replace(
        "            da.empty(shape=0, dtype=np.int8)\n"
        "            if is_dask\n"
        "            else np.empty(shape=0, dtype=np.int8)\n",
        "            np.empty(shape=0, dtype=np.int8)\n",
    )
    src = src.replace(
        '            "predictions": da.array(pred_inst)\n'
        "            if isinstance(raw_maps[0], da.Array)\n"
        "            else pred_inst,\n",
        '            "predictions": pred_inst,  # vendored: numpy-only\n',
    )
    src = src.replace(
        "        inst_info_dict_[key] = (\n"
        '            da.from_array(col_np, chunks="auto") if is_dask else col_np\n'
        "        )\n",
        "        inst_info_dict_[key] = col_np  # vendored: numpy-only\n",
    )
    assert "da." not in src, "dask reference survived: " + str(
        [ln for ln in src.splitlines() if "da." in ln and "nda" not in ln][:5]
    )

    # 3. ModelABC -> nn.Module
    src = src.replace("class HoVerNet(ModelABC):", "class HoVerNet(nn.Module):")

    # 4. header + helpers
    doc_end = src.index('"""', src.index('"""') + 3) + 3  # end of module docstring
    body = src[doc_end:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(src[:doc_end] + "\n" + HEADER + "\n" + body + UTILS_SNIPPET)
    print(f"wrote {OUT} ({len(src.splitlines()) + len(UTILS_SNIPPET.splitlines())} lines)")

    # carry the license file
    lic_src = WHEEL.parent / "tiatoolbox-2.1.3.dist-info/licenses/LICENSE"
    (OUT.parent / "LICENSE.tiatoolbox").write_text(lic_src.read_text())
    print("license copied:", lic_src)


if __name__ == "__main__":
    main()
