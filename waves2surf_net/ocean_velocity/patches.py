"""Spatial patching for multiplying training samples from a regional map.

The Agulhas crop is ~1/12° (see ``infer_resolution_deg``). A geographic
``2° × 3°`` (lat × lon) window is therefore about ``24 × 36`` pixels.

Design choices (pick one; this module supports the first two):

1. **Grid wrapper (recommended first test)** — expand each time snapshot into
   a fixed lattice of crops. Deterministic, multiplies ``len(dataset)`` by the
   number of origins, easy to reproduce. Apply on train (+ optionally val).
2. **Random crop** — one (or ``k``) random window(s) per ``__getitem__``.
   Classic augmentation; epoch length stays ~1× unless you oversample.
3. **Sub-bbox NetCDF reads** — pass a tighter lon/lat box into ``read_field``.
   Less RAM, but many small I/O calls; usually slower than one full read +
   in-memory crop unless the full map is huge.
4. **Iterable / multi-patch-from-one-file** — load a day once, yield many
   patches. Best I/O amortization; more awkward with a map-style ``Dataset``.

Vector fields (``uo``, ``vo``, ``KX``, ``KY``) may be cropped as-is. Do **not**
use generic image flips/rotations without rotating those components.

U-Net note: four stride-2 levels prefer sizes divisible by 16. ``24×36`` is
not; the decoder already resizes skips, so it still runs. For a cleaner first
ablation you can use ``32×48`` (~2.67°×4°) instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PatchGeometry:
    """Fixed pixel crop size and optional stride (same units: pixels)."""

    height: int
    width: int
    stride_y: int
    stride_x: int

    def __post_init__(self) -> None:
        if self.height < 1 or self.width < 1:
            raise ValueError("patch height/width must be >= 1")
        if self.stride_y < 1 or self.stride_x < 1:
            raise ValueError("patch strides must be >= 1")


def infer_resolution_deg(
    bbox: dict[str, float], height: int, width: int
) -> tuple[float, float]:
    """Return ``(dlat, dlon)`` in degrees from an inclusive bbox and array shape.

    Matches how sources crop lon/lat masks: ``n`` points span ``(max-min)``
    with spacing ``(max-min)/(n-1)`` when ``n > 1``.
    """
    if height < 2 or width < 2:
        raise ValueError("need at least a 2×2 field to infer resolution")
    dlat = (float(bbox["lat_max"]) - float(bbox["lat_min"])) / (height - 1)
    dlon = (float(bbox["lon_max"]) - float(bbox["lon_min"])) / (width - 1)
    return dlat, dlon


def degrees_to_patch_pixels(
    lat_deg: float,
    lon_deg: float,
    dlat: float,
    dlon: float,
) -> tuple[int, int]:
    """Convert a geographic window to ``(height, width)`` in pixels.

    Uses ``round(deg / dres)`` so a ``2°`` span on a ``1/12°`` grid → 24 px.
    """
    if lat_deg <= 0 or lon_deg <= 0:
        raise ValueError("geographic patch extents must be positive")
    if dlat <= 0 or dlon <= 0:
        raise ValueError("grid resolution must be positive")
    height = max(1, int(round(lat_deg / dlat)))
    width = max(1, int(round(lon_deg / dlon)))
    return height, width


def patch_geometry_from_degrees(
    bbox: dict[str, float],
    field_shape: tuple[int, int],
    *,
    size_deg: tuple[float, float],
    stride_deg: tuple[float, float] | None = None,
) -> PatchGeometry:
    """Build pixel geometry from geographic size (and optional stride).

    ``size_deg`` / ``stride_deg`` are ``(lat_deg, lon_deg)``. If ``stride_deg``
    is omitted, stride equals the patch size (non-overlapping grid).
    """
    height, width = field_shape
    dlat, dlon = infer_resolution_deg(bbox, height, width)
    ph, pw = degrees_to_patch_pixels(size_deg[0], size_deg[1], dlat, dlon)
    if stride_deg is None:
        sy, sx = ph, pw
    else:
        sy, sx = degrees_to_patch_pixels(
            stride_deg[0], stride_deg[1], dlat, dlon
        )
    return PatchGeometry(ph, pw, sy, sx)


def iter_patch_origins(
    field_height: int,
    field_width: int,
    geometry: PatchGeometry,
) -> list[tuple[int, int]]:
    """Return all top-left ``(y0, x0)`` for a sliding window that fits in-bounds."""
    if geometry.height > field_height or geometry.width > field_width:
        raise ValueError(
            f"patch {geometry.height}×{geometry.width} does not fit in "
            f"field {field_height}×{field_width}"
        )
    ys = range(0, field_height - geometry.height + 1, geometry.stride_y)
    xs = range(0, field_width - geometry.width + 1, geometry.stride_x)
    return [(y, x) for y in ys for x in xs]


def crop_tensor(tensor: torch.Tensor, y0: int, x0: int, height: int, width: int) -> torch.Tensor:
    """Crop last two dims of a ``[..., H, W]`` tensor."""
    return tensor[..., y0 : y0 + height, x0 : x0 + width]


def crop_sample(
    sample: dict[str, torch.Tensor],
    y0: int,
    x0: int,
    height: int,
    width: int,
) -> dict[str, torch.Tensor]:
    """Crop spatial keys; leave scalars (metadata, flags, index) unchanged."""
    spatial_keys = ("x", "y", "valid_mask")
    out = dict(sample)
    for key in spatial_keys:
        if key in out and torch.is_tensor(out[key]) and out[key].ndim >= 2:
            out[key] = crop_tensor(out[key], y0, x0, height, width).contiguous()
    out["patch_y0"] = torch.tensor(y0, dtype=torch.int64)
    out["patch_x0"] = torch.tensor(x0, dtype=torch.int64)
    return out


def apply_patches(
    dataset: Dataset,
    config: dict[str, Any],
    split: str,
) -> Dataset:
    """Optionally wrap ``dataset`` with :class:`SpatialPatchDataset` from config.

    Expected ``data.patches`` shape (notebook / experiment configs)::

        {
          "train": true,
          "validation": true,
          "test": false,
          "size_deg": [5.0, 5.0],
          "stride_deg": null,
          "mode": "grid",
          "random_patches_per_item": 1
        }

    If ``data.patches`` is missing, or the split flag is false, the base
    dataset is returned unchanged.
    """
    patches = config.get("data", {}).get("patches")
    if not patches:
        return dataset
    enabled = patches.get(split)
    if enabled is None:
        enabled = bool(patches.get("enabled", False))
    if not enabled:
        return dataset

    bbox = config["data"].get("bbox")
    if not bbox:
        raise ValueError("data.patches requires data.bbox to infer grid resolution")

    size_deg = patches.get("size_deg", [2.0, 3.0])
    stride_raw = patches.get("stride_deg")
    stride_deg = None if stride_raw in (None, [], ()) else tuple(stride_raw)
    return SpatialPatchDataset.from_degrees(
        dataset,
        bbox,
        size_deg=tuple(size_deg),
        stride_deg=stride_deg,
        mode=str(patches.get("mode", "grid")),
        seed=int(config.get("training", {}).get("seed", 42)),
        random_patches_per_item=int(patches.get("random_patches_per_item", 1)),
    )


class SpatialPatchDataset(Dataset):
    """Wrap a map-style dataset and expose spatial crops as extra samples.

    Modes
    -----
    ``grid``
        Index space is ``base_index * n_patches + patch_index``. Length is
        ``len(base) * n_patches``. Best first test: multiplies examples cleanly.
    ``random``
        Length equals ``len(base)`` (or ``len(base) * k`` if
        ``random_patches_per_item > 1``). Each access draws a random valid
        top-left corner (seeded by ``seed`` + item for reproducibility).

    The base dataset should already return the full regional crop
    (e.g. Agulhas bbox). Patching is done in memory after that load.
    """

    def __init__(
        self,
        base: Dataset,
        geometry: PatchGeometry,
        *,
        mode: str = "grid",
        field_shape: tuple[int, int] | None = None,
        random_patches_per_item: int = 1,
        seed: int = 0,
        spatial_keys: Sequence[str] = ("x", "y", "valid_mask"),
    ) -> None:
        if mode not in {"grid", "random"}:
            raise ValueError(f"Unknown patch mode {mode!r}; use 'grid' or 'random'")
        if random_patches_per_item < 1:
            raise ValueError("random_patches_per_item must be >= 1")
        self.base = base
        self.geometry = geometry
        self.mode = mode
        self.random_patches_per_item = int(random_patches_per_item)
        self.seed = int(seed)
        self.spatial_keys = tuple(spatial_keys)

        if field_shape is None:
            probe = base[0]
            field_shape = _spatial_hw(probe, self.spatial_keys)
        self.field_height, self.field_width = field_shape
        self.origins = iter_patch_origins(
            self.field_height, self.field_width, geometry
        )
        if not self.origins:
            raise ValueError("no valid patch origins for the given geometry")

    @classmethod
    def from_degrees(
        cls,
        base: Dataset,
        bbox: dict[str, float],
        *,
        size_deg: tuple[float, float] = (2.0, 3.0),
        stride_deg: tuple[float, float] | None = None,
        mode: str = "grid",
        seed: int = 0,
        random_patches_per_item: int = 1,
    ) -> "SpatialPatchDataset":
        """Infer pixel geometry from ``bbox`` + first sample shape, then wrap."""
        probe = base[0]
        hw = _spatial_hw(probe, ("x", "y", "valid_mask"))
        geometry = patch_geometry_from_degrees(
            bbox, hw, size_deg=size_deg, stride_deg=stride_deg
        )
        return cls(
            base,
            geometry,
            mode=mode,
            field_shape=hw,
            seed=seed,
            random_patches_per_item=random_patches_per_item,
        )

    def __len__(self) -> int:
        n_base = len(self.base)
        if self.mode == "grid":
            return n_base * len(self.origins)
        return n_base * self.random_patches_per_item

    def _random_origin(self, item: int) -> tuple[int, int]:
        rng = np.random.default_rng(self.seed + item)
        y0 = int(rng.integers(0, self.field_height - self.geometry.height + 1))
        x0 = int(rng.integers(0, self.field_width - self.geometry.width + 1))
        return y0, x0

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self.mode == "grid":
            n_patches = len(self.origins)
            base_index, patch_index = divmod(int(item), n_patches)
            y0, x0 = self.origins[patch_index]
        else:
            n_k = self.random_patches_per_item
            base_index, _ = divmod(int(item), n_k)
            y0, x0 = self._random_origin(int(item))

        sample = self.base[base_index]
        cropped = crop_sample(
            sample, y0, x0, self.geometry.height, self.geometry.width
        )
        # Keep a pointer to the parent time index for debugging / viz.
        if "index" in cropped:
            cropped["base_index"] = cropped["index"].clone()
        cropped["index"] = torch.tensor(item, dtype=torch.int64)
        return cropped

    def describe(self) -> str:
        """Short human-readable summary for notebooks / logs."""
        return (
            f"SpatialPatchDataset(mode={self.mode!r}, "
            f"patch={self.geometry.height}×{self.geometry.width}, "
            f"stride={self.geometry.stride_y}×{self.geometry.stride_x}, "
            f"origins={len(self.origins)}, "
            f"len={len(self)} from base={len(self.base)})"
        )


def _spatial_hw(
    sample: dict[str, torch.Tensor], keys: Sequence[str]
) -> tuple[int, int]:
    for key in keys:
        value = sample.get(key)
        if torch.is_tensor(value) and value.ndim >= 2:
            return int(value.shape[-2]), int(value.shape[-1])
    raise KeyError(f"No spatial tensor among keys {list(keys)}")


def patch_bboxes(
    parent_bbox: dict[str, float],
    origins: Sequence[tuple[int, int]],
    geometry: PatchGeometry,
    dlat: float,
    dlon: float,
) -> Iterator[dict[str, float]]:
    """Yield lon/lat boxes for each pixel origin (for option-3 NetCDF crops).

    Assumes south→north rows and west→east columns, matching
    ``ensure_south_to_north`` outputs.
    """
    lat0 = float(parent_bbox["lat_min"])
    lon0 = float(parent_bbox["lon_min"])
    for y0, x0 in origins:
        yield {
            "lat_min": lat0 + y0 * dlat,
            "lat_max": lat0 + (y0 + geometry.height - 1) * dlat,
            "lon_min": lon0 + x0 * dlon,
            "lon_max": lon0 + (x0 + geometry.width - 1) * dlon,
        }
