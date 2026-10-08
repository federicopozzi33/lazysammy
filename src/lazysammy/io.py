"""Result serialization helpers for lazysammy.

This module is the **single source of truth** for writing predictions to disk.
The convenience re-exports in :mod:`lazysammy.utils` and on the high-level
wrappers (:class:`~lazysammy.image.ImageSegmenter`,
:class:`~lazysammy.video.VideoSession`, :class:`~lazysammy.auto_mask.AutoSegmenter`)
all delegate here.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from lazysammy.types import AutoMaskResult, ConceptPrediction, ImagePrediction, VideoResults
from lazysammy.utils import (
    save_masks_as_coco_rle,
    save_masks_as_npy,
    save_masks_as_png,
)
from lazysammy.validation import validate_save_format

MaskMapping = Mapping[str, npt.NDArray[np.bool_]]


class SaveFormat(str, Enum):
    """Supported output serialization formats.

    Subclasses :class:`str` so members compare equal to their string value
    (``SaveFormat.PNG == "png"``) and render as ``"png"`` rather than
    ``"SaveFormat.PNG"`` when interpolated.
    """

    PNG = "png"
    NPY = "npy"
    COCO_RLE = "coco_rle"

    def __str__(self) -> str:
        """Return the plain format string (e.g. ``"png"``)."""
        return self.value


def _save_mask_mapping(
    masks_dict: MaskMapping,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat,
) -> Path:
    """Write a named mask mapping to *output_dir* in the requested format."""
    normalized = validate_save_format(str(fmt))
    out = Path(output_dir)
    mapping = dict(masks_dict)

    if normalized == SaveFormat.PNG.value:
        save_masks_as_png(mapping, out)
        return out
    if normalized == SaveFormat.NPY.value:
        save_masks_as_npy(mapping, out)
        return out
    save_masks_as_coco_rle(mapping, out)
    return out


def save_image_prediction(
    prediction: ImagePrediction,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save an image prediction to disk.

    Args:
        prediction: The prediction whose masks should be written.
        output_dir: Target directory (created if needed).
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the output directory.

    Example::

        save_image_prediction(pred, "out/", fmt="png")
    """
    masks_dict = {f"mask_{i:04d}": m.data for i, m in enumerate(prediction.masks)}
    return _save_mask_mapping(masks_dict, output_dir, fmt=fmt)


def save_auto_mask_result(
    result: AutoMaskResult,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save automatic mask generation output to disk.

    Args:
        result: The auto-mask result to write.
        output_dir: Target directory (created if needed).
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the output directory.
    """
    masks_dict = {f"auto_mask_{i:04d}": m.data for i, m in enumerate(result.masks)}
    return _save_mask_mapping(masks_dict, output_dir, fmt=fmt)


def save_concept_prediction(
    prediction: ConceptPrediction,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save a SAM 3 concept prediction to disk.

    Args:
        prediction: The concept prediction whose instance masks should be written.
        output_dir: Target directory (created if needed).
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the output directory.
    """
    masks_dict = {f"instance_{i:04d}": m.data for i, m in enumerate(prediction.masks)}
    return _save_mask_mapping(masks_dict, output_dir, fmt=fmt)


def save_video_results(
    results: VideoResults,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save full video tracking results to disk.

    Directory structure::

        output_dir/
            frame_000000/
                obj_0001.png
                obj_0002.png
            frame_000001/
                ...

    Args:
        results: The tracking results to write.
        output_dir: Root output directory (created if needed).
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the root output directory.
    """
    normalized = validate_save_format(str(fmt))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_masks in results:
        frame_dir = out / f"frame_{frame_idx:06d}"
        masks_dict = {f"obj_{obj_id:04d}": mask for obj_id, mask in frame_masks.masks.items()}
        _save_mask_mapping(masks_dict, frame_dir, fmt=normalized)
    return out
