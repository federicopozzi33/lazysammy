"""Result serialization helpers for lazysammy."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from lazysammy.types import AutoMaskResult, ImagePrediction, VideoResults
from lazysammy.utils import (
    save_masks_as_coco_rle,
    save_masks_as_npy,
    save_masks_as_png,
)
from lazysammy.validation import validate_save_format


class SaveFormat(str, Enum):
    """Supported output serialization formats."""

    PNG = "png"
    NPY = "npy"
    COCO_RLE = "coco_rle"



def _save_mask_mapping(
    masks_dict: dict[str, object],
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat,
) -> Path:
    normalized = validate_save_format(str(fmt))
    out = Path(output_dir)

    if normalized == SaveFormat.PNG.value:
        save_masks_as_png(masks_dict, out)
        return out
    if normalized == SaveFormat.NPY.value:
        save_masks_as_npy(masks_dict, out)
        return out
    save_masks_as_coco_rle(masks_dict, out)
    return out



def save_image_prediction(
    prediction: ImagePrediction,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save an image prediction to disk."""
    masks_dict = {f"mask_{i:04d}": m.data for i, m in enumerate(prediction.masks)}
    return _save_mask_mapping(masks_dict, output_dir, fmt=fmt)



def save_auto_mask_result(
    result: AutoMaskResult,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save automatic mask generation output to disk."""
    masks_dict = {f"auto_mask_{i:04d}": m.data for i, m in enumerate(result.masks)}
    return _save_mask_mapping(masks_dict, output_dir, fmt=fmt)



def save_video_results(
    results: VideoResults,
    output_dir: str | Path,
    *,
    fmt: str | SaveFormat = SaveFormat.PNG,
) -> Path:
    """Save full video tracking results to disk."""
    normalized = validate_save_format(str(fmt))
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for frame_idx, frame_masks in results:
        frame_dir = out / f"frame_{frame_idx:06d}"
        masks_dict = {
            f"obj_{obj_id:04d}": mask for obj_id, mask in frame_masks.masks.items()
        }
        _save_mask_mapping(masks_dict, frame_dir, fmt=normalized)
    return out
