"""Validation helpers for lazysammy public APIs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

_VALID_SAVE_FORMATS = {"png", "npy", "coco_rle"}


def validate_image_array(image: npt.NDArray[np.generic], *, name: str = "image") -> None:
    """Validate an in-memory image array.

    Accepted inputs are grayscale ``(H, W)``, RGB/BGR ``(H, W, 3)``, and
    RGBA/BGRA ``(H, W, 4)`` arrays.
    """
    if image.ndim == 2:
        return
    if image.ndim != 3:
        msg = f"{name} must have shape (H, W), (H, W, 3), or (H, W, 4); got {image.shape!r}."
        raise ValueError(msg)
    if image.shape[2] not in {3, 4}:
        msg = f"{name} must have 3 or 4 channels; got shape {image.shape!r}."
        raise ValueError(msg)


def validate_points_and_labels(
    points: Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]] | None,
    labels: Sequence[int] | npt.NDArray[np.integer[Any]] | None,
) -> None:
    """Validate point prompt coordinates and labels."""
    if points is None and labels is None:
        return
    if points is None or labels is None:
        msg = "points and labels must be provided together."
        raise ValueError(msg)

    points_arr = np.asarray(points)
    labels_arr = np.asarray(labels)

    if points_arr.ndim != 2 or points_arr.shape[1] != 2:
        msg = f"points must have shape (N, 2); got {points_arr.shape!r}."
        raise ValueError(msg)
    if labels_arr.ndim != 1:
        msg = f"labels must have shape (N,); got {labels_arr.shape!r}."
        raise ValueError(msg)
    if len(points_arr) != len(labels_arr):
        msg = (
            "points and labels must have the same length; "
            f"got {len(points_arr)} and {len(labels_arr)}."
        )
        raise ValueError(msg)
    if not np.isin(labels_arr, [0, 1]).all():
        msg = "labels must contain only 0 (background) or 1 (foreground)."
        raise ValueError(msg)


def validate_box(
    box: Sequence[float] | npt.NDArray[np.floating[Any]] | None,
    *,
    name: str = "box",
) -> None:
    """Validate a bounding box in ``[x1, y1, x2, y2]`` format."""
    if box is None:
        return
    box_arr = np.asarray(box, dtype=np.float32)
    if box_arr.shape != (4,):
        msg = f"{name} must have shape (4,); got {box_arr.shape!r}."
        raise ValueError(msg)
    x1, y1, x2, y2 = [float(v) for v in box_arr]
    if not np.isfinite(box_arr).all():
        msg = f"{name} must contain only finite coordinates."
        raise ValueError(msg)
    if x1 == x2 or y1 == y2:
        msg = f"{name} must span a non-zero area; got {box_arr.tolist()!r}."
        raise ValueError(msg)


def normalize_box(
    box: Sequence[float] | npt.NDArray[np.floating[Any]],
    *,
    name: str = "box",
) -> npt.NDArray[np.float32]:
    """Validate and normalize a bounding box to ``[x1, y1, x2, y2]``.

    This helper accepts corner coordinates in any order and always returns
    the canonical min/max ordering.
    """
    validate_box(box, name=name)
    box_arr = np.asarray(box, dtype=np.float32)
    x1, y1, x2, y2 = [float(v) for v in box_arr]
    return np.array(
        [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
        dtype=np.float32,
    )


def validate_mask_array(
    mask: npt.NDArray[np.generic] | None,
    *,
    name: str = "mask",
    allow_3d: bool = False,
) -> None:
    """Validate a mask or mask-logit array."""
    if mask is None:
        return
    if mask.ndim == 2:
        return
    if allow_3d and mask.ndim == 3:
        return
    dims = "2D or 3D" if allow_3d else "2D"
    msg = f"{name} must be {dims}; got shape {mask.shape!r}."
    raise ValueError(msg)


def validate_segment_prompts(
    *,
    points: Sequence[Sequence[float]] | npt.NDArray[np.floating[Any]] | None,
    labels: Sequence[int] | npt.NDArray[np.integer[Any]] | None,
    box: Sequence[float] | npt.NDArray[np.floating[Any]] | None,
    mask_input: npt.NDArray[np.generic] | None,
) -> None:
    """Validate prompt combinations for image segmentation."""
    validate_points_and_labels(points, labels)
    validate_box(box)
    validate_mask_array(mask_input, name="mask_input", allow_3d=True)
    if points is None and box is None and mask_input is None:
        msg = "At least one prompt is required: points+labels, box, or mask_input."
        raise ValueError(msg)


def validate_sequence_length(name: str, seq: Sequence[Any] | None, expected: int) -> None:
    """Validate that an optional sequence matches the expected length."""
    if seq is not None and len(seq) != expected:
        msg = f"{name} must have length {expected}; got {len(seq)}."
        raise ValueError(msg)


def validate_frame_index(frame_idx: int, num_frames: int) -> None:
    """Validate that a frame index exists in a video."""
    if frame_idx < 0 or frame_idx >= num_frames:
        msg = f"frame_idx {frame_idx} out of range (0-{num_frames - 1})"
        raise IndexError(msg)


def validate_positive_int(name: str, value: int | None, *, minimum: int = 1) -> None:
    """Validate an optional positive integer."""
    if value is None:
        return
    if value < minimum:
        msg = f"{name} must be >= {minimum}; got {value}."
        raise ValueError(msg)


def validate_nonempty_masks(count: int, *, context: str) -> None:
    """Validate that a result contains at least one mask."""
    if count == 0:
        msg = f"{context} does not contain any masks."
        raise ValueError(msg)


def validate_save_format(fmt: str) -> str:
    """Validate and normalize a save format string."""
    normalized = fmt.strip().lower()
    if normalized not in _VALID_SAVE_FORMATS:
        msg = f"Unsupported save format: {fmt!r}. Use 'png', 'npy', or 'coco_rle'."
        raise ValueError(msg)
    return normalized
