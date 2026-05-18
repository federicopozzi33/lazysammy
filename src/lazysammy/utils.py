"""Shared utility helpers for lazysammy."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

from lazysammy.types import AutoMaskResult, ImagePrediction, VideoResults
from lazysammy.validation import validate_image_array

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def auto_detect_device(preferred: str | None = None) -> str:
    """Detect the best available device string.

    Priority: explicit *preferred* > CUDA > MPS > CPU.

    Args:
        preferred: Force a specific device string (e.g. ``"cuda:1"``).

    Returns:
        A device string such as ``"cuda"``, ``"mps"``, or ``"cpu"``.
    """
    if preferred is not None:
        return preferred
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_autocast_dtype(device: torch.device) -> torch.dtype:
    """Return the recommended autocast dtype for *device*.

    Args:
        device: Target device.

    Returns:
        ``torch.bfloat16`` for CUDA, ``torch.float16`` for MPS, ``torch.float32`` for CPU.
    """
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def load_image(source: str | Path | npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Load an image as an RGB numpy array.

    Args:
        source: File path or an already-loaded ``(H, W, 3)`` uint8 array.

    Returns:
        ``(H, W, 3)`` RGB uint8 array.

    Raises:
        FileNotFoundError: If *source* is a path that does not exist.
        ValueError: If the loaded image is ``None``.
    """
    if isinstance(source, np.ndarray):
        validate_image_array(source)
        if source.ndim == 2:
            return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
        if source.shape[2] == 4:
            return cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        return source

    path = Path(source)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise FileNotFoundError(msg)
    img = cv2.imread(str(path))
    if img is None:
        msg = f"Failed to read image: {path}"
        raise ValueError(msg)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".m4v"}


def is_video_file(path: str | Path) -> bool:
    """Return ``True`` if *path* looks like a video file (by extension)."""
    return Path(path).suffix.lower() in _VIDEO_EXTS


def extract_frames(
    video_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    every_n: int = 1,
    max_frames: int | None = None,
    frame_format: str = "jpg",
) -> Path:
    """Extract frames from a video file into a directory.

    Args:
        video_path: Path to a video file (mp4, avi, mov, …).
        output_dir: Directory for extracted frames.  If ``None``, a
            sibling directory ``<video_stem>_frames/`` is created next
            to the video.
        every_n: Keep every *n*-th frame (1 = all frames).
        max_frames: Stop after saving this many frames (``None`` = all).
        frame_format: Image format for saved frames (``"jpg"`` or ``"png"``).

    Returns:
        Path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    video_path = Path(video_path)
    if not video_path.is_file():
        msg = f"Video file not found: {video_path}"
        raise FileNotFoundError(msg)

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Failed to open video: {video_path}"
        raise RuntimeError(msg)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Extracting frames from %s (%d frames, %.1f FPS, every_n=%d)",
        video_path.name, total, fps, every_n,
    )

    saved = 0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = output_dir / f"{saved:05d}.{frame_format}"
            cv2.imwrite(str(out_path), frame)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()
    logger.info("Extracted %d frames to %s", saved, output_dir)
    return output_dir


def list_frame_files(video_dir: str | Path) -> list[Path]:
    """List JPEG/PNG frame files in a directory, sorted numerically.

    Args:
        video_dir: Directory containing video frames.

    Returns:
        Sorted list of :class:`Path` objects.

    Raises:
        FileNotFoundError: If *video_dir* does not exist.
        ValueError: If no image files are found.
    """
    video_dir = Path(video_dir)
    if not video_dir.is_dir():
        msg = f"Video directory not found: {video_dir}"
        raise FileNotFoundError(msg)
    frames = sorted(
        [f for f in video_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS],
        key=lambda p: p.stem,
    )
    if not frames:
        msg = f"No image files found in {video_dir}"
        raise ValueError(msg)
    return frames


# ---------------------------------------------------------------------------
# Mask saving
# ---------------------------------------------------------------------------


def save_masks_as_png(
    masks: dict[str, npt.NDArray[np.bool_]],
    output_dir: str | Path,
) -> list[Path]:
    """Save binary masks as individual PNG files.

    Args:
        masks: Mapping of name to ``(H, W)`` boolean mask array.
        output_dir: Target directory (created if needed).

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, mask in masks.items():
        p = out / f"{name}.png"
        cv2.imwrite(str(p), mask.astype(np.uint8) * 255)
        paths.append(p)
    return paths


def save_masks_as_npy(
    masks: dict[str, npt.NDArray[np.bool_]],
    output_dir: str | Path,
) -> list[Path]:
    """Save masks as individual ``.npy`` files.

    Args:
        masks: Mapping of name to ``(H, W)`` boolean array.
        output_dir: Destination directory.

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, mask in masks.items():
        p = out / f"{name}.npy"
        np.save(p, mask)
        paths.append(p)
    return paths


def masks_to_rle(masks: npt.NDArray[np.bool_]) -> list[dict[str, Any]]:
    """Encode binary masks to COCO-style RLE format.

    Args:
        masks: ``(N, H, W)`` boolean array.

    Returns:
        List of dicts with ``"size"`` and ``"counts"`` keys.
    """
    from pycocotools import mask as mask_utils

    rles: list[dict[str, Any]] = []
    for m in masks:
        fortran = np.asfortranarray(m.astype(np.uint8))
        rle = mask_utils.encode(fortran)
        rle["counts"] = rle["counts"].decode("utf-8")
        rles.append(rle)
    return rles


def save_masks_as_coco_rle(
    masks: dict[str, npt.NDArray[np.bool_]],
    output_dir: str | Path,
) -> list[Path]:
    """Save masks as individual JSON files with COCO RLE encoding.

    Each file is named ``<key>.json`` and contains the RLE dict directly
    (with ``"size"`` and ``"counts"`` keys).

    Args:
        masks: Mapping of name to ``(H, W)`` boolean array.
        output_dir: Destination directory.

    Returns:
        List of written file paths.
    """
    from pycocotools import mask as mask_utils

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, mask in masks.items():
        fortran = np.asfortranarray(mask.astype(np.uint8))
        rle = mask_utils.encode(fortran)
        rle["counts"] = rle["counts"].decode("utf-8")
        p = out / f"{name}.json"
        p.write_text(json.dumps(rle))
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# Video results saving
# ---------------------------------------------------------------------------


def save_video_results(
    results: VideoResults,
    output_dir: str | Path,
    *,
    fmt: str = "png",
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
        results: The :class:`VideoResults` to save.
        output_dir: Root output directory.
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the root output directory.
    """
    from lazysammy.io import save_video_results as _save_video_results

    out = _save_video_results(results, output_dir, fmt=fmt)
    logger.info("Saved %d frames to %s (format=%s)", len(results), out, fmt)
    return out


def save_image_prediction(
    prediction: ImagePrediction,
    output_dir: str | Path,
    *,
    fmt: str = "png",
    prefix: str = "mask",
) -> Path:
    """Save an image prediction to disk.

    Args:
        prediction: The :class:`ImagePrediction` to save.
        output_dir: Target directory.
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.
        prefix: Filename prefix for PNG output.

    Returns:
        Path to the output directory (or npy/json file).
    """
    del prefix
    from lazysammy.io import save_image_prediction as _save_image_prediction

    return _save_image_prediction(prediction, output_dir, fmt=fmt)


# ---------------------------------------------------------------------------
# Auto mask results saving
# ---------------------------------------------------------------------------


def save_auto_mask_result(
    result: AutoMaskResult,
    output_dir: str | Path,
    *,
    fmt: str = "png",
) -> Path:
    """Save automatic mask generation result to disk.

    Args:
        result: The :class:`AutoMaskResult` to save.
        output_dir: Target directory.
        fmt: ``"png"``, ``"npy"``, or ``"coco_rle"``.

    Returns:
        Path to the output.
    """
    from lazysammy.io import save_auto_mask_result as _save_auto_mask_result

    return _save_auto_mask_result(result, output_dir, fmt=fmt)


# ---------------------------------------------------------------------------
# Mask manipulation helpers
# ---------------------------------------------------------------------------


def masks_to_colored_overlay(
    image: npt.NDArray[np.uint8],
    masks: npt.NDArray[np.bool_],
    *,
    alpha: float = 0.5,
    colors: Sequence[tuple[int, int, int]] | None = None,
) -> npt.NDArray[np.uint8]:
    """Overlay colored masks on an image.

    Args:
        image: ``(H, W, 3)`` RGB base image.
        masks: ``(N, H, W)`` boolean masks.
        alpha: Overlay transparency.
        colors: Optional ``(R, G, B)`` tuples; auto-generated if ``None``.

    Returns:
        ``(H, W, 3)`` RGB image with overlaid masks.
    """
    if colors is None:
        rng = np.random.default_rng(42)
        colors = [tuple(int(c) for c in rng.integers(60, 220, size=3)) for _ in range(len(masks))]

    overlay = image.copy()
    for mask, color in zip(masks, colors, strict=False):
        overlay[mask] = (
            np.array(color, dtype=np.float32) * alpha
            + overlay[mask].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)
    return overlay


def combine_masks(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Merge multiple masks into a single union mask.

    Args:
        masks: ``(N, H, W)`` boolean array.

    Returns:
        ``(H, W)`` boolean mask.
    """
    return np.any(masks, axis=0)


def mask_to_bbox(mask: npt.NDArray[np.bool_]) -> list[int]:
    """Compute the bounding box of a binary mask.

    Args:
        mask: ``(H, W)`` boolean mask.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` bounding box.
    """
    if mask.ndim != 2:
        msg = f"mask must be 2D; got shape {mask.shape!r}."
        raise ValueError(msg)
    if not np.any(mask):
        msg = "mask is empty; cannot compute a bounding box."
        raise ValueError(msg)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [int(x_min), int(y_min), int(x_max), int(y_max)]


def mask_iou(mask_a: npt.NDArray[np.bool_], mask_b: npt.NDArray[np.bool_]) -> float:
    """Compute Intersection-over-Union between two binary masks.

    Args:
        mask_a: ``(H, W)`` boolean mask.
        mask_b: ``(H, W)`` boolean mask.

    Returns:
        IoU value in ``[0, 1]``.
    """
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)
