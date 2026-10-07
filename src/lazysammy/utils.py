"""Shared utility helpers for lazysammy."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
import torch

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

    Autocast is only meaningful for a small set of float dtypes.  On CPU,
    ``torch.autocast`` only supports ``bfloat16`` and ``float16``; using
    ``float32`` emits a warning and silently disables autocast.  Callers should
    prefer :func:`autocast` which handles the CPU/none case for them.

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


def autocast(device: torch.device) -> Any:
    """Return an autocast context manager appropriate for *device*.

    Unlike a bare ``torch.autocast(device.type, dtype=...)`` call, this returns
    a no-op ``nullcontext`` when autocast would be unsupported (CPU/float32),
    avoiding the "target dtype is not supported. Disabling autocast." warning
    that PyTorch otherwise emits on every call.

    Args:
        device: Target device.

    Returns:
        A context manager usable with ``with``.
    """
    dtype = get_autocast_dtype(device)
    if dtype is torch.float32:
        # CPU autocast does not support float32; run without autocast.
        return nullcontext()
    return torch.autocast(device.type, dtype=dtype)


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
        arr = np.asarray(source)
        if arr.ndim == 2:
            return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB), dtype=np.uint8)
        if arr.shape[2] == 4:
            return np.ascontiguousarray(cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB), dtype=np.uint8)
        return np.ascontiguousarray(arr, dtype=np.uint8)

    path = Path(source)
    if not path.exists():
        msg = f"Image file not found: {path}"
        raise FileNotFoundError(msg)
    img = cv2.imread(str(path))
    if img is None:
        msg = f"Failed to read image: {path}"
        raise ValueError(msg)
    return np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dtype=np.uint8)


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
    clean: bool = False,
) -> Path:
    """Extract frames from a video file into a directory.

    Frames are written as zero-padded ``<index>.jpg`` files so they always
    sort correctly.

    Because the output directory is reused, **re-extracting into a directory
    that already holds frames can mix results from two different settings**:
    the new run only writes the frames it produces, leaving older ones in
    place. Pass ``clean=True`` when re-extracting with a different ``every_n``
    or ``max_frames``.

    Args:
        video_path: Path to a video file (mp4, avi, mov, ...).
        output_dir: Directory for extracted frames.  If ``None``, a
            sibling directory ``<video_stem>_frames/`` is created next
            to the video.
        every_n: Keep every *n*-th frame (1 = all frames).
        max_frames: Stop after saving this many frames (``None`` = all).
        frame_format: Image format for saved frames (``"jpg"`` or ``"png"``).
        clean: Delete pre-existing frames in *output_dir* before extracting,
            so the result reflects only the current settings.

    Returns:
        Path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
        ValueError: If *every_n* or *max_frames* is not positive.

    Example::

        frames_dir = extract_frames("clip.mp4", every_n=2, max_frames=100)
    """
    if every_n < 1:
        msg = f"every_n must be >= 1; got {every_n}."
        raise ValueError(msg)
    if max_frames is not None and max_frames < 1:
        msg = f"max_frames must be >= 1 when provided; got {max_frames}."
        raise ValueError(msg)
    if frame_format not in {"jpg", "jpeg", "png"}:
        msg = f"frame_format must be 'jpg' or 'png'; got {frame_format!r}."
        raise ValueError(msg)

    video_path = Path(video_path)
    if not video_path.is_file():
        msg = f"Video file not found: {video_path}"
        raise FileNotFoundError(msg)

    if output_dir is None:
        output_dir = video_path.parent / f"{video_path.stem}_frames"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if clean:
        stale = [f for f in output_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
        for f in stale:
            f.unlink()
        if stale:
            logger.info("Removed %d stale frame(s) from %s", len(stale), output_dir)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        msg = f"Failed to open video: {video_path}"
        raise RuntimeError(msg)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    logger.info(
        "Extracting frames from %s (%d frames, %.1f FPS, every_n=%d)",
        video_path.name,
        total,
        fps,
        every_n,
    )

    saved = 0
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if idx % every_n == 0:
            out_path = output_dir / f"{saved:05d}.{frame_format}"
            if not cv2.imwrite(str(out_path), frame):
                cap.release()
                msg = f"Failed to write frame to {out_path}"
                raise RuntimeError(msg)
            saved += 1
            if max_frames is not None and saved >= max_frames:
                break
        idx += 1
    cap.release()

    present = [f for f in output_dir.iterdir() if f.suffix.lower() in _IMAGE_EXTS]
    if not clean and len(present) > saved:
        logger.warning(
            "Extracted %d frame(s) into %s, but the directory already contained "
            "%d image file(s) from a previous run. The frame set now mixes both "
            "runs. Re-run with clean=True (or point output_dir at a fresh "
            "directory) to get a consistent set.",
            saved,
            output_dir,
            len(present),
        )

    logger.info("Extracted %d frames to %s", saved, output_dir)
    return output_dir


def natural_sort_key(path: Path) -> tuple[Any, ...]:
    """Build a sort key that orders numeric filenames naturally.

    Plain lexicographic sorting puts ``"10.jpg"`` before ``"2.jpg"``, which
    silently scrambles the frame order of videos whose frames are not
    zero-padded.  This key splits the filename stem into digit and
    non-digit runs so ``2`` sorts before ``10``.

    Args:
        path: Frame path whose ``stem`` is used for ordering.

    Returns:
        A tuple suitable as a ``sorted(key=...)`` argument.
    """
    parts = re.split(r"(\d+)", path.stem)
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part.lower()) for part in parts if part != ""
    )


def list_frame_files(video_dir: str | Path) -> list[Path]:
    """List JPEG/PNG frame files in a directory, sorted naturally.

    Frames are ordered by numeric value embedded in their name, so
    ``2.jpg`` comes before ``10.jpg`` even when zero-padding is absent.

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
        key=natural_sort_key,
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


def mask_to_rle(mask: npt.NDArray[np.bool_]) -> dict[str, Any]:
    """Encode a single binary mask to COCO-style RLE.

    Args:
        mask: ``(H, W)`` boolean mask.

    Returns:
        A dict with ``"size"`` (``[H, W]``) and ``"counts"`` (UTF-8 string).
    """
    from pycocotools import mask as mask_utils

    fortran = np.asfortranarray(mask.astype(np.uint8))
    rle = mask_utils.encode(fortran)
    rle["counts"] = rle["counts"].decode("utf-8")
    return dict(rle)


def rle_to_mask(rle: dict[str, Any]) -> npt.NDArray[np.bool_]:
    """Decode a COCO-style RLE dict into a boolean ``(H, W)`` mask.

    Args:
        rle: A dict with ``"size"`` and ``"counts"`` keys, as produced by
            :func:`mask_to_rle` or :meth:`lazysammy.types.Mask.to_rle`.

    Returns:
        ``(H, W)`` boolean mask.
    """
    from pycocotools import mask as mask_utils

    counts = rle["counts"]
    if isinstance(counts, str):
        counts = counts.encode("utf-8")
    decoded = mask_utils.decode({"size": rle["size"], "counts": counts})
    return np.asarray(decoded).astype(bool)


def masks_to_rle(masks: npt.NDArray[np.bool_]) -> list[dict[str, Any]]:
    """Encode binary masks to COCO-style RLE format.

    Args:
        masks: ``(N, H, W)`` boolean array.

    Returns:
        List of dicts with ``"size"`` and ``"counts"`` keys.
    """
    return [mask_to_rle(m) for m in masks]


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
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for name, mask in masks.items():
        rle = mask_to_rle(mask)
        p = out / f"{name}.json"
        p.write_text(json.dumps(rle))
        paths.append(p)
    return paths


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
        colors = [
            (int(c[0]), int(c[1]), int(c[2])) for c in rng.integers(60, 220, size=(len(masks), 3))
        ]

    overlay: npt.NDArray[np.uint8] = image.copy()
    for mask, color in zip(masks, colors, strict=False):
        blended = (
            np.array(color, dtype=np.float32) * alpha
            + overlay[mask].astype(np.float32) * (1 - alpha)
        ).astype(np.uint8)
        overlay[mask] = blended
    return overlay


def combine_masks(masks: npt.NDArray[np.bool_]) -> npt.NDArray[np.bool_]:
    """Merge multiple masks into a single union mask.

    Args:
        masks: ``(N, H, W)`` boolean array.

    Returns:
        ``(H, W)`` boolean mask.
    """
    combined: npt.NDArray[np.bool_] = np.any(masks, axis=0)
    return combined


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
