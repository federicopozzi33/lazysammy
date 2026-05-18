"""Visualisation helpers for masks, bounding boxes, and tracking results.

Requires ``matplotlib`` (install via ``pip install lazysammy2[viz]``).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from lazysammy2.types import AutoMaskResult, FrameMasks, ImagePrediction, VideoResults
from lazysammy2.utils import load_image, masks_to_colored_overlay

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------

_DEFAULT_PALETTE: list[tuple[int, int, int]] = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 128, 128),
    (128, 255, 128),
    (128, 128, 255),
]


def _get_color(idx: int) -> tuple[int, int, int]:
    return _DEFAULT_PALETTE[idx % len(_DEFAULT_PALETTE)]


# ---------------------------------------------------------------------------
# Image-level visualisation (pure OpenCV, no matplotlib)
# ---------------------------------------------------------------------------


def draw_masks_on_image(
    image: str | Path | npt.NDArray[np.uint8],
    masks: npt.NDArray[np.bool_],
    *,
    alpha: float = 0.5,
    colors: Sequence[tuple[int, int, int]] | None = None,
    draw_contours: bool = True,
    contour_thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw coloured masks on an image with optional contours.

    Args:
        image: Source image (path or array).
        masks: ``(N, H, W)`` boolean mask array.
        alpha: Overlay transparency.
        colors: Per-mask RGB colours (auto-generated if ``None``).
        draw_contours: Whether to draw mask contour lines.
        contour_thickness: Contour line thickness in pixels.

    Returns:
        ``(H, W, 3)`` RGB image with overlaid masks.
    """
    img = load_image(image) if not isinstance(image, np.ndarray) else image.copy()
    if colors is None:
        colors = [_get_color(i) for i in range(len(masks))]

    overlay = masks_to_colored_overlay(img, masks, alpha=alpha, colors=colors)

    if draw_contours:
        for i, mask in enumerate(masks):
            contours, _ = cv2.findContours(
                mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, colors[i], contour_thickness)

    return overlay


def draw_points_on_image(
    image: npt.NDArray[np.uint8],
    points: npt.NDArray[np.floating[Any]],
    labels: npt.NDArray[np.integer[Any]],
    *,
    radius: int = 5,
    fg_color: tuple[int, int, int] = (0, 255, 0),
    bg_color: tuple[int, int, int] = (255, 0, 0),
) -> npt.NDArray[np.uint8]:
    """Draw point prompts on an image.

    Args:
        image: ``(H, W, 3)`` RGB image.
        points: ``(N, 2)`` array of ``(x, y)`` coordinates.
        labels: Length-N array (``1`` = foreground, ``0`` = background).
        radius: Circle radius in pixels.
        fg_color: Colour for foreground points.
        bg_color: Colour for background points.

    Returns:
        Image with drawn points.
    """
    out = image.copy()
    for pt, lab in zip(points, labels, strict=False):
        color = fg_color if int(lab) == 1 else bg_color
        cv2.circle(out, (int(pt[0]), int(pt[1])), radius, color, -1)
        cv2.circle(out, (int(pt[0]), int(pt[1])), radius, (255, 255, 255), 1)
    return out


def draw_box_on_image(
    image: npt.NDArray[np.uint8],
    box: Sequence[float],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> npt.NDArray[np.uint8]:
    """Draw a bounding box on an image.

    Args:
        image: ``(H, W, 3)`` RGB image.
        box: ``[x1, y1, x2, y2]`` bounding box.
        color: Box colour (RGB).
        thickness: Line thickness.

    Returns:
        Image with drawn box.
    """
    out = image.copy()
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
    return out


# ---------------------------------------------------------------------------
# Matplotlib-based visualisation
# ---------------------------------------------------------------------------


def show_image_prediction(
    image: str | Path | npt.NDArray[np.uint8],
    prediction: ImagePrediction,
    *,
    alpha: float = 0.5,
    figsize: tuple[int, int] = (12, 8),
    show_scores: bool = True,
) -> Any:
    """Display an image prediction using matplotlib.

    Args:
        image: Source image.
        prediction: The prediction to visualise.
        alpha: Mask overlay transparency.
        figsize: Figure size.
        show_scores: Show IoU score annotations.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    img = load_image(image) if not isinstance(image, np.ndarray) else image
    n = len(prediction.masks)

    fig, axes = plt.subplots(1, n + 1, figsize=figsize)
    if n + 1 == 1:
        axes = [axes]

    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, mask_obj in enumerate(prediction.masks):
        vis = draw_masks_on_image(img, mask_obj.data[None], alpha=alpha)
        axes[i + 1].imshow(vis)
        title = f"Mask {i}"
        if show_scores:
            title += f" (IoU: {mask_obj.score:.3f})"
        axes[i + 1].set_title(title)
        axes[i + 1].axis("off")

    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig


def show_auto_masks(
    image: str | Path | npt.NDArray[np.uint8],
    result: AutoMaskResult,
    *,
    alpha: float = 0.4,
    figsize: tuple[int, int] = (14, 8),
    max_masks: int | None = None,
) -> Any:
    """Display automatic mask generation results with matplotlib.

    Args:
        image: Source image.
        result: The auto-mask result to visualise.
        alpha: Overlay transparency.
        figsize: Figure size.
        max_masks: Maximum number of masks to display (``None`` for all).

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    img = load_image(image) if not isinstance(image, np.ndarray) else image
    masks_to_show = result.masks[:max_masks] if max_masks else result.masks

    all_masks = np.stack([m.data for m in masks_to_show]) if masks_to_show else np.zeros(
        (0, *img.shape[:2]), dtype=bool
    )
    overlay = draw_masks_on_image(img, all_masks, alpha=alpha)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].imshow(img)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title(f"Auto Masks ({len(masks_to_show)} shown)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig


def show_video_frame(
    image: str | Path | npt.NDArray[np.uint8],
    frame_masks: FrameMasks,
    *,
    alpha: float = 0.5,
    figsize: tuple[int, int] = (12, 8),
    show_ids: bool = True,
) -> Any:
    """Display a single video frame with tracked object masks.

    Args:
        image: The frame image.
        frame_masks: Masks for this frame.
        alpha: Overlay transparency.
        figsize: Figure size.
        show_ids: Annotate object IDs on the image.

    Returns:
        The matplotlib figure.
    """
    import matplotlib.pyplot as plt

    img = load_image(image) if not isinstance(image, np.ndarray) else image
    oids = frame_masks.object_ids
    colors = [_get_color(oid) for oid in oids]
    masks_arr = (
        np.stack([frame_masks.masks[oid] for oid in oids])
        if oids
        else np.zeros((0, *img.shape[:2]), dtype=bool)
    )
    overlay = draw_masks_on_image(img, masks_arr, alpha=alpha, colors=colors)

    if show_ids:
        for oid in oids:
            mask = frame_masks.masks[oid]
            ys, xs = np.where(mask)
            if len(xs) > 0:
                cx, cy = int(xs.mean()), int(ys.mean())
                cv2.putText(
                    overlay,
                    str(oid),
                    (cx - 10, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(overlay)
    ax.set_title(f"Frame {frame_masks.frame_idx}")
    ax.axis("off")
    fig.tight_layout()
    fig.canvas.draw_idle()
    return fig


def save_video_overlay(
    video_dir: str | Path,
    results: VideoResults,
    output_dir: str | Path,
    *,
    alpha: float = 0.5,
    draw_contours: bool = True,
) -> Path:
    """Save an overlaid video as a folder of PNGs.

    For each frame in *results*, overlays the tracked masks on the original
    frame and saves the composite image.

    Args:
        video_dir: Source frame directory.
        results: Tracking results from :meth:`VideoSession.propagate`.
        output_dir: Output directory for overlay frames.
        alpha: Mask overlay transparency.
        draw_contours: Draw mask contours.

    Returns:
        Path to the output directory.
    """
    from lazysammy2.utils import list_frame_files

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frame_files = list_frame_files(video_dir)

    for frame_idx, fm in results:
        if frame_idx >= len(frame_files):
            continue
        img = load_image(frame_files[frame_idx])
        oids = fm.object_ids
        colors = [_get_color(oid) for oid in oids]
        masks_arr = (
            np.stack([fm.masks[oid] for oid in oids])
            if oids
            else np.zeros((0, *img.shape[:2]), dtype=bool)
        )
        overlay = draw_masks_on_image(
            img, masks_arr, alpha=alpha, colors=colors, draw_contours=draw_contours
        )
        out_path = out / f"frame_{frame_idx:06d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return out


def save_video_overlay_mp4(
    video_dir: str | Path,
    results: VideoResults,
    output_path: str | Path,
    *,
    fps: float = 24.0,
    alpha: float = 0.5,
    draw_contours: bool = True,
    show_ids: bool = True,
    show_frame_number: bool = True,
    codec: str = "mp4v",
) -> Path:
    """Render tracking results into an MP4 video file with mask overlays.

    Every frame from the source directory is written to the video.  Frames
    that appear in *results* get coloured mask overlays; frames without
    results are written as-is so you always get a complete video.

    Args:
        video_dir: Source frame directory.
        results: Tracking results from :meth:`VideoSession.propagate`.
        output_path: Destination ``.mp4`` file path.
        fps: Frames per second for the output video.
        alpha: Mask overlay transparency.
        draw_contours: Draw mask contour lines.
        show_ids: Annotate object IDs on each mask.
        show_frame_number: Burn frame index into the top-left corner.
        codec: FourCC codec string (default ``"mp4v"``; try ``"avc1"``
            for H.264 if your OpenCV build supports it).

    Returns:
        Path to the written video file.
    """
    from lazysammy2.utils import list_frame_files

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    frame_files = list_frame_files(video_dir)
    if not frame_files:
        msg = f"No frames found in {video_dir}"
        raise FileNotFoundError(msg)

    # Read first frame to determine dimensions
    sample = load_image(frame_files[0])
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    if not writer.isOpened():
        msg = f"Failed to open VideoWriter for {out} (codec={codec!r})"
        raise RuntimeError(msg)

    try:
        for idx, fpath in enumerate(frame_files):
            img = load_image(fpath)
            fm = results._frame_index().get(idx)

            if fm is not None and len(fm.object_ids) > 0:
                oids = fm.object_ids
                colors = [_get_color(oid) for oid in oids]
                masks_arr = np.stack([fm.masks[oid] for oid in oids])
                frame = draw_masks_on_image(
                    img, masks_arr, alpha=alpha, colors=colors,
                    draw_contours=draw_contours,
                )
                if show_ids:
                    for oid in oids:
                        mask = fm.masks[oid]
                        ys, xs = np.where(mask)
                        if len(xs) > 0:
                            cx, cy = int(xs.mean()), int(ys.mean())
                            cv2.putText(
                                frame, str(oid), (cx - 10, cy + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (255, 255, 255), 2, cv2.LINE_AA,
                            )
            else:
                frame = img.copy()

            if show_frame_number:
                cv2.putText(
                    frame, f"Frame {idx}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA,
                )

            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()

    return out
