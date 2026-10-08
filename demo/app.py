"""Gradio demo for lazysammy.

Launch with::

    uv run --extra demo python demo/app.py

Or::

    uv run --extra demo gradio demo/app.py
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import cv2
import gradio as gr
import numpy as np

from lazysammy import (
    SAM2,
    SAM3,
    ConceptPrediction,
    ImagePrediction,
    draw_masks_on_image,
    extract_frames,
    load_image,
    natural_sort_key,
    save_video_overlay_mp4,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_VIDEO_FRAME_EXTS = {".jpg", ".jpeg", ".png"}

# Shown wherever the prompt list is still empty.
_NO_PROMPTS_HTML = (
    '<div class="prompt-summary"><em>No prompts yet. Click on the frame to start.</em></div>'
)

# ---------------------------------------------------------------------------
# Global model cache (lazy: loaded once on first use per model size)
#
# SAM 2 predictors hold GPU memory and are not thread-safe, so construction is
# guarded by a lock and every inference call is serialised through it. This
# keeps concurrent Gradio requests from corrupting shared session state.
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, SAM2] = {}
# SAM 3 is a separate, optional model (open-vocabulary concepts). It is cached
# under a single key because it has no size variants.
_SAM3_CACHE: dict[str, SAM3] = {}
# Guards model construction only, so loading multi-GB weights never happens
# twice for the same size.
_MODEL_LOAD_LOCK = threading.Lock()
# SAM 2 predictors are not thread-safe, so every inference call is serialised.
# This lock is intentionally coarse: it also covers video propagation and MP4
# encoding, which keeps concurrent Gradio requests from corrupting shared state.
_INFERENCE_LOCK = threading.Lock()

# Errors the demo surfaces as a status message instead of crashing the request.
_DEMO_ERRORS = (ValueError, RuntimeError, IndexError, OSError)

# Output directory for rendered tracking videos. Created once so repeated
# tracks do not leak a temp directory each; Gradio serves the file from here.
_OUTPUT_DIR = Path(tempfile.mkdtemp(prefix="lazysammy_demo_"))


def _get_model(model_size: str) -> SAM2:
    """Return a cached :class:`SAM2` for *model_size*, loading it if needed."""
    with _MODEL_LOAD_LOCK:
        if model_size not in _MODEL_CACHE:
            logger.info("Loading SAM2 model: %s", model_size)
            _MODEL_CACHE[model_size] = SAM2(model_size)
        return _MODEL_CACHE[model_size]


def _get_sam3_model() -> SAM3:
    """Return the cached :class:`SAM3` model, loading it on first use.

    SAM 3 is an optional extra; if it is not installed the underlying import
    raises ``ImportError``, which the handlers surface as a status message.
    """
    with _MODEL_LOAD_LOCK:
        if "sam3" not in _SAM3_CACHE:
            logger.info("Loading SAM3 model")
            _SAM3_CACHE["sam3"] = SAM3()
        return _SAM3_CACHE["sam3"]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 0, 0),
    (0, 200, 0),
    (0, 80, 255),
    (255, 200, 0),
    (255, 0, 200),
    (0, 200, 200),
    (180, 0, 0),
    (0, 160, 0),
]


def _color_for_obj(obj_id: int) -> tuple[int, int, int]:
    return _COLORS[obj_id % len(_COLORS)]


def _draw_points_on_image(
    img: np.ndarray,
    points: list[list[float]],
    *,
    radius: int = 8,
) -> np.ndarray:
    """Draw point prompts with fg/bg colouring and +/- labels."""
    out = img.copy()
    for p in points:
        x, y, label = int(p[0]), int(p[1]), int(p[2])
        color = (0, 220, 0) if label == 1 else (220, 50, 50)
        cv2.circle(out, (x, y), radius, color, -1)
        cv2.circle(out, (x, y), radius, (255, 255, 255), 2)
        _draw_label(out, "+" if label == 1 else "-", x + radius + 2, y + 4, color, scale=0.55)
    return out


def _draw_box_corners_on_image(
    img: np.ndarray,
    corners: list[list[float]],
) -> np.ndarray:
    """Draw partial / complete box prompt."""
    out = img.copy()
    if len(corners) >= 1:
        x1, y1 = int(corners[0][0]), int(corners[0][1])
        cv2.drawMarker(out, (x1, y1), (0, 200, 255), cv2.MARKER_CROSS, 16, 2)
        cv2.putText(
            out,
            "corner 1",
            (x1 + 10, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 200, 255),
            1,
            cv2.LINE_AA,
        )
    if len(corners) >= 2:
        x2, y2 = int(corners[1][0]), int(corners[1][1])
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
    return out


def _draw_label(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    *,
    scale: float = 0.45,
) -> None:
    """Draw *text* with a white outline so it stays legible on any frame."""
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Video-tab prompt model
#
# Prompts are typed objects rather than ``dict``s with a ``"type"`` string, so
# drawing, session application, and summary rendering are polymorphic instead
# of re-branched on the type at every call site.
# ---------------------------------------------------------------------------


@dataclass
class PointPrompt:
    """One object's point prompts on a single frame."""

    frame_idx: int
    obj_id: int
    points: list[list[float]]
    labels: list[int]

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw the points onto *img* in place."""
        for pt, lab in zip(self.points, self.labels, strict=False):
            x, y = int(pt[0]), int(pt[1])
            point_color = color if lab == 1 else (128, 128, 128)
            cv2.circle(img, (x, y), 7, point_color, -1)
            cv2.circle(img, (x, y), 7, (255, 255, 255), 2)
            tag = f"obj{self.obj_id}+" if lab == 1 else f"obj{self.obj_id}-"
            _draw_label(img, tag, x + 10, y - 6, color)

    def apply(self, session: Any) -> None:
        """Register this prompt on a video tracking session."""
        session.add_points(
            frame_idx=self.frame_idx,
            obj_id=self.obj_id,
            points=self.points,
            labels=self.labels,
        )

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        fg_tag = '<span style="color:green">fg</span>'
        bg_tag = '<span style="color:red">bg</span>'
        pts = ", ".join(
            f"({pt[0]:.0f},{pt[1]:.0f}) {fg_tag if lb == 1 else bg_tag}"
            for pt, lb in zip(self.points, self.labels, strict=False)
        )
        return f"Obj {self.obj_id} | Frame {self.frame_idx} | {pts}"


@dataclass
class BoxPrompt:
    """One object's bounding-box prompt on a single frame."""

    frame_idx: int
    obj_id: int
    box: list[float]

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw the box onto *img* in place."""
        bx = self.box
        cv2.rectangle(img, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
        _draw_label(img, f"obj{self.obj_id}", int(bx[0]) + 4, int(bx[1]) - 6, color, scale=0.5)

    def apply(self, session: Any) -> None:
        """Register this prompt on a video tracking session."""
        session.add_box(frame_idx=self.frame_idx, obj_id=self.obj_id, box=self.box)

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        b = self.box
        return (
            f"Obj {self.obj_id} | Frame {self.frame_idx} | "
            f"box ({b[0]:.0f},{b[1]:.0f}) to ({b[2]:.0f},{b[3]:.0f})"
        )


VideoPrompt = PointPrompt | BoxPrompt


@dataclass
class TextPrompt:
    """A SAM 3 text concept prompt for a video frame.

    Unlike point/box prompts, a text prompt is not tied to one object id: SAM 3
    detects every matching instance and assigns each its own id.
    """

    frame_idx: int
    text: str

    def draw(self, img: np.ndarray, color: tuple[int, int, int]) -> None:
        """Draw a label for the concept in the top-left corner."""
        _draw_label(img, f'"{self.text}"', 10, 24, color, scale=0.6)

    def apply(self, session: Any) -> None:
        """Register this concept prompt on a SAM 3 video session."""
        session.add_text(frame_idx=self.frame_idx, text=self.text)

    def summary_html(self) -> str:
        """Render the prompt body for the prompt-list summary."""
        return f'Concept "{self.text}" | Frame {self.frame_idx}'


@dataclass
class ImagePrompts:
    """Accumulated image-tab prompts for a single object on frame 0.

    ``point`` holds the committed point clicks; ``box_corners`` holds the
    pending box corners (0, 1, or 2) before they become a :class:`BoxPrompt`.
    """

    point: PointPrompt | None = None
    box_corners: list[list[float]] = field(default_factory=list)


class ImageTabState(NamedTuple):
    """Outputs of the image-tab click/undo/clear handlers, in Gradio order."""

    image: np.ndarray | None
    prompts: ImagePrompts
    display: str


class VideoTabState(NamedTuple):
    """Outputs of :func:`_on_extract`, in Gradio output order."""

    frame_files: list[str]
    info: str
    frames_dir: str | None
    num_frames: int
    slider: Any
    preview: np.ndarray | None
    clean_frame: np.ndarray | None
    prompts: list[VideoPrompt]
    box: list[Any]
    counter: str
    prompts_html: str


# =========================================================================
# Tab 1: Image Segmentation
# =========================================================================


def _run_image_segmentation(
    image: np.ndarray,
    model_size: str,
    predict: Callable[[SAM2], ImagePrediction],
    *,
    annotate: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Run one image prediction under the inference lock and format the result.

    Owns the model lookup, the lock, the error handling, and the overlay/info
    formatting, so the point and box paths differ only in *predict*/*annotate*.
    """
    try:
        sam = _get_model(model_size)
        with _INFERENCE_LOCK:
            pred = predict(sam)
    except _DEMO_ERRORS as exc:
        logger.exception("Image segmentation failed")
        return image, f"Error: Segmentation failed: {exc}"

    best = pred.best_mask
    overlay = draw_masks_on_image(image, best.numpy()[np.newaxis], alpha=0.5)
    if annotate is not None:
        overlay = annotate(overlay)
    info = f"Best mask: IoU: {best.score:.3f} | area: {best.area:,} px"
    return overlay, info


def _segment_with_points(
    image: np.ndarray | None,
    points: list[list[float]],
    model_size: str,
    multimask: bool,
) -> tuple[np.ndarray | None, str]:
    """Segment using accumulated point prompts."""
    if image is None:
        return None, "Upload an image first."
    if not points:
        return image, "Click on the image to place point prompts."

    coords = [[p[0], p[1]] for p in points]
    labels = [int(p[2]) for p in points]
    return _run_image_segmentation(
        image,
        model_size,
        lambda sam: sam.segment(image, points=coords, labels=labels, multimask_output=multimask),
        annotate=lambda overlay: _draw_points_on_image(overlay, points),
    )


def _segment_with_box(
    image: np.ndarray | None,
    box: list[float],
    model_size: str,
) -> tuple[np.ndarray | None, str]:
    """Segment using a bounding-box prompt."""
    if image is None:
        return None, "Upload an image first."
    if len(box) < 4:
        return image, "Click two corners on the image to define a box."

    x1, y1 = min(box[0], box[2]), min(box[1], box[3])
    x2, y2 = max(box[0], box[2]), max(box[1], box[3])
    return _run_image_segmentation(
        image,
        model_size,
        lambda sam: sam.segment_box(image, x1, y1, x2, y2),
        annotate=lambda overlay: cv2.rectangle(
            overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2
        ),
    )


# =========================================================================
# Tab 2: Auto Segment Everything
# =========================================================================


def _render_segmentation(
    image: np.ndarray,
    result: Any,
    *,
    alpha: float,
    empty_message: str,
    info: str,
    annotate: Callable[[np.ndarray], np.ndarray] | None = None,
) -> tuple[np.ndarray, str]:
    """Draw a mask result and format its info line, or report an empty result.

    Shared tail of the auto-segment and concept-segment handlers so the
    overlay/info formatting lives in one place.
    """
    if not result.masks:
        return image, empty_message
    overlay = draw_masks_on_image(image, result.numpy(), alpha=alpha)
    if annotate is not None:
        overlay = annotate(overlay)
    return overlay, info


def _auto_segment_image(
    image: np.ndarray | None,
    model_size: str,
    min_area: int,
    min_iou: float,
) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "Upload an image first."

    try:
        sam = _get_model(model_size)
        with _INFERENCE_LOCK:
            result = sam.auto_segment(image)
    except _DEMO_ERRORS as exc:
        logger.exception("Auto segmentation failed")
        return image, f"Error: Auto segmentation failed: {exc}"

    if min_area > 0:
        result = result.filter_by_area(min_area=int(min_area))
    if min_iou > 0:
        result = result.filter_by_iou(min_iou=float(min_iou))

    return _render_segmentation(
        image,
        result,
        alpha=0.4,
        empty_message="No masks found with the current filters.",
        info=f"Found {len(result.masks)} masks",
    )


# =========================================================================
# Tab 3: Concept Segmentation (SAM 3)
# =========================================================================


def _concept_segment_image(
    image: np.ndarray | None,
    text: str,
    min_score: float,
) -> tuple[np.ndarray | None, str]:
    """Segment every instance of a text concept with SAM 3."""
    if image is None:
        return None, "Upload an image first."
    if not text.strip():
        return image, "Type a concept to segment (e.g. 'a player in white')."

    try:
        sam = _get_sam3_model()
        with _INFERENCE_LOCK:
            result = sam.segment_text(image, text.strip())
    except ImportError:
        return image, "SAM 3 is not installed. Run: uv sync --extra sam3"
    except _DEMO_ERRORS as exc:
        logger.exception("Concept segmentation failed")
        return image, f"Error: Concept segmentation failed: {exc}"

    if min_score > 0:
        result = result.filter_by_score(float(min_score))

    return _render_segmentation(
        image,
        result,
        alpha=0.4,
        empty_message=f"No instances of {text.strip()!r} found.",
        info=f"Found {len(result.masks)} instance(s) of {text.strip()!r}",
        annotate=lambda overlay: _draw_concept_boxes(overlay, result),
    )


def _draw_concept_boxes(
    image: np.ndarray,
    prediction: ConceptPrediction,
) -> np.ndarray:
    """Draw each detected instance's bounding box and score."""
    out = image.copy()
    for i, (mask, box) in enumerate(zip(prediction.masks, prediction.boxes, strict=False)):
        color = _color_for_obj(i)
        x1, y1, x2, y2 = (int(v) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        _draw_label(out, f"{mask.score:.2f}", x1 + 4, y1 - 6, color, scale=0.5)
    return out


# =========================================================================
# Tab 4: Video Object Tracking
# =========================================================================


def _extract_video_frames(
    video_path: str | None,
    every_n: int,
    max_frames: int,
) -> VideoTabState:
    """Extract frames from an uploaded video and build the reset tab state."""
    if not video_path:
        return _empty_video_state("Upload a video first.")

    try:
        frames_dir = extract_frames(
            video_path,
            every_n=int(every_n),
            max_frames=int(max_frames) or None,
            clean=True,
        )
    except _DEMO_ERRORS as exc:
        return _empty_video_state(f"Error: Frame extraction failed: {exc}")

    frame_paths = sorted(
        (f for f in Path(frames_dir).iterdir() if f.suffix.lower() in _VIDEO_FRAME_EXTS),
        key=natural_sort_key,
    )
    if not frame_paths:
        return _empty_video_state("No frames were extracted from this video.")

    first = _get_frame_image([str(f) for f in frame_paths], 0)
    return VideoTabState(
        frame_files=[str(f) for f in frame_paths],
        info=f"Extracted {len(frame_paths)} frames to {frames_dir}",
        frames_dir=str(frames_dir),
        num_frames=len(frame_paths),
        slider=gr.Slider(maximum=max(0, len(frame_paths) - 1), value=0),
        preview=first,
        clean_frame=first,
        prompts=[],
        box=[],
        counter=_frame_counter(0, len(frame_paths)),
        prompts_html=_NO_PROMPTS_HTML,
    )


def _empty_video_state(info: str) -> VideoTabState:
    """A reset video-tab state carrying only a status message."""
    return VideoTabState(
        frame_files=[],
        info=info,
        frames_dir=None,
        num_frames=0,
        slider=gr.Slider(maximum=0, value=0),
        preview=None,
        clean_frame=None,
        prompts=[],
        box=[],
        counter=_frame_counter(0, 0),
        prompts_html=_NO_PROMPTS_HTML,
    )


def _frame_counter(idx: int, num_frames: int) -> str:
    """Render the frame counter HTML for *idx* of *num_frames*."""
    return f'<div class="frame-counter">Frame {idx} / {max(0, num_frames - 1)}</div>'


def _get_frame_image(frame_files: list[str], frame_idx: int) -> np.ndarray | None:
    """Load frame *frame_idx* from *frame_files* as an RGB array."""
    if not frame_files or frame_idx < 0 or frame_idx >= len(frame_files):
        return None
    try:
        return load_image(frame_files[frame_idx])
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Could not load frame %s: %s", frame_idx, exc)
        return None


def _render_frame(
    clean_frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    frame_idx: int,
) -> np.ndarray | None:
    """Draw committed prompts and any pending box corners for one frame."""
    annotated = _draw_video_prompts(clean_frame, prompts, frame_idx)
    if vid_box and annotated is not None:
        annotated = _draw_box_corners_on_image(annotated, vid_box)
    return annotated


def _draw_video_prompts(
    frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    frame_idx: int,
) -> np.ndarray | None:
    """Draw all prompts for a given frame."""
    if frame is None:
        return None
    out = frame.copy()
    for prompt in prompts:
        if prompt.frame_idx == frame_idx:
            if isinstance(prompt, (PointPrompt, BoxPrompt)):
                color = _color_for_obj(prompt.obj_id)
            else:
                color = (255, 255, 255)
            prompt.draw(out, color)
    return out


def _format_prompts_html(prompts: list[VideoPrompt]) -> str:
    """Render prompts as styled HTML for the video tab summary."""
    if not prompts:
        return _NO_PROMPTS_HTML

    obj_ids = sorted({p.obj_id for p in prompts if isinstance(p, (PointPrompt, BoxPrompt))})
    frames_used = sorted({p.frame_idx for p in prompts})
    n_points = sum(len(p.points) for p in prompts if isinstance(p, PointPrompt))
    n_boxes = sum(1 for p in prompts if isinstance(p, BoxPrompt))
    n_text = sum(1 for p in prompts if isinstance(p, TextPrompt))

    header = (
        f"<b>{len(obj_ids)} object(s)</b> | "
        f"{n_points} point(s) | {n_boxes} box(es) | {n_text} concept(s) | "
        f"on frame(s) {', '.join(str(f) for f in frames_used)}"
    )

    rows = []
    for p in prompts:
        if isinstance(p, TextPrompt):
            badge = (
                '<span style="display:inline-block;width:10px;height:10px;'
                "border-radius:50%;background:#ffffff;border:1px solid #94a3b8;"
                'margin-right:4px"></span>'
            )
        else:
            color = "#{:02x}{:02x}{:02x}".format(*_color_for_obj(p.obj_id))
            badge = (
                f'<span style="display:inline-block;width:10px;height:10px;'
                f'border-radius:50%;background:{color};margin-right:4px"></span>'
            )
        rows.append(f"{badge}{p.summary_html()}")

    body = "<br>".join(rows)
    return f'<div class="prompt-summary">{header}<hr style="margin:4px 0">{body}</div>'


def _track_and_render(
    frames_dir_str: str | None,
    prompts_state: list[VideoPrompt],
    model_size: str,
    fps: float,
    alpha: float,
    bidirectional: bool,
) -> tuple[str | None, str]:
    """Build a tracking session from the accumulated prompts and render an MP4.

    Point/box prompts use SAM 2. If any text concept prompt is present, SAM 3
    is used instead (it also supports points and boxes, so a mixed prompt set
    still works).
    """
    if not frames_dir_str:
        return None, "Extract frames first."
    if not prompts_state:
        return None, "Add at least one prompt before tracking."

    frames_dir = Path(frames_dir_str)
    if not frames_dir.is_dir():
        return None, f"Frame directory is gone: {frames_dir}"

    uses_concepts = any(isinstance(p, TextPrompt) for p in prompts_state)
    try:
        with _INFERENCE_LOCK:
            if uses_concepts:
                session = _get_sam3_model().video(frames_dir, offload_video_to_cpu=True)
            else:
                session = _get_model(model_size).video(frames_dir, offload_video_to_cpu=True)
            for prompt in prompts_state:
                prompt.apply(session)
            if uses_concepts:
                results = session.propagate(direction="both" if bidirectional else "forward")
                session.close()
            elif bidirectional:
                results = session.propagate_bidirectional()
            else:
                results = session.propagate()
            direction = "bidirectional" if bidirectional else "forward"

            # Gradio needs a concrete file. Write into the shared output dir
            # with a unique name so concurrent tracks do not collide.
            out_path = _OUTPUT_DIR / f"tracking_{uuid4().hex}.mp4"
            save_video_overlay_mp4(
                frames_dir,
                results,
                out_path,
                fps=float(fps),
                alpha=float(alpha),
                show_ids=True,
                show_frame_number=True,
            )
    except ImportError:
        return None, "SAM 3 is not installed. Run: uv sync --extra sam3"
    except _DEMO_ERRORS as exc:
        logger.exception("Tracking failed")
        return None, f"Error: Tracking failed: {exc}"

    backend = "SAM 3" if uses_concepts else "SAM 2"
    info = (
        f"{backend}: tracked {len(results.object_ids)} object(s) "
        f"across {len(results)} frames ({direction})"
    )
    return str(out_path), info


# =========================================================================
# Build the Gradio UI
# =========================================================================

_CSS = """
.click-instruction {
    text-align: center; padding: 8px 12px; border-radius: 8px;
    font-weight: 600; font-size: 0.95em; margin: 4px 0;
}
.click-fg { background: #dcfce7; color: #166534; }
.click-bg { background: #fee2e2; color: #991b1b; }
.click-box { background: #fef9c3; color: #854d0e; }
.step-header {
    margin: 0 0 4px 0; padding: 6px 14px; border-radius: 8px;
    font-weight: 700; font-size: 1.05em;
    background: #f0f4ff; color: #1e3a5f; border-left: 4px solid #3b82f6;
}
.prompt-summary {
    padding: 6px 12px; border-radius: 6px;
    background: #f8fafc; border: 1px solid #e2e8f0;
    font-size: 0.88em; line-height: 1.5;
}
.frame-counter {
    text-align: center; font-size: 1.1em; font-weight: 600;
    padding: 4px 0; color: #374151;
}
"""


MODE_FOREGROUND = "Add foreground point"
MODE_BACKGROUND = "Add background point"
MODE_BOX = "Draw a box (2 clicks)"

MODE_CHOICES = [MODE_FOREGROUND, MODE_BACKGROUND, MODE_BOX]


def _is_box_mode(mode: str) -> bool:
    """Return True when *mode* is the two-click box mode."""
    return mode == MODE_BOX


def _label_for_mode(mode: str) -> int:
    """Return the point label (1 foreground, 0 background) for *mode*."""
    return 0 if mode == MODE_BACKGROUND else 1


def _render_mode_instruction(mode: str) -> str:
    if mode == MODE_BACKGROUND:
        return (
            '<div class="click-instruction click-bg">'
            "Click on the image to add <b>background</b> points "
            "(areas to exclude)</div>"
        )
    if mode == MODE_BOX:
        return (
            '<div class="click-instruction click-box">'
            "Click <b>two corners</b> on the image to define a "
            "bounding box</div>"
        )
    return (
        '<div class="click-instruction click-fg">'
        "Click on the image to add <b>foreground</b> points "
        "(objects to include)</div>"
    )


def _gradio_major_version() -> int:
    """Return the installed Gradio major version (0 when undetectable)."""
    raw = getattr(gr, "__version__", "") or ""
    try:
        return int(str(raw).split(".")[0])
    except (ValueError, IndexError):
        return 0


# =========================================================================
# Image-tab prompt bookkeeping (pure helpers, unit-testable)
#
# These live at module level rather than as closures inside build_app() so the
# prompt logic can be tested directly, without a running Gradio server.
# =========================================================================


def _format_img_prompts(prompts: ImagePrompts) -> str:
    """Render the image-tab prompt list as a compact one-line string."""
    parts = []
    if prompts.point is not None:
        for pt, lb in zip(prompts.point.points, prompts.point.labels, strict=False):
            kind = "fg" if lb == 1 else "bg"
            parts.append(f"({pt[0]:.0f}, {pt[1]:.0f}) {kind}")
    corners = prompts.box_corners
    if len(corners) == 1:
        parts.append(f"box corner 1: ({corners[0][0]:.0f}, {corners[0][1]:.0f}): click 2nd corner")
    if len(corners) >= 2:
        parts.append(
            f"box: ({corners[0][0]:.0f},{corners[0][1]:.0f}) to "
            f"({corners[1][0]:.0f},{corners[1][1]:.0f})"
        )
    return " | ".join(parts) if parts else ""


def _redraw_input(original: np.ndarray | None, prompts: ImagePrompts) -> np.ndarray | None:
    """Return a copy of *original* with the current prompts drawn on it."""
    if original is None:
        return None
    out = original.copy()
    if prompts.point is not None:
        prompts.point.draw(out, (0, 220, 0))
    if prompts.box_corners:
        out = _draw_box_corners_on_image(out, prompts.box_corners)
    return out


def _image_state(original: np.ndarray | None, prompts: ImagePrompts) -> ImageTabState:
    """Build the annotated image, prompt state, and display text."""
    return ImageTabState(
        image=_redraw_input(original, prompts),
        prompts=prompts,
        display=_format_img_prompts(prompts),
    )


def apply_image_click(
    original: np.ndarray | None,
    prompts: ImagePrompts,
    mode: str,
    x: float,
    y: float,
) -> ImageTabState:
    """Apply one image-tab click and return the updated state."""
    if original is None:
        return ImageTabState(original, prompts, "")

    if _is_box_mode(mode):
        corners = [*prompts.box_corners, [x, y]]
        if len(corners) > 2:
            corners = [[x, y]]
        prompts = replace(prompts, box_corners=corners)
    else:
        label = _label_for_mode(mode)
        if prompts.point is None:
            prompts = replace(prompts, point=PointPrompt(0, 0, [[x, y]], [label]))
        else:
            prompts = replace(
                prompts,
                point=replace(
                    prompts.point,
                    points=[*prompts.point.points, [x, y]],
                    labels=[*prompts.point.labels, label],
                ),
            )
    return _image_state(original, prompts)


def undo_last_prompt(
    original: np.ndarray | None,
    prompts: ImagePrompts,
    mode: str,
) -> ImageTabState:
    """Undo the most recent image-tab prompt (point or box corner)."""
    if _is_box_mode(mode):
        prompts = replace(prompts, box_corners=prompts.box_corners[:-1])
    elif prompts.point is not None:
        point = prompts.point
        if len(point.points) > 1:
            prompts = replace(
                prompts,
                point=replace(point, points=point.points[:-1], labels=point.labels[:-1]),
            )
        else:
            prompts = replace(prompts, point=None)
    return _image_state(original, prompts)


def clear_prompts(original: np.ndarray | None) -> ImageTabState:
    """Clear all image-tab prompts, returning a fresh copy of the image."""
    out = original.copy() if original is not None else None
    return ImageTabState(out, ImagePrompts(), "")


def select_segmentation_mode(prompts: ImagePrompts) -> BoxPrompt | PointPrompt | None:
    """Choose the prompt to segment with; a completed box wins over points."""
    corners = prompts.box_corners
    if len(corners) >= 2:
        (x1, y1), (x2, y2) = corners
        return BoxPrompt(0, 0, [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
    return prompts.point


def _on_image_upload(
    img: np.ndarray | None,
    mode: str,
) -> tuple[np.ndarray | None, ImagePrompts, str]:
    """Reset prompt state when a new image is uploaded."""
    return img, ImagePrompts(), _render_mode_instruction(mode)


def _on_image_click(
    original: np.ndarray | None,
    prompts: ImagePrompts,
    mode: str,
    evt: gr.SelectData,
) -> ImageTabState:
    """Gradio ``select`` adapter: unpack the click coordinates."""
    x, y = evt.index[0], evt.index[1]
    return apply_image_click(original, prompts, mode, x, y)


def _segment(
    original: np.ndarray | None,
    prompts: ImagePrompts,
    ms: str,
    multimask: bool,
) -> tuple[np.ndarray | None, str]:
    """Dispatch image-tab segmentation to box or point mode."""
    if original is None:
        return None, "Upload an image first."
    prompt = select_segmentation_mode(prompts)
    if isinstance(prompt, BoxPrompt):
        return _segment_with_box(original, prompt.box, ms)
    if isinstance(prompt, PointPrompt):
        points = [[pt[0], pt[1], lb] for pt, lb in zip(prompt.points, prompt.labels, strict=False)]
        return _segment_with_points(original, points, ms, multimask)
    return original, "Add some prompts first (click on the image)."


# =========================================================================
# Video-tab event handlers (module level, unit-testable)
# =========================================================================


def _on_extract(video_path: str | None, every_n: int, max_frames: int) -> VideoTabState:
    """Extract frames from an uploaded video and reset the video-tab state."""
    return _extract_video_frames(video_path, every_n, int(max_frames) or 0)


def _on_slider_change(
    frame_files: list[str],
    idx: int,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    num_frames: int,
) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Redraw the preview when the frame slider moves."""
    idx = int(idx)
    clean = _get_frame_image(frame_files, idx)
    annotated = _render_frame(clean, prompts, vid_box, idx)
    return annotated, clean, _frame_counter(idx, int(num_frames))


def _append_point(
    prompts: list[VideoPrompt],
    frame_idx: int,
    obj_id: int,
    x: float,
    y: float,
    label: int,
) -> list[VideoPrompt]:
    """Append a point to the matching prompt, or start a new one."""
    for i, p in enumerate(prompts):
        if isinstance(p, PointPrompt) and p.frame_idx == frame_idx and p.obj_id == obj_id:
            updated = replace(p, points=[*p.points, [x, y]], labels=[*p.labels, label])
            return [*prompts[:i], updated, *prompts[i + 1 :]]
    return [*prompts, PointPrompt(frame_idx, obj_id, [[x, y]], [label])]


def _on_frame_click(
    clean_frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    frame_slider_val: int,
    obj_id: int,
    mode: str,
    evt: gr.SelectData,
) -> tuple[np.ndarray | None, list[VideoPrompt], list[Any], str]:
    """Gradio ``select`` adapter for the video frame preview."""
    if clean_frame is None:
        return None, prompts, vid_box, _format_prompts_html(prompts)

    x, y = evt.index[0], evt.index[1]
    frame_idx = int(frame_slider_val)
    obj_id = int(obj_id)

    if _is_box_mode(mode):
        vid_box = [*vid_box, [x, y]]
        if len(vid_box) == 2:
            (x1, y1), (x2, y2) = vid_box
            prompts = [
                *prompts,
                BoxPrompt(
                    frame_idx,
                    obj_id,
                    [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                ),
            ]
            vid_box = []
    else:
        prompts = _append_point(prompts, frame_idx, obj_id, x, y, _label_for_mode(mode))

    display = _format_prompts_html(prompts)
    annotated = _render_frame(clean_frame, prompts, vid_box, frame_idx)
    return annotated, prompts, vid_box, display


def _vid_undo(
    clean_frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    frame_slider_val: int,
    mode: str,
) -> tuple[np.ndarray | None, list[VideoPrompt], list[Any], str]:
    """Undo the most recent video-tab prompt (point, box corner, or concept)."""
    frame_idx = int(frame_slider_val)
    if _is_box_mode(mode) and vid_box:
        vid_box = vid_box[:-1]
    elif prompts:
        last = prompts[-1]
        if isinstance(last, PointPrompt) and len(last.points) > 1:
            prompts = [
                *prompts[:-1],
                replace(last, points=last.points[:-1], labels=last.labels[:-1]),
            ]
        else:
            prompts = prompts[:-1]
    display = _format_prompts_html(prompts)
    annotated = _render_frame(clean_frame, prompts, vid_box, frame_idx)
    return annotated, prompts, vid_box, display


def _add_text_prompt(
    clean_frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    frame_slider_val: int,
    text: str,
) -> tuple[np.ndarray | None, list[VideoPrompt], list[Any], str]:
    """Add a SAM 3 text concept prompt for the current frame."""
    if clean_frame is None:
        return None, prompts, vid_box, _format_prompts_html(prompts)
    if not text.strip():
        return clean_frame, prompts, vid_box, _format_prompts_html(prompts)

    frame_idx = int(frame_slider_val)
    prompts = [*prompts, TextPrompt(frame_idx, text.strip())]
    display = _format_prompts_html(prompts)
    annotated = _render_frame(clean_frame, prompts, vid_box, frame_idx)
    return annotated, prompts, vid_box, display


def _vid_clear(
    clean_frame: np.ndarray | None,
) -> tuple[np.ndarray | None, list[VideoPrompt], list[Any], str]:
    """Clear all video-tab prompts."""
    return clean_frame, [], [], _NO_PROMPTS_HTML


def _style_kwargs() -> dict[str, Any]:
    """Theme and CSS kwargs for the installed Gradio version.

    Gradio 6 moved ``theme`` and ``css`` from the ``Blocks`` constructor to
    ``launch()``; this is the single place that knows which one applies.
    """
    return {"theme": gr.themes.Soft(), "css": _CSS}


def _build_image_tab(model_size: gr.Dropdown) -> None:
    """Build the Image Segmentation tab and wire its events."""
    with gr.Tab("Image Segmentation"):
        gr.Markdown(
            "Upload an image, then **click directly on it** to place "
            "prompts. Choose a mode below before clicking."
        )

        # -- Prompt mode selection --
        with gr.Row():
            prompt_mode = gr.Radio(
                choices=MODE_CHOICES,
                value=MODE_FOREGROUND,
                label="Click Mode",
                info="Select what happens when you click on the image.",
            )
            multimask_toggle = gr.Checkbox(
                label="Multi-mask output",
                value=True,
            )

        click_instruction = gr.HTML(_render_mode_instruction(MODE_FOREGROUND))

        # -- Images side-by-side --
        with gr.Row():
            with gr.Column(scale=1):
                img_input = gr.Image(
                    label="Input Image: click to add prompts",
                    type="numpy",
                    height=512,
                )
            with gr.Column(scale=1):
                img_output = gr.Image(
                    label="Segmentation Result",
                    height=512,
                )
                img_info = gr.Textbox(label="Info", interactive=False)

        # -- Hidden states --
        points_state = gr.State(ImagePrompts())
        current_img_state = gr.State(None)  # clean original image

        # -- Action buttons --
        with gr.Row():
            segment_btn = gr.Button(
                "Segment",
                variant="primary",
                size="lg",
            )
            undo_btn = gr.Button("Undo Last", variant="secondary")
            clear_btn = gr.Button("Clear All", variant="stop")

        points_display = gr.Textbox(
            label="Current Prompts",
            interactive=False,
            lines=2,
            placeholder="No prompts yet. Click on the image above.",
        )

        # ---- Image tab helpers ----

        # ---- Image tab events ----

        # NOTE: the upload/reset listener must use `.input()`, not `.change()`.
        # `change` also fires for *programmatic* updates, and the click
        # handler writes the annotated image back into `img_input`, so a
        # `change` listener here would wipe the prompt state on every click.
        img_input.input(
            fn=_on_image_upload,
            inputs=[img_input, prompt_mode],
            outputs=[current_img_state, points_state, click_instruction],
        )

        prompt_mode.change(
            fn=_render_mode_instruction,
            inputs=[prompt_mode],
            outputs=[click_instruction],
        )

        img_input.select(
            fn=_on_image_click,
            inputs=[current_img_state, points_state, prompt_mode],
            outputs=[img_input, points_state, points_display],
        )

        undo_btn.click(
            fn=undo_last_prompt,
            inputs=[current_img_state, points_state, prompt_mode],
            outputs=[img_input, points_state, points_display],
        )

        clear_btn.click(
            fn=clear_prompts,
            inputs=[current_img_state],
            outputs=[img_input, points_state, points_display],
        )

        segment_btn.click(
            fn=_segment,
            inputs=[current_img_state, points_state, model_size, multimask_toggle],
            outputs=[img_output, img_info],
        )


def _build_auto_tab(model_size: gr.Dropdown) -> None:
    """Build the Auto Segment tab and wire its events."""
    with gr.Tab("Auto Segment"):
        gr.Markdown("Upload an image to automatically segment **every object**: no prompts needed.")
        with gr.Row():
            with gr.Column(scale=1):
                auto_img_input = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=480,
                )
                with gr.Row():
                    auto_min_area = gr.Slider(
                        minimum=0,
                        maximum=50_000,
                        step=100,
                        value=0,
                        label="Min mask area (px)",
                    )
                    auto_min_iou = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        step=0.05,
                        value=0.7,
                        label="Min predicted IoU",
                    )
                auto_btn = gr.Button("Auto Segment", variant="primary")

            with gr.Column(scale=1):
                auto_img_output = gr.Image(label="All Masks", height=480)
                auto_info = gr.Textbox(label="Info", interactive=False)

        auto_btn.click(
            fn=_auto_segment_image,
            inputs=[auto_img_input, model_size, auto_min_area, auto_min_iou],
            outputs=[auto_img_output, auto_info],
        )


def _build_concept_tab() -> None:
    """Build the SAM 3 Concept Segmentation tab and wire its events."""
    with gr.Tab("Concept Segment (SAM 3)"):
        gr.Markdown(
            "Describe what you want in words and SAM 3 segments **every** "
            "matching instance. Requires the optional extra: "
            "`uv sync --extra sam3`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                concept_img_input = gr.Image(
                    label="Input Image",
                    type="numpy",
                    height=480,
                )
                concept_text = gr.Textbox(
                    label="Concept",
                    placeholder="e.g. a player in white",
                    info="A short noun phrase. SAM 3 finds all matching instances.",
                )
                concept_min_score = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.5,
                    label="Min score",
                )
                concept_btn = gr.Button("Segment Concept", variant="primary")

            with gr.Column(scale=1):
                concept_img_output = gr.Image(label="Instances", height=480)
                concept_info = gr.Textbox(label="Info", interactive=False)

        concept_btn.click(
            fn=_concept_segment_image,
            inputs=[concept_img_input, concept_text, concept_min_score],
            outputs=[concept_img_output, concept_info],
        )


def _build_video_tab(model_size: gr.Dropdown) -> None:
    """Build the Video Tracking tab and wire its events."""
    with gr.Tab("Video Tracking"):
        gr.Markdown(
            "1. Upload a video. 2. Click on frames to mark objects. "
            "3. Run tracking to get an overlay video.  \n"
            "> **Tip:** Enable *Bidirectional* if your prompts are on "
            "a mid-video frame so tracking runs both forward *and* backward."
        )

        # -- State --
        frame_files_state = gr.State([])
        frames_dir_state = gr.State(None)
        num_frames_state = gr.State(0)
        video_prompts_state = gr.State([])
        clean_frame_state = gr.State(None)
        vid_box_state = gr.State([])

        # -- Step 1: Upload & extract --
        with gr.Group():
            gr.HTML('<div class="step-header">1. Load Video</div>')
            with gr.Row():
                with gr.Column(scale=2):
                    video_input = gr.Video(label="Upload Video")
                with gr.Column(scale=1):
                    extract_every_n = gr.Slider(
                        minimum=1,
                        maximum=30,
                        step=1,
                        value=1,
                        label="Extract every N-th frame",
                        info="Use higher values for long videos to speed up extraction.",
                    )
                    extract_max = gr.Slider(
                        minimum=0,
                        maximum=500,
                        step=10,
                        value=200,
                        label="Max frames (0 = all)",
                    )
                    extract_btn = gr.Button(
                        "Extract Frames",
                        variant="primary",
                    )
            extract_info = gr.Textbox(label="Status", interactive=False)

        # -- Step 2: Prompt placement --
        with gr.Group():
            gr.HTML('<div class="step-header">2. Add Prompts: click on the frame</div>')

            with gr.Row():
                # Left column: frame navigation + click controls
                with gr.Column(scale=1):
                    frame_slider = gr.Slider(
                        minimum=0,
                        maximum=1,
                        step=1,
                        value=0,
                        label="Frame",
                        interactive=True,
                    )
                    frame_counter_html = gr.HTML('<div class="frame-counter">Frame 0 / 0</div>')
                    with gr.Row():
                        vid_prompt_obj_id = gr.Number(
                            label="Object ID",
                            value=1,
                            precision=0,
                            info="Each unique ID gets its own colour.",
                        )
                        vid_prompt_mode = gr.Radio(
                            choices=MODE_CHOICES,
                            value=MODE_FOREGROUND,
                            label="Click Mode",
                        )
                    vid_click_instruction = gr.HTML(_render_mode_instruction(MODE_FOREGROUND))

                    # SAM 3 concept prompt: describe an object in words.
                    with gr.Row():
                        vid_text_prompt = gr.Textbox(
                            label="Concept (SAM 3)",
                            placeholder="e.g. person",
                            info="Adds a text prompt; tracking then uses SAM 3.",
                            scale=3,
                        )
                        vid_text_btn = gr.Button("Add Concept", size="sm", scale=1)

                # Right column: prompt list + actions
                with gr.Column(scale=1):
                    vid_prompts_display = gr.HTML(_NO_PROMPTS_HTML)
                    with gr.Row():
                        vid_undo_btn = gr.Button(
                            "Undo Last",
                            variant="secondary",
                            size="sm",
                        )
                        vid_clear_btn = gr.Button(
                            "Clear All Prompts",
                            variant="stop",
                            size="sm",
                        )

            frame_preview = gr.Image(
                label="Frame Preview: click to add prompts",
                height=500,
                interactive=False,
            )

        # -- Step 3: Track --
        with gr.Group():
            gr.HTML('<div class="step-header">3. Track & Render</div>')
            with gr.Row():
                with gr.Column(scale=1):
                    track_bidirectional = gr.Checkbox(
                        label="Bidirectional propagation",
                        value=False,
                        info=(
                            "Track both forward and backward from the "
                            "prompt frame. Enable this when your prompts "
                            "are in the middle of the video."
                        ),
                    )
                    with gr.Row():
                        track_fps = gr.Slider(
                            minimum=1,
                            maximum=60,
                            step=1,
                            value=24,
                            label="Output FPS",
                        )
                        track_alpha = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            step=0.05,
                            value=0.5,
                            label="Overlay Alpha",
                        )
                    track_btn = gr.Button(
                        "Track & Render Video",
                        variant="primary",
                        size="lg",
                    )
                    track_info = gr.Textbox(label="Status", interactive=False)
                with gr.Column(scale=1):
                    track_output = gr.Video(label="Result Video")

        # ---- Video tab events ----

        extract_outputs = [
            frame_files_state,
            extract_info,
            frames_dir_state,
            num_frames_state,
            frame_slider,
            frame_preview,
            clean_frame_state,
            video_prompts_state,
            vid_box_state,
            frame_counter_html,
            vid_prompts_display,
        ]
        # Keep the wiring honest: the outputs must match VideoTabState order.
        assert len(extract_outputs) == len(VideoTabState._fields)
        extract_btn.click(
            fn=_on_extract,
            inputs=[video_input, extract_every_n, extract_max],
            outputs=extract_outputs,
        )

        # The slider redraws the preview, so it must use `.input()` (user
        # only) rather than `.change()`, which also fires on programmatic
        # updates and would re-enter this handler.
        frame_slider.input(
            fn=_on_slider_change,
            inputs=[
                frame_files_state,
                frame_slider,
                video_prompts_state,
                vid_box_state,
                num_frames_state,
            ],
            outputs=[frame_preview, clean_frame_state, frame_counter_html],
        )

        vid_prompt_mode.change(
            fn=_render_mode_instruction,
            inputs=[vid_prompt_mode],
            outputs=[vid_click_instruction],
        )

        frame_preview.select(
            fn=_on_frame_click,
            inputs=[
                clean_frame_state,
                video_prompts_state,
                vid_box_state,
                frame_slider,
                vid_prompt_obj_id,
                vid_prompt_mode,
            ],
            outputs=[
                frame_preview,
                video_prompts_state,
                vid_box_state,
                vid_prompts_display,
            ],
        )

        vid_undo_btn.click(
            fn=_vid_undo,
            inputs=[
                clean_frame_state,
                video_prompts_state,
                vid_box_state,
                frame_slider,
                vid_prompt_mode,
            ],
            outputs=[
                frame_preview,
                video_prompts_state,
                vid_box_state,
                vid_prompts_display,
            ],
        )

        vid_clear_btn.click(
            fn=_vid_clear,
            inputs=[clean_frame_state],
            outputs=[
                frame_preview,
                video_prompts_state,
                vid_box_state,
                vid_prompts_display,
            ],
        )

        vid_text_btn.click(
            fn=_add_text_prompt,
            inputs=[
                clean_frame_state,
                video_prompts_state,
                vid_box_state,
                frame_slider,
                vid_text_prompt,
            ],
            outputs=[
                frame_preview,
                video_prompts_state,
                vid_box_state,
                vid_prompts_display,
            ],
        )

        track_btn.click(
            fn=_track_and_render,
            inputs=[
                frames_dir_state,
                video_prompts_state,
                model_size,
                track_fps,
                track_alpha,
                track_bidirectional,
            ],
            outputs=[track_output, track_info],
        )


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks app.

    Returns:
        The assembled :class:`gradio.Blocks` application.
    """
    blocks_kwargs: dict[str, Any] = {"title": "lazysammy Demo"}
    if _gradio_major_version() < 6:
        blocks_kwargs.update(_style_kwargs())

    with gr.Blocks(**blocks_kwargs) as app:  # type: ignore[arg-type]
        gr.Markdown(
            "# lazysammy Demo\n"
            "Interactive segmentation & video tracking powered by "
            "[SAM 2](https://github.com/facebookresearch/sam2) and "
            "[SAM 3](https://github.com/facebookresearch/sam3) through "
            "[lazysammy](https://github.com/federicopozzi33/easier-sam2).\n\n"
            "**Tip:** start with `tiny` or `small` for fast iteration, then "
            "switch to `large` for best quality. The Concept tab needs the "
            "optional `sam3` extra."
        )

        model_size = gr.Dropdown(
            choices=["tiny", "small", "base_plus", "large"],
            value="small",
            label="Model Size",
            info="Larger = more accurate but slower and heavier.",
        )

        _build_image_tab(model_size)
        _build_auto_tab(model_size)
        _build_concept_tab()
        _build_video_tab(model_size)

    return app


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    demo = build_app()
    # Bind to localhost by default. Set LAZYSAMMY_DEMO_HOST=0.0.0.0 (and pass
    # --share or a tunnel) if you deliberately want to expose the app on your
    # network: a segmentation demo loads multi-GB models on demand.
    host = os.environ.get("LAZYSAMMY_DEMO_HOST", "127.0.0.1")
    port = int(os.environ.get("LAZYSAMMY_DEMO_PORT", "7860"))

    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "server_port": port,
        "share": False,
    }
    if _gradio_major_version() >= 6:
        launch_kwargs.update(_style_kwargs())

    demo.launch(**launch_kwargs)  # type: ignore[arg-type]
