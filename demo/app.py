"""Gradio demo for lazysammy.

Launch with::

    uv run --extra demo python demo/app.py

Or::

    uv run --extra demo gradio demo/app.py

The demo is split into a small package (:mod:`lazysammy_demo`) so this file
stays a readable wiring layer: drawing, the typed prompt model, click modes,
the model runtime, and the video tab live there. The names below are
re-exported so the demo's public surface (and its tests) stay stable.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import replace
from typing import Any, NamedTuple, Protocol

import cv2
import gradio as gr
import numpy as np
from lazysammy_demo.drawing import (
    color_for_obj as _color_for_obj,
)
from lazysammy_demo.drawing import (
    draw_box_corners_on_image as _draw_box_corners_on_image,
)
from lazysammy_demo.drawing import (
    draw_concept_boxes as _draw_concept_boxes,
)
from lazysammy_demo.drawing import (
    draw_label as _draw_label,
)
from lazysammy_demo.drawing import (
    draw_points_on_image as _draw_points_on_image,
)
from lazysammy_demo.modes import (
    MODE_BACKGROUND,
    MODE_BOX,
    MODE_CHOICES,
    MODE_FOREGROUND,
)
from lazysammy_demo.modes import (
    is_box_mode as _is_box_mode,
)
from lazysammy_demo.modes import (
    label_for_mode as _label_for_mode,
)
from lazysammy_demo.modes import (
    render_mode_instruction as _render_mode_instruction,
)
from lazysammy_demo.prompts import (
    BoxPrompt,
    ImagePrompts,
    PointPrompt,
    TextPrompt,
    VideoPrompt,
)
from lazysammy_demo.runtime import (
    DEMO_ERRORS as _DEMO_ERRORS,
)
from lazysammy_demo.runtime import (
    INFERENCE_LOCK as _INFERENCE_LOCK,
)
from lazysammy_demo.runtime import (
    MODEL_LOAD_LOCK as _MODEL_LOAD_LOCK,
)
from lazysammy_demo.runtime import (
    get_model as _get_model,
)
from lazysammy_demo.runtime import (
    get_sam3_model as _get_sam3_model,
)
from lazysammy_demo.video_tab import (
    NO_PROMPTS_HTML as _NO_PROMPTS_HTML,
)
from lazysammy_demo.video_tab import (
    VideoTabState,
    _add_text_prompt,
    _empty_video_state,
    _extract_video_frames,
    _format_prompts_html,
    _frame_counter,
    _get_frame_image,
    _on_extract,
    _on_frame_click,
    _on_slider_change,
    _render_frame,
    _track_and_render,
    _vid_clear,
    _vid_undo,
)
from lazysammy_demo.video_tab import (
    build_video_tab as _build_video_tab,
)

from lazysammy import (
    ImagePrediction,
    draw_masks_on_image,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# The demo's public surface. The underscore-prefixed names are re-exported from
# the `lazysammy_demo` package so the demo's tests and any external wiring keep
# a stable import path; listing them here marks them as intentional re-exports.
__all__ = [
    "MODE_BACKGROUND",
    "MODE_BOX",
    "MODE_CHOICES",
    "MODE_FOREGROUND",
    "_DEMO_ERRORS",
    "_INFERENCE_LOCK",
    "_MODEL_LOAD_LOCK",
    "_NO_PROMPTS_HTML",
    "BoxPrompt",
    "ImagePrompts",
    "ImageTabState",
    "PointPrompt",
    "TextPrompt",
    "VideoPrompt",
    "VideoTabState",
    "_add_text_prompt",
    "_build_video_tab",
    "_color_for_obj",
    "_draw_box_corners_on_image",
    "_draw_concept_boxes",
    "_draw_label",
    "_draw_points_on_image",
    "_empty_video_state",
    "_extract_video_frames",
    "_format_prompts_html",
    "_frame_counter",
    "_get_frame_image",
    "_get_model",
    "_get_sam3_model",
    "_is_box_mode",
    "_label_for_mode",
    "_on_extract",
    "_on_frame_click",
    "_on_slider_change",
    "_render_frame",
    "_render_mode_instruction",
    "_track_and_render",
    "_vid_clear",
    "_vid_undo",
    "apply_image_click",
    "build_app",
    "clear_prompts",
    "select_segmentation_mode",
    "undo_last_prompt",
]


class ImageTabState(NamedTuple):
    """Outputs of the image-tab click/undo/clear handlers, in Gradio order."""

    image: np.ndarray | None
    prompts: ImagePrompts
    display: str


class MaskResult(Protocol):
    """A mask result the demo can overlay: auto-mask or concept prediction."""

    masks: list[Any]

    def numpy(self) -> np.ndarray: ...


# =========================================================================
# Tab 1: Image Segmentation
# =========================================================================


def _run_image_segmentation(
    image: np.ndarray,
    model_size: str,
    predict: Callable[[Any], ImagePrediction],
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
    result: MaskResult,
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


def _gradio_major_version() -> int:
    """Return the installed Gradio major version (0 when undetectable)."""
    raw = getattr(gr, "__version__", "") or ""
    try:
        return int(str(raw).split(".")[0])
    except (ValueError, IndexError):
        return 0


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
