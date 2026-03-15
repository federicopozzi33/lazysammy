"""Gradio demo for easier-sam2.

Launch with::

    uv run --extra demo python demo/app.py

Or::

    uv run --extra demo gradio demo/app.py
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from easier_sam2 import SAM2, draw_masks_on_image, extract_frames, save_video_overlay_mp4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global model cache (lazy — loaded once on first use per model size)
# ---------------------------------------------------------------------------

_MODEL_CACHE: dict[str, SAM2] = {}


def _get_model(model_size: str) -> SAM2:
    if model_size not in _MODEL_CACHE:
        logger.info("Loading SAM2 model: %s", model_size)
        _MODEL_CACHE[model_size] = SAM2(model_size)
    return _MODEL_CACHE[model_size]


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

_COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 200, 0),
    (255, 0, 200), (0, 200, 200), (180, 0, 0), (0, 160, 0),
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
        tag = "+" if label == 1 else "-"
        cv2.putText(out, tag, (x + radius + 2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, tag, (x + radius + 2, y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
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
        cv2.putText(out, "corner 1", (x1 + 10, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)
    if len(corners) >= 2:
        x2, y2 = int(corners[1][0]), int(corners[1][1])
        bx1, by1 = min(x1, x2), min(y1, y2)
        bx2, by2 = max(x1, x2), max(y1, y2)
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (0, 200, 255), 2)
    return out


# =========================================================================
# Tab 1 — Image Segmentation
# =========================================================================


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

    sam = _get_model(model_size)
    pred = sam.segment(image, points=coords, labels=labels, multimask_output=multimask)
    best = pred.best_mask
    overlay = draw_masks_on_image(image, best.mask[np.newaxis], alpha=0.5)
    overlay = _draw_points_on_image(overlay, points)
    info = f"Best mask — IoU: {best.score:.3f} | area: {best.area:,} px"
    return overlay, info


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

    sam = _get_model(model_size)
    pred = sam.segment_box(image, x1, y1, x2, y2)
    best = pred.best_mask
    overlay = draw_masks_on_image(image, best.mask[np.newaxis], alpha=0.5)
    cv2.rectangle(overlay, (int(x1), int(y1)), (int(x2), int(y2)), (0, 200, 255), 2)
    info = f"Best mask — IoU: {best.score:.3f} | area: {best.area:,} px"
    return overlay, info


# =========================================================================
# Tab 2 — Auto Segment Everything
# =========================================================================


def _auto_segment_image(
    image: np.ndarray | None,
    model_size: str,
    min_area: int,
    min_iou: float,
) -> tuple[np.ndarray | None, str]:
    if image is None:
        return None, "Upload an image first."

    sam = _get_model(model_size)
    result = sam.auto_segment(image)

    if min_area > 0:
        result = result.filter_by_area(min_area=min_area)
    if min_iou > 0:
        result = result.filter_by_iou(min_iou=min_iou)

    if not result.masks:
        return image, "No masks found with current filters."

    all_masks = result.numpy()
    overlay = draw_masks_on_image(image, all_masks, alpha=0.4)
    info = f"Found {len(result.masks)} masks"
    return overlay, info


# =========================================================================
# Tab 3 — Video Object Tracking
# =========================================================================


def _extract_video_frames(
    video_path: str | None,
    every_n: int,
    max_frames: int,
) -> tuple[list[str], str, str | None, int]:
    if not video_path:
        return [], "Upload a video first.", None, 0

    frames_dir = extract_frames(video_path, every_n=every_n, max_frames=max_frames or None)
    frame_files = sorted(
        [str(f) for f in Path(frames_dir).iterdir()
         if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    )
    if not frame_files:
        return [], "No frames extracted.", None, 0

    info = f"Extracted {len(frame_files)} frames to {frames_dir}"
    return frame_files, info, str(frames_dir), len(frame_files)


def _get_frame_image(frame_files: list[str], frame_idx: int) -> np.ndarray | None:
    if not frame_files or frame_idx < 0 or frame_idx >= len(frame_files):
        return None
    img = cv2.imread(frame_files[frame_idx])
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if img is not None else None


def _draw_video_prompts(
    frame: np.ndarray | None,
    prompts: list[dict[str, Any]],
    frame_idx: int,
) -> np.ndarray | None:
    """Draw all prompts for a given frame."""
    if frame is None:
        return None
    out = frame.copy()
    for prompt in prompts:
        if prompt["frame_idx"] != frame_idx:
            continue
        color = _color_for_obj(prompt["obj_id"])
        oid = prompt["obj_id"]
        if prompt["type"] == "point":
            for pt, lab in zip(prompt["points"], prompt["labels"]):
                c = color if lab == 1 else (128, 128, 128)
                cv2.circle(out, (int(pt[0]), int(pt[1])), 7, c, -1)
                cv2.circle(out, (int(pt[0]), int(pt[1])), 7, (255, 255, 255), 2)
                tag = f"obj{oid}+" if lab == 1 else f"obj{oid}-"
                cv2.putText(out, tag, (int(pt[0]) + 10, int(pt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(out, tag, (int(pt[0]) + 10, int(pt[1]) - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        elif prompt["type"] == "box":
            bx = prompt["box"]
            cv2.rectangle(out, (int(bx[0]), int(bx[1])), (int(bx[2]), int(bx[3])), color, 2)
            cv2.putText(out, f"obj{oid}", (int(bx[0]) + 4, int(bx[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(out, f"obj{oid}", (int(bx[0]) + 4, int(bx[1]) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def _format_prompts(prompts: list[dict]) -> str:
    lines = []
    for i, p in enumerate(prompts):
        if p["type"] == "point":
            pts = ", ".join(
                f"({pt[0]:.0f},{pt[1]:.0f}){'fg' if lb == 1 else 'bg'}"
                for pt, lb in zip(p["points"], p["labels"])
            )
            lines.append(f"[{i}] Frame {p['frame_idx']} | Obj {p['obj_id']} | points: {pts}")
        elif p["type"] == "box":
            b = p["box"]
            lines.append(
                f"[{i}] Frame {p['frame_idx']} | Obj {p['obj_id']} | "
                f"box: ({b[0]:.0f},{b[1]:.0f})-({b[2]:.0f},{b[3]:.0f})"
            )
    return "\n".join(lines) if lines else "(no prompts yet)"


def _format_prompts_html(prompts: list[dict]) -> str:
    """Render prompts as styled HTML for the video tab summary."""
    if not prompts:
        return (
            '<div class="prompt-summary"><em>No prompts yet — '
            'click on the frame below to start.</em></div>'
        )

    # Gather stats
    obj_ids = sorted({p["obj_id"] for p in prompts})
    frames_used = sorted({p["frame_idx"] for p in prompts})
    n_points = sum(
        len(p["points"]) for p in prompts if p["type"] == "point"
    )
    n_boxes = sum(1 for p in prompts if p["type"] == "box")

    header = (
        f"<b>{len(obj_ids)} object(s)</b> &middot; "
        f"{n_points} point(s) &middot; {n_boxes} box(es) &middot; "
        f"on frame(s) {', '.join(str(f) for f in frames_used)}"
    )

    rows = []
    for i, p in enumerate(prompts):
        color = "#{:02x}{:02x}{:02x}".format(*_color_for_obj(p["obj_id"]))
        badge = (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'border-radius:50%;background:{color};margin-right:4px"></span>'
        )
        if p["type"] == "point":
            pts = ", ".join(
                f"({pt[0]:.0f},{pt[1]:.0f}) "
                f"{'<span style=\"color:green\">fg</span>' if lb == 1 else '<span style=\"color:red\">bg</span>'}"
                for pt, lb in zip(p["points"], p["labels"])
            )
            rows.append(
                f"{badge}Obj {p['obj_id']} · Frame {p['frame_idx']} · {pts}"
            )
        elif p["type"] == "box":
            b = p["box"]
            rows.append(
                f"{badge}Obj {p['obj_id']} · Frame {p['frame_idx']} · "
                f"box ({b[0]:.0f},{b[1]:.0f})→({b[2]:.0f},{b[3]:.0f})"
            )

    body = "<br>".join(rows)
    return f'<div class="prompt-summary">{header}<hr style="margin:4px 0">{body}</div>'


def _track_and_render(
    video_path: str | None,
    frames_dir_str: str | None,
    prompts_state: list[dict[str, Any]],
    model_size: str,
    every_n: int,
    max_frames: int,
    fps: float,
    alpha: float,
    bidirectional: bool,
) -> tuple[str | None, str]:
    if not frames_dir_str:
        return None, "Extract frames first."
    if not prompts_state:
        return None, "Add at least one prompt before tracking."

    frames_dir = Path(frames_dir_str)
    sam = _get_model(model_size)
    session = sam.video(frames_dir, offload_video_to_cpu=True)

    for prompt in prompts_state:
        ptype = prompt["type"]
        if ptype == "point":
            session.add_points(
                frame_idx=prompt["frame_idx"],
                obj_id=prompt["obj_id"],
                points=prompt["points"],
                labels=prompt["labels"],
            )
        elif ptype == "box":
            session.add_box(
                frame_idx=prompt["frame_idx"],
                obj_id=prompt["obj_id"],
                box=prompt["box"],
            )

    if bidirectional:
        results = session.propagate_bidirectional()
        direction = "bidirectional"
    else:
        results = session.propagate()
        direction = "forward"

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    save_video_overlay_mp4(
        frames_dir, results, out_path,
        fps=fps, alpha=alpha, show_ids=True, show_frame_number=True,
    )

    info = (
        f"Tracked {len(results.object_ids)} object(s) across "
        f"{len(results)} frames ({direction})"
    )
    return out_path, info


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


def _render_mode_instruction(mode: str) -> str:
    if "Foreground" in mode:
        return (
            '<div class="click-instruction click-fg">'
            '🟢 Click on the image to add <b>foreground</b> points '
            '(objects to include)</div>'
        )
    elif "Background" in mode:
        return (
            '<div class="click-instruction click-bg">'
            '🔴 Click on the image to add <b>background</b> points '
            '(areas to exclude)</div>'
        )
    else:
        return (
            '<div class="click-instruction click-box">'
            '📦 Click <b>two corners</b> on the image to define a '
            'bounding box</div>'
        )


def build_app() -> gr.Blocks:
    """Construct and return the Gradio Blocks app."""

    with gr.Blocks(title="easier-sam2 Demo", theme=gr.themes.Soft(), css=_CSS) as app:
        gr.Markdown(
            "# 🎯 easier-sam2 Demo\n"
            "Interactive segmentation & video tracking powered by "
            "[SAM 2](https://github.com/facebookresearch/sam2) via "
            "[easier-sam2](https://github.com/fpozzi/easier-sam2)."
        )

        model_size = gr.Dropdown(
            choices=["tiny", "small", "base_plus", "large"],
            value="large",
            label="Model Size",
            info="Larger = more accurate but slower.",
        )

        # ==============================================================
        # Tab 1: Image Segmentation (click-to-prompt)
        # ==============================================================

        with gr.Tab("🖼️ Image Segmentation"):
            gr.Markdown(
                "Upload an image, then **click directly on it** to place "
                "prompts. Choose a mode below before clicking."
            )

            # -- Prompt mode selection --
            with gr.Row():
                prompt_mode = gr.Radio(
                    choices=[
                        "➕ Foreground point",
                        "➖ Background point",
                        "📦 Box (2 clicks)",
                    ],
                    value="➕ Foreground point",
                    label="Click Mode",
                    info="Select what happens when you click on the image.",
                )
                multimask_toggle = gr.Checkbox(
                    label="Multi-mask output", value=True,
                )

            click_instruction = gr.HTML(
                _render_mode_instruction("➕ Foreground point")
            )

            # -- Images side-by-side --
            with gr.Row():
                with gr.Column(scale=1):
                    img_input = gr.Image(
                        label="Input Image — click to add prompts",
                        type="numpy",
                        height=512,
                    )
                with gr.Column(scale=1):
                    img_output = gr.Image(
                        label="Segmentation Result", height=512,
                    )
                    img_info = gr.Textbox(label="Info", interactive=False)

            # -- Hidden states --
            points_state = gr.State([])       # [[x, y, label], ...]
            box_state = gr.State([])           # [[x,y], [x,y]]
            current_img_state = gr.State(None) # clean original image

            # -- Action buttons --
            with gr.Row():
                segment_btn = gr.Button(
                    "🔍 Segment", variant="primary", size="lg",
                )
                undo_btn = gr.Button("↩ Undo Last", variant="secondary")
                clear_btn = gr.Button("🗑 Clear All", variant="stop")

            points_display = gr.Textbox(
                label="Current Prompts",
                interactive=False,
                lines=2,
                placeholder="No prompts yet — click on the image above",
            )

            # ---- Image tab helpers ----

            def _format_img_prompts(pts, box):
                parts = []
                for p in pts:
                    kind = "fg" if int(p[2]) == 1 else "bg"
                    parts.append(f"({p[0]:.0f}, {p[1]:.0f}) {kind}")
                if len(box) == 1:
                    parts.append(
                        f"box corner 1: ({box[0][0]:.0f}, {box[0][1]:.0f}) "
                        "— click 2nd corner"
                    )
                if len(box) >= 2:
                    parts.append(
                        f"box: ({box[0][0]:.0f},{box[0][1]:.0f})"
                        f"→({box[1][0]:.0f},{box[1][1]:.0f})"
                    )
                return " | ".join(parts) if parts else ""

            def _redraw_input(original, pts, box):
                if original is None:
                    return None
                out = original.copy()
                if pts:
                    out = _draw_points_on_image(out, pts)
                if box:
                    out = _draw_box_corners_on_image(out, box)
                return out

            # ---- Image tab events ----

            def _on_image_upload(img):
                return img, [], [], _render_mode_instruction("➕ Foreground point")

            img_input.change(
                fn=_on_image_upload,
                inputs=[img_input],
                outputs=[current_img_state, points_state, box_state, click_instruction],
            )

            prompt_mode.change(
                fn=_render_mode_instruction,
                inputs=[prompt_mode],
                outputs=[click_instruction],
            )

            def _on_image_click(original, points, box_corners, mode, evt: gr.SelectData):
                if original is None:
                    return original, points, box_corners, ""
                x, y = evt.index[0], evt.index[1]

                if "Box" in mode:
                    box_corners = box_corners + [[x, y]]
                    if len(box_corners) > 2:
                        box_corners = [[x, y]]
                else:
                    label = 1 if "Foreground" in mode else 0
                    points = points + [[x, y, label]]

                display = _format_img_prompts(points, box_corners)
                annotated = _redraw_input(original, points, box_corners)
                return annotated, points, box_corners, display

            img_input.select(
                fn=_on_image_click,
                inputs=[current_img_state, points_state, box_state, prompt_mode],
                outputs=[img_input, points_state, box_state, points_display],
            )

            def _undo(original, points, box_corners, mode):
                if "Box" in mode:
                    box_corners = box_corners[:-1] if box_corners else []
                else:
                    points = points[:-1] if points else []
                display = _format_img_prompts(points, box_corners)
                annotated = _redraw_input(original, points, box_corners)
                return annotated, points, box_corners, display

            undo_btn.click(
                fn=_undo,
                inputs=[current_img_state, points_state, box_state, prompt_mode],
                outputs=[img_input, points_state, box_state, points_display],
            )

            def _clear_all(original):
                out = original.copy() if original is not None else None
                return out, [], [], ""

            clear_btn.click(
                fn=_clear_all,
                inputs=[current_img_state],
                outputs=[img_input, points_state, box_state, points_display],
            )

            def _segment(original, points, box_corners, ms, multimask):
                if original is None:
                    return None, "Upload an image first."
                # Prefer box if two corners are set
                if len(box_corners) >= 2:
                    flat = [
                        box_corners[0][0], box_corners[0][1],
                        box_corners[1][0], box_corners[1][1],
                    ]
                    return _segment_with_box(original, flat, ms)
                if points:
                    return _segment_with_points(original, points, ms, multimask)
                return original, "Add some prompts first (click on the image)."

            segment_btn.click(
                fn=_segment,
                inputs=[
                    current_img_state, points_state, box_state,
                    model_size, multimask_toggle,
                ],
                outputs=[img_output, img_info],
            )

        # ==============================================================
        # Tab 2: Auto Segment Everything
        # ==============================================================

        with gr.Tab("✨ Auto Segment"):
            gr.Markdown(
                "Upload an image to automatically segment **every object** — "
                "no prompts needed."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    auto_img_input = gr.Image(
                        label="Input Image", type="numpy", height=480,
                    )
                    with gr.Row():
                        auto_min_area = gr.Slider(
                            minimum=0, maximum=50_000, step=100, value=0,
                            label="Min mask area (px)",
                        )
                        auto_min_iou = gr.Slider(
                            minimum=0.0, maximum=1.0, step=0.05, value=0.7,
                            label="Min predicted IoU",
                        )
                    auto_btn = gr.Button("✨ Auto Segment", variant="primary")

                with gr.Column(scale=1):
                    auto_img_output = gr.Image(label="All Masks", height=480)
                    auto_info = gr.Textbox(label="Info", interactive=False)

            auto_btn.click(
                fn=_auto_segment_image,
                inputs=[auto_img_input, model_size, auto_min_area, auto_min_iou],
                outputs=[auto_img_output, auto_info],
            )

        # ==============================================================
        # Tab 3: Video Object Tracking (click-to-prompt)
        # ==============================================================

        with gr.Tab("🎬 Video Tracking"):
            gr.Markdown(
                "> Upload a video → click on frames to mark objects → "
                "run tracking to get an overlay video.  \n"
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
                gr.HTML('<div class="step-header">① Load Video</div>')
                with gr.Row():
                    with gr.Column(scale=2):
                        video_input = gr.Video(label="Upload Video")
                    with gr.Column(scale=1):
                        extract_every_n = gr.Slider(
                            minimum=1, maximum=30, step=1, value=1,
                            label="Extract every N-th frame",
                            info="Use higher values for long videos to speed up extraction.",
                        )
                        extract_max = gr.Slider(
                            minimum=0, maximum=500, step=10, value=200,
                            label="Max frames (0 = all)",
                        )
                        extract_btn = gr.Button(
                            "📂 Extract Frames", variant="primary",
                        )
                extract_info = gr.Textbox(label="Status", interactive=False)

            # -- Step 2: Prompt placement --
            with gr.Group():
                gr.HTML('<div class="step-header">② Add Prompts — click on the frame</div>')

                with gr.Row():
                    # Left column: frame navigation + click controls
                    with gr.Column(scale=1):
                        frame_slider = gr.Slider(
                            minimum=0, maximum=1, step=1, value=0,
                            label="Frame", interactive=True,
                        )
                        frame_counter_html = gr.HTML(
                            '<div class="frame-counter">Frame 0 / 0</div>'
                        )
                        with gr.Row():
                            vid_prompt_obj_id = gr.Number(
                                label="Object ID", value=1, precision=0,
                                info="Give each object a unique ID (different ID = different colour).",
                            )
                            vid_prompt_mode = gr.Radio(
                                choices=[
                                    "➕ Foreground",
                                    "➖ Background",
                                    "📦 Box (2 clicks)",
                                ],
                                value="➕ Foreground",
                                label="Click Mode",
                            )
                        vid_click_instruction = gr.HTML(
                            _render_mode_instruction("➕ Foreground")
                        )

                    # Right column: prompt list + actions
                    with gr.Column(scale=1):
                        vid_prompts_display = gr.HTML(
                            '<div class="prompt-summary"><em>No prompts yet — '
                            'click on the frame below to start.</em></div>'
                        )
                        with gr.Row():
                            vid_undo_btn = gr.Button(
                                "↩ Undo Last", variant="secondary", size="sm",
                            )
                            vid_clear_btn = gr.Button(
                                "🗑 Clear All Prompts", variant="stop", size="sm",
                            )

                frame_preview = gr.Image(
                    label="Frame Preview — click to add prompts",
                    height=500,
                    interactive=False,
                )

            # -- Step 3: Track --
            with gr.Group():
                gr.HTML('<div class="step-header">③ Track & Render</div>')
                with gr.Row():
                    with gr.Column(scale=1):
                        track_bidirectional = gr.Checkbox(
                            label="↔ Bidirectional propagation",
                            value=False,
                            info=(
                                "Track both forward and backward from the "
                                "prompt frame. Enable this when your prompts "
                                "are in the middle of the video."
                            ),
                        )
                        with gr.Row():
                            track_fps = gr.Slider(
                                minimum=1, maximum=60, step=1, value=24,
                                label="Output FPS",
                            )
                            track_alpha = gr.Slider(
                                minimum=0.0, maximum=1.0, step=0.05, value=0.5,
                                label="Overlay Alpha",
                            )
                        track_btn = gr.Button(
                            "🚀 Track & Render Video",
                            variant="primary", size="lg",
                        )
                        track_info = gr.Textbox(label="Status", interactive=False)
                    with gr.Column(scale=1):
                        track_output = gr.Video(label="Result Video")

            # ---- Video tab events ----

            def _on_extract(video_path, every_n, max_frames):
                ff, info, fd, nf = _extract_video_frames(
                    video_path, every_n, int(max_frames) or 0,
                )
                first = _get_frame_image(ff, 0)
                slider_update = gr.Slider(maximum=max(0, nf - 1), value=0)
                counter = f'<div class="frame-counter">Frame 0 / {max(0, nf - 1)}</div>'
                empty_prompts = (
                    '<div class="prompt-summary"><em>No prompts yet — '
                    'click on the frame below to start.</em></div>'
                )
                return (
                    ff, info, fd, nf, slider_update, first, first,
                    [], [], counter, empty_prompts,
                )

            extract_btn.click(
                fn=_on_extract,
                inputs=[video_input, extract_every_n, extract_max],
                outputs=[
                    frame_files_state, extract_info, frames_dir_state,
                    num_frames_state, frame_slider, frame_preview,
                    clean_frame_state, video_prompts_state, vid_box_state,
                    frame_counter_html, vid_prompts_display,
                ],
            )

            def _on_slider_change(frame_files, idx, prompts, vid_box, num_frames):
                idx = int(idx)
                total = max(0, int(num_frames) - 1)
                clean = _get_frame_image(frame_files, idx)
                annotated = _draw_video_prompts(clean, prompts, idx)
                if vid_box and annotated is not None:
                    annotated = _draw_box_corners_on_image(annotated, vid_box)
                counter = f'<div class="frame-counter">Frame {idx} / {total}</div>'
                return annotated, clean, counter

            frame_slider.change(
                fn=_on_slider_change,
                inputs=[
                    frame_files_state, frame_slider,
                    video_prompts_state, vid_box_state,
                    num_frames_state,
                ],
                outputs=[frame_preview, clean_frame_state, frame_counter_html],
            )

            vid_prompt_mode.change(
                fn=_render_mode_instruction,
                inputs=[vid_prompt_mode],
                outputs=[vid_click_instruction],
            )

            def _on_frame_click(
                clean_frame, frame_files, prompts, vid_box,
                frame_slider_val, obj_id, mode,
                evt: gr.SelectData,
            ):
                if clean_frame is None:
                    return None, prompts, vid_box, _format_prompts_html(prompts)

                x, y = evt.index[0], evt.index[1]
                frame_idx = int(frame_slider_val)
                obj_id = int(obj_id)

                if "Box" in mode:
                    vid_box = vid_box + [[x, y]]
                    if len(vid_box) == 2:
                        bx1 = min(vid_box[0][0], vid_box[1][0])
                        by1 = min(vid_box[0][1], vid_box[1][1])
                        bx2 = max(vid_box[0][0], vid_box[1][0])
                        by2 = max(vid_box[0][1], vid_box[1][1])
                        prompts = prompts + [{
                            "type": "box",
                            "frame_idx": frame_idx,
                            "obj_id": obj_id,
                            "box": [bx1, by1, bx2, by2],
                        }]
                        vid_box = []
                    elif len(vid_box) > 2:
                        vid_box = [[x, y]]
                else:
                    label = 1 if "Foreground" in mode else 0
                    existing = None
                    for p in prompts:
                        if (p["type"] == "point"
                                and p["frame_idx"] == frame_idx
                                and p["obj_id"] == obj_id):
                            existing = p
                            break
                    if existing is not None:
                        prompts = [
                            {**p,
                             "points": p["points"] + [[x, y]],
                             "labels": p["labels"] + [label]}
                            if p is existing else p
                            for p in prompts
                        ]
                    else:
                        prompts = prompts + [{
                            "type": "point",
                            "frame_idx": frame_idx,
                            "obj_id": obj_id,
                            "points": [[x, y]],
                            "labels": [label],
                        }]

                display = _format_prompts_html(prompts)
                annotated = _draw_video_prompts(clean_frame, prompts, frame_idx)
                if vid_box:
                    annotated = _draw_box_corners_on_image(annotated, vid_box)
                return annotated, prompts, vid_box, display

            frame_preview.select(
                fn=_on_frame_click,
                inputs=[
                    clean_frame_state, frame_files_state,
                    video_prompts_state, vid_box_state,
                    frame_slider, vid_prompt_obj_id, vid_prompt_mode,
                ],
                outputs=[
                    frame_preview, video_prompts_state,
                    vid_box_state, vid_prompts_display,
                ],
            )

            def _vid_undo(clean_frame, prompts, vid_box, frame_slider_val, mode):
                frame_idx = int(frame_slider_val)
                if "Box" in mode and vid_box:
                    vid_box = vid_box[:-1]
                elif prompts:
                    last = prompts[-1]
                    if last["type"] == "point" and len(last["points"]) > 1:
                        prompts = prompts[:-1] + [{
                            **last,
                            "points": last["points"][:-1],
                            "labels": last["labels"][:-1],
                        }]
                    else:
                        prompts = prompts[:-1]
                display = _format_prompts_html(prompts)
                annotated = _draw_video_prompts(clean_frame, prompts, frame_idx)
                if vid_box:
                    annotated = _draw_box_corners_on_image(annotated, vid_box)
                return annotated, prompts, vid_box, display

            vid_undo_btn.click(
                fn=_vid_undo,
                inputs=[
                    clean_frame_state, video_prompts_state,
                    vid_box_state, frame_slider, vid_prompt_mode,
                ],
                outputs=[
                    frame_preview, video_prompts_state,
                    vid_box_state, vid_prompts_display,
                ],
            )

            def _vid_clear(clean_frame):
                empty = (
                    '<div class="prompt-summary"><em>No prompts yet — '
                    'click on the frame below to start.</em></div>'
                )
                return clean_frame, [], [], empty

            vid_clear_btn.click(
                fn=_vid_clear,
                inputs=[clean_frame_state],
                outputs=[
                    frame_preview, video_prompts_state,
                    vid_box_state, vid_prompts_display,
                ],
            )

            track_btn.click(
                fn=_track_and_render,
                inputs=[
                    video_input, frames_dir_state, video_prompts_state,
                    model_size, extract_every_n, extract_max,
                    track_fps, track_alpha, track_bidirectional,
                ],
                outputs=[track_output, track_info],
            )

    return app


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
