"""Video Tracking tab: frame extraction, prompt placement, and rendering.

Owns the video-tab state model and event handlers so ``demo/app.py`` only wires
Gradio components to them.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any, NamedTuple
from uuid import uuid4

import gradio as gr
import numpy as np

from lazysammy import extract_frames, load_image, natural_sort_key, save_video_overlay_mp4
from lazysammy_demo.drawing import color_for_obj, draw_box_corners_on_image
from lazysammy_demo.modes import (
    MODE_CHOICES,
    MODE_FOREGROUND,
    is_box_mode,
    label_for_mode,
    render_mode_instruction,
)
from lazysammy_demo.prompts import BoxPrompt, PointPrompt, TextPrompt, VideoPrompt
from lazysammy_demo.runtime import (
    DEMO_ERRORS,
    INFERENCE_LOCK,
    OUTPUT_DIR,
    get_model,
    get_sam3_model,
)

logger = logging.getLogger(__name__)

_VIDEO_FRAME_EXTS = {".jpg", ".jpeg", ".png"}

# Shown wherever the prompt list is still empty.
NO_PROMPTS_HTML = (
    '<div class="prompt-summary"><em>No prompts yet. Click on the frame to start.</em></div>'
)


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


def _frame_counter(idx: int, num_frames: int) -> str:
    """Render the frame counter HTML for *idx* of *num_frames*."""
    return f'<div class="frame-counter">Frame {idx} / {max(0, num_frames - 1)}</div>'


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
        prompts_html=NO_PROMPTS_HTML,
    )


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
    except DEMO_ERRORS as exc:
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
        prompts_html=NO_PROMPTS_HTML,
    )


def _get_frame_image(frame_files: list[str], frame_idx: int) -> np.ndarray | None:
    """Load frame *frame_idx* from *frame_files* as an RGB array."""
    if not frame_files or frame_idx < 0 or frame_idx >= len(frame_files):
        return None
    try:
        return load_image(frame_files[frame_idx])
    except (FileNotFoundError, ValueError) as exc:
        logger.warning("Could not load frame %s: %s", frame_idx, exc)
        return None


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
                color = color_for_obj(prompt.obj_id)
            else:
                color = (255, 255, 255)
            prompt.draw(out, color)
    return out


def _render_frame(
    clean_frame: np.ndarray | None,
    prompts: list[VideoPrompt],
    vid_box: list[Any],
    frame_idx: int,
) -> np.ndarray | None:
    """Draw committed prompts and any pending box corners for one frame."""
    annotated = _draw_video_prompts(clean_frame, prompts, frame_idx)
    if vid_box and annotated is not None:
        annotated = draw_box_corners_on_image(annotated, vid_box)
    return annotated


def _format_prompts_html(prompts: list[VideoPrompt]) -> str:
    """Render prompts as styled HTML for the video tab summary."""
    if not prompts:
        return NO_PROMPTS_HTML

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
            color = "#{:02x}{:02x}{:02x}".format(*color_for_obj(p.obj_id))
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
        with INFERENCE_LOCK:
            if uses_concepts:
                session = get_sam3_model().video(frames_dir, offload_video_to_cpu=True)
            else:
                session = get_model(model_size).video(frames_dir, offload_video_to_cpu=True)
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
            out_path = OUTPUT_DIR / f"tracking_{uuid4().hex}.mp4"
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
    except DEMO_ERRORS as exc:
        logger.exception("Tracking failed")
        return None, f"Error: Tracking failed: {exc}"

    backend = "SAM 3" if uses_concepts else "SAM 2"
    info = (
        f"{backend}: tracked {len(results.object_ids)} object(s) "
        f"across {len(results)} frames ({direction})"
    )
    return str(out_path), info


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------


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

    if is_box_mode(mode):
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
        prompts = _append_point(prompts, frame_idx, obj_id, x, y, label_for_mode(mode))

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
    if is_box_mode(mode) and vid_box:
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
    return clean_frame, [], [], NO_PROMPTS_HTML


def build_video_tab(model_size: gr.Dropdown) -> None:
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
                    vid_click_instruction = gr.HTML(render_mode_instruction(MODE_FOREGROUND))

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
                    vid_prompts_display = gr.HTML(NO_PROMPTS_HTML)
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
            fn=render_mode_instruction,
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
