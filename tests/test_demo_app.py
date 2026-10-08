"""Tests for the Gradio demo (`demo/app.py`).

The demo regressed badly once already: every image click silently reset the
prompt state, so Segment always answered "Add some prompts first". The cause was
wiring, not model code -- ``gr.Image.change`` fires on *programmatic* updates
too, and the click handler writes the annotated image back into the same
component. That re-entered the upload handler, which cleared the prompt state.

These tests therefore cover three layers:

1. The pure prompt bookkeeping helpers (fast, no Gradio needed beyond import).
2. The built Gradio config, which is where the regression actually lived. These
   assertions are the ones that would have caught the bug.
3. A real client/server round trip via ``gradio_client``, using a stub model so
   no SAM 2 weights or GPU are required.
"""

from __future__ import annotations

import importlib.util
import socket
import sys
import threading
import time
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np
import pytest

gr = pytest.importorskip("gradio", reason="demo tests need the 'demo' extra")

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "demo" / "app.py"

# `change` fires for user input AND programmatic updates; `input` only for user
# input. The image click handlers write into the same component, so only
# `input` is safe for the upload/reset listener.
USER_ONLY_TRIGGER = "input"


@pytest.fixture(scope="module")
def app_module() -> Any:
    """Import `demo/app.py` as a module. It is a script, not a package member."""
    spec = importlib.util.spec_from_file_location("lazysammy_demo_app", APP_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Register before exec so dataclasses can resolve the module namespace.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture
def sample_rgb() -> np.ndarray:
    """A small deterministic RGB image."""
    img = np.zeros((80, 100, 3), dtype=np.uint8)
    img[20:60, 30:70] = (200, 120, 40)
    return img


class _StubMask:
    """Minimal stand-in for `lazysammy.types.Mask`."""

    def __init__(self, data: np.ndarray, score: float = 0.9) -> None:
        self._data = data
        self.score = score
        self.area = int(data.sum())

    def numpy(self) -> np.ndarray:
        return self._data


class _StubPrediction:
    """Minimal stand-in for `lazysammy.types.ImagePrediction`."""

    def __init__(self, masks: list[_StubMask]) -> None:
        self.masks = masks

    @property
    def best_mask(self) -> _StubMask:
        return max(self.masks, key=lambda m: m.score)


class _StubAutoResult:
    """Minimal stand-in for the auto-segment result."""

    def __init__(self, masks: list[_StubMask]) -> None:
        self.masks = masks

    def filter_by_area(self, *, min_area: int) -> _StubAutoResult:
        return _StubAutoResult([m for m in self.masks if m.area >= min_area])

    def filter_by_iou(self, *, min_iou: float) -> _StubAutoResult:
        return _StubAutoResult([m for m in self.masks if m.score >= min_iou])

    def numpy(self) -> np.ndarray:
        return np.stack([m.numpy() for m in self.masks])


class _StubSAM:
    """Records calls and returns deterministic masks."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def _mask(self, shape: tuple[int, int]) -> _StubMask:
        data = np.zeros(shape, dtype=bool)
        data[20:60, 30:70] = True
        return _StubMask(data)

    def segment(self, image: np.ndarray, **kwargs: Any) -> _StubPrediction:
        self.calls.append(("segment", kwargs))
        return _StubPrediction([self._mask(image.shape[:2])])

    def segment_box(self, image: np.ndarray, *args: Any) -> _StubPrediction:
        self.calls.append(("segment_box", {"args": args}))
        return _StubPrediction([self._mask(image.shape[:2])])

    def auto_segment(self, image: np.ndarray) -> _StubAutoResult:
        self.calls.append(("auto_segment", {}))
        return _StubAutoResult([self._mask(image.shape[:2])])


@pytest.fixture
def stub_sam(app_module: Any, monkeypatch: pytest.MonkeyPatch) -> _StubSAM:
    """Replace the demo's model loader with a stub so no weights are needed."""
    stub = _StubSAM()
    monkeypatch.setattr(app_module, "_get_model", lambda _size: stub)
    return stub


# ---------------------------------------------------------------------------
# Prompt bookkeeping helpers
# ---------------------------------------------------------------------------


def test_point_click_appends_labeled_point(app_module: Any, sample_rgb: np.ndarray) -> None:
    state = app_module.apply_image_click(
        sample_rgb, app_module.ImagePrompts(), app_module.MODE_FOREGROUND, 40.0, 50.0
    )
    assert state.prompts.point is not None
    assert state.prompts.point.points == [[40.0, 50.0]]
    assert state.prompts.point.labels == [1]
    assert state.prompts.box_corners == []
    assert "fg" in state.display
    assert state.image is not None
    assert bool((state.image != sample_rgb).any())


def test_background_click_uses_label_zero(app_module: Any, sample_rgb: np.ndarray) -> None:
    state = app_module.apply_image_click(
        sample_rgb, app_module.ImagePrompts(), app_module.MODE_BACKGROUND, 10.0, 20.0
    )
    assert state.prompts.point is not None
    assert state.prompts.point.labels == [0]
    assert "bg" in state.display


def test_box_mode_two_clicks_defines_box(app_module: Any, sample_rgb: np.ndarray) -> None:
    state = app_module.apply_image_click(
        sample_rgb, app_module.ImagePrompts(), app_module.MODE_BOX, 10.0, 10.0
    )
    assert state.prompts.box_corners == [[10.0, 10.0]]

    state = app_module.apply_image_click(sample_rgb, state.prompts, app_module.MODE_BOX, 60.0, 70.0)
    assert state.prompts.box_corners == [[10.0, 10.0], [60.0, 70.0]]
    assert state.prompts.point is None
    assert "box" in state.display.lower()


def test_box_mode_third_click_starts_new_box(app_module: Any, sample_rgb: np.ndarray) -> None:
    prompts = app_module.ImagePrompts(box_corners=[[10.0, 10.0], [60.0, 70.0]])
    state = app_module.apply_image_click(sample_rgb, prompts, app_module.MODE_BOX, 5.0, 5.0)
    assert state.prompts.box_corners == [[5.0, 5.0]]


def test_click_before_upload_is_a_noop(app_module: Any) -> None:
    state = app_module.apply_image_click(
        None, app_module.ImagePrompts(), app_module.MODE_FOREGROUND, 1.0, 2.0
    )
    assert state.image is None
    assert state.prompts.point is None
    assert state.prompts.box_corners == []
    assert state.display == ""


def test_undo_removes_last_point(app_module: Any) -> None:
    prompts = app_module.ImagePrompts(
        point=app_module.PointPrompt(0, 0, [[1.0, 2.0], [3.0, 4.0]], [1, 1])
    )
    state = app_module.undo_last_prompt(None, prompts, app_module.MODE_FOREGROUND)
    assert state.prompts.point is not None
    assert state.prompts.point.points == [[1.0, 2.0]]


def test_undo_on_empty_state_is_safe(app_module: Any) -> None:
    state = app_module.undo_last_prompt(None, app_module.ImagePrompts(), app_module.MODE_FOREGROUND)
    assert state.prompts.point is None
    assert state.prompts.box_corners == []
    assert state.display == ""


def test_clear_prompts_resets_state_and_copies_image(
    app_module: Any, sample_rgb: np.ndarray
) -> None:
    state = app_module.clear_prompts(sample_rgb)
    assert state.prompts.point is None
    assert state.prompts.box_corners == []
    assert state.display == ""
    assert state.image is not None
    assert state.image is not sample_rgb


def test_clear_prompts_tolerates_missing_image(app_module: Any) -> None:
    state = app_module.clear_prompts(None)
    assert state.image is None
    assert state.prompts.point is None
    assert state.prompts.box_corners == []


def test_segment_prefers_box_when_two_corners_present(app_module: Any) -> None:
    prompts = app_module.ImagePrompts(
        point=app_module.PointPrompt(0, 0, [[1.0, 2.0]], [1]),
        box_corners=[[10.0, 10.0], [60.0, 70.0]],
    )
    prompt = app_module.select_segmentation_mode(prompts)
    assert isinstance(prompt, app_module.BoxPrompt)
    assert prompt.box == [10.0, 10.0, 60.0, 70.0]


def test_segment_uses_points_when_no_box(app_module: Any) -> None:
    prompts = app_module.ImagePrompts(point=app_module.PointPrompt(0, 0, [[1.0, 2.0]], [1]))
    prompt = app_module.select_segmentation_mode(prompts)
    assert isinstance(prompt, app_module.PointPrompt)


def test_segment_returns_none_without_prompts(app_module: Any) -> None:
    assert app_module.select_segmentation_mode(app_module.ImagePrompts()) is None


# ---------------------------------------------------------------------------
# The regression: event wiring in the built Gradio app
# ---------------------------------------------------------------------------


def _trigger_map(config: dict[str, Any]) -> dict[int, set[str]]:
    """Map component id -> set of user-facing trigger names it exposes."""
    triggers: dict[int, set[str]] = {}
    for dep in config["dependencies"]:
        for target in dep.get("targets") or []:
            component_id, trigger = target
            triggers.setdefault(component_id, set()).add(trigger)
    return triggers


def _component_id(config: dict[str, Any], label: str) -> int:
    for component in config["components"]:
        if component.get("props", {}).get("label") == label:
            return int(component["id"])
    raise AssertionError(f"no component labelled {label!r}")


def test_upload_listener_never_uses_change(app_module: Any) -> None:
    """Regression: `.change()` on the clickable image cleared prompts on every click.

    The click handler returns the annotated image into ``img_input``. Because
    ``change`` also fires for programmatic updates, wiring the upload/reset
    handler with ``change`` made each click wipe ``points_state``/``box_state``
    before Segment could read them.
    """
    config = app_module.build_app().get_config_file()
    triggers = _trigger_map(config)

    image_id = _component_id(config, "Input Image: click to add prompts")
    image_triggers = triggers.get(image_id, set())

    assert "select" in image_triggers, "clicking the image must be wired to select"
    assert USER_ONLY_TRIGGER in image_triggers, "uploads must be detected via input"
    assert "change" not in image_triggers, (
        "`change` fires for programmatic updates; the click handler writes into this "
        "component, so `change` here resets the prompt state on every click"
    )


def test_video_preview_is_wired_for_select(app_module: Any) -> None:
    config = app_module.build_app().get_config_file()
    assert "select" in _trigger_map(config).get(
        _component_id(config, "Frame Preview: click to add prompts"), set()
    )


def test_frame_slider_avoids_change(app_module: Any) -> None:
    """The slider redraws the preview, so it must not use the update-sensitive event."""
    config = app_module.build_app().get_config_file()
    slider_triggers = _trigger_map(config).get(_component_id(config, "Frame"), set())
    assert USER_ONLY_TRIGGER in slider_triggers
    assert "change" not in slider_triggers


def test_build_app_returns_gradio_blocks(app_module: Any) -> None:
    assert isinstance(app_module.build_app(), gr.Blocks)


# ---------------------------------------------------------------------------
# Segment / auto-segment pipelines (stubbed model)
# ---------------------------------------------------------------------------


def test_segment_with_points_reports_score_and_area(
    app_module: Any, stub_sam: _StubSAM, sample_rgb: np.ndarray
) -> None:
    overlay, info = app_module._segment_with_points(sample_rgb, [[40.0, 50.0, 1]], "tiny", True)
    assert overlay is not None
    assert "IoU" in info
    assert "area" in info
    assert stub_sam.calls
    assert stub_sam.calls[0][0] == "segment"


def test_segment_with_points_asks_for_an_image(app_module: Any) -> None:
    overlay, info = app_module._segment_with_points(None, [[1.0, 2.0, 1]], "tiny", True)
    assert overlay is None
    assert "Upload an image" in info


def test_segment_with_points_prompts_for_clicks(app_module: Any, sample_rgb: np.ndarray) -> None:
    _, info = app_module._segment_with_points(sample_rgb, [], "tiny", True)
    assert "Click on the image" in info


def test_segment_with_box_requires_two_corners(app_module: Any, sample_rgb: np.ndarray) -> None:
    _, info = app_module._segment_with_box(sample_rgb, [10.0, 10.0], "tiny")
    assert "two corners" in info


def test_auto_segment_reports_mask_count(
    app_module: Any, stub_sam: _StubSAM, sample_rgb: np.ndarray
) -> None:
    overlay, info = app_module._auto_segment_image(sample_rgb, "tiny", 0, 0.0)
    assert overlay is not None
    assert "Found 1 masks" in info


def test_auto_segment_honours_filters(
    app_module: Any, stub_sam: _StubSAM, sample_rgb: np.ndarray
) -> None:
    _, info = app_module._auto_segment_image(sample_rgb, "tiny", 10_000_000, 0.0)
    assert "No masks found" in info


# ---------------------------------------------------------------------------
# End-to-end through a real Gradio server and the official client
# ---------------------------------------------------------------------------


def _free_port() -> int:
    with closing(socket.socket()) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def served_demo(app_module: Any, stub_sam: _StubSAM) -> Any:
    """Launch the demo on a free port and yield a connected `gradio_client`."""
    gradio_client = pytest.importorskip("gradio_client")
    port = _free_port()
    demo = app_module.build_app()
    demo.launch(
        server_name="127.0.0.1",
        server_port=port,
        prevent_thread_lock=True,
        quiet=True,
    )
    try:
        client = gradio_client.Client(f"http://127.0.0.1:{port}/", verbose=False)
        # A launch is asynchronous; poll until the API surface is readable.
        for _ in range(60):
            try:
                client.view_api(return_format="dict")
                break
            except Exception:
                time.sleep(0.5)
        else:
            pytest.fail("demo server did not become ready")
        yield client
    finally:
        demo.close()
        time.sleep(0.2)


@pytest.mark.slow
def test_client_can_reach_segment_endpoint(served_demo: Any) -> None:
    """The Segment endpoint is reachable.

    With no image in the session state it must explain that an upload is needed
    rather than raising. This exercises the exact endpoint that silently failed
    in the regression, because the prompt state had been wiped by each click.
    """
    _, info = served_demo.predict("tiny", True, api_name="/_segment")
    assert "Upload an image first." in info


@pytest.mark.slow
def test_client_can_run_auto_segment(served_demo: Any, sample_rgb: np.ndarray) -> None:
    """Auto Segment works over the wire with the stubbed model."""
    from gradio_client import handle_file
    from PIL import Image

    tmp = REPO_ROOT / ".pytest_demo_input.png"
    try:
        Image.fromarray(sample_rgb).save(tmp)
        _, info = served_demo.predict(
            handle_file(str(tmp)), "tiny", 0, 0.0, api_name="/_auto_segment_image"
        )
        assert "masks" in info
    finally:
        tmp.unlink(missing_ok=True)


@pytest.mark.slow
def test_upload_endpoint_resets_prompts_and_returns_instruction(
    served_demo: Any, app_module: Any, sample_rgb: np.ndarray
) -> None:
    """Uploading an image resets the click instruction to the foreground mode.

    The handler also clears the prompt states; only the instruction component is
    observable over the API, so that is what we assert.
    """
    from gradio_client import handle_file
    from PIL import Image

    tmp = REPO_ROOT / ".pytest_demo_upload.png"
    try:
        Image.fromarray(sample_rgb).save(tmp)
        result = served_demo.predict(
            handle_file(str(tmp)),
            app_module.MODE_FOREGROUND,
            api_name="/_on_image_upload",
        )
        instruction = result[-1] if isinstance(result, tuple) else result
        assert "foreground" in instruction
    finally:
        tmp.unlink(missing_ok=True)


def test_thread_safety_locks_exist(app_module: Any) -> None:
    """Model loading and inference are serialised; SAM 2 predictors are not thread-safe."""
    lock_type = type(threading.Lock())
    assert isinstance(app_module._MODEL_LOAD_LOCK, lock_type)
    assert isinstance(app_module._INFERENCE_LOCK, lock_type)


# ---------------------------------------------------------------------------
# SAM 3 concept segmentation and text prompts
# ---------------------------------------------------------------------------


class _StubConceptPrediction:
    """Minimal stand-in for `lazysammy.types.ConceptPrediction`."""

    def __init__(self, masks: list[_StubMask], boxes: list[list[float]]) -> None:
        self.masks = masks
        self.boxes = boxes

    def filter_by_score(self, min_score: float) -> _StubConceptPrediction:
        kept = [
            (m, b) for m, b in zip(self.masks, self.boxes, strict=False) if m.score >= min_score
        ]
        return _StubConceptPrediction([m for m, _ in kept], [b for _, b in kept])

    def numpy(self) -> np.ndarray:
        return np.stack([m.numpy() for m in self.masks])


class _StubSAM3:
    """Records calls and returns deterministic concept detections."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def segment_text(self, image: np.ndarray, text: str, **kwargs: Any) -> _StubConceptPrediction:
        self.calls.append(("segment_text", {"text": text, **kwargs}))
        shape = image.shape[:2]
        masks = [_StubMask(np.ones(shape, dtype=bool), score=0.9)]
        return _StubConceptPrediction(masks, [[10.0, 10.0, 40.0, 40.0]])


@pytest.fixture
def stub_sam3(app_module: Any, monkeypatch: pytest.MonkeyPatch) -> _StubSAM3:
    """Replace the demo's SAM 3 loader with a stub."""
    stub = _StubSAM3()
    monkeypatch.setattr(app_module, "_get_sam3_model", lambda: stub)
    return stub


def test_concept_segment_reports_instance_count(
    app_module: Any, stub_sam3: _StubSAM3, sample_rgb: np.ndarray
) -> None:
    overlay, info = app_module._concept_segment_image(sample_rgb, "a player in white", 0.0)
    assert overlay is not None
    assert "1 instance(s)" in info
    assert stub_sam3.calls[0][0] == "segment_text"
    assert stub_sam3.calls[0][1]["text"] == "a player in white"


def test_concept_segment_requires_text(app_module: Any, sample_rgb: np.ndarray) -> None:
    _, info = app_module._concept_segment_image(sample_rgb, "   ", 0.0)
    assert "Type a concept" in info


def test_concept_segment_requires_image(app_module: Any) -> None:
    overlay, info = app_module._concept_segment_image(None, "cat", 0.0)
    assert overlay is None
    assert "Upload an image" in info


def test_concept_segment_reports_missing_sam3(
    app_module: Any, monkeypatch: pytest.MonkeyPatch, sample_rgb: np.ndarray
) -> None:
    def _raise() -> Any:
        raise ImportError("no sam3")

    monkeypatch.setattr(app_module, "_get_sam3_model", _raise)
    _, info = app_module._concept_segment_image(sample_rgb, "cat", 0.0)
    assert "SAM 3 is not installed" in info


def test_add_text_prompt_appends_concept(app_module: Any, sample_rgb: np.ndarray) -> None:
    _, prompts, _, display = app_module._add_text_prompt(sample_rgb, [], [], 0, "person")
    assert len(prompts) == 1
    assert isinstance(prompts[0], app_module.TextPrompt)
    assert prompts[0].text == "person"
    assert "person" in display


def test_add_text_prompt_ignores_blank(app_module: Any, sample_rgb: np.ndarray) -> None:
    _, prompts, _, _ = app_module._add_text_prompt(sample_rgb, [], [], 0, "  ")
    assert prompts == []


def test_text_prompt_summary_and_draw(app_module: Any, sample_rgb: np.ndarray) -> None:
    prompt = app_module.TextPrompt(0, "person")
    assert "person" in prompt.summary_html()
    drawn = sample_rgb.copy()
    prompt.draw(drawn, (255, 255, 255))
    assert bool((drawn != sample_rgb).any())


def test_video_prompts_html_counts_concepts(app_module: Any) -> None:
    prompts = [app_module.TextPrompt(0, "person"), app_module.PointPrompt(0, 1, [[1, 1]], [1])]
    html = app_module._format_prompts_html(prompts)
    assert "1 concept(s)" in html
    assert "1 point(s)" in html


def test_track_uses_sam3_when_text_prompt_present(
    app_module: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A text prompt routes tracking to SAM 3 instead of SAM 2."""
    calls: dict[str, Any] = {}

    class _StubSession:
        def add_text(self, frame_idx: int, text: str) -> None:
            calls["text"] = text

        def propagate(self, direction: str = "both") -> Any:
            calls["direction"] = direction
            return _StubVideoResults()

        def close(self) -> None:
            calls["closed"] = True

    class _StubSAM3Video:
        def video(self, frames_dir: Any, **kwargs: Any) -> _StubSession:
            calls["video"] = frames_dir
            return _StubSession()

    monkeypatch.setattr(app_module, "_get_sam3_model", lambda: _StubSAM3Video())
    monkeypatch.setattr(app_module, "save_video_overlay_mp4", lambda *a, **k: None)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    prompts = [app_module.TextPrompt(0, "person")]
    out, info = app_module._track_and_render(str(frames_dir), prompts, "small", 24.0, 0.5, False)

    assert out is not None
    assert "SAM 3" in info
    assert calls["text"] == "person"
    assert calls["closed"] is True


class _StubVideoResults:
    """Minimal VideoResults stand-in for the tracking test."""

    def __init__(self) -> None:
        self.object_ids = {1, 2}

    def __len__(self) -> int:
        return 3
