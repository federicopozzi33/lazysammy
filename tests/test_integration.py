"""End-to-end integration tests exercising real SAM 2 inference.

These tests load actual SAM 2 weights, so they are marked ``integration`` and
excluded from the default run. Run them explicitly with::

    uv run pytest -m integration

The smallest model (``tiny``) is used by default; override with the
``LAZYSAMMY_TEST_MODEL`` environment variable. Set ``HF_HUB_OFFLINE=1`` to
require the weights to already be in the local HuggingFace cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from lazysammy import SAM2

pytestmark = [pytest.mark.integration, pytest.mark.slow]

# Reuse one model across every test in this module: loading is the slow part.
_MODEL_SIZE = os.environ.get("LAZYSAMMY_TEST_MODEL", "small")
_DEVICE = os.environ.get("LAZYSAMMY_TEST_DEVICE", "cpu")

_SQUARE = (slice(40, 90), slice(40, 80))  # 50 rows x 40 cols = 2000 px


@pytest.fixture(scope="module")
def sam() -> SAM2:
    """A SAM2 facade backed by real weights, shared across the module."""
    return SAM2(_MODEL_SIZE, device=_DEVICE)


@pytest.fixture
def square_image() -> np.ndarray:
    """A 128x128 image with a 50x40 red rectangle on a black background."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    img[_SQUARE] = (230, 60, 60)
    return img


@pytest.fixture
def square_frames(tmp_path: Path, square_image: np.ndarray) -> Path:
    """Six frames of a rectangle translating to the right."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(6):
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        x = 20 + i * 8
        img[40:90, x : x + 40] = (230, 60, 60)
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return frames


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def _rectangle_reference(shape: tuple[int, int] = (128, 128)) -> np.ndarray:
    """Full-frame boolean mask matching the fixture rectangle."""
    reference = np.zeros(shape, dtype=bool)
    reference[_SQUARE] = True
    return reference


class TestImageSegmentationEndToEnd:
    def test_point_prompt_segments_the_rectangle(self, sam: SAM2, square_image: np.ndarray) -> None:
        pred = sam.segment_point(square_image, x=60, y=65)

        assert len(pred) >= 1
        best = pred.best_mask
        assert best.score > 0.8
        # The model should recover the rectangle almost exactly.
        assert _iou(best.data, _rectangle_reference()) > 0.9

    def test_box_prompt_matches_rectangle(self, sam: SAM2, square_image: np.ndarray) -> None:
        pred = sam.segment_box(square_image, 40, 40, 80, 90)
        assert pred.best_mask.area == pytest.approx(2000, rel=0.05)

    def test_multi_box_returns_one_prediction_per_box(
        self, sam: SAM2, square_image: np.ndarray
    ) -> None:
        preds = sam.segment_multi_box(square_image, [[40, 40, 80, 90], [0, 0, 20, 20]])
        assert len(preds) == 2
        assert preds[0].best_mask.area > preds[1].best_mask.area

    def test_rejects_promptless_call(self, sam: SAM2, square_image: np.ndarray) -> None:
        with pytest.raises(ValueError, match="At least one prompt"):
            sam.segment(square_image)

    def test_accepts_file_path(self, sam: SAM2, tmp_path: Path, square_image: np.ndarray) -> None:
        path = tmp_path / "img.png"
        cv2.imwrite(str(path), cv2.cvtColor(square_image, cv2.COLOR_RGB2BGR))

        pred = sam.segment_point(path, x=60, y=65)
        assert pred.best_mask.area == pytest.approx(2000, rel=0.1)
        assert pred.image_shape == (128, 128)

    def test_mask_save_roundtrip(self, sam: SAM2, square_image: np.ndarray, tmp_path: Path) -> None:
        pred = sam.segment_point(square_image, x=60, y=65)
        out = tmp_path / "mask.png"
        pred.best_mask.save(out)

        reloaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
        assert reloaded is not None
        assert (reloaded > 0).sum() == pytest.approx(pred.best_mask.area, rel=0.02)


class TestRefinementEndToEnd:
    def test_refine_with_logits_preserves_region(self, sam: SAM2, square_image: np.ndarray) -> None:
        first = sam.segment(square_image, points=[[60, 65]], labels=[1])
        logits = first.best_mask.logits
        assert logits is not None
        assert logits.ndim == 2  # squeeze(0) applied by SAM2's predict()

        refined = sam.refine(square_image, logits, points=[[62, 66]], labels=[1])
        assert refined.best_mask.area == pytest.approx(2000, rel=0.1)

    def test_background_point_excludes_a_region(self, sam: SAM2, square_image: np.ndarray) -> None:
        """A label-0 click on a second blob must exclude that blob."""
        img = np.zeros((128, 128, 3), dtype=np.uint8)
        img[20:60, 20:60] = (230, 60, 60)  # blob A
        img[70:110, 70:110] = (60, 60, 230)  # blob B

        selected = sam.segment(
            img, points=[[40, 40], [90, 90]], labels=[1, 1], multimask_output=False
        ).best_mask
        excluded = sam.segment(
            img, points=[[40, 40], [90, 90]], labels=[1, 0], multimask_output=False
        ).best_mask

        assert selected.data[90, 90]  # blob B selected as foreground
        assert excluded.data[40, 40]  # blob A still selected
        assert not excluded.data[90, 90]  # blob B excluded
        assert excluded.area < selected.area


class TestAutoSegmentEndToEnd:
    def test_generates_at_least_one_mask(self, sam: SAM2, square_image: np.ndarray) -> None:
        result = sam.auto_segment(square_image, points_per_side=8)

        assert len(result) >= 1
        assert result.image_shape == (128, 128)
        assert all(m.data.dtype == bool for m in result)

    def test_filters_narrow_the_result(self, sam: SAM2, square_image: np.ndarray) -> None:
        result = sam.auto_segment(square_image, points_per_side=8)
        filtered = result.filter_by_iou(min_iou=0.99)
        assert len(filtered) <= len(result)


class TestVideoTrackingEndToEnd:
    def test_tracks_translating_rectangle(self, sam: SAM2, square_frames: Path) -> None:
        session = sam.video(square_frames)
        session.add_points(frame_idx=0, obj_id=1, points=[[40, 65]], labels=[1])
        results = session.propagate()

        assert len(results) == 6
        assert results.object_ids == {1}
        # The rectangle keeps constant area while translating.
        for _frame_idx, frame in results:
            assert frame.masks[1].sum() == pytest.approx(2000, rel=0.15)

    def test_bidirectional_covers_whole_clip(self, sam: SAM2, square_frames: Path) -> None:
        session = sam.video(square_frames)
        session.add_points(frame_idx=3, obj_id=1, points=[[64, 65]], labels=[1])
        results = session.propagate_bidirectional()

        tracked = [idx for idx, _ in results]
        assert tracked == sorted(tracked)
        assert tracked[0] == 0
        assert tracked[-1] == 5

    def test_reset_allows_reuse(self, sam: SAM2, square_frames: Path) -> None:
        session = sam.video(square_frames)
        session.add_points(frame_idx=0, obj_id=1, points=[[40, 65]], labels=[1])
        session.propagate()
        assert session.results is not None

        session.reset()
        assert session.results is None

        session.add_points(frame_idx=0, obj_id=2, points=[[40, 65]], labels=[1])
        second = session.propagate()
        assert second.object_ids == {2}

    def test_save_results_writes_frames(
        self, sam: SAM2, square_frames: Path, tmp_path: Path
    ) -> None:
        session = sam.video(square_frames)
        session.add_points(frame_idx=0, obj_id=1, points=[[40, 65]], labels=[1])
        session.propagate()

        out = session.save(tmp_path / "masks")
        assert (out / "frame_000000" / "obj_0001.png").exists()

    def test_overlay_mp4_is_written(self, sam: SAM2, square_frames: Path, tmp_path: Path) -> None:
        session = sam.video(square_frames)
        session.add_points(frame_idx=0, obj_id=1, points=[[40, 65]], labels=[1])
        session.propagate()

        out = session.save_overlay(tmp_path / "overlay.mp4", fps=6)
        assert out.exists()
        cap = cv2.VideoCapture(str(out))
        try:
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
        finally:
            cap.release()

    def test_video_file_input_extracts_frames(
        self, sam: SAM2, square_frames: Path, tmp_path: Path
    ) -> None:
        """Passing an .mp4 should trigger extraction and still track."""
        video = tmp_path / "clip.mp4"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter.fourcc(*"mp4v"), 6.0, (128, 128))
        if not writer.isOpened():  # pragma: no cover - codec availability
            pytest.skip("No MP4 codec available in this OpenCV build.")
        try:
            for path in sorted(square_frames.glob("*.jpg")):
                writer.write(cv2.imread(str(path)))
        finally:
            writer.release()

        session = sam.video(video, frames_dir=tmp_path / "extracted")
        assert session.num_frames == 6
        session.add_points(frame_idx=0, obj_id=1, points=[[40, 65]], labels=[1])
        assert len(session.propagate()) == 6
