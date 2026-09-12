"""Tests for visualization and overlay helpers (no SAM 2 required)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lazysammy.types import FrameMasks, VideoResults
from lazysammy.visualization import (
    draw_box_on_image,
    draw_masks_on_image,
    draw_points_on_image,
    save_video_overlay,
    save_video_overlay_mp4,
)


@pytest.fixture
def frames_dir(tmp_path: Path, small_image: np.ndarray) -> Path:
    """Write three frames using *zero-padded* names."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for i in range(3):
        img = small_image.copy()
        img[0, 0] = i * 40
        cv2.imwrite(str(frames / f"{i:05d}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    return frames


def _mask() -> np.ndarray:
    m = np.zeros((64, 64), dtype=bool)
    m[20:40, 20:40] = True
    return m


class TestDrawMasksOnImage:
    def test_returns_same_shape_and_dtype(self, small_image: np.ndarray) -> None:
        out = draw_masks_on_image(small_image, _mask()[None])
        assert out.shape == small_image.shape
        assert out.dtype == np.uint8

    def test_does_not_mutate_input(self, small_image: np.ndarray) -> None:
        original = small_image.copy()
        draw_masks_on_image(small_image, _mask()[None])
        np.testing.assert_array_equal(small_image, original)

    def test_draws_something_on_mask_area(self, small_image: np.ndarray) -> None:
        out = draw_masks_on_image(small_image, _mask()[None], alpha=1.0)
        # Inside the mask must differ from the source; outside must not.
        assert not np.array_equal(out[30, 30], small_image[30, 30])
        np.testing.assert_array_equal(out[0, 0], small_image[0, 0])

    def test_handles_multiple_masks(self, small_image: np.ndarray) -> None:
        masks = np.stack([_mask(), np.roll(_mask(), 5, axis=1)])
        out = draw_masks_on_image(small_image, masks)
        assert out.shape == small_image.shape

    def test_accepts_image_path(self, tmp_path: Path, small_image: np.ndarray) -> None:
        path = tmp_path / "img.png"
        cv2.imwrite(str(path), cv2.cvtColor(small_image, cv2.COLOR_RGB2BGR))
        out = draw_masks_on_image(path, _mask()[None])
        assert out.shape == small_image.shape

    def test_empty_masks_returns_image_copy(self, small_image: np.ndarray) -> None:
        empty = np.zeros((0, 64, 64), dtype=bool)
        out = draw_masks_on_image(small_image, empty)
        np.testing.assert_array_equal(out, small_image)

    def test_custom_colors_are_used(self, small_image: np.ndarray) -> None:
        out = draw_masks_on_image(small_image, _mask()[None], alpha=1.0, colors=[(255, 0, 0)])
        np.testing.assert_array_equal(out[30, 30], np.array([255, 0, 0], dtype=np.uint8))


class TestDrawPointsAndBox:
    def test_points_drawn_in_both_colors(self, small_image: np.ndarray) -> None:
        points = np.array([[10, 10], [40, 40]], dtype=np.float32)
        labels = np.array([1, 0])
        out = draw_points_on_image(small_image, points, labels)
        assert not np.array_equal(out[10, 10], small_image[10, 10])
        assert not np.array_equal(out[40, 40], small_image[40, 40])

    def test_box_is_drawn(self, small_image: np.ndarray) -> None:
        out = draw_box_on_image(small_image, [5, 5, 50, 50])
        assert not np.array_equal(out[5, 5], small_image[5, 5])

    def test_box_does_not_mutate_input(self, small_image: np.ndarray) -> None:
        original = small_image.copy()
        draw_box_on_image(small_image, [5, 5, 50, 50])
        np.testing.assert_array_equal(small_image, original)


class TestSaveVideoOverlayFrames:
    def test_writes_one_png_per_frame(self, frames_dir: Path, tmp_path: Path) -> None:
        results = VideoResults(
            frames=[FrameMasks(frame_idx=i, masks={1: _mask()}) for i in range(3)],
            num_frames=3,
        )
        out = save_video_overlay(frames_dir, results, tmp_path / "overlay")
        assert sorted(p.name for p in out.glob("*.png")) == [
            "frame_000000.png",
            "frame_000001.png",
            "frame_000002.png",
        ]

    def test_skips_frame_indices_beyond_available_files(
        self, frames_dir: Path, tmp_path: Path
    ) -> None:
        results = VideoResults(
            frames=[
                FrameMasks(frame_idx=0, masks={1: _mask()}),
                FrameMasks(frame_idx=99, masks={1: _mask()}),
            ],
            num_frames=100,
        )
        out = save_video_overlay(frames_dir, results, tmp_path / "partial")
        assert len(list(out.glob("*.png"))) == 1


class TestSaveVideoOverlayMp4:
    def test_writes_playable_video_with_all_frames(self, frames_dir: Path, tmp_path: Path) -> None:
        results = VideoResults(
            frames=[FrameMasks(frame_idx=0, masks={1: _mask()})],
            num_frames=3,
        )
        out = save_video_overlay_mp4(frames_dir, results, tmp_path / "out.mp4")

        assert out.exists()
        cap = cv2.VideoCapture(str(out))
        try:
            assert cap.isOpened()
            # Frames without results are still written, so all 3 appear.
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        finally:
            cap.release()

    def test_rejects_bad_codec_length(self, frames_dir: Path, tmp_path: Path) -> None:
        results = VideoResults(frames=[], num_frames=3)
        with pytest.raises(ValueError, match="FourCC"):
            save_video_overlay_mp4(frames_dir, results, tmp_path / "bad.mp4", codec="mp4")

    def test_raises_when_no_frames_present(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        results = VideoResults(frames=[], num_frames=0)
        with pytest.raises(ValueError, match="No image files"):
            save_video_overlay_mp4(empty, results, tmp_path / "out.mp4")
