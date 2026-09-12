"""Tests for the unified serialization layer (``lazysammy.io``).

These verify the *format* contract of PNG / NPY / COCO-RLE output for all
three result types, without loading SAM 2.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lazysammy.io import (
    SaveFormat,
    save_auto_mask_result,
    save_image_prediction,
    save_video_results,
)
from lazysammy.types import (
    AutoMask,
    AutoMaskResult,
    FrameMasks,
    ImagePrediction,
    Mask,
    VideoResults,
)


def _mask(offset: int = 0) -> np.ndarray:
    m = np.zeros((20, 20), dtype=bool)
    m[2 + offset : 8 + offset, 2 + offset : 8 + offset] = True
    return m


def _prediction() -> ImagePrediction:
    masks = [
        Mask(data=_mask(0), score=0.9),
        Mask(data=_mask(1), score=0.5),
    ]
    return ImagePrediction(masks=masks, image_shape=(20, 20))


def _auto_result() -> AutoMaskResult:
    masks = [
        AutoMask(
            data=_mask(0),
            score=0.9,
            area=int(_mask(0).sum()),
            bbox=[2, 2, 6, 6],
            stability_score=0.95,
            crop_box=[0, 0, 20, 20],
        )
    ]
    return AutoMaskResult(masks=masks, image_shape=(20, 20))


def _video_results() -> VideoResults:
    return VideoResults(
        frames=[
            FrameMasks(frame_idx=0, masks={1: _mask(0), 2: _mask(1)}),
            FrameMasks(frame_idx=1, masks={1: _mask(1)}),
        ],
        num_frames=2,
    )


class TestSaveImagePrediction:
    def test_png_creates_one_file_per_mask(self, tmp_path: Path) -> None:
        out = save_image_prediction(_prediction(), tmp_path / "png", fmt="png")
        assert sorted(p.name for p in out.glob("*.png")) == [
            "mask_0000.png",
            "mask_0001.png",
        ]

    def test_npy_roundtrip_preserves_values(self, tmp_path: Path) -> None:
        out = save_image_prediction(_prediction(), tmp_path / "npy", fmt="npy")
        loaded = np.load(out / "mask_0000.npy")
        np.testing.assert_array_equal(loaded, _mask(0))

    def test_coco_rle_contains_size_and_counts(self, tmp_path: Path) -> None:
        out = save_image_prediction(_prediction(), tmp_path / "rle", fmt="coco_rle")
        data = json.loads((out / "mask_0000.json").read_text())
        assert "counts" in data
        assert "size" in data
        assert isinstance(data["counts"], str)


class TestSaveAutoMaskResult:
    def test_png_naming_uses_auto_mask_prefix(self, tmp_path: Path) -> None:
        out = save_auto_mask_result(_auto_result(), tmp_path / "auto")
        assert [p.name for p in out.glob("*.png")] == ["auto_mask_0000.png"]

    def test_accepts_save_format_enum(self, tmp_path: Path) -> None:
        out = save_auto_mask_result(_auto_result(), tmp_path / "enum", fmt=SaveFormat.NPY)
        assert (out / "auto_mask_0000.npy").exists()


class TestSaveVideoResults:
    def test_directory_layout_is_frame_major(self, tmp_path: Path) -> None:
        out = save_video_results(_video_results(), tmp_path / "video")
        assert (out / "frame_000000" / "obj_0001.png").exists()
        assert (out / "frame_000000" / "obj_0002.png").exists()
        assert (out / "frame_000001" / "obj_0001.png").exists()

    def test_empty_results_create_only_root(self, tmp_path: Path) -> None:
        out = save_video_results(VideoResults(frames=[], num_frames=0), tmp_path / "empty")
        assert out.is_dir()
        assert list(out.iterdir()) == []

    def test_video_npy_roundtrip(self, tmp_path: Path) -> None:
        out = save_video_results(_video_results(), tmp_path / "vnpy", fmt="npy")
        loaded = np.load(out / "frame_000000" / "obj_0001.npy")
        np.testing.assert_array_equal(loaded, _mask(0))


class TestSaveFormatValidation:
    def test_unknown_format_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported save format"):
            save_image_prediction(_prediction(), tmp_path, fmt="jpeg")

    def test_format_is_case_insensitive(self, tmp_path: Path) -> None:
        out = save_image_prediction(_prediction(), tmp_path / "upper", fmt="PNG")
        assert (out / "mask_0000.png").exists()
