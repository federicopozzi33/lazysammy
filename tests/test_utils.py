"""Tests for utility functions – no SAM2 model required."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lazysammy2.utils import (
    auto_detect_device,
    combine_masks,
    list_frame_files,
    mask_iou,
    mask_to_bbox,
    save_masks_as_coco_rle,
    save_masks_as_npy,
    save_masks_as_png,
)


class TestAutoDetectDevice:
    """Device detection tests."""

    def test_returns_string(self) -> None:
        device = auto_detect_device()
        assert isinstance(device, str)
        assert device in {"cuda", "mps", "cpu"}


class TestListFrameFiles:
    """Frame listing tests."""

    def test_lists_jpg_frames(self, tmp_frames_dir: Path) -> None:
        frames = list_frame_files(tmp_frames_dir)
        assert len(frames) == 5
        # Should be sorted
        names = [f.name for f in frames]
        assert names == sorted(names)


class TestSaveMasks:
    """Mask saving tests."""

    def _masks_dict(self) -> dict[str, np.ndarray]:
        return {
            "obj0": np.ones((10, 10), dtype=bool),
            "obj1": np.zeros((10, 10), dtype=bool),
        }

    def test_save_as_png(self, tmp_path: Path) -> None:
        out = Path(str(tmp_path)) / "png"
        save_masks_as_png(self._masks_dict(), out)
        files = list(out.glob("*.png"))
        assert len(files) == 2

    def test_save_as_npy(self, tmp_path: Path) -> None:
        out = Path(str(tmp_path)) / "npy"
        save_masks_as_npy(self._masks_dict(), out)
        files = list(out.glob("*.npy"))
        assert len(files) == 2
        loaded = np.load(files[0])
        assert loaded.dtype == bool

    def test_save_as_coco_rle(self, tmp_path: Path) -> None:
        import json

        out = Path(str(tmp_path)) / "rle"
        save_masks_as_coco_rle(self._masks_dict(), out)
        files = list(out.glob("*.json"))
        assert len(files) == 2
        with open(files[0]) as f:
            data = json.load(f)
        assert "counts" in data
        assert "size" in data


class TestMaskOps:
    """Mask combination and IoU tests."""

    def test_combine_masks(self) -> None:
        m1 = np.zeros((10, 10), dtype=bool)
        m1[0:5, 0:5] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[5:10, 5:10] = True
        combined = combine_masks([m1, m2])
        assert combined.sum() == m1.sum() + m2.sum()

    def test_mask_to_bbox(self) -> None:
        m = np.zeros((100, 100), dtype=bool)
        m[10:30, 20:50] = True
        bbox = mask_to_bbox(m)
        assert bbox == [20, 10, 49, 29]

    def test_mask_iou_identical(self) -> None:
        m = np.ones((10, 10), dtype=bool)
        assert mask_iou(m, m) == 1.0

    def test_mask_iou_disjoint(self) -> None:
        m1 = np.zeros((10, 10), dtype=bool)
        m1[0:5] = True
        m2 = np.zeros((10, 10), dtype=bool)
        m2[5:10] = True
        assert mask_iou(m1, m2) == 0.0
