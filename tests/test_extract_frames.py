"""Tests for frame extraction and image loading utilities."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from lazysammy.utils import (
    combine_masks,
    extract_frames,
    is_video_file,
    load_image,
    mask_iou,
    mask_to_bbox,
    masks_to_colored_overlay,
    masks_to_rle,
)


class TestIsVideoFile:
    @pytest.mark.parametrize("name", ["a.mp4", "a.avi", "a.mov", "a.MKV", "a.webm", "a.m4v"])
    def test_recognises_video_extensions(self, name: str) -> None:
        assert is_video_file(name)

    @pytest.mark.parametrize("name", ["a.jpg", "a.png", "a.txt", "no_ext"])
    def test_rejects_non_video_extensions(self, name: str) -> None:
        assert not is_video_file(name)


class TestExtractFrames:
    def test_extracts_all_frames(self, tmp_video_file: Path, tmp_path: Path) -> None:
        out = extract_frames(tmp_video_file, tmp_path / "frames")
        frames = sorted(out.glob("*.jpg"))
        assert len(frames) == 8

    def test_every_n_subsamples(self, tmp_video_file: Path, tmp_path: Path) -> None:
        out = extract_frames(tmp_video_file, tmp_path / "n2", every_n=2)
        assert len(list(out.glob("*.jpg"))) == 4

    def test_max_frames_limits_output(self, tmp_video_file: Path, tmp_path: Path) -> None:
        out = extract_frames(tmp_video_file, tmp_path / "max", max_frames=3)
        assert len(list(out.glob("*.jpg"))) == 3

    def test_frames_are_zero_padded(self, tmp_video_file: Path, tmp_path: Path) -> None:
        out = extract_frames(tmp_video_file, tmp_path / "pad")
        names = sorted(p.name for p in out.glob("*.jpg"))
        assert names[0] == "00000.jpg"
        assert "00007.jpg" in names

    def test_default_output_dir_is_sibling(self, tmp_video_file: Path) -> None:
        out = extract_frames(tmp_video_file)
        assert out == tmp_video_file.parent / f"{tmp_video_file.stem}_frames"
        assert out.is_dir()

    def test_missing_video_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            extract_frames(tmp_path / "nope.mp4")

    def test_invalid_parameters_raise(self, tmp_video_file: Path) -> None:
        with pytest.raises(ValueError, match="every_n"):
            extract_frames(tmp_video_file, every_n=0)
        with pytest.raises(ValueError, match="max_frames"):
            extract_frames(tmp_video_file, max_frames=-1)


class TestLoadImage:
    def test_accepts_rgb_array(self, small_image: np.ndarray) -> None:
        out = load_image(small_image)
        assert out.shape == small_image.shape
        assert out.dtype == np.uint8

    def test_converts_grayscale_to_rgb(self) -> None:
        gray = np.zeros((10, 10), dtype=np.uint8)
        out = load_image(gray)
        assert out.shape == (10, 10, 3)

    def test_converts_rgba_to_rgb(self) -> None:
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        out = load_image(rgba)
        assert out.shape == (10, 10, 3)

    def test_reads_bgr_file_as_rgb(self, tmp_path: Path) -> None:
        # Write a pure-red BGR image; it must load as pure red in RGB.
        bgr = np.zeros((8, 8, 3), dtype=np.uint8)
        bgr[:, :] = (0, 0, 255)
        path = tmp_path / "red.png"
        cv2.imwrite(str(path), bgr)

        out = load_image(path)
        np.testing.assert_array_equal(out[0, 0], np.array([255, 0, 0], dtype=np.uint8))

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            load_image(tmp_path / "nope.png")

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            load_image(np.zeros((2, 2, 2, 2), dtype=np.uint8))


class TestMaskHelpers:
    def test_overlay_colors_are_deterministic(self, small_image: np.ndarray) -> None:
        masks = np.stack([small_image[:, :, 0] > 200])
        a = masks_to_colored_overlay(small_image, masks, alpha=1.0)
        b = masks_to_colored_overlay(small_image, masks, alpha=1.0)
        np.testing.assert_array_equal(a, b)

    def test_combine_masks_returns_bool(self) -> None:
        m1 = np.zeros((4, 4), dtype=bool)
        m1[0, 0] = True
        m2 = np.zeros((4, 4), dtype=bool)
        m2[1, 1] = True
        combined = combine_masks(np.stack([m1, m2]))
        assert combined.dtype == bool
        assert combined.sum() == 2

    def test_mask_to_bbox_matches_manual_computation(self) -> None:
        m = np.zeros((50, 50), dtype=bool)
        m[5:15, 10:25] = True
        assert mask_to_bbox(m) == [10, 5, 24, 14]

    def test_mask_iou_half_overlap(self) -> None:
        a = np.zeros((4, 4), dtype=bool)
        a[:, :2] = True
        b = np.zeros((4, 4), dtype=bool)
        b[:, 1:3] = True
        # intersection = 4, union = 12
        assert mask_iou(a, b) == pytest.approx(4 / 12)

    def test_mask_iou_both_empty_is_zero(self) -> None:
        empty = np.zeros((4, 4), dtype=bool)
        assert mask_iou(empty, empty) == 0.0

    def test_masks_to_rle_encodes_each_mask(self) -> None:
        masks = np.stack([np.ones((4, 4), dtype=bool), np.zeros((4, 4), dtype=bool)])
        rles = masks_to_rle(masks)
        assert len(rles) == 2
        assert all(isinstance(r["counts"], str) for r in rles)
