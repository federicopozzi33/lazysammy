"""Tests for the automatic mask generation layer, using a stub generator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from lazysammy.auto_mask import AutoSegmenter
from lazysammy.types import AutoMaskResult


class _StubGenerator:
    """Stub for ``SAM2AutomaticMaskGenerator`` returning fixed annotations."""

    def __init__(self, anns: list[dict[str, Any]] | None = None) -> None:
        self.generate_calls = 0
        self._anns = anns

    def generate(self, image: np.ndarray) -> list[dict[str, Any]]:
        self.generate_calls += 1
        if self._anns is not None:
            return self._anns
        h, w = image.shape[:2]
        small = np.zeros((h, w), dtype=bool)
        small[2:4, 2:4] = True
        large = np.zeros((h, w), dtype=bool)
        large[0 : h // 2, 0 : w // 2] = True
        return [
            {
                "segmentation": small,
                "predicted_iou": 0.7,
                "area": int(small.sum()),
                "bbox": [2, 2, 2, 2],
                "stability_score": 0.8,
                "point_coords": [[3, 3]],
                "crop_box": [0, 0, w, h],
            },
            {
                "segmentation": large,
                "predicted_iou": 0.95,
                "area": int(large.sum()),
                "bbox": [0, 0, w // 2, h // 2],
                "stability_score": 0.97,
                "point_coords": [[1, 1]],
                "crop_box": [0, 0, w, h],
            },
        ]


def _make_segmenter(
    anns: list[dict[str, Any]] | None = None,
) -> tuple[AutoSegmenter, _StubGenerator]:
    seg = AutoSegmenter.__new__(AutoSegmenter)
    gen = _StubGenerator(anns)
    seg._generator = gen
    seg._device = torch.device("cpu")
    seg._dtype = torch.float32
    return seg, gen


class TestGenerate:
    def test_returns_automask_result(self, small_image: np.ndarray) -> None:
        seg, gen = _make_segmenter()
        result = seg.generate(small_image)

        assert isinstance(result, AutoMaskResult)
        assert gen.generate_calls == 1
        assert result.image_shape == small_image.shape[:2]

    def test_masks_sorted_by_area_descending(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        areas = [m.area for m in result.masks]
        assert areas == sorted(areas, reverse=True)

    def test_segmentation_converted_to_bool(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        assert all(m.data.dtype == bool for m in result.masks)

    def test_scores_and_stability_preserved(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        top = result.masks[0]
        assert top.score == pytest.approx(0.95)
        assert top.stability_score == pytest.approx(0.97)

    def test_accepts_file_path(self, tmp_path: Path, small_image: np.ndarray) -> None:
        import cv2

        path = tmp_path / "img.png"
        cv2.imwrite(str(path), cv2.cvtColor(small_image, cv2.COLOR_RGB2BGR))

        seg, _ = _make_segmenter()
        result = seg.generate(path)
        assert len(result.masks) == 2

    def test_empty_annotations_produce_empty_result(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter(anns=[])
        result = seg.generate(small_image)
        assert result.masks == []
        assert result.numpy().shape == (0, *small_image.shape[:2])

    def test_non_array_segmentation_falls_back_to_empty(self, small_image: np.ndarray) -> None:
        anns = [
            {
                "segmentation": "not-an-array",
                "predicted_iou": 0.5,
                "area": 10,
                "bbox": [0, 0, 1, 1],
                "stability_score": 0.5,
                "point_coords": [[0, 0]],
                "crop_box": [0, 0, 4, 4],
            }
        ]
        seg, _ = _make_segmenter(anns=anns)
        result = seg.generate(small_image)
        assert len(result.masks) == 1
        assert result.masks[0].data.sum() == 0


class TestFiltering:
    def test_filter_by_area(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        filtered = result.filter_by_area(min_area=100)
        assert all(m.area >= 100 for m in filtered.masks)
        assert len(filtered.masks) == 1

    def test_filter_preserves_image_shape(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        assert result.filter_by_iou(0.9).image_shape == small_image.shape[:2]

    def test_filter_does_not_mutate_original(self, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        before = len(result.masks)
        result.filter_by_iou(0.99)
        assert len(result.masks) == before


class TestSaveAndProps:
    def test_save_writes_pngs(self, tmp_path: Path, small_image: np.ndarray) -> None:
        seg, _ = _make_segmenter()
        result = seg.generate(small_image)
        out = seg.save(result, tmp_path / "out", fmt="png")
        assert len(list(out.glob("*.png"))) == 2

    def test_device_property(self) -> None:
        seg, _ = _make_segmenter()
        assert seg.device == torch.device("cpu")

    def test_generator_property(self) -> None:
        seg, gen = _make_segmenter()
        assert seg.generator is gen
