"""Unit tests for image segmentation helpers that do not require loading SAM2."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from lazysammy.image import ImageSegmenter
from lazysammy.utils import ImageEmbeddingCache


class _DummyPredictor:
    """Minimal predictor stub for testing ``segment_multi_box``."""

    def __init__(self) -> None:
        self.set_image_calls = 0
        self.predict_boxes: list[np.ndarray] = []

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1

    def predict(
        self,
        *,
        point_coords: object,
        point_labels: object,
        box: np.ndarray,
        mask_input: object,
        multimask_output: bool,
        return_logits: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        del point_coords, point_labels, mask_input, multimask_output, return_logits
        self.predict_boxes.append(box.copy())
        masks = np.ones((1, 4, 4), dtype=bool)
        scores = np.array([0.9], dtype=np.float32)
        logits = None
        return masks, scores, logits


class TestSegmentMultiBox:
    def test_uses_single_image_setup_and_normalizes_boxes(self) -> None:
        segmenter = ImageSegmenter.__new__(ImageSegmenter)
        segmenter._device = torch.device("cpu")
        segmenter._dtype = torch.bfloat16
        segmenter._image_cache = ImageEmbeddingCache()
        predictor = _DummyPredictor()
        segmenter._predictor = predictor

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        results = segmenter.segment_multi_box(
            image,
            boxes=[[10, 20, 1, 2], [3, 4, 7, 8]],
        )

        assert predictor.set_image_calls == 1
        assert len(predictor.predict_boxes) == 2
        np.testing.assert_array_equal(
            predictor.predict_boxes[0],
            np.array([1, 2, 10, 20], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            predictor.predict_boxes[1],
            np.array([3, 4, 7, 8], dtype=np.float32),
        )
        assert len(results) == 2
        assert results[0].best_mask.score == pytest.approx(0.9, rel=1e-6)

    def test_empty_boxes_returns_empty_list(self) -> None:
        segmenter = ImageSegmenter.__new__(ImageSegmenter)
        segmenter._device = torch.device("cpu")
        segmenter._dtype = torch.bfloat16
        segmenter._image_cache = ImageEmbeddingCache()
        segmenter._predictor = _DummyPredictor()

        image = np.zeros((8, 8, 3), dtype=np.uint8)
        assert segmenter.segment_multi_box(image, boxes=[]) == []


class _CountingPredictor:
    """Predictor stub that counts ``set_image`` calls."""

    def __init__(self) -> None:
        self.set_image_calls = 0
        self.predict_calls = 0

    def set_image(self, image: np.ndarray) -> None:
        self.set_image_calls += 1

    def predict(self, **kwargs: object) -> tuple[np.ndarray, np.ndarray, None]:
        self.predict_calls += 1
        return (
            np.ones((1, 4, 4), dtype=bool),
            np.array([0.9], dtype=np.float32),
            None,
        )


def _counting_segmenter() -> tuple[ImageSegmenter, _CountingPredictor]:
    segmenter = ImageSegmenter.__new__(ImageSegmenter)
    segmenter._device = torch.device("cpu")
    segmenter._image_cache = ImageEmbeddingCache()
    predictor = _CountingPredictor()
    segmenter._predictor = predictor
    return segmenter, predictor


class TestImageCaching:
    """Regression: every prompt re-ran the image encoder."""

    def test_repeated_prompts_on_same_array_encode_once(self) -> None:
        segmenter, predictor = _counting_segmenter()
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        segmenter.segment(image, points=[[1, 1]], labels=[1])
        segmenter.segment(image, points=[[2, 2]], labels=[1])
        segmenter.segment(image, points=[[3, 3]], labels=[1])

        assert predictor.set_image_calls == 1
        assert predictor.predict_calls == 3

    def test_different_arrays_re_encode(self) -> None:
        segmenter, predictor = _counting_segmenter()
        segmenter.segment(np.zeros((8, 8, 3), dtype=np.uint8), points=[[1, 1]], labels=[1])
        segmenter.segment(np.ones((8, 8, 3), dtype=np.uint8), points=[[1, 1]], labels=[1])
        assert predictor.set_image_calls == 2

    def test_set_image_then_predict_skips_encoding(self) -> None:
        segmenter, predictor = _counting_segmenter()
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        segmenter.set_image(image)
        segmenter.predict(points=[[1, 1]], labels=[1])
        segmenter.predict(points=[[2, 2]], labels=[1])

        assert predictor.set_image_calls == 1
        assert predictor.predict_calls == 2

    def test_predict_without_set_image_raises(self) -> None:
        segmenter, _ = _counting_segmenter()
        with pytest.raises(RuntimeError, match="set_image"):
            segmenter.predict(points=[[1, 1]], labels=[1])


class TestSegmentMultiPoint:
    def test_one_prediction_per_object(self) -> None:
        segmenter, predictor = _counting_segmenter()
        image = np.zeros((8, 8, 3), dtype=np.uint8)

        results = segmenter.segment_multi_point(
            image,
            points_per_object=[[[1, 1]], [[2, 2], [3, 3]]],
        )

        assert len(results) == 2
        assert predictor.set_image_calls == 1
        assert predictor.predict_calls == 2

    def test_empty_returns_empty(self) -> None:
        segmenter, _ = _counting_segmenter()
        assert segmenter.segment_multi_point(np.zeros((8, 8, 3), dtype=np.uint8), []) == []

    def test_mismatched_labels_raise(self) -> None:
        segmenter, _ = _counting_segmenter()
        with pytest.raises(ValueError, match="same length"):
            segmenter.segment_multi_point(
                np.zeros((8, 8, 3), dtype=np.uint8),
                points_per_object=[[[1, 1]], [[2, 2]]],
                labels_per_object=[[1]],
            )
